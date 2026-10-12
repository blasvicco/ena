#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA vs. PPO / SAC on non-stationary HalfCheetah — same evaluation/stopping rule for all, frozen test (continuous control)."""

# Structurally parallel to cartpole_challenge.py by design (CLAUDE.md: same experiment lifecycle per domain).
# pylint: disable=duplicate-code

# General imports
import argparse
import functools
import os
import platform

# Libs imports
import gymnasium as gym
import tensorflow as tf
from keras import Input, Model, Sequential, initializers, ops, random
from keras.layers import Dense

# App imports
from agents import ENAgent, PPOAgent, SACAgent
from cartpole_ena_ablation import (
	MAX_WORKERS,
	ABLATION_VARIANTS,
	build_seeded,
	parse_args,
	post_update,
	run_study,
	seed_experiment,
	select_variants,
	train_and_test,
)
from tools.visualization import vprint

# Constants
NUM_EXPERIMENTS = 20  # author, 2026-10-12: 20 replicates in one go (detects d >= 1.0 at 80% power, alpha 0.05 / 2)
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "half_cheetah_outputs")
ENA_NAME = "ENA-Combined"
ENA_BRAIN_HIDDEN = (32, 16)  # ENA's hidden layers, frozen by the search tuning (see ENA_SETTINGS)

# The lifecycle of cartpole_challenge.py with one protocol: every agent trains FIXED_BUDGET_PER_ENV env-steps per
# environment (ENA's include its candidate evaluations), learning throughout, with no evaluation checks; then a
# frozen test on 5 worlds, 2 of them never trained (Ice-Floor, High-Wind). The analysis compares each environment's
# test return with the training return at the end of its phase (tools/cheetah_results.py).
SOLVED_SCORE = 4800.0  # gymnasium's registered reward_threshold for HalfCheetah-v5: a reference line, not a metric
FIXED_BUDGET_PER_ENV = 1_000_000
HEALTH_EVERY = 10_000  # env-steps between readings of each agent's internal health (health_{agent}.csv; SAC's alpha and Q)
TESTING_EPISODES = [50, 50, 50, 50, 50]
# Ignored (training length is the step budget); one entry per training world.
TRAINING_EPISODES = [500, 500, 500]
Y_LIM = (-1000, 12000)
AIR_DENSITY = 1.2  # kg/m^3, sea-level air; only windy worlds have a medium

# DQN and SAC-Discrete are discrete-action agents and have no place in this continuous domain.
CHALLENGE_AGENTS = [{"name": ENA_NAME}, {"name": "PPO"}, {"name": "SAC"}]


# Helper methods
def make_env(
	friction=0.4,
	gravity=-9.8,
	name="Earth",
	render_mode=None,
	wind=(0, 0, 0),
):
	"""Create a HalfCheetah-v5 world with custom physics (gravity, ground friction, wind)."""
	env = gym.make("HalfCheetah-v5", render_mode=render_mode)
	model = env.unwrapped.model

	# 1. Gravity (Z-axis) - Negative pulls DOWN
	model.opt.gravity[2] = gravity
	env.unwrapped.gravity = gravity  # Expose for ENAgent

	# 2. Friction: MuJoCo uses the larger of the two contacting geoms' sliding frictions, and every geom
	# of the stock model is at 0.4, so the friction is set on every geom (floor and cheetah alike).
	model.geom_friction[:, 0] = friction
	env.unwrapped.friction = friction  # Expose for ENAgent

	# 3. Wind (3D vector): it only acts through fluid forces, which need a medium; windy worlds get air
	# density, calm worlds stay vacuum like the stock model.
	model.opt.wind[:] = wind
	model.opt.density = AIR_DENSITY if any(wind) else 0.0
	env.unwrapped.wind = tuple(float(component) for component in wind)  # Expose for ENAgent (it re-creates worlds from these)

	env.unwrapped.name = name
	return env


def make_training_worlds() -> list:
	"""The 3 training worlds, in training order (author's definitions, 2026-10-11)."""
	return [
		# Earth's gravity and the stock friction, with a soft head wind (5 m/s, a quarter of High-Wind's).
		make_env(friction=0.4, gravity=-9.8, wind=(-5, 0, 0), name="Earth-Standard"),
		# 1.5x Earth's gravity, the top of the range HalfCheetah benchmarks use (CARL; multi-task HalfCheetah), on rough
		# ground. It replaced Jupiter (-24.8, about 2.5 g): beyond any published setting, and learned by too few agents
		# (ENA on about half of its seeds, SAC never) for forgetting, which needs a learned environment, to be measured.
		make_env(friction=1.0, gravity=-14.7, name="Heavy-Rough"),
		# Mars's gravity on a slippery floor: friction 0.2, half the stock 0.4 (0.1 let SAC exploit the physics, and lies
		# beyond the published training ranges; Packer et al. 2018 extrapolate down to 0.25x their default).
		make_env(friction=0.2, gravity=-3.7, name="Mars-Slippery"),
	]


def make_unseen_worlds() -> list:
	"""The 2 never-trained test worlds: Earth gravity with an icy floor, and Earth with a strong head wind."""
	return [
		make_env(friction=0.05, gravity=-9.8, name="Ice-Floor"),
		make_env(friction=0.4, gravity=-9.8, wind=(-20, 0, 0), name="High-Wind"),
	]


def make_test_worlds(worlds: list) -> list:
	"""The frozen test sequence: the training worlds interleaved with the never-trained ones."""
	ice_floor, high_wind = make_unseen_worlds()
	return [worlds[0], ice_floor, worlds[1], high_wind, worlds[2]]


class Normc(initializers.Initializer):
	"""normc (Salimans et al., 2017, evolution-strategies-starter's normc_initializer): Gaussian weights, each unit's
	incoming weight vector (a kernel column) rescaled to norm std. Draws follow Keras' global seed."""

	def __init__(self, std: float = 1.0):
		self.std = std

	def __call__(self, shape, dtype=None):
		weights = random.normal(shape, dtype=dtype)
		return weights * self.std / ops.sqrt(ops.sum(ops.square(weights), axis=0, keepdims=True))

	def get_config(self) -> dict:
		return {"std": self.std}


def ena_brain(action_dim: int, state_dim: int, hidden: int | tuple = ENA_BRAIN_HIDDEN) -> Sequential:
	"""Build a ENA-NN for continuous control: the ES MuJoCo policy network (Salimans et al., 2017), two tanh hidden layers
	with normc weights and zero biases, and a tanh output in the [-1, 1] action range. hidden is both layers' size, or one
	size per layer (32, 16).

	One deviation: the output layer keeps normc std 1.0 instead of their 0.01. ES perturbs a single central policy, so a
	near-still start costs it nothing; ENA's first population should already act, with varied gaits for selection to tell
	apart (on normalized observations, initial actions are about +-0.5: moving, not saturated)."""
	sizes = (hidden, hidden) if isinstance(hidden, int) else tuple(hidden)
	return Sequential(
		[
			Input(shape=(state_dim,)),
			*[Dense(units, activation="tanh", kernel_initializer=Normc(1.0)) for units in sizes],
			Dense(action_dim, activation="tanh", kernel_initializer=Normc(1.0)),
		]
	)


# Baseline settings: validated published configurations, never tuned by us.
#
# Source for both baselines: RL Baselines3 Zoo (Raffin, 2020), the tuned hyperparameters for
# Stable-Baselines3 (Raffin et al., JMLR 2021). Its benchmark (benchmark.md, HalfCheetah-v3, 1M steps,
# 150 evaluation episodes) reports a mean return of 5,819 +/- 664 for PPO and 9,535 +/- 100 for SAC with
# exactly these settings. We take them unchanged, so a reader can check that each baseline is
# competitive on HalfCheetah, and any advantage of ENA cannot come from a weakened baseline. The
# deviations, all forced by the non-stationary protocol or by the shared runner, are listed per agent.
#
# PPO (rl-baselines3-zoo hyperparams/ppo.yml, entry HalfCheetah):
#   - policy and value networks 256x256 ReLU, separate, without orthogonal initialization;
#   - n_steps 512, batch 64, 20 epochs, gamma 0.98, GAE lambda 0.92, clip 0.1;
#   - learning rate 2.0633e-5, entropy coefficient 4.01762e-4, value coefficient 0.58096;
#   - max gradient norm 0.8, initial log std -2;
#   - observations and rewards normalized (VecNormalize: reward scaled by the std of the discounted return).
#   Deviation: constant learning rate (the zoo's schedule is also constant), and no annealing to zero
#   at the end of the budget, which in a non-stationary stream would freeze PPO in its last environment.
#
# SAC (Stable-Baselines3 defaults, as used by the zoo's HalfCheetah entry, hyperparams/sac.yml):
#   - actor and twin critics 256x256 ReLU (Haarnoja et al., 2018);
#   - learning rate 3e-4, batch 256, buffer 1M, tau 0.005, gamma 0.99;
#   - automatic entropy tuning toward -dim(A) = -6;
#   - one gradient step per environment step, after 10,000 steps of uniform random actions.
#   Deviation: the gradient steps are applied in groups of 20 every 20 environment steps (same update
#   ratio), for speed. At test, every baseline acts deterministically (SAC: tanh of the policy mean).
PPO_SETTINGS = {
	"n_steps": 512,
	"batch_size": 64,
	"n_epochs": 20,
	"gamma": 0.98,
	"gae_lambda": 0.92,
	"clip_ratio": 0.1,
	"learning_rate": 2.0633e-5,
	"entropy_coef": 4.01762e-4,
	"value_coef": 0.58096,
	"max_grad_norm": 0.8,
	"log_std_init": -2.0,
	"reward_scaling": "return",
	"anneal_lr": False,
}
SAC_SETTINGS = {
	"batch_size": 256,
	"buffer_capacity": 1_000_000,
	"gamma": 0.99,
	"learning_rate": 3e-4,
	"tau": 0.005,
	"learning_starts": 10_000,
	"update_frequency": 20,
	"gradient_steps": 20,
	# Implementation only: each update's 20 gradient steps in one compiled call (same steps and ratio; tested
	# equal to stepping one by one). About 11% faster on CPU; XLA was slower, so it stays off.
	"fused_updates": True,
}


def mlp_brain(action_dim: int, state_dim: int, hidden: tuple = (256, 256)) -> Model:
	"""256x256 ReLU MLP with Keras' default (Glorot) initialization: SAC's actor and critics, and PPO's policy and
	value networks (the zoo's PPO disables orthogonal initialization)."""
	inputs = Input(shape=(state_dim,))
	hidden_tensor = inputs
	for units in hidden:
		hidden_tensor = Dense(units, activation="relu")(hidden_tensor)
	outputs = Dense(action_dim)(hidden_tensor)
	return Model(inputs, outputs)


# ENA's environment detection on HalfCheetah: the chirp driver (agents/ena/skills/env_detection/driver/chirp.py).
# The transition-model driver CartPole uses (dynamics_model) misfires here: its evidence is the acting
# individual's own transitions, and each new behavior of a learning population (a new gait, a flip, standing
# still) reaches contact states the model has not seen, so it reads as new physics (27 confirmed changes and
# 23 niches for 3 environments on the detection benchmark, tools/detection_benchmark.py). The chirp driver plays
# the same short excitation at every episode end, so its response differs only through the physics. Chosen on
# the benchmark's ablation (tools/detection_ablation.py: screened on seeds 0-4, ablated on held-out seeds 5-34,
# confirmed on seeds 35-64): a 64-step wavelet train of the Mexican hat (the wavelet's waveform at dyadic scales)
# analysed by Haar wavelet packets at level 3 (dyadic, non-overlapping; 8 bands), one energy per channel and band.
ENA_DETECTION = {
	"recognizer_strategy": "chirp",
	"recognizer_chirp": {},  # the driver's defaults (DEFAULTS in chirp.py) are this configuration
}


# ENA's search on HalfCheetah (no prior paper: set by the calibration recorded in docs/PLAN_half_cheetah.md).
# The search settings, frozen from the tuning probes (tools/half_cheetah_probe.py; seeds 200+, disjoint from the
# study's), the rest as CartPole's. Of a 24-configuration screen (brain 32x16 / 32x8, mutation rate 0.05 / 0.1 / 0.25,
# depth 100 / 200, 1M / 2M steps per environment), evaluation depth was the only large effect (200 over 100); 32x16,
# rate 0.25, depth 200 was the best configuration at the study's 1M budget, with the best retention of Mars among the
# leaders. The mutation rate had no measurable effect. The brain's size is ENA_BRAIN_HIDDEN (above).
ENA_SETTINGS = {
	"exploration_rate": 0.2,
	"max_per_niche": 10,
	"max_gladiator_ratio": 0.5,
	"max_eval_steps": 200,
	"mutation_rate": 0.25,
	"mutation_noise": 0.1,
	"pop_size": 100,
	"selection_driver": "bpr",
	"trust_algorithm": "linear_ramp",
}
# The olympic Hall of Fame (the default driver plus periodic Olympiads): HalfCheetah's returns are noisy enough that
# one episode's trust admits lucky gladiators; every 200k steps, gladiators replay 3 episodes on shared seeds.
ENA_HALL_OF_FAME = {
	"hall_of_fame_driver": "olympic",
	"olympics_interval": 200_000,
	"olympics_episodes": 3,
	"olympics_qualifier_seats": 2,
}


def build_ena_agent(max_agent_threads: int, brain_hidden: int | tuple = ENA_BRAIN_HIDDEN, **overrides) -> ENAgent:
	"""ENA-Combined (the ablation's skill set) with HalfCheetah's physics context, calibrated detection and search settings;
	overrides replace any ENAgent argument (the calibration probes use them)."""
	skills = next(row for row in ABLATION_VARIANTS if row["name"] == ENA_NAME)
	settings = {
		**ENA_SETTINGS,
		"plasticity_algorithm": skills["plasticity_algorithm"],
		"hall_of_fame_driver": skills["hall_of_fame_driver"],
		"recognizer_strategy": skills["recognizer_strategy"],
		**ENA_DETECTION,
		**ENA_HALL_OF_FAME,
		**overrides,
	}
	brain = functools.partial(ena_brain, hidden=brain_hidden)
	return ENAgent(brain=brain, make_env=make_env, max_threads=max_agent_threads, physics=["gravity", "friction", "wind"], **settings)


def build_agent(name: str, max_agent_threads: int, **ena_overrides):
	"""One challenge agent by name; its training length comes from the stopping rule (ena_overrides: calibration only)."""
	if name == ENA_NAME:
		agent = build_ena_agent(max_agent_threads, **ena_overrides)
	elif name == "PPO":
		agent = PPOAgent(brain=mlp_brain, critic_brain=mlp_brain, make_env=make_env, **PPO_SETTINGS)
	elif name == "SAC":
		agent = SACAgent(brain=mlp_brain, make_env=make_env, **SAC_SETTINGS)
	else:
		raise ValueError(f"Unknown HalfCheetah challenge agent: {name!r}")
	agent.set_name(name)
	agent.set_episodes(TRAINING_EPISODES)
	return agent


def hardware_line(ena_threads: int) -> str:
	"""Where a replicate ran, so wall-clock times are comparable: CPU, cores, TensorFlow's thread pool, ENA's threads."""
	cpu = platform.processor() or platform.machine()
	try:
		with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
			cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name")), cpu)
	except OSError:
		pass
	tensorflow = tf.config.threading.get_intra_op_parallelism_threads() or "all"
	return f"Hardware: {platform.system()} {cpu}, {os.cpu_count()} cores; TensorFlow threads {tensorflow}; ENA threads {ena_threads}"


def run_experiment(experiment_id: int, output_dir: str, queue, variants: list, include_random: bool) -> None:
	"""Run one replicate: every agent trained to the same env-step budget per environment, then a frozen test.

	include_random is the shared runner's flag; this study has no Random floor, so asking for it is an error."""
	if include_random:
		raise ValueError("The HalfCheetah study has no Random floor")
	names = [row["name"] for row in variants]
	post_update(experiment_id, queue, status="Initializing Agents...", progress=2)
	seed_experiment(experiment_id)

	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "outputs.txt"), "w", encoding="utf-8") as output_file:
		vprint(f"Experiment {experiment_id + 1} Log", file=output_file)
		vprint("=" * 30, file=output_file)

		worlds = make_training_worlds()
		max_agent_threads = max(1, os.cpu_count() // MAX_WORKERS)
		vprint(f"  {hardware_line(max_agent_threads)}", file=output_file)
		agents = [build_seeded(experiment_id, build_agent, name, max_agent_threads) for name in names]

		domain = {"success_threshold": SOLVED_SCORE, "ylim": Y_LIM}
		train_and_test(
			agents,
			experiment_id,
			output_dir,
			output_file,
			queue,
			worlds,
			testing_episodes=TESTING_EPISODES,
			training={"steps_per_env": [FIXED_BUDGET_PER_ENV] * len(worlds), "training_x_axis": "env_steps", "health_every": HEALTH_EVERY, **domain},
			testing={"test_worlds": make_test_worlds(worlds), **domain},
		)

		post_update(experiment_id, queue, status="Done", done=True, progress=2)


def parse_challenge_args(argv: list = None) -> argparse.Namespace:
	"""The shared study flags; no Random floor by default (the study does not run one)."""
	# No Random floor: it adds nothing to the comparison of ENA with PPO and SAC.
	return parse_args(
		argv, variant_table=CHALLENGE_AGENTS, defaults={"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": False}
	)


# =============================================================================
# MAIN: Run N independent experiments to produce statistically valid results
# =============================================================================
if __name__ == "__main__":
	ARGS = parse_challenge_args()
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(CHALLENGE_AGENTS, ARGS.variants),
		ARGS.include_random,
		"HALFCHEETAH CHALLENGE",
		experiment_fn=run_experiment,
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
	)
