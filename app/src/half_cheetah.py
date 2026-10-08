#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA vs. PPO / SAC on non-stationary HalfCheetah — same evaluation/stopping rule for all, frozen test (continuous control)."""

# Structurally parallel to cartpole_challenge.py by design (CLAUDE.md: same experiment lifecycle per domain).
# pylint: disable=duplicate-code

# General imports
import argparse
import functools
import os

# Libs imports
import gymnasium as gym
from keras import Input, Model, Sequential, initializers
from keras.layers import Dense

# App imports
from agents import ENAgent, PPOAgent, RandomAgent, SACAgent
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
from tools.early_stopping import StoppingRule
from tools.visualization import vprint

# Constants
NUM_EXPERIMENTS = 50
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "half_cheetah_outputs")
ENA_NAME = "ENA-Combined"

# Same two protocols and lifecycle as cartpole_challenge.py (see its header): "fixed-budget" (default)
# trains every agent FIXED_BUDGET_PER_ENV env-steps per environment and records the first solve;
# "first-solve" ends an environment at the first solve, a plateau or MAX_STEPS_PER_ENV. Then a frozen
# test on 5 worlds, 2 of them never trained (Ice-Floor, High-Wind).
#
# PROVISIONAL (docs/PLAN_half_cheetah.md, Stage 2): the solved bar, check interval, budgets, test
# episodes and ENA's configuration below are placeholders until the calibration and speed runs
# (Stages 3-5) and the author's decision. Do not launch a study before then.
SOLVED_SCORE = 4800.0  # gymnasium's registered reward_threshold for HalfCheetah-v5
CHECK_EVERY = 25_000  # an evaluation episode is 1000 steps; 5 per check = 5,000 steps
EVAL_EPISODES = 5
PLATEAU_CHECKS = 20  # 500k env-steps without a PLATEAU_MIN_GAIN gain
PLATEAU_MIN_GAIN = 100.0
MAX_STEPS_PER_ENV = 1_500_000
FIXED_BUDGET_PER_ENV = 1_000_000
TESTING_EPISODES = [50, 50, 50, 50, 50]
# Ignored under a stopping rule (training length is the step budget); one entry per training world.
TRAINING_EPISODES = [500, 500, 500]
Y_LIM = (-1000, 12000)
AIR_DENSITY = 1.2  # kg/m^3, sea-level air; only windy worlds have a medium

PROTOCOLS = ("fixed-budget", "first-solve")
OUTPUT_DIRS = {
	"fixed-budget": OUTPUT_DIR,
	"first-solve": os.path.join(os.path.dirname(OUTPUT_DIR), "half_cheetah_first_solve_outputs"),
}
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
	"""The 3 training worlds, in training order."""
	return [
		make_env(friction=0.1, gravity=-3.7, name="Mars-Slippery"),
		make_env(friction=1.5, gravity=-24.8, name="Jupiter-Rough"),
		make_env(friction=0.4, gravity=-9.8, name="Earth-Standard"),
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


def ena_brain(action_dim: int, state_dim: int, hidden: int = 8, activation: str = "relu") -> Sequential:
	"""Build a ENA-NN for continuous control (tanh outputs in the [-1, 1] action range); hidden units and activation per layer."""
	init_hidden = initializers.Orthogonal(gain=1.0)
	init_output = initializers.Orthogonal(gain=0.1)
	return Sequential(
		[
			Input(shape=(state_dim,)),
			Dense(hidden, activation=activation, kernel_initializer=init_hidden),
			Dense(hidden, activation=activation, kernel_initializer=init_hidden),
			Dense(action_dim, activation="tanh", kernel_initializer=init_output),
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


# ENA's environment detection, calibrated on HalfCheetah's training environments with
# tools/detection_calibration.py (output in half_cheetah_probe_outputs/detection_calibration/): each
# environment's linear-Gaussian model fitted on pooled calibration policies, tested on a held-out policy.
#   - Verification window 1,000 transitions (CartPole: 50). With 24 features and 17 outputs, a model fitted
#     fresh on 50 transitions nearly interpolates them, so it beat every niche's model and every alarm became a
#     new niche (47 niches for 3 environments in the first probe). At 1,000 the own-environment gap is <= 1.5
#     nats/step, so alarms caused by a new policy are absorbed instead.
#   - Novelty margin 3 nats/step (CartPole: 2): headroom over that 1.5, far below Jupiter's separation (13.6).
#   - Recent window 3,000 and niche maturity 1,000 transitions: identification and comparison need well over a
#     verification window of data.
#   - Tolerance 3 and CuSum gamma 1e8 are CartPole's: the larger tolerances removed the separation.
#   - Known limit: the linear model does not separate Mars-Slippery from Earth-Standard (gaps 1.4 and 0.35), so
#     they share a niche; Jupiter-Rough, the environment no other policy transfers to, separates clearly.
ENA_DETECTION = {
	"recognizer_residual_tolerance": 3.0,
	"recognizer_novelty_margin": 3.0,
	"recognizer_cusum_false_alarm_steps": 1e8,
	"recognizer_min_verify_samples": 1_000,
	"recognizer_recent_size": 3_000,
	"recognizer_min_samples": 1_000,
}


# ENA's search on HalfCheetah (no prior paper: set by the calibration recorded in docs/PLAN_half_cheetah.md).
ENA_SETTINGS = {
	"exploration_rate": 0.2,
	"max_per_niche": 10,
	"max_gladiator_ratio": 0.5,
	"max_eval_steps": 250,
	"mutation_rate": 0.1,
	"mutation_noise": 0.3,
	"pop_size": 200,
	"selection_driver": "bpr",
	"trust_algorithm": "linear_ramp",
}


def build_ena_agent(max_agent_threads: int, brain_hidden: int = 8, brain_activation: str = "relu", **overrides) -> ENAgent:
	"""ENA-Combined (the ablation's skill set) with HalfCheetah's physics context, calibrated detection and search settings;
	overrides replace any ENAgent argument (the calibration probes use them)."""
	skills = next(row for row in ABLATION_VARIANTS if row["name"] == ENA_NAME)
	settings = {
		**ENA_SETTINGS,
		"plasticity_algorithm": skills["plasticity_algorithm"],
		"hall_of_fame_driver": skills["hall_of_fame_driver"],
		"recognizer_strategy": skills["recognizer_strategy"],
		**ENA_DETECTION,
		**overrides,
	}
	brain = functools.partial(ena_brain, hidden=brain_hidden, activation=brain_activation)
	return ENAgent(brain=brain, make_env=make_env, max_threads=max_agent_threads, physics=["gravity", "friction", "wind"], **settings)


def build_agent(name: str, max_agent_threads: int, **ena_overrides):
	"""One challenge agent by name; its training length comes from the stopping rule (ena_overrides: calibration only)."""
	if name == ENA_NAME:
		agent = build_ena_agent(max_agent_threads, **ena_overrides)
	elif name == "PPO":
		agent = PPOAgent(brain=mlp_brain, critic_brain=mlp_brain, make_env=make_env, **PPO_SETTINGS)
	elif name == "SAC":
		agent = SACAgent(brain=mlp_brain, make_env=make_env, **SAC_SETTINGS)
	elif name == "Random":
		agent = RandomAgent(brain=None, make_env=make_env)
	else:
		raise ValueError(f"Unknown HalfCheetah challenge agent: {name!r}")
	agent.set_name(name)
	agent.set_episodes(TRAINING_EPISODES)
	return agent


def make_stopping_rule(protocol: str = "fixed-budget") -> StoppingRule:
	"""The challenge's evaluation/stopping rule for a protocol, identical for every agent."""
	if protocol not in PROTOCOLS:
		raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
	fixed_budget = protocol == "fixed-budget"
	return StoppingRule(
		check_every=CHECK_EVERY,
		eval_episodes=EVAL_EPISODES,
		solved_score=SOLVED_SCORE,
		plateau_checks=PLATEAU_CHECKS,
		plateau_min_gain=PLATEAU_MIN_GAIN,
		max_steps=FIXED_BUDGET_PER_ENV if fixed_budget else MAX_STEPS_PER_ENV,
		stop_when_solved=not fixed_budget,
		stop_on_plateau=not fixed_budget,
		# Keep checking after a solve: the return at the end of each environment measures forgetting.
		check_after_solve=True,
	)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_experiment(experiment_id: int, output_dir: str, queue, variants: list, include_random: bool, protocol: str = "fixed-budget") -> None:
	"""Run one replicate: every agent trained under the same evaluation/stopping rule, then a frozen test."""
	names = [row["name"] for row in variants]
	post_update(experiment_id, queue, status="Initializing Agents...", progress=2)
	seed_experiment(experiment_id)

	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "outputs.txt"), "w", encoding="utf-8") as output_file:
		vprint(f"Experiment {experiment_id + 1} Log", file=output_file)
		vprint("=" * 30, file=output_file)

		worlds = make_training_worlds()
		max_agent_threads = max(1, os.cpu_count() // MAX_WORKERS)
		agents = [build_seeded(experiment_id, build_agent, name, max_agent_threads) for name in names + (["Random"] if include_random else [])]

		domain = {"success_threshold": SOLVED_SCORE, "ylim": Y_LIM}
		train_and_test(
			agents,
			experiment_id,
			output_dir,
			output_file,
			queue,
			worlds,
			testing_episodes=TESTING_EPISODES,
			training={"stopping": make_stopping_rule(protocol), "eval_worlds": make_training_worlds(), "training_x_axis": "env_steps", **domain},
			testing={"test_worlds": make_test_worlds(worlds), **domain},
		)

		post_update(experiment_id, queue, status="Done", done=True, progress=2)


def parse_challenge_args(argv: list = None) -> argparse.Namespace:
	"""The shared study flags plus --protocol; the default output directory follows the protocol."""
	protocol_parser = argparse.ArgumentParser(add_help=False)
	protocol_parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed-budget")
	known, remaining = protocol_parser.parse_known_args(argv)
	args = parse_args(
		remaining,
		variant_table=CHALLENGE_AGENTS,
		defaults={"output_dir": OUTPUT_DIRS[known.protocol], "num_experiments": NUM_EXPERIMENTS, "include_random": True},
	)
	args.protocol = known.protocol
	return args


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
		f"HALFCHEETAH CHALLENGE ({ARGS.protocol})",
		experiment_fn=functools.partial(run_experiment, protocol=ARGS.protocol),
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
	)
