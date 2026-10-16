#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA vs. state-of-the-art RL on non-stationary CartPole — equal step budget for all, frozen test (50 replicates)."""

# General imports
from queue import Queue
import argparse
import functools
import os

# Libs imports
import numpy as np
from keras import Input, Model, initializers
from keras.layers import Dense

# App imports
from agents import DQNAgent, PPOAgent, RandomAgent, SACDiscreteAgent
from agents.tools.abstract import AAgent
from cartpole_ena_ablation import (
	ABLATION_VARIANTS,
	TRAINING_EPISODES_ENA,
	MAX_WORKERS,
	build_seeded,
	build_ena_agent,
	make_env,
	make_training_worlds,
	parse_args,
	run_study,
	seed_experiment,
	select_variants,
	train_and_test,
)
from tools.visualization import post_update, vprint
from tools.early_stopping import StoppingRule


# Constants
NUM_EXPERIMENTS = 50
# SAC-Discrete target entropy, as a share of the maximum log|A| (see build_agent and tools/sac_discrete_entropy_check.py).
SAC_DISCRETE_TARGET_ENTROPY = 0.5
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartpole_challenge_outputs")
ENA_NAME = "ENA-Combined"

# Two protocols, selected with --protocol:
# - "fixed-budget" (default): every agent trains FIXED_BUDGET_PER_ENV env-steps on each environment.
#   The same periodic evaluation records the first solve per environment (adaptability) but never
#   ends training, so the frozen test compares agents after equal training (performance).
# - "first-solve": training on each environment ends at the first solve, a plateau or the
#   MAX_STEPS_PER_ENV cap (answers "who solves it soonest"; results in cartpole_challenge_first_solve_outputs).
# Common to both:
# - ENA is the ablation's ENA-Combined (Plasticity + Hall of Fame + env detection, bpr
#   selection, linear_ramp plasticity and trust).
# - Training: every agent trains on each environment under the same StoppingRule — a frozen
#   evaluation (EVAL_EPISODES episodes, separate env copy, the agent's test-time policy) every
#   CHECK_EVERY env-steps it consumes (ENA's include its candidate evaluations); the environment
#   ends when the evaluation mean reaches SOLVED_SCORE (CartPole-v1's "solved" threshold), on a
#   plateau, or at MAX_STEPS_PER_ENV. Steps-to-solve per environment measures adaptability;
#   evaluation steps are reported separately and never count as training.
# - Frozen test: no agent learns during the 5 test environments (incl. never-trained Neptune and
#   Brown-dwarf); baselines act greedily, ENA's weights are frozen and only its archive selection
#   (choosing among stored gladiators) adapts.
# Baselines use their implementations' default hyperparameters.
CHECK_EVERY = 5_000
EVAL_EPISODES = 5
SOLVED_SCORE = 475.0
PLATEAU_CHECKS = 40  # 200k env-steps without a 10-point gain
PLATEAU_MIN_GAIN = 10.0
MAX_STEPS_PER_ENV = 3_000_000
# Per-environment budget for the fixed-budget protocol: about ENA-Combined's own consumption in the
# ablation (7.68M env-steps over 3 environments, candidate evaluations included).
FIXED_BUDGET_PER_ENV = 2_500_000
PROTOCOLS = ("fixed-budget", "first-solve")
OUTPUT_DIRS = {
	"fixed-budget": OUTPUT_DIR,
	"first-solve": os.path.join(os.path.dirname(OUTPUT_DIR), "cartpole_challenge_first_solve_outputs"),
}
CHALLENGE_AGENTS = [{"name": ENA_NAME}, {"name": "DQN"}, {"name": "PPO"}, {"name": "SAC-Discrete"}]


# Helper methods
def mlp_brain(action_dim: int, state_dim: int, hidden: tuple = (64, 64)) -> Model:
	"""Build a ReLU MLP: state -> one output per action (DQN Q-network, SAC-Discrete actor and critics)."""
	inputs = Input(shape=(state_dim,))
	hidden_tensor = inputs
	for units in hidden:
		hidden_tensor = Dense(units, activation="relu")(hidden_tensor)
	outputs = Dense(action_dim)(hidden_tensor)
	return Model(inputs, outputs)


def ppo_actor_brain(action_dim: int, state_dim: int, hidden: tuple = (64, 64)) -> Model:
	"""Factory for PPO actor network."""
	init_hidden = initializers.Orthogonal(gain=np.sqrt(2))
	init_output = initializers.Orthogonal(gain=0.01)
	inputs = Input(shape=(state_dim,))
	hidden_tensor = inputs
	for units in hidden:
		hidden_tensor = Dense(units, activation="tanh", kernel_initializer=init_hidden)(hidden_tensor)
	outputs = Dense(action_dim, kernel_initializer=init_output)(hidden_tensor)
	return Model(inputs, outputs)


def ppo_critic_brain(_action_dim: int, state_dim: int, hidden: tuple = (64, 64)) -> Model:
	"""Factory for PPO critic network."""
	init_hidden = initializers.Orthogonal(gain=np.sqrt(2))
	init_output = initializers.Orthogonal(gain=1.0)
	inputs = Input(shape=(state_dim,))
	hidden_tensor = inputs
	for units in hidden:
		hidden_tensor = Dense(units, activation="tanh", kernel_initializer=init_hidden)(hidden_tensor)
	outputs = Dense(1, kernel_initializer=init_output)(hidden_tensor)
	return Model(inputs, outputs)


def build_agent(name: str, max_agent_threads: int) -> AAgent:
	"""One challenge agent by name; its training episodes are set later (or replaced by ENA's step budget)."""
	if name == ENA_NAME:
		return build_ena_agent(next(row for row in ABLATION_VARIANTS if row["name"] == ENA_NAME), max_agent_threads)
	if name == "DQN":
		agent = DQNAgent(brain=mlp_brain, make_env=make_env)
	elif name == "PPO":
		# Constant learning rate (Stable-Baselines3's PPO default): in a non-stationary stream PPO keeps
		# learning at full rate in every environment and isn't told when training will end.
		agent = PPOAgent(brain=ppo_actor_brain, critic_brain=ppo_critic_brain, make_env=make_env, entropy_coef=0.01, anneal_lr=False)
	elif name == "SAC-Discrete":
		# Target entropy 0.5 x log|A| instead of the paper's 0.98: with CartPole's 2 actions, 0.98 is almost
		# the maximum entropy, the policy can't reach it, and the temperature alpha grows without bound
		# and play collapses to random; 0.89, CleanRL's choice, diverges the same way; 0.5 stays bounded
		# (measured by tools/sac_discrete_entropy_check.py).
		agent = SACDiscreteAgent(brain=mlp_brain, make_env=make_env, target_entropy_ratio=SAC_DISCRETE_TARGET_ENTROPY)
	elif name == "Random":
		agent = RandomAgent(brain=None, make_env=make_env)
	else:
		raise ValueError(f"Unknown challenge agent: {name!r}")
	agent.set_name(name)
	agent.set_episodes(TRAINING_EPISODES_ENA)
	return agent


# pylint: disable=too-many-arguments,too-many-positional-arguments
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
	)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_experiment(
	experiment_id: int, output_dir: str, queue: Queue | None, variants: list, include_random: bool, protocol: str = "fixed-budget"
) -> None:
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

		training = {"stopping": make_stopping_rule(protocol), "eval_worlds": make_training_worlds(), "training_x_axis": "env_steps"}
		train_and_test(agents, experiment_id, output_dir, output_file, queue, worlds, training=training)

		post_update(experiment_id, queue, status="Done", done=True, progress=2)


def parse_challenge_args(argv: list = None) -> argparse.Namespace:
	"""The shared study flags plus --protocol; the default output directory follows the protocol."""
	protocol_parser = argparse.ArgumentParser(add_help=False)
	protocol_parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed-budget")
	known, remaining = protocol_parser.parse_known_args(argv)
	args = parse_args(
		remaining,
		variant_table=CHALLENGE_AGENTS,
		# No Random floor by default: it does not learn, so its frozen test is the one the skill ablation already reports.
		defaults={"output_dir": OUTPUT_DIRS[known.protocol], "num_experiments": NUM_EXPERIMENTS, "include_random": False},
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
		f"ENA CHALLENGE ({ARGS.protocol})",
		experiment_fn=functools.partial(run_experiment, protocol=ARGS.protocol),
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
		replicate=ARGS.replicate,
	)
