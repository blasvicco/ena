#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA performance study — each skill's contribution to ENA's non-stationary-RL performance (50-replicate ablation)."""

# General imports
import argparse
import csv
import os
import random
import threading
import time
import traceback
import faulthandler
import concurrent.futures
from multiprocessing import Manager
from typing import Callable


# Libs imports
import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, initializers
from keras.layers import Dense, Input

# App imports
from agents import ENAgent, RandomAgent
from tools.metrics import (
	calculate_adaptation,
	calculate_compute_cost,
	calculate_paper_metrics,
	estimate_eval_cost,
	plot_academic_comparison,
	plot_plasticity_analysis,
	plot_specialist_transitions,
)
from tools.visualization import vprint, ProgressReporter

# Constants
NUM_EXPERIMENTS = 50
MAX_WORKERS = os.cpu_count() - 2
GPUS = tf.config.list_physical_devices("GPU")
TESTING_EPISODES = [200, 200, 200, 200, 200]
TRAINING_EPISODES_ENA = [500, 500, 500]
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartpole_ena_performance_outputs")
# Per-experiment seed = SEED_OFFSET + experiment index (Python, NumPy and TensorFlow RNGs).
SEED_OFFSET = 1000
INCLUDE_RANDOM_FLOOR = True
# Trust driver for every row that doesn't set its own "trust_algorithm" (Trust is held constant, not ablated).
TRUST_ALGORITHM = "linear_ramp"

# Every meaningful on/off combination of the three studied skills, every other ENAgent argument
# held constant (see run_experiment()). "Off" is plasticity "zero" (2 candidates x 10 steps per
# episode, minimal mutation), hall_of_fame "none", env detection "none". Env detection only
# exists through the Hall of Fame — niches are created when it inducts a gladiator — so rows with
# env detection but no Hall of Fame would equal their Hall-of-Fame-less twin and are omitted.
# Each skill's contribution is measured alone and in the presence of the others:
#   env detection: HoF+EnvDetection vs. HoF, and Combined vs. Plasticity+HoF
#   Hall of Fame:  HoF vs. NoSkill, and Plasticity+HoF vs. Plasticity
#   Plasticity:    Plasticity vs. NoSkill, Plasticity+HoF vs. HoF, and Combined vs. HoF+EnvDetection
PERFORMANCE_VARIANTS = [
	{"name": "ENA-NoSkill", "plasticity_algorithm": "zero", "hall_of_fame_driver": "none", "recognizer_strategy": "none"},
	{"name": "ENA-Plasticity", "plasticity_algorithm": "linear_ramp", "hall_of_fame_driver": "none", "recognizer_strategy": "none"},
	{"name": "ENA-HoF", "plasticity_algorithm": "zero", "hall_of_fame_driver": "default", "recognizer_strategy": "none"},
	{"name": "ENA-HoF+EnvDetection", "plasticity_algorithm": "zero", "hall_of_fame_driver": "default", "recognizer_strategy": "dynamics_model"},
	{"name": "ENA-Plasticity+HoF", "plasticity_algorithm": "linear_ramp", "hall_of_fame_driver": "default", "recognizer_strategy": "none"},
	{"name": "ENA-Combined", "plasticity_algorithm": "linear_ramp", "hall_of_fame_driver": "default", "recognizer_strategy": "dynamics_model"},
]


# Helper methods
def ena_brain(action_dim: int, state_dim: int) -> Sequential:
	"""Build a ENA-NN."""
	initializer = initializers.Orthogonal(gain=1.0)
	return Sequential(
		[
			Input(shape=(state_dim,)),
			Dense(4, activation="relu", kernel_initializer=initializer),
			Dense(2, activation="relu", kernel_initializer=initializer),
			Dense(
				action_dim,
				activation="linear",
				kernel_initializer=initializer,
			),
		]
	)


def post_update(experiment_id: int, queue, status: str = None, progress: int = 0, done: bool = False) -> None:
	"""Post update to the reporting queue."""
	if queue:
		queue.put(
			{
				"done": done,
				"exp_id": experiment_id,
				"progress": progress,
				"status": status,
			}
		)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def agent_testing(agents: list, experiment_id: int, output_dir: str, output_file, queue, worlds: list) -> None:
	"""Test agents on worlds."""
	# Neural weights are frozen; only the archive selector updates Trust at test time.
	post_update(experiment_id, queue, status="Phase 2: Testing Setup", progress=8)
	mystery_env = make_env(gravity=11.15, name="Neptune")
	extrap_env = make_env(gravity=50.0, name="Brown-dwarf")
	test_worlds = [worlds[0], mystery_env, worlds[1], extrap_env, worlds[2]]

	agent_ids = []
	execution_times = {}
	history_testing = {}
	known_niche_ids_by_history = []
	for agent in agents:
		agent_name = agent.get_name()
		episodes = agent.get_episodes()
		agent_ids.append(agent_name)
		post_update(experiment_id, queue, status=f"Testing {agent_name}", progress=8)
		# Niches already established by the end of training — so the test chart's niche
		# events classify a niche's first test-phase appearance as a "switch", not a
		# spurious "new niche" (no niche can genuinely be created once train() stops running).
		known_niche_ids_by_history.append(getattr(agent, "known_niche_ids", None))
		agent.set_learning(False, output_dir=output_dir)
		start_time = time.time()
		history_testing[agent_name] = play(
			agent,
			test_worlds,
			episodes,
			live_log_path=os.path.join(output_dir, f"live_testing_{agent_name}.csv"),
			on_episode=make_eta_reporter(experiment_id, queue, "Testing", agent_name, start_time),
		)
		execution_times[agent_name] = time.time() - start_time

	# Calculate and print metrics for this experiment
	print(f"\n  --- Experiment {experiment_id + 1} Test Results ---", file=output_file)
	calculate_paper_metrics(
		history_testing.values(),
		agent_ids,
		file=output_file,
		success_threshold=475.0,
		times=execution_times,
	)

	# Testing Plots
	post_update(experiment_id, queue, status="Generating Testing Plots", progress=8)
	plot_academic_comparison(
		history_testing.values(),
		agent_ids,
		save_path=os.path.join(output_dir, "test.png"),
		known_niche_ids_by_history=known_niche_ids_by_history,
	)

	for agent in agents:
		is_ena = isinstance(agent, ENAgent)
		if is_ena:
			agent_name = agent.get_name()
			plot_specialist_transitions(
				history_testing[agent_name],
				pop_size=agent.pop_size,
				save_path=os.path.join(
					output_dir,
					f"specialist_transitions_{agent_name}.png",
				),
			)


def phase_env_steps(history: pd.DataFrame) -> list:
	"""Env-steps consumed in each contiguous environment phase of a play() history, in order."""
	phase_ids = (history["env_id"] != history["env_id"].shift()).cumsum()
	phase_ends = history.groupby(phase_ids, sort=True)["env_steps"].last().tolist()
	return [int(end - start) for start, end in zip([0] + phase_ends[:-1], phase_ends)]


# pylint: disable=too-many-arguments,too-many-positional-arguments
def agent_training(
	agents: list,
	experiment_id: int,
	output_dir: str,
	output_file,
	queue,
	worlds: list,
	match_steps_of: str = None,
	stopping=None,
	eval_worlds: list = None,
) -> None:
	"""Train agents on worlds — by episode count, matched to a reference agent's env-steps, or early-stopped."""
	# stopping (tools.early_stopping.StoppingRule) with eval_worlds (separate copies of worlds) trains
	# every agent on each environment until solved/plateau/cap, writes each agent's checks to
	# live_evaluation_{agent}.csv and adds the ADAPTATION SUMMARY table to outputs.txt.
	# match_steps_of names an agent trained earlier in the list (by episodes): every agent after it
	# trains on each environment until it has consumed the same number of env-steps the reference
	# consumed there (ENA's include its candidate evaluations), equalizing interaction budgets.
	agent_ids = []
	execution_times = {}
	history_training = {}
	step_budget = None
	for agent in agents:
		agent_name = agent.get_name()
		episodes = agent.get_episodes()
		agent_ids.append(agent_name)
		post_update(experiment_id, queue, status=f"Training {agent_name}", progress=8)
		agent.set_learning(True)
		if hasattr(agent, "set_total_steps"):
			# Agents with a training-length schedule (PPO's LR annealing) must know the real total.
			if step_budget is not None:
				agent.set_total_steps(sum(step_budget))
			elif stopping is not None:
				agent.set_total_steps(stopping.max_steps * len(worlds))
		start_time = time.time()
		live_log_path = os.path.join(output_dir, f"live_training_{agent_name}.csv")
		history_training[agent_name] = play(
			agent,
			worlds,
			episodes,
			live_log_path=live_log_path,
			on_episode=make_eta_reporter(experiment_id, queue, "Training", agent_name, start_time),
			steps_per_env=step_budget,
			stopping=stopping,
			eval_environments=eval_worlds,
		)
		execution_times[agent_name] = time.time() - start_time
		evaluations = history_training[agent_name].attrs.get("evaluations")
		if evaluations:
			pd.DataFrame(evaluations).to_csv(os.path.join(output_dir, f"live_evaluation_{agent_name}.csv"), index=False)
		if agent_name == match_steps_of:
			step_budget = phase_env_steps(history_training[agent_name])

	# Training Summary
	print("\n  --- Training Performance vs Time ---", file=output_file)
	for agent_name, duration in execution_times.items():
		env_steps = int(history_training[agent_name]["env_steps"].iloc[-1])
		print(f"  {agent_name}: {duration:.2f} seconds, {env_steps:,} env-steps", file=output_file)

	# Training Metrics
	print("\nTraining:", file=output_file)
	calculate_paper_metrics(
		history_training.values(),
		agent_ids,
		file=output_file,
		success_threshold=475.0,
		times=execution_times,
	)
	calculate_compute_cost(history_training.values(), agent_ids, file=output_file)
	calculate_adaptation(history_training.values(), agent_ids, file=output_file)

	# Training Plots
	post_update(experiment_id, queue, status="Generating Training Plots", progress=8)
	plot_academic_comparison(
		history_training.values(),
		agent_ids,
		save_path=os.path.join(output_dir, "training.png"),
	)

	for agent in agents:
		is_ena = isinstance(agent, ENAgent)
		if is_ena:
			agent_name = agent.get_name()
			plot_plasticity_analysis(
				agent,
				history_training[agent_name],
				save_path=os.path.join(
					output_dir,
					f"plasticity_{agent_name}.png",
				),
			)
			plot_specialist_transitions(
				history_training[agent_name],
				pop_size=agent.pop_size,
				save_path=os.path.join(
					output_dir,
					f"specialist_transitions_TRAINING_{agent_name}.png",
				),
			)


def make_env(
	gravity: float = -9.8,
	masspole: float = 0.1,
	name: str = "Earth",
	length: float = 0.5,
	render_mode: str = None,
) -> gym.Env:
	"""Create the environment."""
	env = gym.make("CartPole-v1", render_mode=render_mode)

	# Physics Parameters
	env.unwrapped.gravity = gravity
	env.unwrapped.length = length
	env.unwrapped.masspole = masspole
	env.unwrapped.name = name

	# Recalculate all internal dependencies used by the physics engine
	env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
	env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length

	if hasattr(env.unwrapped, "masspole_length"):
		env.unwrapped.masspole_length = env.unwrapped.masspole * env.unwrapped.length

	return env


def _play_episode(agent, env) -> tuple:
	"""Play one episode (training the agent on each step while it is learning); return (score, steps)."""
	state, _unused_meta = env.reset()
	state = np.array(state, dtype=np.float32)
	episode_score, steps, is_done = 0, 0, False
	while not is_done:
		# 1. Agent performs the step internally
		next_state, reward, terminated, truncated, action = agent.act(env, state)
		is_done = terminated or truncated
		steps += 1

		# 2. Collect data for training (ENA uses this after the episode ends)
		if agent.learn:
			agent.train(
				env=env,
				step_data={
					"action": action,
					"done": is_done,
					"next_state": next_state,
					"reward": reward,
					"state": state,
					"terminated": terminated,
				},
			)

		episode_score += reward
		state = next_state
	return episode_score, steps


def evaluate_agent(agent, env, episodes: int) -> tuple:
	"""Mean score of `episodes` frozen evaluation episodes on env, and the env-steps they took.

	Uses the agent's own side-effect-free evaluation_episode(env) where it has one (ENA); otherwise
	plays act() with learn temporarily off (greedy for the baselines) and never calls train().
	"""
	scores, total_steps = [], 0
	for _ in range(episodes):
		if hasattr(agent, "evaluation_episode"):
			score, steps = agent.evaluation_episode(env)
		else:
			was_learning, agent.learn = agent.learn, False
			try:
				state, _unused_meta = env.reset()
				state = np.array(state, dtype=np.float32)
				score, steps, is_done = 0.0, 0, False
				while not is_done:
					state, reward, terminated, truncated, _action = agent.act(env, state)
					state = np.array(state, dtype=np.float32)
					is_done = terminated or truncated
					score += reward
					steps += 1
			finally:
				agent.learn = was_learning
		scores.append(float(score))
		total_steps += steps
	return float(np.mean(scores)), total_steps


def _episode_row(agent, env, env_idx: int, episode: int, score: float, env_steps: int) -> dict:
	"""The per-episode history/CSV row for the episode the agent just finished."""
	current_individual = agent.population[getattr(agent, "best_individual_idx", 0)] if hasattr(agent, "population") else {}
	plasticity_history = getattr(agent, "plasticity_history", None)
	plasticity_value = plasticity_history[-1] if plasticity_history else None

	# Compute-cost columns: derived from the same plasticity value, mirroring
	# ena.py train()'s own num_to_eval/eval_step_cap formula (estimate_eval_cost's
	# docstring notes the duplication) — None for agents without these attributes
	# and for any episode before plasticity has a value yet.
	num_to_eval, eval_step_cap, candidate_env_steps = None, None, None
	pop_size = getattr(agent, "pop_size", None)
	min_eval_steps = getattr(agent, "min_eval_steps", None)
	max_eval_steps = getattr(agent, "max_eval_steps", None)
	if plasticity_value is not None and pop_size and min_eval_steps is not None and max_eval_steps:
		num_to_eval, eval_step_cap = estimate_eval_cost(plasticity_value, pop_size, min_eval_steps, max_eval_steps)
		candidate_env_steps = num_to_eval * eval_step_cap

	return {
		"env_id": getattr(env.unwrapped, "name", env_idx),
		"episode": episode,
		"score": score,
		"specialist_id": getattr(agent, "best_individual_idx", 0),
		"detected_env_id": getattr(agent, "current_env_id", -1),
		"internal_env_id": current_individual.get("internal_env_id", -1),
		"is_gladiator": current_individual.get("is_gladiator", False),
		"gladiator_env": current_individual.get("env_name", None),
		"env_shift_detected": getattr(agent, "last_env_shift_detected", False),
		"plasticity": plasticity_value,
		"num_to_eval": num_to_eval,
		"eval_step_cap": eval_step_cap,
		"candidate_env_steps": candidate_env_steps,
		"env_steps": env_steps,
	}


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def play(
	agent,
	environments: list,
	episodes_per_env: list,
	live_log_path: str = None,
	on_episode=None,
	steps_per_env: list = None,
	stopping=None,
	eval_environments: list = None,
) -> pd.DataFrame:
	"""Play the game, optionally live-logging each episode's row and reporting progress."""
	# If live_log_path is given, each episode's row is also appended to that CSV, flushed
	# immediately — lets a separate process (e.g. tools/live_dashboard.py) watch convergence
	# while this run is still in progress, without changing the returned DataFrame or any
	# existing chart in any way.
	# If on_episode is given, it's called after every episode as on_episode(progress, total) —
	# e.g. to report a live ETA — in episodes, or in env-steps when steps_per_env is given.
	# steps_per_env replaces episodes_per_env with an interaction budget: each environment is
	# played until the agent has consumed that many env-steps there (finishing the episode in
	# progress). The env_steps column is the agent's cumulative consumption — its own
	# total_env_steps where it has one (ENA: played + candidate-evaluation steps), else the
	# steps played here.
	# stopping (a tools.early_stopping.StoppingRule) replaces both budgets: each environment is
	# played until the rule declares it solved or plateaued, or its max_steps cap is reached, with
	# evaluations on eval_environments[i] (a separate copy of environments[i]). The returned
	# DataFrame's attrs["phases"] holds each environment's outcome and attrs["evaluations"] every
	# check; evaluation steps are reported there, never added to env_steps.
	history, phases, evaluations = [], [], []
	if stopping is not None:
		total = stopping.max_steps * len(environments)
	else:
		total = sum(steps_per_env) if steps_per_env is not None else sum(episodes_per_env)
	played_steps = 0
	live_log_file = None
	live_log_writer = None
	if live_log_path:
		# pylint: disable=consider-using-with
		live_log_file = open(live_log_path, "w", newline="", encoding="utf-8")

	def consumed() -> int:
		return getattr(agent, "total_env_steps", played_steps)

	try:
		for env_idx, env in enumerate(environments):
			phase_start, phase_episodes, status, phase_eval_steps, solve_steps = consumed(), 0, "budget", 0, None
			if stopping is not None:
				stopping.start_phase()
			while _phase_continues(stopping, steps_per_env, episodes_per_env, env_idx, consumed() - phase_start, phase_episodes):
				episode_score, steps = _play_episode(agent, env)
				played_steps += steps
				phase_episodes += 1
				row = _episode_row(agent, env, env_idx, len(history) + 1, episode_score, consumed())
				history.append(row)

				if on_episode is not None:
					on_episode(_progress(stopping, steps_per_env, env_idx, consumed() - phase_start, len(history)), total)
				live_log_writer = _write_live_row(live_log_file, live_log_writer, row)

				if stopping is not None:
					status, phase_eval_steps, solve_steps = _apply_stopping(
						agent, stopping, eval_environments[env_idx], consumed() - phase_start, row, evaluations, (phase_eval_steps, solve_steps)
					)
					if status != "continue":
						break
			phases.append(
				{
					"env_id": getattr(env.unwrapped, "name", env_idx),
					"status": status,
					"steps": consumed() - phase_start,
					"solve_steps": solve_steps,
					"eval_steps": phase_eval_steps,
				}
			)
	finally:
		if live_log_file is not None:
			live_log_file.close()

	frame = pd.DataFrame(history)
	frame.attrs["phases"] = phases
	frame.attrs["evaluations"] = evaluations
	return frame


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _progress(stopping, steps_per_env, env_idx: int, phase_steps: int, episodes_so_far: int) -> int:
	"""play()'s on_episode progress: env-steps under a stopping rule or step budget, else episodes."""
	if stopping is not None:
		return stopping.max_steps * env_idx + phase_steps
	if steps_per_env is not None:
		return sum(steps_per_env[:env_idx]) + phase_steps
	return episodes_so_far


def _write_live_row(live_log_file, live_log_writer, row: dict):
	"""Append row to the live CSV (header on first write), flushed; return the writer to reuse."""
	if live_log_file is None:
		return live_log_writer
	if live_log_writer is None:
		live_log_writer = csv.DictWriter(live_log_file, fieldnames=list(row.keys()))
		live_log_writer.writeheader()
	live_log_writer.writerow(row)
	live_log_file.flush()
	return live_log_writer


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _phase_continues(stopping, steps_per_env, episodes_per_env, env_idx: int, phase_steps: int, phase_episodes: int) -> bool:
	"""Whether play() keeps playing the current environment (a stopping rule's exits are handled after each episode)."""
	if stopping is not None:
		return True
	if steps_per_env is not None:
		return phase_steps < steps_per_env[env_idx]
	return phase_episodes < episodes_per_env[env_idx]


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _apply_stopping(agent, stopping, eval_env, phase_steps: int, row: dict, evaluations: list, progress: tuple) -> tuple:
	"""Run a due check and apply the budget; return (status, phase evaluation steps, first-solve steps).

	progress is (evaluation steps so far, first-solve steps or None). status "continue" keeps
	playing. A solve or plateau only ends the environment when the rule says so; otherwise the
	environment trains to max_steps and ends "solved" if it was ever solved, else "cap".
	"""
	phase_eval_steps, solve_steps = progress
	status = "continue"
	if stopping.due(phase_steps):
		eval_mean, eval_steps = evaluate_agent(agent, eval_env, stopping.eval_episodes)
		phase_eval_steps += eval_steps
		outcome = stopping.record(phase_steps, eval_mean)
		evaluations.append(
			{
				"env_id": row["env_id"],
				"phase_steps": phase_steps,
				"env_steps": row["env_steps"],
				"eval_mean": eval_mean,
				"eval_steps": eval_steps,
				"status": outcome,
			}
		)
		if outcome == "solved" and solve_steps is None:
			solve_steps = phase_steps
		if (outcome == "solved" and stopping.stop_when_solved) or (outcome == "plateau" and stopping.stop_on_plateau):
			status = outcome
	if status == "continue" and phase_steps >= stopping.max_steps:
		status = "solved" if solve_steps is not None else "cap"
	return status, phase_eval_steps, solve_steps


def format_duration(seconds: float) -> str:
	"""Format a duration in seconds as e.g. '1h 55m 31s', '6m 22s', or '42s'."""
	hours, remainder = divmod(int(seconds), 3600)
	minutes, secs = divmod(remainder, 60)
	if hours:
		return f"{hours}h {minutes}m {secs}s"
	if minutes:
		return f"{minutes}m {secs}s"
	return f"{secs}s"


def make_eta_reporter(experiment_id: int, queue, label: str, agent_name: str, start_time: float) -> Callable[[int, int], None]:
	"""Build a play() on_episode callback that reports a live ETA via the status queue."""
	# Estimated from the average time per episode elapsed so far this phase.

	def _report(episode_num: int, total_episodes: int) -> None:
		"""Post the current ETA, skipping the first and last episode (no rate yet / done)."""
		if episode_num <= 0 or episode_num >= total_episodes:
			return
		elapsed = time.time() - start_time
		remaining = (elapsed / episode_num) * (total_episodes - episode_num)
		eta_str = format_duration(remaining)
		post_update(experiment_id, queue, status=f"{label} {agent_name} (ETA {eta_str})")

	return _report


def make_training_worlds() -> list:
	"""The three training environments, in training order."""
	return [
		make_env(gravity=3.7, masspole=0.05, name="Mars-Light", length=0.1),
		make_env(gravity=24.8, masspole=0.5, name="Jupiter-Heavy", length=1.5),
		make_env(gravity=9.8, masspole=0.1, name="Earth-Standard", length=0.5),
	]


def build_ena_agent(variant: dict, max_agent_threads: int) -> ENAgent:
	"""An ENAgent for one variant row, every non-studied argument held at the study-wide value."""
	agent = ENAgent(
		brain=ena_brain,
		make_env=make_env,
		controller_episode_rate=0.15,
		max_per_niche=5,
		max_gladiator_ratio=0.5,
		max_threads=max_agent_threads,
		mutation_rate=0.05,
		physics=["gravity", "masspole", "length"],
		pop_size=100,
		recognizer_threshold=0.90,
		recognizer_ema_alpha=0.05,
		selection_driver="bpr",
		trust_algorithm=variant.get("trust_algorithm", TRUST_ALGORITHM),
		plasticity_algorithm=variant["plasticity_algorithm"],
		hall_of_fame_driver=variant["hall_of_fame_driver"],
		recognizer_strategy=variant["recognizer_strategy"],
	)
	agent.set_name(variant["name"])
	agent.set_episodes(TRAINING_EPISODES_ENA)
	return agent


def seed_experiment(experiment_id: int) -> int:
	"""Seed Python, NumPy and TensorFlow for one experiment replicate; return the seed used."""
	seed = SEED_OFFSET + experiment_id
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	return seed


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_experiment(experiment_id: int, output_dir: str, queue, variants: list, include_random: bool) -> None:
	"""Run a single experiment: train then test one ENAgent per variant row (plus the Random floor if asked)."""
	post_update(experiment_id, queue, status="Initializing Agents...", progress=2)
	seed_experiment(experiment_id)

	# 0. Prepare Experiment Directory
	os.makedirs(output_dir, exist_ok=True)
	outputs_path = os.path.join(output_dir, "outputs.txt")
	with open(outputs_path, "w", encoding="utf-8") as output_file:
		vprint(f"Experiment {experiment_id + 1} Log", file=output_file)
		vprint("=" * 30, file=output_file)

		# 1. Prepare Environments
		worlds = make_training_worlds()

		max_agent_threads = max(1, os.cpu_count() // MAX_WORKERS)

		# 2. Initialize Agents — optionally the random floor, plus one ENAgent per studied variant
		agents = []
		if include_random:
			random_agent = RandomAgent(brain=None, make_env=make_env)
			random_agent.set_name("Random")
			random_agent.set_episodes(TRAINING_EPISODES_ENA)
			agents.append(random_agent)

		for variant in variants:
			agents.append(build_ena_agent(variant, max_agent_threads))

		# 3. Training Phase
		agent_training(
			agents=agents,
			experiment_id=experiment_id,
			output_dir=output_dir,
			output_file=output_file,
			queue=queue,
			worlds=worlds,
		)

		# 4. Testing Phase
		for agent in agents:
			agent.set_episodes(TESTING_EPISODES)
			if isinstance(agent, ENAgent):
				agent.load_gladiators()

		agent_testing(
			agents=agents,
			experiment_id=experiment_id,
			output_dir=output_dir,
			output_file=output_file,
			queue=queue,
			worlds=worlds,
		)

		post_update(experiment_id, queue, status="Done", done=True, progress=2)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_experiment_worker(experiment_id: int, reporting_queue, study_dir: str, variants: list, include_random: bool, experiment_fn=None) -> None:
	"""Wrapper that runs a single experiment in an isolated process."""
	# Everything the worker needs is passed explicitly rather than read from module globals, so
	# command-line overrides reach it under any multiprocessing start method (spawn re-imports).
	output_dir = os.path.join(study_dir, f"exp_{experiment_id + 1}")
	os.makedirs(output_dir, exist_ok=True)
	crash_log_path = os.path.join(output_dir, "crash.log")

	crash_file = None
	try:
		# pylint: disable=consider-using-with
		crash_file = open(crash_log_path, "w", encoding="utf-8")  # keep open
		faulthandler.enable(file=crash_file)
	except OSError:
		# Crash logging is a best-effort diagnostic, not load-bearing — if the log file
		# itself can't be opened (permissions, disk full, bad path), proceed without it
		# rather than losing the whole experiment over a logging failure.
		pass

	# Broad except is deliberate here, not a silent swallow: this is a process-isolation
	# boundary (one agent's hours-long training run, in its own worker process). Any
	# failure — TF/Keras errors, gymnasium errors, OOM, numerical issues — must be caught
	# so one variant's crash doesn't take down the whole ProcessPoolExecutor pool. The
	# failure still surfaces, not swallowed: full traceback written to crash.log, plus a
	# [bold red]Error[/bold red] status pushed to the reporting queue for the live console.
	try:
		(experiment_fn or run_experiment)(experiment_id, output_dir, reporting_queue, variants, include_random)
	except Exception:  # pylint: disable=broad-except
		tb_str = traceback.format_exc()
		if crash_file and not crash_file.closed:
			crash_file.write(tb_str)
			crash_file.flush()
		reporting_queue.put(
			{
				"done": True,
				"exp_id": experiment_id,
				"status": f"[bold red]Error: {tb_str}[/bold red]",
			}
		)
	finally:
		if crash_file:
			crash_file.close()


def _positive_int(value: str) -> int:
	"""argparse type: a strictly positive integer."""
	number = int(value)
	if number <= 0:
		raise argparse.ArgumentTypeError(f"must be positive, got {number}")
	return number


def select_variants(variant_table: list, names: list) -> list:
	"""The rows of variant_table named in names, in table order."""
	return [row for row in variant_table if row["name"] in names]


def parse_args(argv: list = None, variant_table: list = None, defaults: dict = None) -> argparse.Namespace:
	"""Command-line overrides for which rows to run, how many replicates, and where they go."""
	variant_table = PERFORMANCE_VARIANTS if variant_table is None else variant_table
	defaults = {"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": INCLUDE_RANDOM_FLOOR, **(defaults or {})}
	names = [row["name"] for row in variant_table]
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variants", nargs="+", default=names, choices=names, help="Rows to run (default: all).")
	parser.add_argument("--output-dir", default=defaults["output_dir"], help="Study directory for the exp_* replicates.")
	parser.add_argument("--num-experiments", type=int, default=defaults["num_experiments"], help="Number of independent replicates.")
	parser.add_argument(
		"--include-random", action=argparse.BooleanOptionalAction, default=defaults["include_random"], help="Also run the Random floor."
	)
	parser.add_argument("--workers", type=_positive_int, default=MAX_WORKERS, help="Replicates run in parallel (one process each).")
	return parser.parse_args(argv)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_study(
	num_experiments: int, output_dir: str, variants: list, include_random: bool, title: str, experiment_fn=None, workers: int = None
) -> None:
	"""Run num_experiments independent replicates in parallel worker processes, skipping any already complete."""
	# experiment_fn(experiment_id, output_dir, queue, variants, include_random) runs one replicate —
	# default run_experiment (ENAgent rows); another study can pass its own (e.g. cartpole_challenge.py).
	# workers sets how many replicates run at once (default MAX_WORKERS); running again with a larger
	# num_experiments continues a study in batches, since already-complete replicates are skipped.
	workers = MAX_WORKERS if workers is None else workers
	vprint(
		f"[green]Hardware:[/green] GPU detection successful: {GPUS}." if GPUS else "[yellow]Hardware:[/yellow] No GPU detected. Defaulting to CPU."
	)

	with Manager() as manager:
		# 1. Setup Queue
		reporting_queue = manager.Queue()

		# 2. Initialize and Start the Progress Reporter Thread
		reporter = ProgressReporter(reporting_queue, num_experiments)
		reporter_thread = threading.Thread(target=reporter.run, daemon=True)
		reporter_thread.start()

		vprint(f"\n[bold green]{'='*60}[/bold green]")
		vprint(f"[bold]LAUNCHING {num_experiments} INDEPENDENT {title} EXPERIMENTS[/bold]")
		vprint(f"Agents: {(['Random'] if include_random else []) + [row['name'] if isinstance(row, dict) else row for row in variants]}")
		vprint(f"Running [cyan]{workers}[/cyan] replicates in parallel.")
		vprint(f"[bold green]{'='*60}[/bold green]\n")

		# 3. Launch each experiment in isolated processes
		with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
			futures = {}
			for exp_id in range(num_experiments):
				exp_dir = os.path.join(output_dir, f"exp_{exp_id + 1}")
				analyzed = os.path.exists(os.path.join(exp_dir, "outputs.txt")) and os.path.getsize(os.path.join(exp_dir, "outputs.txt")) > 0
				if analyzed:
					reporting_queue.put(
						{
							"exp_id": exp_id,
							"status": "[yellow]Skipping (already exists)[/yellow]",
							"done": True,
						}
					)
					continue

				future = executor.submit(run_experiment_worker, exp_id, reporting_queue, output_dir, variants, include_random, experiment_fn)
				futures[future] = exp_id

			for future in concurrent.futures.as_completed(futures):
				exp_id = futures[future]
				try:
					future.result()
				except Exception as exc:  # pylint: disable=broad-except
					# run_experiment_worker already catches and reports every failure inside
					# its own run — this only catches what that boundary can't (the worker
					# process itself crashing/killed, or pickling errors from the pool), and
					# still surfaces it (printed here) rather than swallowing it.
					vprint(f"[bold red]Experiment {exp_id + 1} crashed: {exc}[/bold red]")

		# 4. Cleanup — still inside Manager context so queue is alive
		reporting_queue.put(None)
		reporter_thread.join(timeout=2.0)

	vprint("\n[bold green]" + "=" * 60 + "[/bold green]")
	vprint(f"[bold]ALL {num_experiments} EXPERIMENTS COMPLETE[/bold]")
	vprint(f"Check the [cyan]{output_dir}/[/cyan] directory for detailed logs and plots.")
	vprint("[bold green]" + "=" * 60 + "[/bold green]")


# =============================================================================
# MAIN: Run N independent experiments to produce statistically valid results
# =============================================================================
if __name__ == "__main__":
	ARGS = parse_args()
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(PERFORMANCE_VARIANTS, ARGS.variants),
		ARGS.include_random,
		"ENA PERFORMANCE",
		workers=ARGS.workers,
	)
