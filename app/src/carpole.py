#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA Experiments"""

# General imports
import os
import threading
import time
import traceback
import faulthandler
import concurrent.futures
from multiprocessing import Manager


# Libs imports
import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input, initializers, Sequential
from keras.layers import Dense

# App imports
from agents import DQNAgent, ENAgent, PPOAgent
from metrics import (
	calculate_paper_metrics,
	plot_academic_comparison,
	plot_plasticity_analysis,
	plot_specialist_transitions,
)
from visualization import vprint, ProgressReporter

# Constants
NUM_EXPERIMENTS = 30
MAX_WORKERS = os.cpu_count() - 2
GPUS = tf.config.list_physical_devices("GPU")
TESTING_EPISODES = [200, 200, 200, 200, 200]
TRAINING_EPISODES_BASELINE = [1500, 1500, 1500]
TRAINING_EPISODES_ENA = [500, 500, 500]


# Helper methods
def dqn_brain(action_dim, state_dim, hidden=(64, 64)):
	"""Build a Q-network Brain."""
	inputs = Input(shape=(state_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="relu")(_x_)
	outputs = Dense(action_dim)(_x_)
	return Model(inputs, outputs)


def ena_brain(action_dim, state_dim):
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


def ppo_brain(action_dim, state_dim, hidden=(64, 64), output_activation=None):
	"""Shared factory for actor and critic networks."""
	inputs = Input(shape=(state_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="tanh")(_x_)
	outputs = Dense(action_dim, activation=output_activation)(_x_)
	return Model(inputs, outputs)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def agent_testing(agents, experiment_id, output_dir, output_file, queue, worlds):
	"""Test agents on worlds"""
	# Neural weights are frozen; only the archive selector updates Trust at test time.
	post_update(experiment_id, queue, status="Phase 2: Testing Setup", progress=8)
	mystery_env = make_env(gravity=11.15, name="Neptune")
	extrap_env = make_env(gravity=50.0, name="Brown-dwarf")
	test_worlds = [worlds[0], mystery_env, worlds[1], extrap_env, worlds[2]]

	agent_ids = []
	execution_times = {}
	history_testing = {}
	for agent in agents:
		agent_name = agent.get_name()
		episodes = agent.get_episodes()
		agent_ids.append(agent_name)
		post_update(experiment_id, queue, status=f"Testing {agent_name}", progress=8)
		agent.set_learning(False)
		start_time = time.time()
		history_testing[agent_name] = play(agent, test_worlds, episodes)
		execution_times[agent_name] = time.time() - start_time

	# Calculate and print metrics for this experiment
	print(f"\n  --- Experiment {experiment_id + 1} Test Results ---", file=output_file)
	calculate_paper_metrics(
		history_testing.values(),
		agent_ids,
		success_threshold=475.0,
		times=execution_times,
		file=output_file,
	)

	# Testing Plots
	post_update(experiment_id, queue, status="Generating Testing Plots", progress=8)
	plot_academic_comparison(
		history_testing.values(),
		agent_ids,
		save_path=os.path.join(output_dir, "test.png"),
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


def agent_training(agents, experiment_id, output_dir, output_file, queue, worlds):
	"""Train agents on worlds"""
	agent_ids = []
	execution_times = {}
	history_training = {}
	for agent in agents:
		agent_name = agent.get_name()
		episodes = agent.get_episodes()
		agent_ids.append(agent_name)
		post_update(experiment_id, queue, status=f"Training {agent_name}", progress=8)
		agent.set_learning(True)
		start_time = time.time()
		history_training[agent_name] = play(agent, worlds, episodes)
		execution_times[agent_name] = time.time() - start_time

	# Training Summary
	print("\n  --- Training Performance vs Time ---", file=output_file)
	for agent_name, duration in execution_times.items():
		print(f"  {agent_name}: {duration:.2f} seconds", file=output_file)

	# Training Metrics
	print("\nTraining:", file=output_file)
	calculate_paper_metrics(
		history_training.values(),
		agent_ids,
		success_threshold=475.0,
		times=execution_times,
		file=output_file,
	)

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


def make_env(
	gravity=-9.8,
	name="Earth",
	mass_pole=0.1,
	pole_length=0.5,
	render_mode=None,
):
	"""Create the environment"""
	env = gym.make("CartPole-v1", render_mode=render_mode)

	# Physics Parameters
	env.unwrapped.gravity = gravity
	env.unwrapped.length = pole_length
	env.unwrapped.masspole = mass_pole
	env.unwrapped.name = name

	# Recalculate all internal dependencies used by the physics engine
	env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
	env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length

	if hasattr(env.unwrapped, "masspole_length"):
		env.unwrapped.masspole_length = env.unwrapped.masspole * env.unwrapped.length

	return env


def play(agent, environments, episodes_per_env):
	"""Play the game"""
	history = []
	for env_idx, env in enumerate(environments):
		for _ in range(episodes_per_env[env_idx]):
			state, _ = env.reset()
			state = np.array(state, dtype=np.float32)
			episode_score = 0
			is_done = False

			while not is_done:
				# 1. Agent performs the step internally
				# Returns: (next_state, reward, done, action)
				next_state, reward, terminated, truncated, action = agent.act(
					env, state
				)
				is_done = terminated or truncated

				# 2. Collect data for training (DQN uses this per-step, ENA uses this later)
				if agent.learn:
					agent.train(
						env=env,
						step_data={
							"action": action,
							"done": is_done,
							"next_state": next_state,
							"reward": reward,
							"state": state,
						},
					)

				episode_score += reward
				state = next_state

			history.append(
				{
					"env_id": getattr(env.unwrapped, "name", env_idx),
					"episode": len(history) + 1,
					"score": episode_score,
					"specialist_id": getattr(agent, "best_individual_idx", 0),
				}
			)
	return pd.DataFrame(history)


def post_update(experiment_id, queue, status=None, progress=0, done=False):
	"""Post update to the reporting queue."""
	if queue:
		queue.put(
			{
				"exp_id": experiment_id,
				"status": status,
				"progress": progress,
				"done": done,
			}
		)


def run_experiment(experiment_id, output_dir, queue=None):
	"""Run a single experiment."""
	post_update(experiment_id, queue, status="Initializing Agents...", progress=2)

	# 0. Prepare Experiment Directory
	os.makedirs(output_dir, exist_ok=True)
	outputs_path = os.path.join(output_dir, "outputs.txt")
	with open(outputs_path, "w", encoding="utf-8") as output_file:
		vprint(f"Experiment {experiment_id + 1} Log", file=output_file)
		vprint("=" * 30, file=output_file)

		# 1. Prepare Environments
		worlds = [
			make_env(
				gravity=3.7,
				mass_pole=0.05,
				name="Mars-Light",
				pole_length=0.1,
			),
			make_env(
				gravity=24.8,
				mass_pole=0.5,
				name="Jupiter-Heavy",
				pole_length=1.5,
			),
			make_env(
				gravity=9.8,
				mass_pole=0.1,
				name="Earth-Standard",
				pole_length=0.5,
			),
		]

		max_agent_threads = max(1, os.cpu_count() // MAX_WORKERS)

		# Define explicit episode counts (2 episodes per training environment for quick test)
		# 2. Initialize Agents — fresh instances per experiment for independence
		ena_agent_01 = ENAgent(
			action_size=2,
			brain=ena_brain,
			make_env=make_env,
			gladiator_amounts=15,
			max_threads=max_agent_threads,
			mutation_rate=0.05,
			plasticity_algorithm="fuzzylogic",
			pop_size=50,
			trust_algorithm="fuzzylogic",
		)
		ena_agent_01.set_name("ENA-01 Fuzzy-Fuzzy")
		ena_agent_01.set_episodes(TRAINING_EPISODES_ENA)

		ena_agent_02 = ENAgent(
			action_size=2,
			brain=ena_brain,
			make_env=make_env,
			gladiator_amounts=15,
			max_threads=max_agent_threads,
			mutation_rate=0.05,
			plasticity_algorithm="quadratic",
			pop_size=50,
			trust_algorithm="quadratic",
		)
		ena_agent_02.set_name("ENA-02 Quadratic-Quadratic")
		ena_agent_02.set_episodes(TRAINING_EPISODES_ENA)

		dqn_agent = DQNAgent(brain=dqn_brain, make_env=make_env)
		dqn_agent.set_name("DQN-TF")
		dqn_agent.set_episodes(TRAINING_EPISODES_BASELINE)

		ppo_agent = PPOAgent(brain=ppo_brain, make_env=make_env)
		ppo_agent.set_name("PPO-TF")
		ppo_agent.set_episodes(TRAINING_EPISODES_BASELINE)

		# 3. Training Phase
		agent_training(
			agents=[
				ena_agent_01,
				ena_agent_02,
				dqn_agent,
				ppo_agent,
			],
			experiment_id=experiment_id,
			output_dir=output_dir,
			output_file=output_file,
			queue=queue,
			worlds=worlds,
		)

		# 4. Testing Phase
		# Set episodes for testing
		ena_agent_01.set_episodes(TESTING_EPISODES)
		ena_agent_02.set_episodes(TESTING_EPISODES)
		dqn_agent.set_episodes(TESTING_EPISODES)
		ppo_agent.set_episodes(TESTING_EPISODES)

		# Load gladiators for ENA agents
		ena_agent_01.load_gladiators()
		ena_agent_02.load_gladiators()

		agent_testing(
			agents=[
				ena_agent_01,
				ena_agent_02,
				dqn_agent,
				ppo_agent,
			],
			experiment_id=experiment_id,
			output_dir=output_dir,
			output_file=output_file,
			queue=queue,
			worlds=worlds,
		)

		post_update(experiment_id, queue, status="Done", done=True, progress=2)


def run_experiment_worker(experiment_id, reporting_queue):
	"""Wrapper that runs a single experiment in an isolated process."""
	output_dir = os.path.join("outputs", f"exp_{experiment_id + 1}")
	os.makedirs(output_dir, exist_ok=True)
	crash_log_path = os.path.join(output_dir, "crash.log")

	crash_file = None
	try:
		# pylint: disable=consider-using-with
		crash_file = open(crash_log_path, "w", encoding="utf-8")  # keep open
		faulthandler.enable(file=crash_file)
	except Exception:  # pylint: disable=broad-except
		pass

	try:
		run_experiment(experiment_id, output_dir, reporting_queue)
	except Exception:  # pylint: disable=broad-except
		reporting_queue.put(
			{
				"exp_id": experiment_id,
				"status": f"[bold red]Error: {traceback.format_exc()}[/bold red]",
				"done": True,
			}
		)
	finally:
		if crash_file:
			crash_file.close()


# =============================================================================
# MAIN: Run N independent experiments to produce statistically valid results
# =============================================================================
if __name__ == "__main__":
	vprint(
		f"[green]Hardware:[/green] GPU detection successful: {GPUS}."
		if GPUS
		else "[yellow]Hardware:[/yellow] No GPU detected. Defaulting to CPU."
	)

	with Manager() as manager:
		# 1. Setup Queue
		REPORTING_QUEUE = manager.Queue()

		# 2. Initialize and Start the Progress Reporter Thread
		REPORTER = ProgressReporter(REPORTING_QUEUE, NUM_EXPERIMENTS)
		REPORTER_THREAD = threading.Thread(target=REPORTER.run, daemon=True)
		REPORTER_THREAD.start()

		vprint(f"\n[bold green]{'='*60}[/bold green]")
		vprint(f"[bold]LAUNCHING {NUM_EXPERIMENTS} INDEPENDENT EXPERIMENTS[/bold]")
		vprint(f"Utilizing [cyan]{MAX_WORKERS}[/cyan] CPU cores.")
		vprint(f"[bold green]{'='*60}[/bold green]\n")

		# 3. Launch each experiment in isolated processes
		with concurrent.futures.ProcessPoolExecutor(
			max_workers=MAX_WORKERS
		) as executor:
			futures = {}
			for exp_id in range(NUM_EXPERIMENTS):
				exp_dir = os.path.join("outputs", f"exp_{exp_id + 1}")
				analyzed = (
					os.path.exists(os.path.join(exp_dir, "outputs.txt"))
					and os.path.getsize(os.path.join(exp_dir, "outputs.txt")) > 0
				)
				if analyzed:
					REPORTING_QUEUE.put(
						{
							"exp_id": exp_id,
							"status": "[yellow]Skipping (already exists)[/yellow]",
							"done": True,
						}
					)
					continue

				future = executor.submit(run_experiment_worker, exp_id, REPORTING_QUEUE)
				futures[future] = exp_id

			for future in concurrent.futures.as_completed(futures):
				exp_id = futures[future]
				try:
					future.result()
				except Exception as exc:  # pylint: disable=broad-except
					vprint(
						f"[bold red]Experiment {exp_id + 1} crashed: {exc}[/bold red]"
					)

		# 4. Cleanup — still inside Manager context so queue is alive
		REPORTING_QUEUE.put(None)
		REPORTER_THREAD.join(timeout=2.0)

	vprint("\n[bold green]" + "=" * 60 + "[/bold green]")
	vprint(f"[bold]ALL {NUM_EXPERIMENTS} EXPERIMENTS COMPLETE[/bold]")
	vprint("Check the [cyan]outputs/[/cyan] directory for detailed logs and plots.")
	vprint("[bold green]" + "=" * 60 + "[/bold green]")
