#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA Experiments: HalfCheetah"""

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
from agents import ENAgent, PPOAgent, SACAgent
from metrics import (
	calculate_paper_metrics,
	plot_academic_comparison,
	plot_plasticity_analysis,
	plot_specialist_transitions,
)
from visualization import vprint, ProgressReporter

# Constants
NUM_EXPERIMENTS = 1
MAX_WORKERS = os.cpu_count() - 2
GPUS = tf.config.list_physical_devices("GPU")

# HalfCheetah specific constants
SUCCESS_THRESHOLD = 8000.0
Y_LIM = (-1000, 12000)

TESTING_EPISODES = [200, 200, 200, 200, 200]
TRAINING_EPISODES_BASELINE = [1000, 1000, 1000]
TRAINING_EPISODES_ENA = [500, 500, 500]

# Helper methods
def ena_brain(action_dim, state_dim):
	"""Build a ENA-NN for continuous control."""
	init_hidden = initializers.Orthogonal(gain=1.0)
	init_output = initializers.Orthogonal(gain=0.1)
	return Sequential(
		[
			Input(shape=(state_dim,)),
			Dense(64, activation="relu", kernel_initializer=init_hidden),
			Dense(64, activation="relu", kernel_initializer=init_hidden),
			Dense(
				action_dim,
				activation="tanh",
				kernel_initializer=init_output,
			),
		]
	)


def ppo_actor_brain(action_output_dim, state_dim, hidden=(256, 256)):
	"""Factory for PPO actor network."""
	init_hidden = initializers.Orthogonal(gain=np.sqrt(2))
	init_output = initializers.Orthogonal(gain=0.01)
	inputs = Input(shape=(state_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="relu", kernel_initializer=init_hidden)(_x_)
	outputs = Dense(action_output_dim, kernel_initializer=init_output)(_x_)
	return Model(inputs, outputs)


def ppo_critic_brain(action_output_dim, state_dim, hidden=(256, 256)):
	"""Factory for PPO critic network."""
	init_hidden = initializers.Orthogonal(gain=np.sqrt(2))
	init_output = initializers.Orthogonal(gain=1.0)
	inputs = Input(shape=(state_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="relu", kernel_initializer=init_hidden)(_x_)
	outputs = Dense(action_output_dim, kernel_initializer=init_output)(_x_)
	return Model(inputs, outputs)


def sac_brain(action_output_dim, state_dim, hidden=(64, 64)):
	"""Builds the actor network for SAC."""
	inputs = Input(shape=(state_dim,))
	unimportant_tensor = inputs
	for units in hidden:
		unimportant_tensor = Dense(units, activation="relu")(unimportant_tensor)
	outputs = Dense(action_output_dim)(unimportant_tensor)
	return Model(inputs, outputs)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def agent_testing(agents, experiment_id, output_dir, output_file, queue, worlds):
	"""Test agents on worlds"""
	post_update(experiment_id, queue, status="Phase 2: Testing Setup", progress=8)

	# Testing worlds with varying physics (Generalization Test)
	test_worlds = [
		worlds[0],  # Mars-Light
		make_env(gravity=-9.8, friction=0.05, name="Ice-Floor"),
		worlds[1],  # Jupiter-Heavy
		make_env(gravity=-9.8, friction=0.4, wind=(-20, 0, 0), name="High-Wind"),
		worlds[2],  # Earth-Standard
	]

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
		file=output_file,
		success_threshold=SUCCESS_THRESHOLD,
		times=execution_times,
	)

	# Testing Plots
	post_update(experiment_id, queue, status="Generating Testing Plots", progress=8)
	plot_academic_comparison(
		history_testing.values(),
		agent_ids,
		save_path=os.path.join(output_dir, "test.png"),
		success_threshold=SUCCESS_THRESHOLD,
		ylim=Y_LIM,
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
					f"specialist_transitions_TESTING_{agent_name}.png",
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
		success_threshold=SUCCESS_THRESHOLD,
		times=execution_times,
		file=output_file,
	)

	# Training Plots
	post_update(experiment_id, queue, status="Generating Training Plots", progress=8)
	plot_academic_comparison(
		history_training.values(),
		agent_ids,
		save_path=os.path.join(output_dir, "training.png"),
		success_threshold=SUCCESS_THRESHOLD,
		ylim=Y_LIM,
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
	friction=0.4,
	gravity=-9.8,
	name="Earth",
	render_mode=None,
	wind=(0, 0, 0),
):
	"""Create the environment with custom physics"""
	env = gym.make("HalfCheetah-v5", render_mode=render_mode)

	# 1. Gravity (Z-axis) - Negative pulls DOWN
	env.unwrapped.model.opt.gravity[2] = gravity
	env.unwrapped.gravity = gravity  # Expose for ENAgent

	# 2. Friction (modify the floor geom friction)
	env.unwrapped.model.geom_friction[0, 0] = friction
	env.unwrapped.friction = friction  # Expose for ENAgent

	# 3. Wind (3D vector)
	env.unwrapped.model.opt.wind[:] = wind

	env.unwrapped.name = name
	return env


def play(agent, envs, episodes_per_env):
	"""Play the game"""
	history = []
	for env_idx, env in enumerate(envs):
		for _ in range(episodes_per_env[env_idx]):
			state, _ = env.reset()
			state = np.array(state, dtype=np.float32)
			episode_score = 0
			is_done = False

			while not is_done:
				# 1. Agent performs the step internally
				next_state, reward, terminated, truncated, action = agent.act(
					env, state
				)
				is_done = terminated or truncated

				# 2. Collect data for training
				if agent.learn:
					agent.train(
						env=env,
						step_data={
							"action": action,
							"done": is_done,
							"terminated": terminated,  # True only on genuine terminal state
							"next_state": next_state,
							"reward": reward,
							"state": state,
						},
					)

				episode_score += reward
				state = next_state

			current_individual = (
				agent.population[getattr(agent, "best_individual_idx", 0)]
				if hasattr(agent, "population")
				else {}
			)
			history.append(
				{
					"env_id": getattr(env.unwrapped, "name", env_idx),
					"episode": len(history) + 1,
					"score": episode_score,
					"specialist_id": getattr(agent, "best_individual_idx", 0),
					"is_gladiator": current_individual.get("is_gladiator", False),
					"gladiator_env": current_individual.get("env_name", None),
					"detected_env_id": getattr(agent, "current_env_id", -1),
				}
			)
	return pd.DataFrame(history)


def post_update(experiment_id, queue, status=None, progress=0, done=False):
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
			make_env(friction=0.1, gravity=-3.7, name="Mars-Slippery"),
			make_env(friction=1.5, gravity=-24.8, name="Jupiter-Rough"),
			make_env(friction=0.4, gravity=-9.8, name="Earth-Standard"),
		]

		active_experiments = min(NUM_EXPERIMENTS, MAX_WORKERS)
		max_agent_threads = max(1, os.cpu_count() // active_experiments)

		# 2. Initialize Agents
		ena_agent_01 = ENAgent(
			brain=ena_brain,
			make_env=make_env,
			env_max_steps=1000.0,
			gladiator_amounts=50,
			max_eval_steps=250,
			max_threads=max_agent_threads,
			exploration_rate=0.2,
			mutation_rate=0.1,
			physics=["gravity", "friction"],
			plasticity_algorithm="fuzzylogic",
			pop_size=200,
			trust_algorithm="fuzzylogic",
			mutation_noise=0.3,
			env_window_size=int(sum(TRAINING_EPISODES_ENA) * 0.05),
			env_cooldown_window=int(sum(TRAINING_EPISODES_ENA) * 0.05),
			env_detection_threshold=-300.0,
		)
		ena_agent_01.set_name("ENA-01 Fuzzy-Fuzzy")
		ena_agent_01.set_episodes(TRAINING_EPISODES_ENA)

		# 		ena_agent_02 = ENAgent(
		# 			brain=ena_brain,
		# 			make_env=make_env,
		# 			gladiator_amounts=15,
		# 			max_eval_steps=1000,  # HalfCheetah episodes are 1000 steps
		# 			max_threads=max_agent_threads,
		# 			mutation_rate=0.05,
		# 			physics=["gravity", "friction"],
		# 			plasticity_algorithm="quadratic",
		# 			pop_size=50,
		# 			trust_algorithm="quadratic",
		# 			max_score=SUCCESS_THRESHOLD,
		# 		)
		# 		ena_agent_02.set_name("ENA-02 Quadratic-Quadratic")
		# 		ena_agent_02.set_episodes(TRAINING_EPISODES_ENA)

		# 		ppo_agent = PPOAgent(
		# 			brain=ppo_actor_brain, critic_brain=ppo_critic_brain, make_env=make_env
		# 		)
		# 		ppo_agent.set_name("PPO Baseline")
		# 		ppo_agent.set_episodes(TRAINING_EPISODES_BASELINE)

		# 		sac_agent = SACAgent(brain=sac_brain, make_env=make_env, update_frequency=20, gradient_steps=5)
		# 		sac_agent.set_name("SAC Baseline")
		# 		sac_agent.set_episodes(TRAINING_EPISODES_BASELINE)

		# 3. Training Phase
		agent_training(
			agents=[
				ena_agent_01,
				# ena_agent_02,
				# ppo_agent,
				# sac_agent,
			],
			experiment_id=experiment_id,
			output_dir=output_dir,
			output_file=output_file,
			queue=queue,
			worlds=worlds,
		)

		# 4. Testing Phase
		ena_agent_01.set_episodes(TESTING_EPISODES)
		# ena_agent_02.set_episodes(TESTING_EPISODES)
		# ppo_agent.set_episodes(TESTING_EPISODES)
		# sac_agent.set_episodes(TESTING_EPISODES)

		ena_agent_01.load_gladiators()
		# ena_agent_02.load_gladiators()

		agent_testing(
			agents=[
				ena_agent_01,
				# ena_agent_02,
				# ppo_agent,
				# sac_agent,
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
	output_dir = os.path.join("cheetah_outputs", f"exp_{experiment_id + 1}")
	os.makedirs(output_dir, exist_ok=True)
	crash_log_path = os.path.join(output_dir, "crash.log")

	crash_file = None
	try:
		# pylint: disable=consider-using-with
		crash_file = open(crash_log_path, "w", encoding="utf-8")
		faulthandler.enable(file=crash_file)
	except Exception:  # pylint: disable=broad-except
		pass

	try:
		run_experiment(experiment_id, output_dir, reporting_queue)
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


if __name__ == "__main__":
	vprint(
		f"[green]Hardware:[/green] GPU detection successful: {GPUS}."
		if GPUS
		else "[yellow]Hardware:[/yellow] No GPU detected. Defaulting to CPU."
	)

	with Manager() as manager:
		REPORTING_QUEUE = manager.Queue()
		REPORTER = ProgressReporter(REPORTING_QUEUE, NUM_EXPERIMENTS)
		REPORTER_THREAD = threading.Thread(target=REPORTER.run, daemon=True)
		REPORTER_THREAD.start()

		vprint(f"\n[bold green]{'='*60}[/bold green]")
		vprint(
			f"[bold]HALFCHEETAH: LAUNCHING {NUM_EXPERIMENTS} INDEPENDENT EXPERIMENTS[/bold]"
		)
		vprint(f"Utilizing [cyan]{MAX_WORKERS}[/cyan] CPU cores.")
		vprint(f"[bold green]{'='*60}[/bold green]\n")

		with concurrent.futures.ProcessPoolExecutor(
			max_workers=MAX_WORKERS
		) as executor:
			futures = {}
			for exp_id in range(NUM_EXPERIMENTS):
				exp_dir = os.path.join("outputs_cheetah", f"exp_{exp_id + 1}")
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
				except Exception as error:  # pylint: disable=broad-except
					vprint(
						f"[bold red]Experiment {exp_id + 1} crashed: {error}[/bold red]"
					)

		REPORTING_QUEUE.put(None)
		REPORTER_THREAD.join(timeout=2.0)

	vprint("\n[bold green]" + "=" * 60 + "[/bold green]")
	vprint(f"[bold]ALL {NUM_EXPERIMENTS} EXPERIMENTS COMPLETE[/bold]")
	vprint(
		"Check the [cyan]outputs_cheetah/[/cyan] directory for detailed logs and plots."
	)
	vprint("[bold green]" + "=" * 60 + "[/bold green]")
