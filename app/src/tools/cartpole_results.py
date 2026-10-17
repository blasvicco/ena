#!/usr/bin/env python
"""CartPole's paper results beyond the shared tables: reliability on the training environments, and steps to first solve."""
# CartPole's results for the paper beyond the shared tables: each replicate's test reliability on the training
# environments alone (the unseen ones left out), and the comparison's steps to first solve as medians and quartiles,
# compared by a rank test, since a few slow replicates dominate their mean. paper_results.py writes them.

# General imports
import os
from typing import Callable

# Libs imports
import pandas as pd

# App imports
from tools.metrics import reliability
from tools.paper_stats import format_value, mann_whitney_p, p_macro, quartiles

TRAINING_RELIABILITY = "Training Reliability (%)"


def add_training_reliability(results: dict, study_dir: str, train_envs: list, threshold: float) -> None:
	"""Add each agent's test reliability on the training environments to every replicate's Test table."""
	# results is collect_results' {phase: [per-replicate DataFrames with Agent and Exp]}; the reliability is read from
	# each agent's live_testing CSV, over the test episodes played in the training environments.
	for frame in results["Test"]:
		frame[TRAINING_RELIABILITY] = [
			_training_reliability(os.path.join(study_dir, exp_dir, f"live_testing_{agent}.csv"), train_envs, threshold)
			for agent, exp_dir in zip(frame["Agent"], frame["Exp"])
		]


def _training_reliability(path: str, train_envs: list, threshold: float) -> float:
	"""Return the reliability of one agent's test episodes in the training environments."""
	testing = pd.read_csv(path, usecols=["env_id", "score"])
	return reliability(testing.loc[testing["env_id"].isin(train_envs), "score"], threshold)


def first_solve_entries(study: str, values: Callable[[str, str], pd.Series], agents: list, envs: list) -> dict:
	"""Steps to first solve: quartiles and unsolved replicates per agent and environment; the last agent vs each other by rank test."""
	# values(agent, metric) gives the agent's per-replicate values of an Adaptation metric ("<env>:steps", "<env>:solved").
	# A replicate that never solved an environment is counted at the end of its phase, the largest value either way, so
	# the rank test treats it correctly. Keys: <study>/<agent>/adapt/<env>:steps/q1|median|q3|unsolved and
	# <study>/<baseline>-vs-<agent>/adapt/<env>:steps/p.
	agent, entries = agents[-1], {}
	for env in envs:
		metric = f"{env}:steps"
		for name in agents:
			key = f"{study}/{name}/adapt/{metric}"
			low, median, high = (format_value(value, "steps") for value in quartiles(values(name, metric)))
			entries.update({f"{key}/q1": low, f"{key}/median": median, f"{key}/q3": high})
			entries[f"{key}/unsolved"] = str(int((values(name, f"{env}:solved") == 0).sum()))
		for baseline in agents[:-1]:
			entries[f"{study}/{baseline}-vs-{agent}/adapt/{metric}/p"] = p_macro(mann_whitney_p(values(baseline, metric), values(agent, metric)))
	return entries
