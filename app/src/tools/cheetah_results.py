#!/usr/bin/env python
"""HalfCheetah's paper results: end of phase, test return, retention and absolute forgetting, comparisons, hardware."""
# HalfCheetah's results for the paper: per replicate, the end of each training phase (the training return over its last
# 100k env-steps; the CartPole ablation's, over its last episodes), the test return there, and the retention and absolute
# forgetting of the environments left behind; ENA's retention against its gladiators' episodes only; ENA against each
# baseline on both; the hardware the wall-clock times were measured on; its table and figure.
# paper_results.py writes them.

# General imports
import os

# Libs imports
import pandas as pd

# App imports
from tools.replicates import get_experiment_directories
from tools.paper_config import latex_table
from tools.paper_figures import replicate_box_figure
from tools.paper_stats import ALPHA, difference_entries, format_value, holm, p_macro, welch_p


# HalfCheetah: the test sequence, the training environments (in order) and the agents, ENA last.
CHEETAH_ENVS = ["Earth-Standard", "Ice-Floor", "Heavy-Rough", "High-Wind", "Mars-Slippery"]
CHEETAH_TRAIN_ENVS = ["Earth-Standard", "Heavy-Rough", "Mars-Slippery"]
CHEETAH_LEFT_BEHIND = CHEETAH_TRAIN_ENVS[:-1]  # the environments a later phase can make the agent forget
CHEETAH_AGENTS = ["PPO", "SAC", "ENA-Combined"]
# The end of a phase is the mean training return of the episodes ending in its last 100k env-steps (10% of the budget):
# every agent learns, noise included, until the phase ends, and the same interaction counts for each.
END_OF_PHASE_STEPS = 100_000


def end_of_phase(training: pd.DataFrame, env: str, last_steps: int = END_OF_PHASE_STEPS, last_episodes: int | None = None) -> float:
	"""Return the mean training return of the episodes ending in the last `last_steps` env-steps (or `last_episodes` episodes) of env's phase."""
	# training is a live_training CSV: env_id, cumulative env_steps, score. A study with an episode budget (the CartPole
	# ablation) passes last_episodes: its variants play the same episodes but very different numbers of env-steps.
	phase = training[training["env_id"] == env]
	if last_episodes is not None:
		return float(phase["score"].tail(last_episodes).mean())
	return float(phase.loc[phase["env_steps"] > phase["env_steps"].max() - last_steps, "score"].mean())


# pylint: disable=too-many-arguments
def retention_rows(
	study_dir: str,
	agents: list,
	train_envs: list,
	last_steps: int = END_OF_PHASE_STEPS,
	last_episodes: int | None = None,
	*,
	archived_only: bool = False,
) -> pd.DataFrame:
	"""Tabulate, per replicate, agent and training environment, the end of phase, test return, retention and drop."""
	# Per replicate, agent and training environment: the end of its phase (end_of_phase over its last `last_steps`
	# env-steps or `last_episodes` episodes, from the training episodes), the mean test return there, retention = test /
	# end (%, none when the phase ended at or below zero: nothing learned to retain) and the drop, end - test. Any study
	# whose replicates log the same live_training/live_testing CSVs can use it (the CartPole challenge and ablation do).
	# archived_only (ENA) keeps the same window but averages only the episodes a gladiator played (is_gladiator); a
	# window without one has no end of phase.
	rows = []
	for exp_dir in get_experiment_directories(study_dir):
		for agent in agents:
			training = pd.read_csv(os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv"))
			testing = pd.read_csv(os.path.join(study_dir, exp_dir, f"live_testing_{agent}.csv"))
			if archived_only:
				training["score"] = training["score"].where(training["is_gladiator"].astype(bool))
			for env in train_envs:
				end = end_of_phase(training, env, last_steps, last_episodes)
				test = testing.loc[testing["env_id"] == env, "score"].mean()
				retention = 100 * test / end if end > 0 else float("nan")
				rows.append({"Exp": exp_dir, "Agent": agent, "Env": env, "end": end, "test": test, "retention": retention, "drop": end - test})
	return pd.DataFrame(rows, columns=["Exp", "Agent", "Env", "end", "test", "retention", "drop"])


def _left_behind(rows: pd.DataFrame, left_behind: list, column: str = "retention") -> pd.DataFrame:
	"""Each replicate's column (retention or drop) averaged over the environments left behind: one value per (Agent, Exp)."""
	return rows[rows["Env"].isin(left_behind)].groupby(["Agent", "Exp"], sort=False)[column].mean().dropna().reset_index()


def retention_entries(study: str, rows: pd.DataFrame, left_behind: list) -> dict:
	"""Summarize end of phase, retention and absolute forgetting per agent and environment, and over those left behind."""
	# Mean and std per agent and training environment of the end of phase, retention and absolute forgetting (the drop,
	# end - test), and of each replicate's retention and forgetting over the environments left behind ('left'). The test
	# return is summary_entries' (same mean, outputs.txt).
	entries = {}
	for (agent, env), group in rows.groupby(["Agent", "Env"], sort=False):
		values = group["end"].dropna()
		entries[f"{study}/{agent}/end/{env}/mean"] = format_value(values.mean(), "return")
		entries[f"{study}/{agent}/end/{env}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "return")
		values = group["retention"].dropna()
		entries[f"{study}/{agent}/retention/{env}/mean"] = format_value(values.mean() if len(values) else float("nan"), "retention")
		entries[f"{study}/{agent}/retention/{env}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "retention")
		values = group["drop"].dropna()
		entries[f"{study}/{agent}/forgetting/{env}/mean"] = format_value(values.mean() if len(values) else float("nan"), "return")
		entries[f"{study}/{agent}/forgetting/{env}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "return")
	for agent, group in _left_behind(rows, left_behind).groupby("Agent", sort=False):
		values = group["retention"]
		entries[f"{study}/{agent}/retention/left/mean"] = format_value(values.mean(), "retention")
		entries[f"{study}/{agent}/retention/left/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "retention")
	for agent, group in _left_behind(rows, left_behind, "drop").groupby("Agent", sort=False):
		values = group["drop"]
		entries[f"{study}/{agent}/forgetting/left/mean"] = format_value(values.mean(), "return")
		entries[f"{study}/{agent}/forgetting/left/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "return")
	return entries


def archived_reference_entries(study: str, study_dir: str, agent: str) -> dict:
	"""ENA's left-behind retention against its gladiators' end-of-phase episodes, over the replicates that have them."""
	# The end of phase averages every training episode, including those in which ENA plays a specialist outside the Hall
	# of Fame, such as the epsilon-greedy picks, while the test plays gladiators only. This reference keeps the episodes a
	# gladiator played, over the replicates in which one played at the end of every phase left behind. Keys as
	# retention_entries', under <study>/archived, and <study>/archived/n.
	rows = retention_rows(study_dir, [agent], CHEETAH_LEFT_BEHIND, archived_only=True)
	rows = rows[rows.groupby("Exp")["end"].transform("count") == len(CHEETAH_LEFT_BEHIND)]
	return {**retention_entries(f"{study}/archived", rows, CHEETAH_LEFT_BEHIND), f"{study}/archived/n": str(rows["Exp"].nunique())}


def retention_comparisons(study: str, rows: pd.DataFrame, left_behind: list, agent: str, baselines: list) -> dict:
	"""Compare the agent with each baseline on left-behind retention, as one Holm family over the baselines."""
	# agent vs each baseline on the left-behind retention, one Holm family over the baselines: Welch p (raw and
	# Holm-adjusted), difference agent - baseline with its interval, effect size d, and replicates per agent for 80% power
	# at the family's most stringent level.
	per_replicate = _left_behind(rows, left_behind)
	values = {name: per_replicate.loc[per_replicate["Agent"] == name, "retention"] for name in [agent, *baselines]}
	return _holm_comparisons(study, values, agent, baselines, "retention")


def forgetting_comparisons(study: str, rows: pd.DataFrame, left_behind: list, agent: str, baselines: list) -> dict:
	"""Compare the agent with each baseline on left-behind absolute forgetting, as its own Holm family."""
	# agent vs each baseline on the left-behind absolute forgetting (end - test, in return units), its own Holm family
	# over the baselines, with the same entries as retention_comparisons.
	per_replicate = _left_behind(rows, left_behind, "drop")
	values = {name: per_replicate.loc[per_replicate["Agent"] == name, "drop"] for name in [agent, *baselines]}
	return _holm_comparisons(study, values, agent, baselines, "forgetting")


def _holm_comparisons(study: str, values: dict, agent: str, baselines: list, metric: str) -> dict:
	"""Compare the agent with each baseline on per-replicate values, as one Holm family over the baselines."""
	# agent vs each baseline on {name: per-replicate values}, one Holm family over the baselines, under
	# <study>/<baseline>-vs-<agent>/<metric>/left.
	entries, tested = {}, []
	for baseline in baselines:
		key = f"{study}/{baseline}-vs-{agent}/{metric}/left"
		if values[baseline].empty or values[agent].empty:
			continue
		tested.append((key, welch_p(values[baseline], values[agent])))
		entries.update(difference_entries(key, values[baseline], values[agent], ALPHA / len(baselines), metric))
	for (key, _p_value), adjusted in zip(tested, holm([p_value for _key, p_value in tested], family_size=len(baselines))):
		entries[f"{key}/pholm"] = p_macro(adjusted)
	return entries


def hardware_entries(study: str, study_dir: str) -> dict:
	"""List every distinct machine and thread count the study's replicates ran with."""
	# The machine and thread counts the study's replicates ran with (the runner's 'Hardware:' line); every
	# distinct one, so replicates from different machines are never reported as one.
	machines = []
	for exp_dir in get_experiment_directories(study_dir):
		with open(os.path.join(study_dir, exp_dir, "outputs.txt"), encoding="utf-8") as outputs:
			for line in outputs:
				if line.strip().startswith("Hardware:"):
					machine = line.strip()[len("Hardware:") :].strip()
					if machine not in machines:
						machines.append(machine)
	return {f"{study}/hardware": "; ".join(machines)}


# pylint: disable=too-many-arguments,too-many-positional-arguments
def retention_table(rows: pd.DataFrame, left_behind: list, agents: list, minutes: dict | None, caption: str, label: str) -> str:
	"""Build the LaTeX retention table: per agent, end of phase, test return and retention, then the left-behind means."""
	# A LaTeX table, one row per agent: per environment left behind, the end of its phase, the test return and retention
	# (means over replicates); then the mean left-behind retention and absolute forgetting, and, given minutes, the
	# training time (minutes, as given).
	headers = [part for env in left_behind for part in (f"{env.split('-')[0]} end", f"{env.split('-')[0]} test", f"{env.split('-')[0]} ret. (\\%)")]
	headers += ["Retention (\\%)", "Forgetting"] + ([] if minutes is None else ["Time (min)"])
	lines = ["\\hline", "\\textbf{Agent} & " + " & ".join(f"\\textbf{{{header}}}" for header in headers) + " \\\\", "\\hline"]
	means = rows.groupby(["Agent", "Env"])[["end", "test", "retention"]].mean()
	left = _left_behind(rows, left_behind).groupby("Agent")["retention"].mean()
	forgetting = _left_behind(rows, left_behind, "drop").groupby("Agent")["drop"].mean()
	for agent in agents:
		cells = []
		for env in left_behind:
			row = means.loc[(agent, env)]
			cells += [format_value(row["end"], "return"), format_value(row["test"], "return"), format_value(row["retention"], "retention")]
		cells += [format_value(left[agent], "retention"), format_value(forgetting[agent], "return")] + ([] if minutes is None else [minutes[agent]])
		lines.append(f"{agent} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(caption, label, f"|l|{'c|' * len(headers)}", lines, fit="page")


def retention_figure(rows: pd.DataFrame, left_behind: list, agents: list, path: str) -> str:
	"""Box plot of each replicate's left-behind retention per agent, each replicate drawn as a dot; returns the path."""
	per_replicate = _left_behind(rows, left_behind)
	samples = [per_replicate.loc[per_replicate["Agent"] == agent, "retention"].to_numpy() for agent in agents]
	return replicate_box_figure(samples, agents, "Retention (%)", path, reference=100)
