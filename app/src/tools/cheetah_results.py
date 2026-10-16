#!/usr/bin/env python
"""HalfCheetah's results for the paper: per replicate, the end of each training phase (the training return over its last
100k env-steps), the test return there, and the retention and absolute forgetting of the environments left behind; ENA
against each baseline on both; the hardware the wall-clock times were measured on; its table and figure.
paper_results.py writes them."""

# General imports
import math
import os

# Libs imports
import pandas as pd

# App imports
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


def _complete_replicates(study_dir: str) -> list:
	"""The study's finished replicate directories (their outputs.txt holds the test results), sorted."""
	finished = []
	for exp_dir in sorted(os.listdir(study_dir)):
		path = os.path.join(study_dir, exp_dir, "outputs.txt")
		if exp_dir.startswith("exp_") and os.path.exists(path):
			with open(path, encoding="utf-8") as outputs:
				if "Test Results ---" in outputs.read():
					finished.append(exp_dir)
	return finished


def end_of_phase(training: pd.DataFrame, env: str, last_steps: int = END_OF_PHASE_STEPS) -> float:
	"""Mean training return of env's episodes that end within the last `last_steps` env-steps of its phase (NaN if the
	environment was not trained). training is a live_training CSV: env_id, cumulative env_steps, score."""
	phase = training[training["env_id"] == env]
	if phase.empty:
		return float("nan")
	return float(phase.loc[phase["env_steps"] > phase["env_steps"].max() - last_steps, "score"].mean())


def retention_rows(study_dir: str, agents: list, train_envs: list, last_steps: int = END_OF_PHASE_STEPS) -> pd.DataFrame:
	"""Per finished replicate, agent and training environment: the end of its phase (end_of_phase over its last `last_steps`
	env-steps, from the training episodes), the mean test return there, retention = test / end (%, none when the phase
	ended at or below zero: nothing learned to retain) and the drop, end - test. Any study whose replicates log the same
	live_training/live_testing CSVs can use it (the CartPole challenge does)."""
	rows = []
	for exp_dir in _complete_replicates(study_dir):
		for agent in agents:
			training_path = os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv")
			testing_path = os.path.join(study_dir, exp_dir, f"live_testing_{agent}.csv")
			if not os.path.exists(training_path) or not os.path.exists(testing_path):
				continue
			training, testing = pd.read_csv(training_path), pd.read_csv(testing_path)
			for env in train_envs:
				end = end_of_phase(training, env, last_steps)
				test = testing.loc[testing["env_id"] == env, "score"].mean()
				retention = 100 * test / end if end > 0 else float("nan")
				rows.append({"Exp": exp_dir, "Agent": agent, "Env": env, "end": end, "test": test, "retention": retention, "drop": end - test})
	return pd.DataFrame(rows, columns=["Exp", "Agent", "Env", "end", "test", "retention", "drop"])


def _return(value: float) -> str:
	"""A return as the paper prints it: a whole number with thousands separators."""
	return "n/a" if value is None or math.isnan(value) else f"{value:,.0f}"


def _left_behind(rows: pd.DataFrame, left_behind: list, column: str = "retention") -> pd.DataFrame:
	"""Each replicate's column (retention or drop) averaged over the environments left behind: one value per (Agent, Exp)."""
	return rows[rows["Env"].isin(left_behind)].groupby(["Agent", "Exp"], sort=False)[column].mean().dropna().reset_index()


def retention_entries(study: str, rows: pd.DataFrame, left_behind: list) -> dict:
	"""Mean and std per agent and training environment of the end of phase, retention and absolute forgetting (the drop,
	end - test), and of each replicate's retention and forgetting over the environments left behind ('left'). The test
	return is summary_entries' (same mean, outputs.txt)."""
	entries = {}
	for (agent, env), group in rows.groupby(["Agent", "Env"], sort=False):
		values = group["end"].dropna()
		entries[f"{study}/{agent}/end/{env}/mean"] = _return(values.mean())
		entries[f"{study}/{agent}/end/{env}/std"] = _return(values.std() if len(values) > 1 else float("nan"))
		values = group["retention"].dropna()
		entries[f"{study}/{agent}/retention/{env}/mean"] = format_value(values.mean() if len(values) else float("nan"), "retention")
		entries[f"{study}/{agent}/retention/{env}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "retention")
		values = group["drop"].dropna()
		entries[f"{study}/{agent}/forgetting/{env}/mean"] = _return(values.mean() if len(values) else float("nan"))
		entries[f"{study}/{agent}/forgetting/{env}/std"] = _return(values.std() if len(values) > 1 else float("nan"))
	for agent, group in _left_behind(rows, left_behind).groupby("Agent", sort=False):
		values = group["retention"]
		entries[f"{study}/{agent}/retention/left/mean"] = format_value(values.mean(), "retention")
		entries[f"{study}/{agent}/retention/left/std"] = format_value(values.std() if len(values) > 1 else float("nan"), "retention")
	for agent, group in _left_behind(rows, left_behind, "drop").groupby("Agent", sort=False):
		values = group["drop"]
		entries[f"{study}/{agent}/forgetting/left/mean"] = _return(values.mean())
		entries[f"{study}/{agent}/forgetting/left/std"] = _return(values.std() if len(values) > 1 else float("nan"))
	return entries


def retention_comparisons(study: str, rows: pd.DataFrame, left_behind: list, agent: str, baselines: list) -> dict:
	"""agent vs each baseline on the left-behind retention, one Holm family over the baselines: Welch p (raw and
	Holm-adjusted), difference agent - baseline with its interval, effect size d, and replicates per agent for 80% power
	at the family's most stringent level."""
	per_replicate = _left_behind(rows, left_behind)
	values = {name: per_replicate.loc[per_replicate["Agent"] == name, "retention"] for name in [agent, *baselines]}
	return _holm_comparisons(study, values, agent, baselines, "retention")


def forgetting_comparisons(study: str, rows: pd.DataFrame, left_behind: list, agent: str, baselines: list) -> dict:
	"""agent vs each baseline on the left-behind absolute forgetting (end - test, in return units), its own Holm family
	over the baselines, with the same entries as retention_comparisons."""
	per_replicate = _left_behind(rows, left_behind, "drop")
	values = {name: per_replicate.loc[per_replicate["Agent"] == name, "drop"] for name in [agent, *baselines]}
	return _holm_comparisons(study, values, agent, baselines, "forgetting")


def _holm_comparisons(study: str, values: dict, agent: str, baselines: list, metric: str) -> dict:
	"""agent vs each baseline on {name: per-replicate values}, one Holm family over the baselines, under
	<study>/<baseline>-vs-<agent>/<metric>/left."""
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
	"""The machine and thread counts the study's finished replicates ran with (the runner's 'Hardware:' line); every
	distinct one, so replicates from different machines are never reported as one."""
	machines = []
	for exp_dir in _complete_replicates(study_dir):
		with open(os.path.join(study_dir, exp_dir, "outputs.txt"), encoding="utf-8") as outputs:
			for line in outputs:
				if line.strip().startswith("Hardware:"):
					machine = line.strip()[len("Hardware:") :].strip()
					if machine not in machines:
						machines.append(machine)
	return {f"{study}/hardware": "; ".join(machines)} if machines else {}


# pylint: disable=too-many-arguments,too-many-positional-arguments
def retention_table(rows: pd.DataFrame, left_behind: list, agents: list, minutes: dict, caption: str, label: str) -> str:
	"""A LaTeX table, one row per agent: per environment left behind, the end of its phase, the test return and retention
	(means over replicates); then the mean left-behind retention and absolute forgetting, and the training time (minutes,
	as given)."""
	headers = [part for env in left_behind for part in (f"{env.split('-')[0]} end", f"{env.split('-')[0]} test", f"{env.split('-')[0]} ret. (\\%)")]
	lines = [
		"\\hline",
		"\\textbf{Agent} & " + " & ".join(f"\\textbf{{{header}}}" for header in [*headers, "Retention (\\%)", "Forgetting", "Time (min)"]) + " \\\\",
		"\\hline",
	]
	means = rows.groupby(["Agent", "Env"])[["end", "test", "retention"]].mean()
	left = _left_behind(rows, left_behind).groupby("Agent")["retention"].mean()
	forgetting = _left_behind(rows, left_behind, "drop").groupby("Agent")["drop"].mean()
	for agent in agents:
		cells = []
		for env in left_behind:
			if (agent, env) in means.index:
				row = means.loc[(agent, env)]
				cells += [_return(row["end"]), _return(row["test"]), format_value(row["retention"], "retention")]
			else:
				cells += ["--"] * 3
		cells += [format_value(left.get(agent, float("nan")), "retention"), _return(forgetting.get(agent, float("nan"))), minutes.get(agent, "--")]
		lines.append(f"{agent} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(caption, label, f"|l|{'c|' * (len(headers) + 3)}", lines, fit="page")


def retention_figure(rows: pd.DataFrame, left_behind: list, agents: list, path: str) -> str:
	"""Box plot of each replicate's left-behind retention per agent, each replicate drawn as a dot; returns the path."""
	per_replicate = _left_behind(rows, left_behind)
	samples = [per_replicate.loc[per_replicate["Agent"] == agent, "retention"].to_numpy() for agent in agents]
	return replicate_box_figure(samples, agents, "Retention (%)", path, reference=100)
