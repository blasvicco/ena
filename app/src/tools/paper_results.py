#!/usr/bin/env python
"""Generate every number and results table the paper quotes, from the study outputs.

Writes app/latex/generated/results.tex — one \\resdef{key}{value} per number, quoted in the paper as
\\Res{key} — and app/latex/generated/table_*.tex. Re-run it whenever a study gains replicates; the
paper then picks up the new numbers on its next build. Never edit the generated files by hand.

Keys:
	<study>/n                                   replicates parsed
	<study>/<agent>/<phase>/<metric>/mean|std   phase: train, test, compute, adapt
	<study>/<agent>/time/min/mean|std           wall-clock training time, minutes
	<study>/<A>-vs-<B>/<phase>/<metric>/p|diff  Welch's t-test p-value and mean(B) - mean(A)
	<study>/<agent>/end|retention/<env>/mean|std  HalfCheetah: end of phase (its last 3 checks), retention (%)
	<study>/<agent>/retention/left/mean|std     retention over the environments left behind
	<study>/<B>-vs-<agent>/retention/left/p|pholm|diff|ci/low|ci/high|d|nreq
	<study>/hardware                            the machine and thread counts the replicates ran with

Usage:
	python src/tools/paper_results.py
"""

# General imports
import math
import os
import re
import sys

# Libs imports
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from cartpole_ena_ablation import make_training_worlds, make_unseen_worlds
from cartpole_ena_compute import REFERENCES as COMPUTE_REFERENCES
from half_cheetah import SOLVED_SCORE as CHEETAH_SOLVED, Y_LIM as CHEETAH_Y_LIM, build_ena_agent as build_cheetah_ena
from tools.aggregate_metrics import collect_results
from tools.cheetah_results import (
	CHEETAH_AGENTS,
	CHEETAH_ENVS,
	CHEETAH_LEFT_BEHIND,
	CHEETAH_TRAIN_ENVS,
	hardware_entries,
	retention_comparisons,
	retention_entries,
	retention_figure,
	retention_rows,
	retention_table,
)
from tools.detection_ablation import SCREENED, pass_table, study_labels
from tools.detection_stats import study_detection, summarize as summarize_detection
from tools.plot_study_curves import plot as plot_study_curves
from tools.paper_figures import replicate_box_figure
from tools.paper_stats import (
	ALPHA,
	CI_LEVEL,
	POWER,
	difference_entries,
	clopper_pearson,
	detectable_d,
	format_value,
	holm,
	mcnemar_p,
	p_macro,
	welch_p,
)
from tools.paper_config import cartpole_rows, cheetah_rows, config_entries, environment_table, latex_table
from tools.sac_discrete_entropy_check import OUTPUT_DIR as SACD_DIR

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATED_DIR = os.path.join(APP_DIR, "latex", "generated")
IMAGES_DIR = os.path.join(APP_DIR, "latex", "images")
# What each study contributes to the paper besides macros: generated tables and figures. A study whose
# outputs are missing gets placeholder tables and no figures, never an older run's.
STUDY_TABLES = {
	"ablation": ["table_ablation_test.tex", "table_detection.tex"],
	"challenge": ["table_challenge_test.tex", "table_challenge_adapt.tex"],
	"detection_ablation": ["table_chirp_ablation.tex"],
	"cheetah": ["table_cheetah_test.tex", "table_cheetah_retention.tex"],
}
STUDY_FIGURES = {
	"ablation": ["ablation_reliability.pdf"],
	"challenge": ["challenge_reliability.pdf", "challenge_test_curves.pdf"],
	"cheetah": ["cheetah_retention.pdf", "cheetah_training_curves.pdf", "cheetah_test_curves.pdf"],
}
PHASE_KEYS = {"Training": "train", "Test": "test", "Compute Cost": "compute", "Adaptation": "adapt"}
METRIC_SLUGS = {
	"Reliability (%)": "rel",
	"Stability Gap (↓)": "gap",
	"Efficiency (Score/s)": "eff",
	"Total Candidate Env-Steps": "candsteps",
	"Avg Eval Depth": "evaldepth",
	"Avg Candidates/Gen": "candidates",
}
TIME_PATTERN = re.compile(r"^\s+(?P<agent>\S.*?): (?P<seconds>[\d.]+) seconds(?:, (?P<steps>[\d,]+) env-steps)?\s*$")

CARTPOLE_ENVS = ["Mars-Light", "Neptune", "Jupiter-Heavy", "Brown-dwarf", "Earth-Standard"]
CARTPOLE_TRAIN_ENVS = ["Mars-Light", "Jupiter-Heavy", "Earth-Standard"]
ABLATION_AGENTS = [
	"Random",
	"ENA-NoSkill",
	"ENA-Plasticity",
	"ENA-HoF",
	"ENA-HoF+EnvDetection",
	"ENA-Plasticity+HoF",
	"ENA-Combined",
]
# The ablation's planned comparisons, one family (Holm): each skill added alone and on top of the others.
ABLATION_FAMILY = [
	("ENA-NoSkill", "ENA-Plasticity"),
	("ENA-NoSkill", "ENA-HoF"),
	("ENA-HoF", "ENA-HoF+EnvDetection"),
	("ENA-Plasticity", "ENA-Plasticity+HoF"),
	("ENA-HoF", "ENA-Plasticity+HoF"),
	("ENA-Plasticity+HoF", "ENA-Combined"),
	("ENA-HoF+EnvDetection", "ENA-Combined"),
]
# No Random row: the challenge does not run it (the floor is in the ablation table).
CHALLENGE_AGENTS = ["DQN", "PPO", "SAC-Discrete", "ENA-Combined"]
TEST_METRICS = ["Reliability (%)", "Stability Gap (↓)"] + [f"{env} Avg" for env in CARTPOLE_ENVS]
DETECTION_SLUGS = {"unrecognized": "unrec", "changes": "changes", "first_change": "firstchange", "niches": "niches"}
# Environment-recognition table columns: (metric, header).
DETECTION_COLUMNS = [("unrecognized", "Unrec. (\\%)"), ("changes", "Changes"), ("first_change", "First (ep.)"), ("niches", "Niches")]
DETECTION_PHASES = [("train", "Training"), ("test", "Test")]
CONTROLLER_ALTERNATIVES = ["ENA-Combined-QuadraticPlasticity", "ENA-Combined-FuzzyPlasticity"]
STUDIES = {
	"ablation": "cartpole_ena_ablation_outputs",
	"compute": "cartpole_ena_compute_outputs",
	"controller": "cartpole_ena_plasticity_controller_outputs",
	"challenge": "cartpole_challenge_outputs",
	"cheetah": "half_cheetah_outputs",
}
DETECTION_ABLATION_DIR = os.path.join(APP_DIR, "detection_ablation_outputs")  # tools/detection_ablation.py
# detection_ablation.BASELINES label -> macro slug.
BASELINE_SLUGS = {
	"Transition model, changing behavior": "transition",
	"Transition model, one fixed gait": "transitionfixed",
	"Chirp (study), changing behavior": "chirp",
}


def slug(metric: str) -> str:
	"""Short key for a metric column name."""
	if metric in METRIC_SLUGS:
		return METRIC_SLUGS[metric]
	return re.sub(r" Avg$", "", metric).replace(" ", "")


def parse_training_times(outputs_path: str) -> dict:
	"""{agent: (training seconds, env-steps or None)} from a replicate's 'Training Performance vs Time' block."""
	times = {}
	with open(outputs_path, encoding="utf-8") as outputs:
		for line in outputs:
			match = TIME_PATTERN.match(line.rstrip("\n"))
			if match:
				steps = match.group("steps")
				times[match.group("agent")] = (float(match.group("seconds")), int(steps.replace(",", "")) if steps else None)
	return times


def _pooled(results: dict, phase: str) -> pd.DataFrame:
	"""Every replicate's table of one phase, stacked (empty frame if none)."""
	frames = results.get(phase) or []
	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Agent", "Exp"])


def _values(pooled: pd.DataFrame, agent: str, metric: str) -> pd.Series:
	"""One agent's per-replicate values of one metric, numeric."""
	if metric not in pooled:
		return pd.Series(dtype=float)
	return pd.to_numeric(pooled.loc[pooled["Agent"] == agent, metric].astype(str).str.replace(",", ""), errors="coerce").dropna()


def summary_entries(study: str, results: dict) -> dict:
	"""Mean and sample std of every metric for every agent and phase, plus the replicate count."""
	entries = {}
	replicates = set()
	for phase, phase_key in PHASE_KEYS.items():
		pooled = _pooled(results, phase)
		replicates.update(pooled["Exp"].unique())
		for agent in pooled["Agent"].unique():
			for metric in (column for column in pooled.columns if column not in ("Agent", "Exp")):
				values = _values(pooled, agent, metric)
				if values.empty:
					continue
				key = f"{study}/{agent}/{phase_key}/{slug(metric)}"
				entries[f"{key}/mean"] = format_value(values.mean(), slug(metric))
				entries[f"{key}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), slug(metric))
	entries[f"{study}/n"] = str(len(replicates))
	return entries


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def comparison_entries(
	study: str, results_a: dict, agent_a: str, results_b: dict, agent_b: str, phase: str, metrics: list, alpha: float = ALPHA
) -> dict:
	"""agent_b vs. agent_a on each metric (the agents may come from different studies): Welch p, mean difference (b - a)
	with its 95% interval, effect size d, and replicates per agent for 80% power at alpha."""
	entries = {}
	pooled_a, pooled_b = _pooled(results_a, phase), _pooled(results_b, phase)
	for metric in metrics:
		values_a, values_b = _values(pooled_a, agent_a, metric), _values(pooled_b, agent_b, metric)
		if values_a.empty or values_b.empty:
			continue
		key = f"{study}/{agent_a}-vs-{agent_b}/{PHASE_KEYS[phase]}/{slug(metric)}"
		entries.update(difference_entries(key, values_a, values_b, alpha, slug(metric)))
	return entries


def paired_p(pooled_a: pd.DataFrame, agent_a: str, pooled_b: pd.DataFrame, agent_b: str, metric: str) -> tuple:
	"""Paired t-test p-value of b vs. a over the replicates both have (matched by Exp), and the number of pairs."""
	frames = []
	for pooled, agent, column in ((pooled_a, agent_a, "a"), (pooled_b, agent_b, "b")):
		rows = pooled.loc[pooled["Agent"] == agent, ["Exp", metric]].copy()
		rows[column] = pd.to_numeric(rows[metric].astype(str).str.replace(",", ""), errors="coerce")
		frames.append(rows[["Exp", column]].dropna())
	pairs = frames[0].merge(frames[1], on="Exp")
	if len(pairs) < 2:
		return float("nan"), len(pairs)
	differences = pairs["b"] - pairs["a"]
	if differences.std() == 0:
		return (1.0 if differences.mean() == 0 else 0.0), len(pairs)
	return float(stats.ttest_rel(pairs["b"], pairs["a"]).pvalue), len(pairs)


def with_reference_rows(study_results: dict, reference_results: dict, agents: list) -> dict:
	"""study_results plus the reference study's rows of agents for the replicates the study has (same seeds: same runs)."""
	replicates = set(_pooled(study_results, "Test")["Exp"])
	merged = {}
	for phase in set(study_results) | set(reference_results):
		frames = list(study_results.get(phase) or [])
		for frame in reference_results.get(phase) or []:
			rows = frame[frame["Agent"].isin(agents) & frame["Exp"].isin(replicates)]
			if not rows.empty:
				frames.append(rows)
		merged[phase] = frames
	return merged


def compute_ratio_entries(results: dict, full: str, ramp: str, zero: str) -> dict:
	"""Candidate-step ratios within the compute study's replicates: full search over the ramp, the ramp over zero."""
	pooled = _pooled(results, "Compute Cost")
	steps = {agent: _values(pooled, agent, "Total Candidate Env-Steps").mean() for agent in (full, ramp, zero)}
	if any(math.isnan(value) or value == 0 for value in steps.values()):
		return {}
	return {"compute/ratio/full-ramp": f"{steps[full] / steps[ramp]:.0f}", "compute/ratio/ramp-zero": f"{steps[ramp] / steps[zero]:.0f}"}


def compute_rank_entries(results_full: dict, full: str, results_ramp: dict, ramp: str) -> dict:
	"""Where the single full-search run's test reliability falls among the ramp's replicates: how many reached at least it."""
	full_values = _values(_pooled(results_full, "Test"), full, "Reliability (%)")
	ramp_values = _values(_pooled(results_ramp, "Test"), ramp, "Reliability (%)")
	if full_values.empty or ramp_values.empty:
		return {}
	return {"compute/rank/atleast": str(int((ramp_values >= full_values.mean()).sum())), "compute/rank/of": str(len(ramp_values))}


def candidate_share_entries(study: str, results: dict, agent: str) -> dict:
	"""Share (%) of the agent's training env-steps spent evaluating candidates, per replicate, then mean and std."""
	candidates = _pooled(results, "Compute Cost")
	training = _pooled(results, "Adaptation")
	shares = []
	for exp in candidates.loc[candidates["Agent"] == agent, "Exp"]:
		spent = _values(candidates[candidates["Exp"] == exp], agent, "Total Candidate Env-Steps")
		total = _values(training[training["Exp"] == exp], agent, "Train-Env-Steps")
		if not spent.empty and not total.empty and total.iloc[0] > 0:
			shares.append(100 * spent.iloc[0] / total.iloc[0])
	if not shares:
		return {}
	series = pd.Series(shares)
	return {
		f"{study}/{agent}/candshare/mean": f"{series.mean():.1f}",
		f"{study}/{agent}/candshare/std": f"{series.std():.1f}" if len(series) > 1 else "n/a",
	}


def controller_verdict(outcomes: dict) -> str:
	"""The pre-registered rule (docs/PLAN_ena_plasticity_controller.md 3.2), on Holm-adjusted p-values.

	outcomes: {alternative: {"rel": (difference, p), "candsteps": (difference, p), "rel_mean": mean reliability}}.
	An alternative qualifies if its test reliability is significantly higher, or not significantly lower
	with significantly fewer candidate steps; of several, the most reliable is adopted.
	"""
	qualifying = []
	for alternative, outcome in outcomes.items():
		rel_difference, rel_p = outcome["rel"]
		steps_difference, steps_p = outcome["candsteps"]
		higher = rel_difference > 0 and rel_p < ALPHA
		not_lower = not (rel_difference < 0 and rel_p < ALPHA)
		cheaper = steps_difference < 0 and steps_p < ALPHA
		if higher or (not_lower and cheaper):
			qualifying.append((outcome["rel_mean"], alternative))
	return max(qualifying)[1] if qualifying else "linear ramp"


def controller_entries(results_ref: dict, reference: str, results_alt: dict, alternatives: list) -> dict:
	"""The controller study's family: Holm-adjusted Welch p per alternative and metric, paired p, detectable effect, verdict."""
	metrics = (("Test", "Reliability (%)", "rel"), ("Compute Cost", "Total Candidate Env-Steps", "candsteps"))
	entries, outcomes = {}, {alternative: {} for alternative in alternatives}
	for phase, metric, metric_slug in metrics:
		pooled_ref, pooled_alt = _pooled(results_ref, phase), _pooled(results_alt, phase)
		tested = []
		for alternative in alternatives:
			values_ref, values_alt = _values(pooled_ref, reference, metric), _values(pooled_alt, alternative, metric)
			if values_ref.empty or values_alt.empty:
				continue
			key = f"controller/{reference}-vs-{alternative}/{PHASE_KEYS[phase]}/{metric_slug}"
			tested.append((alternative, key, values_alt.mean() - values_ref.mean(), welch_p(values_ref, values_alt)))
			entries[f"{key}/ppaired"] = p_macro(paired_p(pooled_ref, reference, pooled_alt, alternative, metric)[0])
			if metric_slug == "rel":
				outcomes[alternative]["rel_mean"] = values_alt.mean()
		for (alternative, key, difference, _p_value), adjusted in zip(tested, holm([row[3] for row in tested], family_size=len(alternatives))):
			entries[f"{key}/pholm"] = p_macro(adjusted)
			outcomes[alternative][metric_slug] = (difference, adjusted)
	complete = {alternative: outcome for alternative, outcome in outcomes.items() if {"rel", "candsteps", "rel_mean"} <= set(outcome)}
	replicates = _pooled(results_alt, "Test")["Exp"].nunique()
	entries["controller/family"] = str(len(alternatives))
	entries["controller/mde"] = f"{detectable_d(replicates, ALPHA / len(alternatives)):.2f}"
	entries["controller/verdict"] = controller_verdict(complete)
	return entries


def family_entries(study: str, results: dict, pairs: list, metric: str = "Reliability (%)") -> dict:
	"""A study's planned comparisons as one family: Holm-adjusted Welch p per pair, the family size, and the
	smallest effect size the study's replicate count detects with 80% power at the family's most stringent level."""
	pooled = _pooled(results, "Test")
	tested = []
	for agent_a, agent_b in pairs:
		values_a, values_b = _values(pooled, agent_a, metric), _values(pooled, agent_b, metric)
		if not values_a.empty and not values_b.empty:
			tested.append((f"{study}/{agent_a}-vs-{agent_b}/test/{slug(metric)}", welch_p(values_a, values_b)))
	entries = {}
	for (key, _p_value), adjusted in zip(tested, holm([p_value for _key, p_value in tested], family_size=len(pairs))):
		entries[f"{key}/pholm"] = p_macro(adjusted)
	replicates = pooled["Exp"].nunique()
	entries[f"{study}/family"] = str(len(pairs))
	entries[f"{study}/mde"] = f"{detectable_d(replicates, ALPHA / len(pairs)):.2f}"
	return entries


def completed_replicates(study_dir: str) -> int:
	"""How many of the study's replicates finished (their outputs.txt holds the test results)."""
	count = 0
	for exp_dir in os.listdir(study_dir):
		path = os.path.join(study_dir, exp_dir, "outputs.txt")
		if exp_dir.startswith("exp_") and os.path.exists(path):
			with open(path, encoding="utf-8") as outputs:
				count += "Test Results ---" in outputs.read()
	return count


def episode_count_entries(study: str, study_dir: str, agents: list) -> dict:
	"""Mean number of training episodes per environment phase for each agent, over the study's finished replicates."""
	entries = {}
	for agent in agents:
		counts = []
		for exp_dir in sorted(os.listdir(study_dir)):
			path = os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv")
			outputs = os.path.join(study_dir, exp_dir, "outputs.txt")
			if not os.path.exists(path) or not os.path.exists(outputs):
				continue
			with open(outputs, encoding="utf-8") as text:
				if "Test Results ---" not in text.read():
					continue
			env_ids = pd.read_csv(path, usecols=["env_id"])["env_id"]
			counts += env_ids.groupby(env_ids.ne(env_ids.shift()).cumsum()).size().tolist()
		if counts:
			entries[f"{study}/{agent}/episodes/perenv"] = f"{sum(counts) / len(counts):,.0f}"
	return entries


def time_entries(study: str, study_dir: str) -> dict:
	"""Mean and std of each agent's wall-clock training time (minutes) over the study's complete replicates."""
	minutes = {}
	for exp_dir in sorted(os.listdir(study_dir)):
		path = os.path.join(study_dir, exp_dir, "outputs.txt")
		if not exp_dir.startswith("exp_") or not os.path.exists(path):
			continue
		with open(path, encoding="utf-8") as outputs:
			if "Test Results ---" not in outputs.read():
				continue
		for agent, (seconds, _steps) in parse_training_times(path).items():
			minutes.setdefault(agent, []).append(seconds / 60)
	entries = {}
	for agent, values in minutes.items():
		series = pd.Series(values)
		entries[f"{study}/{agent}/time/min/mean"] = format_value(series.mean(), "min")
		entries[f"{study}/{agent}/time/min/std"] = format_value(series.std() if len(series) > 1 else float("nan"), "min")
	return entries


def _latex_safe(value: str) -> str:
	"""Text-mode-safe value ('<' isn't a text-mode character in OT1)."""
	return value if value.startswith("$") else value.replace("<", "\\textless{}")


def write_macros(entries: dict, path: str) -> None:
	"""Write the generated \\resdef file, sorted by key."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as output:
		output.write("% GENERATED by app/src/tools/paper_results.py from the study outputs. Do not edit; re-run the script.\n")
		for key in sorted(entries):
			output.write(f"\\resdef{{{key}}}{{{_latex_safe(entries[key])}}}\n")


# pylint: disable=too-many-arguments,too-many-positional-arguments
def results_table(results: dict, phase: str, agents: list, columns: list, caption: str, label: str) -> str:
	"""A LaTeX table: one row per agent; columns are (metric, header, show_std) with the mean (± std) over replicates."""
	pooled = _pooled(results, phase)
	lines = [
		"\\hline",
		"\\textbf{Agent} & " + " & ".join(f"\\textbf{{{header}}}" for _metric, header, _std in columns) + " \\\\",
		"\\hline",
	]
	for agent in agents:
		cells = []
		for metric, _header, show_std in columns:
			values = _values(pooled, agent, metric)
			if values.empty:
				cells.append("--")
				continue
			cell = format_value(values.mean(), slug(metric))
			if show_std and len(values) > 1:
				cell += f" $\\pm$ {format_value(values.std(), slug(metric))}"
			cells.append(cell)
		lines.append(f"{agent} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(caption, label, f"|l|{'c|' * len(columns)}", lines, fit_width=True)


def sac_discrete_entries(summary_path: str) -> dict:
	"""Median temperature (start, peak) and peak critic value per target-entropy ratio, from sac_discrete_entropy_check.py."""
	if not os.path.exists(summary_path):
		return {}
	entries = {}
	for row in pd.read_csv(summary_path).itertuples():
		key = f"sacd/{row.ratio:g}"
		entries[f"{key}/alphastart"] = f"{row.alpha_start:.1f}"
		entries[f"{key}/alphamax"] = f"{row.alpha_max:,.1f}" if row.alpha_max < 100 else f"{row.alpha_max:,.0f}"
		entries[f"{key}/valuemax"] = f"{row.value_max:,.0f}"
		entries[f"{key}/steps"] = f"{row.steps / 1000:.0f}k"
		entries[f"{key}/seeds"] = str(row.seeds)
	return entries


def detection_entries(study: str, summary: pd.DataFrame) -> dict:
	"""Mean and std of each environment-recognition number (detection_stats.summarize()) as macros."""
	entries = {}
	for row in summary.itertuples():
		key = f"{study}/{row.agent}/detect/{row.phase}/{row.env_id}/{DETECTION_SLUGS[row.metric]}"
		entries[f"{key}/mean"] = format_value(row.mean, "detect")
		entries[f"{key}/std"] = format_value(row.std, "detect")
	return entries


def detection_table(summary: pd.DataFrame, agents: list, caption: str, label: str) -> str:
	"""A LaTeX table of the recognition numbers: one row per agent, phase and environment, training first."""
	lines = [
		"\\hline",
		"\\textbf{Agent} & \\textbf{Phase} & \\textbf{Env.} & "
		+ " & ".join(f"\\textbf{{{header}}}" for _metric, header in DETECTION_COLUMNS)
		+ " \\\\",
		"\\hline",
	]
	values = summary.set_index(["agent", "phase", "env_id", "metric"])
	for agent in agents:
		for phase_key, phase_label in DETECTION_PHASES:
			rows = summary[(summary["agent"] == agent) & (summary["phase"] == phase_key)]
			for env_id in rows["env_id"].drop_duplicates():
				cells = []
				for metric, _header in DETECTION_COLUMNS:
					key = (agent, phase_key, env_id, metric)
					if key not in values.index:
						cells.append("--")
						continue
					mean, std = values.loc[key, "mean"], values.loc[key, "std"]
					cells.append(format_value(mean, "detect") + ("" if math.isnan(std) else f" $\\pm$ {format_value(std, 'detect')}"))
				lines.append(f"{agent} & {phase_label} & {env_id.split('-', maxsplit=1)[0]} & " + " & ".join(cells) + " \\\\")
		lines.append("\\hline")
	return latex_table(caption, label, f"|l|l|l|{'c|' * len(DETECTION_COLUMNS)}", lines, fit_width=True)


def detection_ablation_entries(grid: pd.DataFrame, confirmation: pd.DataFrame, chosen: tuple) -> dict:
	"""The detection benchmark's numbers, each from the seed set that may report it (tools/detection_ablation.py):

	- confirmation seeds (never used for a choice): per baseline, the mean over seeds of changes, false changes, niches and
	  worst purity (%) and the seeds passing; the study's chirp's pass count with its Clopper-Pearson interval, worst and
	  mean switch delay, worst and mean purity, and the seeds with a false change;
	- grid seeds: how many configurations pass on every seed, the chosen excitation's fewest passes over feature types,
	  the best one without a wavelet train, and the screened one;
	- one Holm family of exact McNemar tests (paired by seed): the chirp vs the transition model (confirmation) and the
	  chosen configuration vs the best one without a wavelet train (grid)."""
	entries = {"detbench/seeds": str(grid["seed"].nunique()), "detbench/confirmseeds": str(confirmation["seed"].nunique())}
	for label, rows in confirmation.groupby("baseline", sort=False):
		baseline = BASELINE_SLUGS[label]
		entries[f"detbench/{baseline}/changes"] = f"{rows['changes'].mean():.1f}"
		entries[f"detbench/{baseline}/inside"] = f"{rows['inside'].mean():.1f}"
		entries[f"detbench/{baseline}/niches"] = f"{rows['niches'].mean():.1f}"
		entries[f"detbench/{baseline}/purity"] = f"{100 * rows['min_purity'].mean():.1f}"
		entries[f"detbench/{baseline}/passes"] = str(int(rows["passes"].sum()))
		# Seeds on which two environments share their majority niche (merged physics).
		entries[f"detbench/{baseline}/merged"] = str(int((~rows["distinct"].astype(bool)).sum()))
	study = confirmation[confirmation["baseline"] == "Chirp (study), changing behavior"].sort_values("seed")
	transition = confirmation[confirmation["baseline"] == "Transition model, changing behavior"].sort_values("seed")
	passes, trials = int(study["passes"].sum()), len(study)
	low, high = clopper_pearson(passes, trials)
	entries["detbench/chosen/passes"] = str(passes)
	entries["detbench/chosen/lower"], entries["detbench/chosen/upper"] = f"{100 * low:.1f}", f"{100 * high:.1f}"
	entries["detbench/chosen/maxdelay"] = str(int(study["max_delay"].max()))
	entries["detbench/chosen/delaymean"] = f"{study['max_delay'].mean():.1f}"
	entries["detbench/chosen/minpurity"] = f"{100 * study['min_purity'].min():.0f}"
	entries["detbench/chosen/puritymean"] = f"{100 * study['min_purity'].mean():.1f}"
	entries["detbench/chosen/falseseeds"] = str(int((study["inside"] > 0).sum()))

	def record(excitation, features):
		return grid[(grid["excitation"] == excitation) & (grid["features"] == features)].sort_values("seed")["passes"].to_numpy()

	counts = grid.groupby(["excitation", "features"], sort=False)["passes"].sum()
	others = counts[[not excitation.startswith("Wavelet train") for excitation, _features in counts.index]]
	best_other = others.idxmax()
	entries["detbench/bestother/excitation"], entries["detbench/bestother/features"] = best_other
	entries["detbench/bestother/passes"] = str(int(others[best_other]))
	entries["detbench/screened/passes"] = str(int(counts[SCREENED]))
	entries["detbench/chosengrid/passes"] = str(int(counts[chosen]))
	# The chosen excitation with every feature type: its fewest passes (how much the excitation alone carries).
	entries["detbench/chosenexcitation/minpasses"] = str(int(counts[[excitation == chosen[0] for excitation, _features in counts.index]].min()))
	entries["detbench/passing"] = str(int((pass_table(grid) == grid["seed"].nunique()).fillna(False).to_numpy().sum()))
	adjusted = holm([mcnemar_p(study["passes"].to_numpy(), transition["passes"].to_numpy()), mcnemar_p(record(*chosen), record(*best_other))])
	entries["detbench/cmp/chirp/p"], entries["detbench/cmp/other/p"] = p_macro(adjusted[0]), p_macro(adjusted[1])
	return entries


def cartpole_check_entries(cartpole: pd.DataFrame) -> dict:
	"""CartPole's detector on CartPole's benchmark (the benchmark's sanity check): seeds meeting the target."""
	return {"detbench/cartpole/passes": str(int(cartpole["passes"].sum())), "detbench/cartpole/seeds": str(len(cartpole))}


def chirp_ablation_table(grid: pd.DataFrame, chosen: tuple) -> str:
	"""Seeds meeting the detection target per excitation (row) and response features (column); the study's choice in bold."""
	table, seeds = pass_table(grid), grid["seed"].nunique()
	lines = ["\\hline", "\\textbf{Excitation} & " + " & ".join(f"\\textbf{{{column}}}" for column in table.columns) + " \\\\", "\\hline"]
	for excitation, row in table.iterrows():
		cells = [f"{count}/{seeds}" for count in row]
		cells = [f"\\textbf{{{cell}}}" if (excitation, column) == chosen else cell for cell, column in zip(cells, table.columns)]
		lines.append(f"{excitation} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(
		f"Chirp detector ablation on the HalfCheetah detection benchmark: benchmark seeds (of {seeds}) on which an excitation and "
		"response features catch both switches, confirm no false change and give each environment its own niche",
		"tab:chirp_ablation",
		f"|l|{'c|' * len(table.columns)}",
		lines,
		fit_width=True,
	)


def clear_absent_outputs(present: set, generated_dir: str = GENERATED_DIR, images_dir: str = IMAGES_DIR) -> None:
	"""Replace the tables of every study not present with placeholders and delete its figures."""
	for study, tables in STUDY_TABLES.items():
		if study in present:
			continue
		for table in tables:
			label = "tab:" + table[len("table_") : -len(".tex")]
			_write(
				os.path.join(generated_dir, table),
				latex_table(f"\\todo{{Generated once the {study} study has run.}}", label, "|c|", ["\\hline", "-- \\\\", "\\hline"]),
			)
		for figure in STUDY_FIGURES.get(study, []):
			path = os.path.join(images_dir, figure)
			if os.path.exists(path):
				os.remove(path)


def reliability_figure(results: dict, agents: list, path: str) -> str:
	"""Box plot of per-replicate test reliability per agent, each replicate drawn as a dot; returns the path written."""
	pooled = _pooled(results, "Test")
	samples = [_values(pooled, agent, "Reliability (%)").to_numpy() for agent in agents]
	return replicate_box_figure(samples, agents, "Test reliability (%)", path, ylim=(-3, 103), rotate=True)


def _write(path: str, text: str) -> None:
	"""Write one generated file."""
	with open(path, "w", encoding="utf-8") as output:
		output.write("% GENERATED by app/src/tools/paper_results.py. Do not edit; re-run the script.\n" + text)


def study_comparisons(results: dict) -> dict:
	"""Every pairwise comparison the paper cites, for the studies present."""
	entries = {}
	test_metrics = TEST_METRICS
	if "ablation" in results:
		pairs = [(variant, "ENA-Combined") for variant in ABLATION_AGENTS[:-1]] + ABLATION_FAMILY
		for variant_a, variant_b in pairs:
			entries.update(
				comparison_entries(
					"ablation",
					results["ablation"],
					variant_a,
					results["ablation"],
					variant_b,
					"Test",
					test_metrics,
					alpha=ALPHA / len(ABLATION_FAMILY),
				)
			)
		entries.update(family_entries("ablation", results["ablation"], ABLATION_FAMILY))
	if "challenge" in results:
		entries.update(candidate_share_entries("challenge", results["challenge"], "ENA-Combined"))
		# ENA-Combined against each baseline on test reliability: one Holm family.
		entries.update(family_entries("challenge", results["challenge"], [(baseline, "ENA-Combined") for baseline in CHALLENGE_AGENTS[:-1]]))
		adapt_metrics = [f"{env}:steps" for env in CARTPOLE_TRAIN_ENVS] + ["Train-Env-Steps"]
		for baseline in CHALLENGE_AGENTS[:-1]:
			for phase, metrics in (("Test", test_metrics), ("Adaptation", adapt_metrics)):
				family = len(CHALLENGE_AGENTS) - 1 if phase == "Test" else 1
				entries.update(
					comparison_entries(
						"challenge", results["challenge"], baseline, results["challenge"], "ENA-Combined", phase, metrics, alpha=ALPHA / family
					)
				)
	if "cheetah" in results:
		# ENA-Combined against each baseline on the test return per environment: a family of the two baselines.
		for baseline in CHEETAH_AGENTS[:-1]:
			entries.update(
				comparison_entries(
					"cheetah",
					results["cheetah"],
					baseline,
					results["cheetah"],
					"ENA-Combined",
					"Test",
					[f"{env} Avg" for env in CHEETAH_ENVS],
					alpha=ALPHA / (len(CHEETAH_AGENTS) - 1),
				)
			)
	if "controller" in results and "ablation" in results:
		# Each alternative Plasticity controller against the linear ramp (the ablation's own ENA-Combined).
		for alternative in CONTROLLER_ALTERNATIVES:
			for phase, metrics in (("Test", test_metrics), ("Compute Cost", ["Total Candidate Env-Steps"])):
				entries.update(
					comparison_entries(
						"controller",
						results["ablation"],
						"ENA-Combined",
						results["controller"],
						alternative,
						phase,
						metrics,
						alpha=ALPHA / len(CONTROLLER_ALTERNATIVES),
					)
				)
		entries.update(controller_entries(results["ablation"], "ENA-Combined", results["controller"], CONTROLLER_ALTERNATIVES))

	return entries


# pylint: disable=too-many-locals
def collect_studies() -> tuple:
	"""Every study with at least one finished replicate: ({study: results}, the per-study macros)."""
	results, entries = {}, {}
	for study, directory in STUDIES.items():
		study_dir = os.path.join(APP_DIR, directory)
		if not os.path.isdir(study_dir):
			print(f"Skipping {study}: {study_dir} not found")
			continue
		study_results, parsed, found = collect_results([study_dir])
		print(f"{study}: {parsed}/{found} replicates parsed")
		if not completed_replicates(study_dir):
			# A study that has started but finished no replicate is not run yet: placeholders, not empty tables.
			continue
		results[study] = study_results
		entries.update(summary_entries(study, results[study]))
		entries.update(time_entries(study, study_dir))
		if study == "challenge":
			entries.update(episode_count_entries(study, study_dir, CHALLENGE_AGENTS))
		if study == "cheetah":
			entries.update(episode_count_entries(study, study_dir, CHEETAH_AGENTS))
			entries.update(hardware_entries(study, study_dir))

	if "compute" in results and "ablation" in results:
		# The compute study runs full search only; zero and the ramp are the ablation's runs of the same replicates.
		results["compute"] = with_reference_rows(results["compute"], results["ablation"], list(COMPUTE_REFERENCES.values()))
		entries.update(summary_entries("compute", results["compute"]))
		entries.update(
			compute_ratio_entries(results["compute"], "ENA-Combined-PlasticityFull", COMPUTE_REFERENCES["linear_ramp"], COMPUTE_REFERENCES["zero"])
		)
		entries.update(
			compute_rank_entries(results["compute"], "ENA-Combined-PlasticityFull", results["ablation"], COMPUTE_REFERENCES["linear_ramp"])
		)
	return results, entries


def write_cheetah_outputs(results: dict, rows: pd.DataFrame, entries: dict) -> None:
	"""HalfCheetah's tables (test return per environment; retention and training time) and figures (retention; training
	curves by env-steps; test curves)."""
	cheetah_columns = [(f"{env} Avg", env.split("-", maxsplit=1)[0], True) for env in CHEETAH_ENVS]
	_write(
		os.path.join(GENERATED_DIR, "table_cheetah_test.tex"),
		results_table(
			results,
			"Test",
			CHEETAH_AGENTS,
			cheetah_columns,
			f"HalfCheetah comparison, frozen test: mean return per environment ({entries['cheetah/n']} replicates)",
			"tab:cheetah_test",
		),
	)
	minutes = {agent: entries.get(f"cheetah/{agent}/time/min/mean", "--") for agent in CHEETAH_AGENTS}
	_write(
		os.path.join(GENERATED_DIR, "table_cheetah_retention.tex"),
		retention_table(
			rows,
			CHEETAH_LEFT_BEHIND,
			CHEETAH_AGENTS,
			minutes,
			f"HalfCheetah comparison: end-of-phase and test return, and retention, of the environments left behind, and "
			f"wall-clock training time ({entries['cheetah/n']} replicates)",
			"tab:cheetah_retention",
		),
	)
	retention_figure(rows, CHEETAH_LEFT_BEHIND, CHEETAH_AGENTS, os.path.join(IMAGES_DIR, "cheetah_retention.pdf"))
	# Training at a step budget: replicates play different numbers of episodes, so the curves are binned by env-steps.
	plot_study_curves(
		os.path.join(APP_DIR, STUDIES["cheetah"]),
		"training",
		CHEETAH_AGENTS,
		os.path.join(IMAGES_DIR, "cheetah_training_curves.pdf"),
		mastery_score=CHEETAH_SOLVED,
		ylim=CHEETAH_Y_LIM,
		x_axis="env_steps",
	)
	plot_study_curves(
		os.path.join(APP_DIR, STUDIES["cheetah"]),
		"testing",
		CHEETAH_AGENTS,
		os.path.join(IMAGES_DIR, "cheetah_test_curves.pdf"),
		mastery_score=CHEETAH_SOLVED,
		ylim=CHEETAH_Y_LIM,
	)


def main() -> None:
	"""Collect every study present, then write the macros and tables."""
	results, entries = collect_studies()
	entries.update(study_comparisons(results))
	entries.update(config_entries())
	entries.update(sac_discrete_entries(os.path.join(SACD_DIR, "summary.csv")))
	entries.update({"stats/alpha": f"{ALPHA:g}", "stats/power": f"{POWER * 100:g}", "stats/ci": f"{CI_LEVEL * 100:g}"})
	detection = {}
	for study in ("ablation",):
		study_dir = os.path.join(APP_DIR, STUDIES[study])
		if study in results:
			detection[study] = summarize_detection(study_detection(study_dir))
			entries.update(detection_entries(study, detection[study]))

	present = set(results)
	ablation_grid, ablation_confirmation = os.path.join(DETECTION_ABLATION_DIR, "grid.csv"), os.path.join(DETECTION_ABLATION_DIR, "confirmation.csv")
	if os.path.exists(ablation_grid) and os.path.exists(ablation_confirmation):
		grid, confirmation = pd.read_csv(ablation_grid), pd.read_csv(ablation_confirmation)
		# The study's chirp configuration, read from the HalfCheetah ENA as the study builds it.
		chosen = study_labels(build_cheetah_ena(1)._skills["env_detection"]._instance._settings)  # pylint: disable=protected-access
		entries.update(detection_ablation_entries(grid, confirmation, chosen))
		cartpole_path = os.path.join(DETECTION_ABLATION_DIR, "cartpole.csv")
		if os.path.exists(cartpole_path):
			entries.update(cartpole_check_entries(pd.read_csv(cartpole_path)))
		present.add("detection_ablation")

	if "cheetah" in results:
		# Retention reads each replicate's evaluation checks and test episodes, not the outputs.txt tables.
		rows = retention_rows(os.path.join(APP_DIR, STUDIES["cheetah"]), CHEETAH_AGENTS, CHEETAH_TRAIN_ENVS)
		entries.update(retention_entries("cheetah", rows, CHEETAH_LEFT_BEHIND))
		entries.update(retention_comparisons("cheetah", rows, CHEETAH_LEFT_BEHIND, "ENA-Combined", CHEETAH_AGENTS[:-1]))

	os.makedirs(GENERATED_DIR, exist_ok=True)
	write_macros(entries, os.path.join(GENERATED_DIR, "results.tex"))
	if "detection_ablation" in present:
		_write(os.path.join(GENERATED_DIR, "table_chirp_ablation.tex"), chirp_ablation_table(grid, chosen))
	test_columns = [(f"{env} Avg", env.split("-", maxsplit=1)[0], False) for env in CARTPOLE_ENVS] + [("Reliability (%)", "Rel. (\\%)", True)]
	if "ablation" in results:
		_write(
			os.path.join(GENERATED_DIR, "table_ablation_test.tex"),
			results_table(
				results["ablation"],
				"Test",
				ABLATION_AGENTS,
				test_columns,
				f"Skill ablation, frozen test: mean return per environment and reliability ({entries['ablation/n']} replicates)",
				"tab:ablation_test",
			),
		)
	if "challenge" in results:
		_write(
			os.path.join(GENERATED_DIR, "table_challenge_test.tex"),
			results_table(
				results["challenge"],
				"Test",
				CHALLENGE_AGENTS,
				test_columns,
				f"CartPole comparison, frozen test ({entries['challenge/n']} replicates)",
				"tab:challenge_test",
			),
		)
		adapt_columns = [(f"{env}:steps", env.split("-", maxsplit=1)[0], True) for env in CARTPOLE_TRAIN_ENVS]
		_write(
			os.path.join(GENERATED_DIR, "table_challenge_adapt.tex"),
			results_table(
				results["challenge"],
				"Adaptation",
				CHALLENGE_AGENTS,
				adapt_columns,
				f"CartPole comparison: training steps to first solve per environment ({entries['challenge/n']} replicates)",
				"tab:challenge_adapt",
			),
		)
	if "ablation" in detection and not detection["ablation"].empty:
		_write(
			os.path.join(GENERATED_DIR, "table_detection.tex"),
			detection_table(
				detection["ablation"],
				[agent for agent in ABLATION_AGENTS if agent in set(detection["ablation"]["agent"])],
				f"Environment recognition per phase in the skill ablation ({entries['ablation/n']} replicates): unrecognized episodes, "
				"confirmed changes, episodes into the phase of the first one, and Hall of Fame niches per training environment",
				"tab:detection",
			),
		)
	_write(
		os.path.join(GENERATED_DIR, "table_cartpole_envs.tex"),
		environment_table(
			cartpole_rows(make_training_worlds(), make_unseen_worlds()),
			["Environment", "$g$ (m/s$^2$)", "$m_p$ (kg)", "$\\ell$ (m)", "Seen in training"],
			"CartPole Environments",
			"tab:cartpole_worlds",
		),
	)
	_write(
		os.path.join(GENERATED_DIR, "table_cheetah_envs.tex"),
		environment_table(
			cheetah_rows(),
			["Environment", "$g$ (m/s$^2$)", "Friction", "Wind (m/s)", "Seen"],
			"HalfCheetah Environments",
			"tab:cheetah_worlds",
		),
	)
	images_dir = IMAGES_DIR
	clear_absent_outputs(present)
	if "ablation" in results:
		reliability_figure(results["ablation"], ABLATION_AGENTS[1:], os.path.join(images_dir, "ablation_reliability.pdf"))
	if "challenge" in results:
		reliability_figure(results["challenge"], CHALLENGE_AGENTS, os.path.join(images_dir, "challenge_reliability.pdf"))
		plot_study_curves(
			os.path.join(APP_DIR, STUDIES["challenge"]), "testing", CHALLENGE_AGENTS, os.path.join(images_dir, "challenge_test_curves.pdf")
		)
	if "cheetah" in results:
		write_cheetah_outputs(results["cheetah"], rows, entries)
	print(f"Wrote {len(entries)} numbers and the tables to {GENERATED_DIR}")


if __name__ == "__main__":
	main()
