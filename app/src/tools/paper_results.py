#!/usr/bin/env python
"""Generate every number and results table the paper quotes, from the study outputs."""
# Writes app/latex/generated/results.tex — one \resdef{key}{value} per number, quoted in the paper as
# \Res{key} — and app/latex/generated/table_*.tex. Re-run it whenever a study gains replicates; the
# paper then picks up the new numbers on its next build. Never edit the generated files by hand.
#
# Keys:
#         <study>/n                                   replicates parsed
#         <study>/<agent>/<phase>/<metric>/mean|std   phase: train, test, compute, adapt
#         <study>/<agent>/time/min/mean|std           wall-clock training time, minutes
#         <study>/<A>-vs-<B>/<phase>/<metric>/p|diff  Welch's t-test p-value and mean(B) - mean(A)
#         <study>/<agent>/end|retention/<env>/mean|std  end of phase (last 10% of the phase), retention (%)
#         <study>/<agent>/retention/left/mean|std     retention over the environments left behind
#         <study>/<B>-vs-<agent>/retention/left/p|pholm|diff|ci/low|ci/high|d|nreq
#         <study>/<agent>/forgetting/<env>|left/mean|std  HalfCheetah and the CartPole challenge: absolute forgetting, end of phase - test
#         <study>/<B>-vs-<agent>/forgetting/left/p|pholm|diff|ci/low|ci/high|d|nreq
#         <study>/<agent>/recog/<env>/<training env|unrecognized|other>  replicates whose test env is mostly recognized so
#         <study>/<agent>/played/<env>/<training env>  replicates whose test env is mostly played by that env's gladiators
#         <study>/<agent>/detect/<phase>/changes/total  confirmed changes per replicate and phase (training or test)
#         seeds/<study>/first|last                    the RNG seeds of the study's first and last replicate
#         <study>/hardware                            the machine and thread counts the replicates ran with
#         selection/identity/identical|checked        selection study: training logs byte-identical to the ablation's ENA-Combined
#         selection/<agent>/played/mean               selection study: distinct individuals played per test
#         selection/<A>-vs-<B>/test/rel/lower|higher|pairs  replicates where B's reliability is lower / higher than A's
#
# Usage:
#         python src/tools/paper_results.py

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
from cartpole_ena_ablation import ABLATION_VARIANTS, SEED_OFFSET, TRAINING_EPISODES_ENA, make_training_worlds, make_unseen_worlds
from cartpole_ena_compute import REFERENCES as COMPUTE_REFERENCES
from half_cheetah import SOLVED_SCORE as CHEETAH_SOLVED, Y_LIM as CHEETAH_Y_LIM, build_ena_agent as build_cheetah_ena
from tools.aggregate_metrics import collect_results
from tools.replicates import get_experiment_directories
from tools.cheetah_results import (
	CHEETAH_AGENTS,
	CHEETAH_ENVS,
	CHEETAH_LEFT_BEHIND,
	CHEETAH_TRAIN_ENVS,
	forgetting_comparisons,
	hardware_entries,
	retention_comparisons,
	retention_entries,
	retention_figure,
	retention_rows,
	retention_table,
)
from tools.detection_ablation import SCREENED, pass_table, study_labels
from tools.detection_stats import majority_recognition, study_detection, summarize as summarize_detection
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
from tools.selection_results import (
	SELECTION_AGENTS,
	SELECTION_FAMILY,
	SELECTION_REFERENCE,
	paired_count_entries,
	played_entries,
	training_identity_entries,
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATED_DIR = os.path.join(APP_DIR, "latex", "generated")
IMAGES_DIR = os.path.join(APP_DIR, "latex", "images")
PHASE_KEYS = {"Training": "train", "Test": "test", "Compute Cost": "compute", "Adaptation": "adapt"}
METRIC_SLUGS = {
	"Reliability (%)": "rel",
	"Stability Gap (↓)": "gap",
	"Efficiency (Score/s)": "eff",
	"Total Candidate Env-Steps": "candsteps",
	"Avg Eval Depth": "evaldepth",
	"Avg Candidates/Gen": "candidates",
}
TIME_PATTERN = re.compile(r"^\s+(?P<agent>\S.*?): (?P<seconds>[\d.]+) seconds, (?P<steps>[\d,]+) env-steps\s*$")

CARTPOLE_ENVS = ["Mars-Light", "Neptune", "Jupiter-Heavy", "Brown-dwarf", "Earth-Standard"]
CARTPOLE_TRAIN_ENVS = ["Mars-Light", "Jupiter-Heavy", "Earth-Standard"]
# The CartPole challenge's end of phase: the last 10% of its per-environment budget, as HalfCheetah's. The ablation
# trains a fixed number of episodes per environment (its variants play very different env-steps), so its end of phase is
# the last 10% of those episodes.
CHALLENGE_END_OF_PHASE_STEPS = 250_000
ABLATION_END_OF_PHASE_EPISODES = TRAINING_EPISODES_ENA[0] // 10
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
# Descriptive, outside the family: the archive organized by detection against no archive at all, without Plasticity
# (with Plasticity it is ENA-Plasticity vs ENA-Combined, reported with every variant against ENA-Combined).
ABLATION_DESCRIPTIVE = [("ENA-NoSkill", "ENA-HoF+EnvDetection")]
# Adding the Hall of Fame, then detection, with and without Plasticity: what each costs in candidate evaluation.
ABLATION_ARCHIVE_COST_PAIRS = [
	("ENA-NoSkill", "ENA-HoF"),
	("ENA-HoF", "ENA-HoF+EnvDetection"),
	("ENA-Plasticity", "ENA-Plasticity+HoF"),
	("ENA-Plasticity+HoF", "ENA-Combined"),
]
# No Random row: the challenge does not run it (the floor is in the ablation table).
CHALLENGE_AGENTS = ["DQN", "PPO", "SAC-Discrete", "ENA-Combined"]
TEST_METRICS = ["Reliability (%)", "Stability Gap (↓)"] + [f"{env} Avg" for env in CARTPOLE_ENVS]
DETECTION_SLUGS = {"unrecognized": "unrec", "changes": "changes", "first_change": "firstchange", "niches": "niches"}
# Environment-recognition table columns: (metric, header).
DETECTION_COLUMNS = [("unrecognized", "Unrec. (\\%)"), ("changes", "Changes"), ("first_change", "First (ep.)"), ("niches", "Niches")]
DETECTION_PHASES = [("train", "Training"), ("test", "Test")]
CONTROLLER_ALTERNATIVES = ["ENA-Combined-QuadraticPlasticity", "ENA-Combined-FuzzyPlasticity"]
# HalfCheetah's per-environment returns are thousands: printed as whole numbers.
WHOLE_RETURN_STUDIES = ("cheetah",)
# Each skill as the ablation table switches it on: (name, variant-row setting, its "off" value).
ABLATION_SKILLS = (
	("Plasticity", "plasticity_algorithm", "zero"),
	("Hall of Fame", "hall_of_fame_driver", "none"),
	("Detection", "recognizer_strategy", "none"),
)
STUDIES = {
	"ablation": "cartpole_ena_ablation_outputs",
	"compute": "cartpole_ena_compute_outputs",
	"controller": "cartpole_ena_plasticity_controller_outputs",
	"selection": "cartpole_ena_selection_outputs",
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


def excitation_name(label: str) -> str:
	"""An excitation's name as the paper prints it: a swept sine is a frequency sweep, since "chirp" names ENA's probe."""
	return label.replace(" chirp", " sweep")


def slug(metric: str) -> str:
	"""Short key for a metric column name."""
	if metric in METRIC_SLUGS:
		return METRIC_SLUGS[metric]
	return re.sub(r" Avg$", "", metric).replace(" ", "")


def value_slug(study: str, metric: str) -> str:
	"""The format slug of a study's metric: as slug(), except HalfCheetah's per-environment returns, printed whole."""
	return "return" if study in WHOLE_RETURN_STUDIES and metric.endswith(" Avg") else slug(metric)


def parse_training_times(outputs_path: str) -> dict:
	"""Return {agent: (training seconds, env-steps)} from a replicate's 'Training Performance vs Time' block."""
	times = {}
	with open(outputs_path, encoding="utf-8") as outputs:
		for line in outputs:
			match = TIME_PATTERN.match(line.rstrip("\n"))
			if match:
				times[match.group("agent")] = (float(match.group("seconds")), int(match.group("steps").replace(",", "")))
	return times


def _pooled(results: dict, phase: str) -> pd.DataFrame:
	"""Return every replicate's table of one phase, stacked."""
	return pd.concat(results[phase], ignore_index=True)


def _values(pooled: pd.DataFrame, agent: str, metric: str) -> pd.Series:
	"""Return one agent's per-replicate values of one metric, numeric."""
	return pd.to_numeric(pooled.loc[pooled["Agent"] == agent, metric].astype(str).str.replace(",", ""), errors="coerce").dropna()


def summary_entries(study: str, results: dict) -> dict:
	"""Mean and sample std of every metric for every agent and phase, plus the replicate count."""
	entries = {}
	replicates = set()
	for phase in results:
		phase_key = PHASE_KEYS[phase]
		pooled = _pooled(results, phase)
		replicates.update(pooled["Exp"].unique())
		for agent in pooled["Agent"].unique():
			for metric in (column for column in pooled.columns if column not in ("Agent", "Exp")):
				values = _values(pooled, agent, metric)
				if values.empty:
					continue
				key = f"{study}/{agent}/{phase_key}/{slug(metric)}"
				entries[f"{key}/mean"] = format_value(values.mean(), value_slug(study, metric))
				entries[f"{key}/std"] = format_value(values.std() if len(values) > 1 else float("nan"), value_slug(study, metric))
	entries[f"{study}/n"] = str(len(replicates))
	return entries


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def comparison_entries(
	study: str, results_a: dict, agent_a: str, results_b: dict, agent_b: str, phase: str, metrics: list, alpha: float = ALPHA
) -> dict:
	"""agent_b vs."""
	# agent_a on each metric (the agents may come from different studies): Welch p, mean difference (b - a) with its 95% interval, effect size d, and
	# replicates per agent for 80% power at alpha.
	entries = {}
	pooled_a, pooled_b = _pooled(results_a, phase), _pooled(results_b, phase)
	for metric in metrics:
		values_a, values_b = _values(pooled_a, agent_a, metric), _values(pooled_b, agent_b, metric)
		key = f"{study}/{agent_a}-vs-{agent_b}/{PHASE_KEYS[phase]}/{slug(metric)}"
		entries.update(difference_entries(key, values_a, values_b, alpha, value_slug(study, metric)))
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
	return {"compute/ratio/full-ramp": f"{steps[full] / steps[ramp]:.0f}", "compute/ratio/ramp-zero": f"{steps[ramp] / steps[zero]:.0f}"}


def compute_rank_entries(results_full: dict, full: str, results_ramp: dict, ramp: str) -> dict:
	"""Where the single full-search run's test reliability falls among the ramp's replicates: how many reached at least it."""
	full_values = _values(_pooled(results_full, "Test"), full, "Reliability (%)")
	ramp_values = _values(_pooled(results_ramp, "Test"), ramp, "Reliability (%)")
	return {"compute/rank/atleast": str(int((ramp_values >= full_values.mean()).sum())), "compute/rank/of": str(len(ramp_values))}


def candidate_share_entries(study: str, results: dict, agent: str) -> dict:
	"""Share (%) of the agent's training env-steps spent evaluating candidates, per replicate, then mean and std."""
	candidates = _pooled(results, "Compute Cost")
	training = _pooled(results, "Adaptation")
	shares = []
	for exp in candidates.loc[candidates["Agent"] == agent, "Exp"]:
		spent = _values(candidates[candidates["Exp"] == exp], agent, "Total Candidate Env-Steps")
		total = _values(training[training["Exp"] == exp], agent, "Train-Env-Steps")
		shares.append(100 * spent.iloc[0] / total.iloc[0])
	series = pd.Series(shares)
	return {
		f"{study}/{agent}/candshare/mean": f"{series.mean():.1f}",
		f"{study}/{agent}/candshare/std": f"{series.std():.1f}" if len(series) > 1 else "n/a",
	}


def controller_verdict(outcomes: dict) -> str:
	"""Apply the pre-registered decision rule on Holm-adjusted p-values."""
	# outcomes: {alternative: {"rel": (difference, p), "candsteps": (difference, p), "rel_mean": mean reliability}}.
	# An alternative qualifies if its test reliability is significantly higher, or not significantly lower
	# with significantly fewer candidate steps; of several, the most reliable is adopted.
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
			key = f"controller/{reference}-vs-{alternative}/{PHASE_KEYS[phase]}/{metric_slug}"
			tested.append((alternative, key, values_alt.mean() - values_ref.mean(), welch_p(values_ref, values_alt)))
			entries[f"{key}/ppaired"] = p_macro(paired_p(pooled_ref, reference, pooled_alt, alternative, metric)[0])
			if metric_slug == "rel":
				outcomes[alternative]["rel_mean"] = values_alt.mean()
		for (alternative, key, difference, _p_value), adjusted in zip(tested, holm([row[3] for row in tested], family_size=len(alternatives))):
			entries[f"{key}/pholm"] = p_macro(adjusted)
			outcomes[alternative][metric_slug] = (difference, adjusted)
	replicates = _pooled(results_alt, "Test")["Exp"].nunique()
	entries["controller/family"] = str(len(alternatives))
	entries["controller/mde"] = f"{detectable_d(replicates, ALPHA / len(alternatives)):.2f}"
	entries["controller/verdict"] = controller_verdict(outcomes)
	return entries


def family_entries(study: str, results: dict, pairs: list, metric: str = "Reliability (%)") -> dict:
	"""Report a study's planned comparisons as one Holm family, with its size and detectable effect."""
	# A study's planned comparisons as one family: Holm-adjusted Welch p per pair, the family size, and the
	# smallest effect size the study's replicate count detects with 80% power at the family's most stringent level.
	pooled = _pooled(results, "Test")
	tested = []
	for agent_a, agent_b in pairs:
		values_a, values_b = _values(pooled, agent_a, metric), _values(pooled, agent_b, metric)
		tested.append((f"{study}/{agent_a}-vs-{agent_b}/test/{slug(metric)}", welch_p(values_a, values_b)))
	entries = {}
	for (key, _p_value), adjusted in zip(tested, holm([p_value for _key, p_value in tested], family_size=len(pairs))):
		entries[f"{key}/pholm"] = p_macro(adjusted)
	replicates = pooled["Exp"].nunique()
	entries[f"{study}/family"] = str(len(pairs))
	entries[f"{study}/mde"] = f"{detectable_d(replicates, ALPHA / len(pairs)):.2f}"
	return entries


def episode_count_entries(study: str, study_dir: str, agents: list) -> dict:
	"""Return each agent's mean number of training episodes per environment phase, over the study's replicates."""
	entries = {}
	for agent in agents:
		counts = []
		for exp_dir in get_experiment_directories(study_dir):
			env_ids = pd.read_csv(os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv"), usecols=["env_id"])["env_id"]
			counts += env_ids.groupby(env_ids.ne(env_ids.shift()).cumsum()).size().tolist()
		entries[f"{study}/{agent}/episodes/perenv"] = f"{sum(counts) / len(counts):,.0f}"
	return entries


def time_entries(study: str, study_dir: str) -> dict:
	"""Return the mean and std of each agent's wall-clock training time (minutes) over the study's replicates."""
	minutes = {}
	for exp_dir in get_experiment_directories(study_dir):
		for agent, (seconds, _steps) in parse_training_times(os.path.join(study_dir, exp_dir, "outputs.txt")).items():
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
def results_table(study: str, results: dict, phase: str, agents: list, columns: list, caption: str, label: str) -> str:
	"""A LaTeX table of one study: one row per agent; columns are (metric, header, show_std) with the mean (± std) over replicates."""
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
			cell = format_value(values.mean(), value_slug(study, metric))
			if show_std and len(values) > 1:
				cell += f" $\\pm$ {format_value(values.std(), value_slug(study, metric))}"
			cells.append(cell)
		lines.append(f"{agent} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(caption, label, f"|l|{'c|' * len(columns)}", lines, fit="column")


def sac_discrete_entries(summary_path: str) -> dict:
	"""Median temperature (start, peak) and peak critic value per target-entropy ratio, from sac_discrete_entropy_check.py."""
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
	return latex_table(caption, label, f"|l|l|l|{'c|' * len(DETECTION_COLUMNS)}", lines, fit="column")


def detection_ablation_entries(grid: pd.DataFrame, confirmation: pd.DataFrame, chosen: tuple) -> dict:
	"""The detection benchmark's numbers, each from the seed set that may report it (tools/detection_ablation.py):"""
	# - confirmation seeds (never used for a choice): per baseline, the mean over seeds of changes, false changes, niches and
	#   worst purity (%) and the seeds passing; the study's chirp's pass count with its Clopper-Pearson interval, worst and
	#   mean switch delay, worst and mean purity, and the seeds with a false change;
	# - grid seeds: how many configurations pass on every seed, the chosen excitation's fewest passes over feature types,
	#   the best one without a wavelet train, and the screened one;
	# - one Holm family of exact McNemar tests (paired by seed): the chirp vs the transition model (confirmation) and the
	#   chosen configuration vs the best one without a wavelet train (grid).
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
	excitation, entries["detbench/bestother/features"] = best_other
	entries["detbench/bestother/excitation"] = excitation_name(excitation).lower()
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
	"""Tabulate the seeds meeting the detection target per excitation (row) and response features (column); the study's choice in bold."""
	table, seeds = pass_table(grid), grid["seed"].nunique()
	lines = ["\\hline", "\\textbf{Excitation} & " + " & ".join(f"\\textbf{{{column}}}" for column in table.columns) + " \\\\", "\\hline"]
	for excitation, row in table.iterrows():
		cells = [f"{count}/{seeds}" for count in row]
		cells = [f"\\textbf{{{cell}}}" if (excitation, column) == chosen else cell for cell, column in zip(cells, table.columns)]
		lines.append(f"{excitation_name(excitation)} & " + " & ".join(cells) + " \\\\")
	lines.append("\\hline")
	return latex_table(
		f"Chirp detector ablation on the HalfCheetah detection benchmark: benchmark seeds (of {seeds}) on which an excitation and "
		"response features catch both switches, confirm no false change and give each environment its own niche",
		"tab:chirp_ablation",
		f"|l|{'c|' * len(table.columns)}",
		lines,
		fit="column",
	)


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
	"""Return every pairwise comparison the paper cites."""
	entries = {}
	test_metrics = TEST_METRICS
	pairs = [(variant, "ENA-Combined") for variant in ABLATION_AGENTS[:-1]] + ABLATION_FAMILY + ABLATION_DESCRIPTIVE
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
	# What the archive and detection cost in candidate evaluation: exploratory, uncorrected.
	for variant_a, variant_b in ABLATION_ARCHIVE_COST_PAIRS:
		entries.update(
			comparison_entries(
				"ablation", results["ablation"], variant_a, results["ablation"], variant_b, "Compute Cost", ["Total Candidate Env-Steps"]
			)
		)
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
	entries.update(selection_comparisons(results["selection"]))

	return entries


def selection_comparisons(results: dict) -> dict:
	"""Report the selection study's planned family, its per-environment differences and paired replicate counts."""
	# The selection study's planned family (Holm over 2) on test reliability, per-environment differences (descriptive)
	# and, per pair, how many replicates the second driver is lower or higher in.
	entries = family_entries("selection", results, SELECTION_FAMILY)
	pooled = _pooled(results, "Test")
	for agent_a, agent_b in SELECTION_FAMILY:
		entries.update(comparison_entries("selection", results, agent_a, results, agent_b, "Test", TEST_METRICS, alpha=ALPHA / len(SELECTION_FAMILY)))
		by_replicate = [
			pd.to_numeric(pooled[pooled["Agent"] == agent].set_index("Exp")["Reliability (%)"].astype(str).str.replace(",", ""), errors="coerce")
			for agent in (agent_a, agent_b)
		]
		entries.update(paired_count_entries(f"selection/{agent_a}-vs-{agent_b}/test/rel", *by_replicate))
	return entries


def collect_studies() -> tuple:
	"""Return every study's parsed results and its per-study macros: ({study: results}, entries)."""
	results, entries = {}, {}
	for study, directory in STUDIES.items():
		study_dir = os.path.join(APP_DIR, directory)
		results[study] = collect_results(study_dir)
		entries.update(summary_entries(study, results[study]))
		entries.update(time_entries(study, study_dir))
		entries.update(seed_entries(study, study_dir))
	entries.update(episode_count_entries("challenge", os.path.join(APP_DIR, STUDIES["challenge"]), CHALLENGE_AGENTS))
	entries.update(episode_count_entries("cheetah", os.path.join(APP_DIR, STUDIES["cheetah"]), CHEETAH_AGENTS))
	entries.update(hardware_entries("cheetah", os.path.join(APP_DIR, STUDIES["cheetah"])))
	# The compute study runs full search only; zero and the ramp are the ablation's runs of the same replicates.
	results["compute"] = with_reference_rows(results["compute"], results["ablation"], list(COMPUTE_REFERENCES.values()))
	entries.update(summary_entries("compute", results["compute"]))
	entries.update(
		compute_ratio_entries(results["compute"], "ENA-Combined-PlasticityFull", COMPUTE_REFERENCES["linear_ramp"], COMPUTE_REFERENCES["zero"])
	)
	entries.update(compute_rank_entries(results["compute"], "ENA-Combined-PlasticityFull", results["ablation"], COMPUTE_REFERENCES["linear_ramp"]))
	# The BPR reference is the ablation's ENA-Combined of the same replicates (identical training, only the test differs).
	results["selection"] = with_reference_rows(results["selection"], results["ablation"], [SELECTION_REFERENCE])
	entries.update(summary_entries("selection", results["selection"]))
	selection_dir, ablation_dir = os.path.join(APP_DIR, STUDIES["selection"]), os.path.join(APP_DIR, STUDIES["ablation"])
	entries.update(training_identity_entries(selection_dir, ablation_dir))
	entries.update(played_entries(selection_dir, ablation_dir))
	return results, entries


def forgetting_entries() -> tuple:
	"""Compute retention and absolute forgetting (HalfCheetah, the CartPole challenge), and HalfCheetah's retention rows."""
	# They read each replicate's training and test episodes, not the outputs.txt tables; the rows feed HalfCheetah's
	# table and figure.
	rows = retention_rows(os.path.join(APP_DIR, STUDIES["cheetah"]), CHEETAH_AGENTS, CHEETAH_TRAIN_ENVS)
	return {**cheetah_retention_entries(rows), **challenge_forgetting_entries()}, rows


def challenge_forgetting_entries() -> dict:
	"""Compute how much each CartPole challenge agent forgets its training environments (descriptive)."""
	# How much each CartPole challenge agent forgets the environments it trained on: end of phase, retention and absolute
	# forgetting, descriptive only (the challenge's planned tests are on reliability).
	rows = retention_rows(os.path.join(APP_DIR, STUDIES["challenge"]), CHALLENGE_AGENTS, CARTPOLE_TRAIN_ENVS, CHALLENGE_END_OF_PHASE_STEPS)
	return retention_entries("challenge", rows, CARTPOLE_TRAIN_ENVS[:-1])


def cheetah_retention_entries(rows: pd.DataFrame) -> dict:
	"""HalfCheetah's retention and absolute forgetting per agent, and ENA-Combined against each baseline on both."""
	baselines = CHEETAH_AGENTS[:-1]
	return {
		**retention_entries("cheetah", rows, CHEETAH_LEFT_BEHIND),
		**retention_comparisons("cheetah", rows, CHEETAH_LEFT_BEHIND, "ENA-Combined", baselines),
		**forgetting_comparisons("cheetah", rows, CHEETAH_LEFT_BEHIND, "ENA-Combined", baselines),
	}


def write_cheetah_outputs(results: dict, rows: pd.DataFrame, entries: dict) -> None:
	"""Write HalfCheetah's tables and figures."""
	# HalfCheetah's tables (test return per environment; retention and training time) and figures (retention; training
	# curves by env-steps; test curves).
	cheetah_columns = [(f"{env} Avg", env.split("-", maxsplit=1)[0], True) for env in CHEETAH_ENVS]
	_write(
		os.path.join(GENERATED_DIR, "table_cheetah_test.tex"),
		results_table(
			"cheetah",
			results,
			"Test",
			CHEETAH_AGENTS,
			cheetah_columns,
			f"HalfCheetah comparison, frozen test: mean return per environment ({entries['cheetah/n']} replicates)",
			"tab:cheetah_test",
		),
	)
	minutes = {agent: entries[f"cheetah/{agent}/time/min/mean"] for agent in CHEETAH_AGENTS}
	_write(
		os.path.join(GENERATED_DIR, "table_cheetah_retention.tex"),
		retention_table(
			rows,
			CHEETAH_LEFT_BEHIND,
			CHEETAH_AGENTS,
			minutes,
			f"HalfCheetah comparison: end-of-phase and test return, retention and absolute forgetting (end $-$ test) of the "
			f"environments left behind, and "
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


def seed_entries(study: str, study_dir: str) -> dict:
	"""The RNG seeds of a study's first and last replicate: exp_<k> runs experiment k - 1, seeded SEED_OFFSET + k - 1."""
	numbers = [int(name.split("_")[1]) for name in get_experiment_directories(study_dir)]
	return {f"seeds/{study}/first": str(SEED_OFFSET + min(numbers) - 1), f"seeds/{study}/last": str(SEED_OFFSET + max(numbers) - 1)}


def variant_skills(variant: str) -> set:
	"""The skills an ablation variant has switched on (Plasticity, Hall of Fame, Detection)."""
	row = next(row for row in ABLATION_VARIANTS if row["name"] == variant)
	return {name for name, setting, off in ABLATION_SKILLS if row[setting] != off}


def comparison_table(entries: dict, rows: list, headers: list, caption: str, label: str) -> str:
	"""A LaTeX table of planned comparisons: per (label, label, macro key) row, the difference, its interval, d and the Holm-adjusted p."""
	head = [*headers, "Difference", f"{CI_LEVEL * 100:g}\\% CI", "$d$", "Holm"]
	lines = ["\\hline", " & ".join(f"\\textbf{{{header}}}" for header in head) + " \\\\", "\\hline"]
	for first, second, key in rows:
		interval = f"{entries[f'{key}/ci/low']} to {entries[f'{key}/ci/high']}"
		lines.append(" & ".join([first, second, entries[f"{key}/diff"], interval, entries[f"{key}/d"], entries[f"{key}/pholm"]]) + " \\\\")
	lines.append("\\hline")
	return latex_table(caption, label, f"|l|l|{'c|' * 4}", lines, fit="column")


def ablation_family_table(entries: dict) -> str:
	"""The ablation's planned comparisons as a table: which skill each adds, and to which variant."""
	rows = [
		(", ".join(sorted(variant_skills(with_b) - variant_skills(base))), base, f"ablation/{base}-vs-{with_b}/test/rel")
		for base, with_b in ABLATION_FAMILY
	]
	caption = (
		f"Skill ablation: the {len(ABLATION_FAMILY)} planned comparisons of test reliability (difference in points), "
		f"one Holm family ({entries['ablation/n']} replicates)"
	)
	return comparison_table(entries, rows, ["Skill added", "To"], caption, "tab:ablation_family")


def controller_table(entries: dict) -> str:
	"""The Plasticity-controller comparisons with the linear ramp as a table: per alternative, reliability and candidate steps."""
	metrics = (("Reliability (points)", "test/rel"), ("Candidate steps", "compute/candsteps"))
	rows = [
		(alternative.removeprefix("ENA-Combined-").removesuffix("Plasticity"), name, f"controller/ENA-Combined-vs-{alternative}/{path}")
		for alternative in CONTROLLER_ALTERNATIVES
		for name, path in metrics
	]
	caption = f"Each Plasticity controller against the linear ramp ({entries['controller/n']} replicates); each metric is one Holm family"
	return comparison_table(entries, rows, ["Controller", "Metric"], caption, "tab:controller")


def recognition_entries(study: str, agent: str, train_envs: list) -> dict:
	"""Replicates that mostly recognize each test environment as each training environment's niche (or none), and mostly play its gladiators."""
	study_dir = os.path.join(APP_DIR, STUDIES[study])
	frames = [majority_recognition(os.path.join(study_dir, exp_dir), agent) for exp_dir in get_experiment_directories(study_dir)]
	key = f"{study}/{agent}"
	entries = {f"{key}/recog/distinct": str(sum(bool(frame["distinct"].iloc[0]) for frame in frames))}
	for env, at_env in pd.concat(frames, ignore_index=True).groupby("env_id", sort=False):
		entries.update(
			{f"{key}/recog/{env}/{label}": str(int((at_env["recognized"] == label).sum())) for label in [*train_envs, "unrecognized", "other"]}
		)
		entries.update({f"{key}/played/{env}/{label}": str(int((at_env["played"] == label).sum())) for label in train_envs})
	return entries


def detection_total_entries(study: str, detection: pd.DataFrame) -> dict:
	"""Mean confirmed changes per replicate in each phase (training, test), summed over the phase's environments."""
	changes = detection[detection["metric"] == "changes"].groupby(["agent", "phase", "exp"])["value"].sum()
	return {
		f"{study}/{agent}/detect/{phase}/changes/total": format_value(total, "detect")
		for (agent, phase), total in changes.groupby(["agent", "phase"]).mean().items()
	}


def detection_outputs() -> tuple:
	"""Recognition numbers (macros) and tables of the ablation's and HalfCheetah's environment detection, and the recognition counts."""
	entries, tables = {}, {}
	for study, label, caption in (
		("ablation", "tab:detection", "Environment recognition per phase in the skill ablation"),
		("cheetah", "tab:cheetah_detection", "Environment recognition per phase by ENA-Combined's chirp on HalfCheetah"),
	):
		detection = study_detection(os.path.join(APP_DIR, STUDIES[study]))
		summary = summarize_detection(detection)
		entries.update({**detection_entries(study, summary), **detection_total_entries(study, detection)})
		caption += (
			f" ({detection['exp'].nunique()} replicates): unrecognized episodes, confirmed changes, episodes into the phase of the first one, "
			"and Hall of Fame niches per training environment"
		)
		tables[f"table_{label.split(':')[1]}.tex"] = detection_table(
			summary, [agent for agent in ABLATION_AGENTS if agent in set(summary["agent"])], caption, label
		)
	for study, train_envs in (("ablation", CARTPOLE_TRAIN_ENVS), ("challenge", CARTPOLE_TRAIN_ENVS), ("cheetah", CHEETAH_TRAIN_ENVS)):
		entries.update(recognition_entries(study, "ENA-Combined", train_envs))
	return entries, tables


def ablation_retention() -> tuple:
	"""The ablation variants' end of phase, test return, retention and forgetting per training environment (descriptive), and its table."""
	agents, left_behind = ABLATION_AGENTS[1:], CARTPOLE_TRAIN_ENVS[:-1]
	rows = retention_rows(os.path.join(APP_DIR, STUDIES["ablation"]), agents, CARTPOLE_TRAIN_ENVS, last_episodes=ABLATION_END_OF_PHASE_EPISODES)
	caption = (
		f"Skill ablation: return at the end of each training phase (its last {ABLATION_END_OF_PHASE_EPISODES} episodes) and at test, "
		f"retention and absolute forgetting (end $-$ test) of the environments left behind ({rows['Exp'].nunique()} replicates)"
	)
	return retention_entries("ablation", rows, left_behind), retention_table(rows, left_behind, agents, None, caption, "tab:ablation_retention")


def main() -> None:
	"""Collect every study, then write the macros, tables and figures."""
	results, entries = collect_studies()
	entries.update(study_comparisons(results))
	entries.update(config_entries())
	entries.update(sac_discrete_entries(os.path.join(SACD_DIR, "summary.csv")))
	entries.update({"stats/alpha": f"{ALPHA:g}", "stats/power": f"{POWER * 100:g}", "stats/ci": f"{CI_LEVEL * 100:g}"})
	detection, tables = detection_outputs()
	entries.update(detection)
	grid, confirmation = pd.read_csv(os.path.join(DETECTION_ABLATION_DIR, "grid.csv")), pd.read_csv(
		os.path.join(DETECTION_ABLATION_DIR, "confirmation.csv")
	)
	# The study's chirp configuration, read from the HalfCheetah ENA as the study builds it.
	chosen = study_labels(build_cheetah_ena(1)._skills["env_detection"]._instance._settings)  # pylint: disable=protected-access
	entries.update(detection_ablation_entries(grid, confirmation, chosen))
	entries.update(cartpole_check_entries(pd.read_csv(os.path.join(DETECTION_ABLATION_DIR, "cartpole.csv"))))
	retention, rows = forgetting_entries()
	entries.update(retention)
	ablation_forgetting, tables["table_ablation_retention.tex"] = ablation_retention()
	entries.update(ablation_forgetting)

	test_columns = [(f"{env} Avg", env.split("-", maxsplit=1)[0], False) for env in CARTPOLE_ENVS] + [("Reliability (%)", "Rel. (\\%)", True)]
	adapt_columns = [(f"{env}:steps", env.split("-", maxsplit=1)[0], True) for env in CARTPOLE_TRAIN_ENVS]
	tables.update(
		{
			"table_chirp_ablation.tex": chirp_ablation_table(grid, chosen),
			"table_ablation_family.tex": ablation_family_table(entries),
			"table_controller.tex": controller_table(entries),
			"table_ablation_test.tex": results_table(
				"ablation",
				results["ablation"],
				"Test",
				ABLATION_AGENTS,
				test_columns,
				f"Skill ablation, frozen test: mean return per environment and reliability ({entries['ablation/n']} replicates)",
				"tab:ablation_test",
			),
			"table_challenge_test.tex": results_table(
				"challenge",
				results["challenge"],
				"Test",
				CHALLENGE_AGENTS,
				test_columns,
				f"CartPole comparison, frozen test ({entries['challenge/n']} replicates)",
				"tab:challenge_test",
			),
			"table_challenge_adapt.tex": results_table(
				"challenge",
				results["challenge"],
				"Adaptation",
				CHALLENGE_AGENTS,
				adapt_columns,
				f"CartPole comparison: training steps to first solve per environment ({entries['challenge/n']} replicates)",
				"tab:challenge_adapt",
			),
			"table_selection_test.tex": results_table(
				"selection",
				results["selection"],
				"Test",
				[SELECTION_REFERENCE] + SELECTION_AGENTS,
				test_columns,
				f"Selection ablation, frozen test: ENA-Combined (Bayesian policy reuse) and the same agents with simpler test-time selection "
				f"({entries['selection/n']} replicates)",
				"tab:selection_test",
			),
			"table_cartpole_envs.tex": environment_table(
				cartpole_rows(make_training_worlds(), make_unseen_worlds()),
				["Environment", "$g$ (m/s$^2$)", "$m_p$ (kg)", "$\\ell$ (m)", "Seen in training"],
				"CartPole Environments",
				"tab:cartpole_worlds",
			),
			"table_cheetah_envs.tex": environment_table(
				cheetah_rows(),
				["Environment", "$g$ (m/s$^2$)", "Friction", "Wind (m/s)", "Air (kg/m$^3$)", "Seen"],
				"HalfCheetah Environments",
				"tab:cheetah_worlds",
			),
		}
	)
	os.makedirs(GENERATED_DIR, exist_ok=True)
	write_macros(entries, os.path.join(GENERATED_DIR, "results.tex"))
	for name, table in tables.items():
		_write(os.path.join(GENERATED_DIR, name), table)
	reliability_figure(results["ablation"], ABLATION_AGENTS[1:], os.path.join(IMAGES_DIR, "ablation_reliability.pdf"))
	reliability_figure(results["challenge"], CHALLENGE_AGENTS, os.path.join(IMAGES_DIR, "challenge_reliability.pdf"))
	plot_study_curves(os.path.join(APP_DIR, STUDIES["challenge"]), "testing", CHALLENGE_AGENTS, os.path.join(IMAGES_DIR, "challenge_test_curves.pdf"))
	write_cheetah_outputs(results["cheetah"], rows, entries)
	print(f"Wrote {len(entries)} numbers and the tables to {GENERATED_DIR}")


if __name__ == "__main__":
	main()
