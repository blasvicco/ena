#!/usr/bin/env python
"""The paper's statistical tests and number formatting, shared by every results generator."""
# The paper's statistical tests and number formatting, shared by every results generator (paper_results.py and its
# per-domain modules). Every agent of a replicate starts from the same seed, so two agents are compared over the
# replicates both have, paired by replicate: the paired t-test with its interval and effect size d_z, the Wilcoxon
# signed-rank test for heavy-tailed metrics, power analysis, Holm's correction (and how many of a family's decisions
# Welch's unpaired test keeps), exact tests for pass rates, and how values, p-values and table cells are printed.

# General imports
import math
from typing import Callable

# Libs imports
import numpy as np
import pandas as pd
from scipy import optimize, stats

# Statistical conventions, stated in the paper from these same constants (stats/* macros).
ALPHA = 0.05
POWER = 0.8
CI_LEVEL = 0.95


# Metrics printed as whole numbers with thousands separators: step counts, and returns and absolute forgetting in
# HalfCheetah's return units (thousands, where a decimal adds nothing); everything else to one decimal.
COUNT_METRICS = ("steps", "Steps", "candsteps", "forgetting", "return")
# A true minus sign in text and in math mode alike (a bare "-" prints as a hyphen in text).
MINUS = "\\ensuremath{-}"


def signed(text: str) -> str:
	"""Print a formatted number's leading minus as a true minus sign (and none on a value that rounds to zero)."""
	if not text.startswith("-"):
		return text
	return MINUS + text[1:] if text.strip("-0.,") else text[1:]


def format_value(value: float, metric_slug: str) -> str:
	"""A value as the paper prints it: counts with thousands separators, the rest to one decimal."""
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "n/a"
	if any(marker in metric_slug for marker in COUNT_METRICS):
		return signed(f"{value:,.0f}")
	return signed(f"{value:.1f}")


def format_p(p_value: float) -> str:
	"""p-value as printed: bounded below 0.001, else three decimals."""
	if p_value is None or math.isnan(p_value):
		return "n/a"
	return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def p_macro(p_value: float) -> str:
	"""A p-value as the paper prints it, in math mode."""
	p_text = format_p(p_value)
	return f"$p{p_text}$" if p_text.startswith("<") else f"$p={p_text}$"


def quartiles(values: pd.Series) -> tuple:
	"""The first quartile, median and third quartile of the values."""
	return tuple(float(value) for value in values.quantile([0.25, 0.5, 0.75]))


# A results table's cell styles: each prints one agent's per-replicate values of one metric.
def mean_cell(values: pd.Series, metric_slug: str) -> str:
	"""A table cell: the mean over replicates."""
	return format_value(values.mean(), metric_slug)


def mean_std_cell(values: pd.Series, metric_slug: str) -> str:
	"""A table cell: the mean over replicates, with the standard deviation when there are several."""
	spread = f" $\\pm$ {format_value(values.std(), metric_slug)}" if len(values) > 1 else ""
	return mean_cell(values, metric_slug) + spread


def median_iqr_cell(values: pd.Series, metric_slug: str) -> str:
	"""A table cell: the median over replicates, with the interquartile range in parentheses on the line below."""
	low, median, high = (format_value(value, metric_slug) for value in quartiles(values))
	return f"\\begin{{tabular}}{{@{{}}c@{{}}}}{median}\\\\({low}--{high})\\end{{tabular}}"


def clopper_pearson(passes: int, trials: int, level: float = CI_LEVEL) -> tuple:
	"""Exact (Clopper-Pearson) two-sided confidence interval of a pass rate."""
	tail = (1.0 - level) / 2.0
	low = 0.0 if passes == 0 else float(stats.beta.ppf(tail, passes, trials - passes + 1))
	high = 1.0 if passes == trials else float(stats.beta.ppf(1.0 - tail, passes + 1, trials - passes))
	return low, high


def mcnemar_p(passes_a: np.ndarray | list, passes_b: np.ndarray | list) -> float:
	"""Exact two-sided McNemar test of two pass/fail records paired by seed: a binomial test on the discordant seeds."""
	passes_a, passes_b = np.asarray(passes_a, dtype=bool), np.asarray(passes_b, dtype=bool)
	only_a, only_b = int(np.sum(passes_a & ~passes_b)), int(np.sum(~passes_a & passes_b))
	if only_a + only_b == 0:
		return 1.0
	return float(stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)


def paired_samples(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list) -> tuple:
	"""The two samples over the replicates both have, as float arrays: Series are matched by index (replicate), sequences by position."""
	if isinstance(sample_a, pd.Series) and isinstance(sample_b, pd.Series):
		pairs = pd.concat({"a": sample_a, "b": sample_b}, axis=1, join="inner").dropna()
		return pairs["a"].to_numpy(dtype=float), pairs["b"].to_numpy(dtype=float)
	sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
	if len(sample_a) != len(sample_b):
		raise ValueError("Paired samples need one value per replicate each.")
	return sample_a, sample_b


def paired_p(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list) -> float:
	"""Two-sided paired t-test p-value over the shared replicates; 1.0 or 0.0 for constant differences, NaN for fewer than 2 pairs."""
	sample_a, sample_b = paired_samples(sample_a, sample_b)
	if len(sample_a) < 2:
		return float("nan")
	differences = sample_b - sample_a
	if np.std(differences) == 0:
		return 1.0 if differences.mean() == 0 else 0.0
	return float(stats.ttest_rel(sample_b, sample_a).pvalue)


def paired_ci(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list, level: float = CI_LEVEL) -> tuple:
	"""Confidence interval of the mean of b - a over the shared replicates; (NaN, NaN) for fewer than 2 pairs or constant differences."""
	sample_a, sample_b = paired_samples(sample_a, sample_b)
	if len(sample_a) < 2 or np.std(sample_b - sample_a) == 0:
		return float("nan"), float("nan")
	interval = stats.ttest_rel(sample_b, sample_a).confidence_interval(level)
	return float(interval.low), float(interval.high)


def cohens_dz(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list) -> float:
	"""Paired effect size d_z of b over a: the mean of b - a over its standard deviation (NaN for fewer than 2 pairs or no spread)."""
	sample_a, sample_b = paired_samples(sample_a, sample_b)
	if len(sample_a) < 2:
		return float("nan")
	differences = sample_b - sample_a
	spread = differences.std(ddof=1)
	return float(differences.mean() / spread) if spread > 0 else float("nan")


def welch_p(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list) -> float:
	"""Two-sided Welch's (unpaired) t-test p-value; 1.0 for two identical constant samples, NaN when either has n < 2."""
	sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
	if len(sample_a) < 2 or len(sample_b) < 2:
		return float("nan")
	if np.std(sample_a) == 0 and np.std(sample_b) == 0:
		return 1.0 if np.mean(sample_a) == np.mean(sample_b) else 0.0
	return float(stats.ttest_ind(sample_a, sample_b, equal_var=False).pvalue)


def wilcoxon_p(sample_a: pd.Series | np.ndarray | list, sample_b: pd.Series | np.ndarray | list) -> float:
	"""Two-sided Wilcoxon signed-rank p-value over the shared replicates; 1.0 when every difference is zero, NaN without pairs."""
	sample_a, sample_b = paired_samples(sample_a, sample_b)
	if len(sample_a) == 0:
		return float("nan")
	if not np.any(sample_b - sample_a):
		return 1.0
	return float(stats.wilcoxon(sample_b, sample_a).pvalue)


def _power(effect: float, replicates: int, alpha: float) -> float:
	"""Power of a two-sided paired t-test over `replicates` pairs at paired effect size `effect` (d_z)."""
	freedom = replicates - 1
	return float(stats.nct.sf(stats.t.ppf(1 - alpha / 2, freedom), freedom, abs(effect) * math.sqrt(replicates)))


def _power_independent(effect: float, replicates: int, alpha: float) -> float:
	"""Power of a two-sided two-sample t-test with `replicates` per group at effect size `effect` (the pre-run power analysis)."""
	freedom = 2 * replicates - 2
	return float(stats.nct.sf(stats.t.ppf(1 - alpha / 2, freedom), freedom, abs(effect) * math.sqrt(replicates / 2)))


def replicates_for_power(effect: float, alpha: float, power: float = POWER, limit: int = 100_000) -> int | None:
	"""Smallest number of paired replicates reaching `power` at paired effect size `effect` (None for no effect)."""
	if not effect or math.isnan(effect):
		return None
	low, high = 2, 2
	while _power(effect, high, alpha) < power:
		low, high = high, high * 2
		if high > limit:
			return None
	while low < high:
		middle = (low + high) // 2
		low, high = (middle + 1, high) if _power(effect, middle, alpha) < power else (low, middle)
	return low


def _detectable(power_of: Callable[[float, int, float], float], replicates: int, alpha: float, power: float) -> float:
	"""Smallest effect size a power function reaches `power` for at `replicates` and `alpha`."""
	return float(optimize.brentq(lambda effect: power_of(effect, replicates, alpha) - power, 1e-3, 20.0))


def detectable_d(replicates: int, alpha: float, power: float = POWER) -> float:
	"""Smallest paired effect size d_z a two-sided paired t-test detects with `power` over `replicates` pairs."""
	return _detectable(_power, replicates, alpha, power)


def planned_detectable_d(replicates: int, alpha: float, power: float = POWER) -> float:
	"""Smallest effect size d the pre-run power analysis (two independent groups of `replicates`) detects with `power`."""
	return _detectable(_power_independent, replicates, alpha, power)


def holm(p_values: list, family_size: int = None) -> list:
	"""Holm-adjusted p-values, in the input order; family_size defaults to the number of p-values."""
	size = family_size or len(p_values)
	adjusted, running = [0.0] * len(p_values), 0.0
	for rank, index in enumerate(sorted(range(len(p_values)), key=lambda position: p_values[position])):
		running = max(running, min(1.0, (size - rank) * p_values[index]))
		adjusted[index] = running
	return adjusted


def holm_family(comparisons: list, family_size: int = None) -> tuple:
	"""Holm-adjusted paired p-values of a family of (sample a, sample b) comparisons, and how many keep their decision under Welch's test."""
	paired = holm([paired_p(sample_a, sample_b) for sample_a, sample_b in comparisons], family_size)
	unpaired = holm([welch_p(*paired_samples(sample_a, sample_b)) for sample_a, sample_b in comparisons], family_size)
	return paired, sum((paired_value < ALPHA) == (unpaired_value < ALPHA) for paired_value, unpaired_value in zip(paired, unpaired))


def welch_agreement_entries(family: str, agree: int, planned: int) -> dict:
	"""A Holm family's count of decisions Welch's unpaired test keeps, and its number of comparisons."""
	return {f"{family}/welch/agree": str(agree), f"{family}/welch/planned": str(planned)}


# pylint: disable=too-many-arguments,too-many-positional-arguments
def difference_entries(
	key: str, values_a: pd.Series | np.ndarray | list, values_b: pd.Series | np.ndarray | list, alpha: float, metric_slug: str
) -> dict:
	"""b vs."""
	# a under one macro key, over the replicates both have: paired p, the mean of b - a with its interval, effect size d_z,
	# and the paired replicates needed for 80% power at alpha.
	values_a, values_b = paired_samples(values_a, values_b)
	difference = (values_b - values_a).mean() if len(values_a) else float("nan")
	low, high = paired_ci(values_a, values_b)
	effect = cohens_dz(values_a, values_b)
	needed = replicates_for_power(effect, alpha)
	return {
		f"{key}/p": p_macro(paired_p(values_a, values_b)),
		f"{key}/diff": ("+" if difference >= 0 else "") + format_value(difference, metric_slug),
		f"{key}/ci/low": format_value(low, metric_slug),
		f"{key}/ci/high": format_value(high, metric_slug),
		f"{key}/d": "n/a" if math.isnan(effect) else signed(f"{effect:.2f}"),
		f"{key}/nreq": "n/a" if needed is None else f"{needed:,}",
	}
