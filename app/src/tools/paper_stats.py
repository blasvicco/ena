#!/usr/bin/env python
"""The paper's statistical tests and number formatting, shared by every results generator."""
# The paper's statistical tests and number formatting, shared by every results generator (paper_results.py and its
# per-domain modules): Welch's t-test with its interval and effect size, power analysis, Holm's correction, exact tests
# for pass rates, and how values and p-values are printed.

# General imports
import math

# Libs imports
import numpy as np
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


def welch_p(sample_a: np.ndarray | list, sample_b: np.ndarray | list) -> float:
	"""Two-sided Welch's t-test p-value; 1.0 for two identical constant samples, NaN when either has n < 2."""
	sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
	if len(sample_a) < 2 or len(sample_b) < 2:
		return float("nan")
	if np.std(sample_a) == 0 and np.std(sample_b) == 0:
		return 1.0 if np.mean(sample_a) == np.mean(sample_b) else 0.0
	return float(stats.ttest_ind(sample_a, sample_b, equal_var=False).pvalue)


def welch_ci(sample_a: np.ndarray | list, sample_b: np.ndarray | list, level: float = CI_LEVEL) -> tuple:
	"""Welch confidence interval of mean(b) - mean(a); (NaN, NaN) when either sample has n < 2 or both are constant."""
	sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
	if len(sample_a) < 2 or len(sample_b) < 2 or (np.std(sample_a) == 0 and np.std(sample_b) == 0):
		return float("nan"), float("nan")
	interval = stats.ttest_ind(sample_b, sample_a, equal_var=False).confidence_interval(level)
	return float(interval.low), float(interval.high)


def cohens_d(sample_a: np.ndarray | list, sample_b: np.ndarray | list) -> float:
	"""Effect size of b over a: difference of means over the root mean square of the two sample standard deviations."""
	sample_a, sample_b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
	if len(sample_a) < 2 or len(sample_b) < 2:
		return float("nan")  # no spread from one replicate
	spread = math.sqrt((sample_a.var(ddof=1) + sample_b.var(ddof=1)) / 2)
	return float((sample_b.mean() - sample_a.mean()) / spread) if spread > 0 else float("nan")


def _power(effect: float, replicates: int, alpha: float) -> float:
	"""Power of a two-sided two-sample t-test with `replicates` per group at effect size `effect`."""
	freedom = 2 * replicates - 2
	return float(stats.nct.sf(stats.t.ppf(1 - alpha / 2, freedom), freedom, abs(effect) * math.sqrt(replicates / 2)))


def replicates_for_power(effect: float, alpha: float, power: float = POWER, limit: int = 100_000) -> int | None:
	"""Smallest replicates per group reaching `power` at effect size `effect` (None for no effect)."""
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


def detectable_d(replicates: int, alpha: float, power: float = POWER) -> float:
	"""Smallest effect size a two-sided two-sample t-test detects with `power` at `replicates` per group."""
	return float(optimize.brentq(lambda effect: _power(effect, replicates, alpha) - power, 1e-3, 20.0))


def holm(p_values: list, family_size: int = None) -> list:
	"""Holm-adjusted p-values, in the input order; family_size defaults to the number of p-values."""
	size = family_size or len(p_values)
	adjusted, running = [0.0] * len(p_values), 0.0
	for rank, index in enumerate(sorted(range(len(p_values)), key=lambda position: p_values[position])):
		running = max(running, min(1.0, (size - rank) * p_values[index]))
		adjusted[index] = running
	return adjusted


# pylint: disable=too-many-arguments,too-many-positional-arguments
def difference_entries(key: str, values_a: np.ndarray | list, values_b: np.ndarray | list, alpha: float, metric_slug: str) -> dict:
	"""b vs."""
	# a under one macro key: Welch p, mean difference (b - a) with its interval, effect size d, and replicates per agent for 80% power at alpha.
	values_a, values_b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
	difference = values_b.mean() - values_a.mean()
	low, high = welch_ci(values_a, values_b)
	effect = cohens_d(values_a, values_b)
	needed = replicates_for_power(effect, alpha)
	return {
		f"{key}/p": p_macro(welch_p(values_a, values_b)),
		f"{key}/diff": ("+" if difference >= 0 else "") + format_value(difference, metric_slug),
		f"{key}/ci/low": format_value(low, metric_slug),
		f"{key}/ci/high": format_value(high, metric_slug),
		f"{key}/d": "n/a" if math.isnan(effect) else signed(f"{effect:.2f}"),
		f"{key}/nreq": "n/a" if needed is None else f"{needed:,}",
	}
