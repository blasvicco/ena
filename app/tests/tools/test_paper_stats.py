"""Tests for paper_stats.py: a single replicate gives no statistics, the signed-rank test, and how values and table cells print."""

# General imports
import math
import warnings

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.paper_stats import (
	MINUS,
	cohens_dz,
	difference_entries,
	format_value,
	mean_cell,
	mean_std_cell,
	median_iqr_cell,
	signed,
	wilcoxon_p,
)


def test_one_replicate_gives_no_statistics_and_no_warnings() -> None:
	"""With one replicate per agent (a validation run) there is no spread: every statistic is n/a, quietly."""
	with step("Act: compare one value with one value, turning any warning into an error."):
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			entries = difference_entries("k", [1.0], [2.0], 0.05, "x")
			effect = cohens_dz([1.0], [2.0])

	with step("Assert: the difference is known; the rest is n/a."):
		assert entries["k/diff"] == "+1.0"
		assert math.isnan(effect)
		assert (entries["k/d"], entries["k/nreq"], entries["k/ci/low"]) == ("n/a", "n/a", "n/a")


@pytest.mark.parametrize("text, expected", [("-13.8", f"{MINUS}13.8"), ("-3,000", f"{MINUS}3,000"), ("-0", "0"), ("-0.0", "0.0"), ("12.5", "12.5")])
def test_signed_prints_a_true_minus_sign_and_none_on_zero(text, expected) -> None:
	"""A negative number gets a minus sign that prints as one in text and math mode; a value rounding to zero gets none."""
	with step("Act/Assert: the signed text."):
		assert signed(text) == expected


@pytest.mark.parametrize(
	"value, metric_slug, expected", [(-239.4, "return", f"{MINUS}239"), (2208.3, "return", "2,208"), (-13.84, "rel", f"{MINUS}13.8")]
)
def test_format_value_prints_returns_whole_and_negatives_with_a_minus_sign(value, metric_slug, expected) -> None:
	"""HalfCheetah-scale returns print as whole numbers with separators; negatives with a true minus sign."""
	with step("Act/Assert: the formatted value."):
		assert format_value(value, metric_slug) == expected


def test_wilcoxon_p_separates_paired_shifts_and_handles_degenerate_ones() -> None:
	"""A signed-rank test on the pairs: one extreme pair does not hide a shift; no difference gives 1, no pair NaN."""
	with step("Arrange: b is a shifted copy of a, plus one huge outlier in b that would swamp a t-test."):
		sample_a = list(range(20))
		sample_b = [value + 15 for value in range(19)] + [10_000_000]

	with step("Act/Assert: the shift is detected; degenerate samples are handled quietly."):
		assert wilcoxon_p(sample_a, sample_b) < 0.001
		assert wilcoxon_p([5.0, 5.0], [5.0, 5.0]) == 1.0
		assert math.isnan(wilcoxon_p([], []))


@pytest.mark.parametrize(
	"cell, expected",
	[(mean_cell, "4.0"), (mean_std_cell, "4.0 $\\pm$ 3.5"), (median_iqr_cell, "\\begin{tabular}{@{}c@{}}3.0\\\\(2.0--4.0)\\end{tabular}")],
)
def test_table_cells_print_the_mean_the_mean_with_its_spread_or_the_median_with_its_quartiles(cell, expected) -> None:
	"""A results table's cell styles: mean; mean and standard deviation; median and interquartile range."""
	with step("Act/Assert: the cell of five replicates, one of them slow."):
		assert cell(pd.Series([1.0, 2.0, 3.0, 4.0, 10.0]), "rel") == expected
