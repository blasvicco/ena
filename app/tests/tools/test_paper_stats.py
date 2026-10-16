"""Tests for paper_stats.py's handling of a single replicate: no statistics, and no numerical warnings."""

# General imports
import math
import warnings

# Libs imports
import pytest

# App imports
from helpers import step
from tools.paper_stats import MINUS, cohens_d, difference_entries, format_value, signed


def test_one_replicate_gives_no_statistics_and_no_warnings() -> None:
	"""With one replicate per agent (a validation run) there is no spread: every statistic is n/a, quietly."""
	with step("Act: compare one value with one value, turning any warning into an error."):
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			entries = difference_entries("k", [1.0], [2.0], 0.05, "x")
			effect = cohens_d([1.0], [2.0])

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
