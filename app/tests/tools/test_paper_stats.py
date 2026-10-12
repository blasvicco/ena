"""Tests for paper_stats.py's handling of a single replicate: no statistics, and no numerical warnings."""

# General imports
import math
import warnings

# App imports
from helpers import step
from tools.paper_stats import cohens_d, difference_entries


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
