"""Tests for the Predictive-Mixture CuSum change detector."""

# General imports
import math

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.tools.pm_cusum import PMCuSum

THRESHOLD = math.log(1e6)  # average run length to false alarm >= 1e6 observations


def _first_alarm(detector, observations):
	"""Index of the first observation at which the detector fires, or None."""
	for index, observation in enumerate(observations):
		if detector.update(observation):
			return index
	return None


def test_windows_are_powers_of_two_up_to_log2_of_the_threshold() -> None:
	"""W = {2^r : r = 1..ceil(log2 b)}."""
	with step("Arrange/Act: a detector with b = log(1e6) ~ 13.8."):
		detector = PMCuSum(dim=4, threshold=THRESHOLD)

	with step("Assert: windows 2, 4, 8, 16."):
		assert detector.windows == [2, 4, 8, 16]


def test_no_false_alarm_on_pre_change_data() -> None:
	"""On N(0, I) data the statistic must stay under b = log(1e6) for thousands of samples."""
	with step("Arrange: a detector and 5000 pre-change samples."):
		detector = PMCuSum(dim=4, threshold=THRESHOLD)
		observations = np.random.default_rng(0).normal(size=(5000, 4))

	with step("Act/Assert: never fires."):
		assert _first_alarm(detector, observations) is None


@pytest.mark.parametrize(
	"change",
	[
		pytest.param(lambda samples: samples + 1.0, id="mean-shift"),
		pytest.param(lambda samples: samples * 3.0, id="variance-increase"),
	],
)
def test_detects_a_change_quickly(change) -> None:
	"""A mean shift or a variance increase after 500 pre-change samples is caught within 30 samples."""
	with step("Arrange: 500 pre-change samples followed by changed ones."):
		rng = np.random.default_rng(1)
		pre = rng.normal(size=(500, 4))
		post = change(rng.normal(size=(500, 4)))
		detector = PMCuSum(dim=4, threshold=THRESHOLD)

	with step("Act: stream both."):
		alarm = _first_alarm(detector, np.vstack([pre, post]))

	with step("Assert: fires after the change, with a short delay."):
		assert alarm is not None
		assert 500 <= alarm < 530


def test_reset_restarts_the_statistic_and_history() -> None:
	"""After reset() the detector behaves as freshly built."""
	with step("Arrange: a detector that has been driven high by changed data."):
		detector = PMCuSum(dim=2, threshold=THRESHOLD)
		for _ in range(20):
			detector.update(np.full(2, 3.0))
		assert detector.statistic > 0.0

	with step("Act: reset."):
		detector.reset()

	with step("Assert: statistic back to zero."):
		assert detector.statistic == 0.0


def test_a_variance_decrease_is_not_a_change() -> None:
	"""One-sided: residuals narrower than the pre-change (a model fitting *better*) never fire."""
	with step("Arrange: 500 pre-change samples, then 2000 with a tenth of the spread and exact zeros."):
		rng = np.random.default_rng(2)
		narrower = np.hstack([0.1 * rng.normal(size=(2000, 2)), np.zeros((2000, 2))])
		observations = np.vstack([rng.normal(size=(500, 4)), narrower])
		detector = PMCuSum(dim=4, threshold=THRESHOLD)

	with step("Act/Assert: never fires."):
		assert _first_alarm(detector, observations) is None
