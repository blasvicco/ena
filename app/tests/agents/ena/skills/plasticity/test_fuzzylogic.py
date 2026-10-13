"""Tests for the Plasticity skill's Mamdani fuzzy-logic driver."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.skills.plasticity.main import Plasticity

HISTORY = [float(score) for score in range(0, 500, 25)]  # 90th percentile = 452.5; worst seen = 0


def _memory(trend=0.0, history=None):
	"""Memory stub with a reward history, the worst score seen, and a performance derivative (the trend input)."""
	return SimpleNamespace(reward_history=HISTORY if history is None else history, min_seen=0.0, performance_derivative=trend)


def _plasticity(reward, trend=0.0, history=None):
	return Plasticity("fuzzylogic").compute(reward, _memory(trend, history), learn=True)


def test_warmup_returns_max_plasticity() -> None:
	"""Fewer than 10 episodes of history: full search effort, as with every adaptive driver."""
	with step("Act/Assert: 1.0 with a 5-episode history."):
		assert _plasticity(500.0, history=[500.0] * 5) == 1.0


def test_extremes_reach_the_full_output_range() -> None:
	"""Top score with a steady trend -> 0.1; worst score -> 1.0 (centroid shrinkage avoided)."""
	with step("Act/Assert: the two ends of the range."):
		assert _plasticity(500.0) == pytest.approx(0.1, abs=1e-3)
		assert _plasticity(0.0) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("trend", [-500.0, -100.0, 0.0, 100.0, 500.0])
def test_output_stays_within_bounds(trend) -> None:
	"""Every score/trend combination yields plasticity in [0.1, 1.0]."""
	with step("Act/Assert: a sweep of scores."):
		values = [_plasticity(score, trend) for score in np.linspace(-50, 550, 61)]
		assert min(values) >= 0.1 - 1e-9 and max(values) <= 1.0 + 1e-9


@pytest.mark.parametrize("trend", [-300.0, -200.0, -100.0, 0.0, 200.0])
def test_worse_scores_never_get_meaningfully_less_plasticity(trend) -> None:
	"""Approximately monotonic in the score at a fixed trend: doing worse never reduces search effort by more
	than 0.02. Exact monotonicity isn't a property of Mamdani max-min inference: where two rules with the same
	consequent cross (LOW->HIGH fading as MEDIUM-and-FALLING->HIGH rises), their max dips and the centroid
	moves briefly toward MEDIUM (measured: at most +0.012, scores 150-210, trend -200)."""
	with step("Act/Assert: never rises by more than 0.02 over an ascending score sweep, and falls overall."):
		values = [_plasticity(score, trend) for score in np.linspace(0, 500, 51)]
		assert all(later <= earlier + 0.02 for earlier, later in zip(values, values[1:]))
		assert values[-1] < values[0]


def test_a_falling_trend_never_gets_less_plasticity_than_a_steady_one() -> None:
	"""At any score, performance falling triggers at least as much search as performance holding steady."""
	with step("Act/Assert: falling >= steady across the score range."):
		for score in np.linspace(0, 500, 26):
			assert _plasticity(score, trend=-300.0) >= _plasticity(score, trend=0.0) - 1e-9


def test_a_falling_trend_raises_plasticity_for_a_good_score() -> None:
	"""The trend input matters: a high score that's been falling gets more search than the same score held steady."""
	with step("Act/Assert: strictly more effort when falling."):
		assert _plasticity(480.0, trend=-300.0) > _plasticity(480.0, trend=0.0) + 0.1
