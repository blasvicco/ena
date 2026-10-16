"""Tests for the shared curve tool backing the Trust and Plasticity skills."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.tools import curves


def _build_memory(reward_history=(), min_seen=0.0, max_seen=0.0):
	"""A minimal stand-in for AgentMemory, exposing only what the curves read."""
	return SimpleNamespace(reward_history=list(reward_history), min_seen=min_seen, max_seen=max_seen)


def test_zero_always_returns_zero() -> None:
	"""zero() ignores its inputs entirely."""
	with step("Act/Assert: zero() returns 0.0 regardless of reward or memory state."):
		assert curves.zero(500.0, _build_memory()) == 0.0


def test_full_always_returns_one() -> None:
	"""full() ignores its inputs entirely."""
	with step("Act/Assert: full() returns 1.0 regardless of reward or memory state."):
		assert curves.full(0.0, _build_memory()) == 1.0


def test_linear_ramp_returns_max_during_warmup() -> None:
	"""Fewer than 10 reward samples means no reliable percentile estimate yet — max value."""
	with step("Arrange: fewer than 10 samples in reward_history."):
		memory = _build_memory(reward_history=[500.0] * 5)

	with step("Assert: the curve returns 1.0 regardless of the reward passed."):
		assert curves.linear_ramp(500.0, memory) == 1.0


def test_linear_ramp_returns_low_value_at_or_above_high_limit() -> None:
	"""A reward at or above the 90th percentile means the agent is settled — low value."""
	with step("Arrange: a stable, high-scoring reward history."):
		memory = _build_memory(reward_history=[500.0] * 10, min_seen=0.0)

	with step("Assert: the curve returns 0.1 for a reward at the high limit."):
		assert curves.linear_ramp(500.0, memory) == pytest.approx(0.1)


def test_linear_ramp_scales_proportionately_instead_of_flattening_below_a_floor() -> None:
	"""The linear ramp scales continuously down to memory.min_seen instead of flattening below a floor."""
	# Stage 7: the curve must scale continuously down to memory.min_seen, not flatten to a
	# flat 1.0 the moment a score dips below the (possibly very narrow) percentile band — a
	# merely mediocre score must not be treated identically to the worst score ever seen.
	with step("Arrange: reward_history pinned at 500 (high_limit=500), min_seen=0 (the floor)."):
		memory = _build_memory(reward_history=[500.0] * 10, min_seen=0.0)

	with step("Act: compute the curve for a mediocre score and a worse one."):
		mediocre = curves.linear_ramp(400.0, memory)
		worse = curves.linear_ramp(250.0, memory)
		worst = curves.linear_ramp(0.0, memory)

	with step("Assert: the curve scales proportionately — worse scores get strictly higher values, not an identical flat maximum."):
		assert mediocre == pytest.approx(0.28)
		assert worse == pytest.approx(0.55)
		assert worst == pytest.approx(1.0)
		assert mediocre < worse < worst


def test_quadratic_returns_max_during_warmup() -> None:
	"""Fewer than 10 reward samples means no reliable benchmark yet — max value."""
	with step("Arrange: fewer than 10 samples in reward_history."):
		memory = _build_memory(reward_history=[500.0] * 5, max_seen=500.0, min_seen=0.0)

	with step("Assert: the curve returns 1.0 regardless of the reward passed."):
		assert curves.quadratic(500.0, memory) == 1.0


def test_quadratic_returns_low_value_at_the_best_seen_score() -> None:
	"""A reward matching the best score ever seen should minimize the curve's output."""
	with step("Arrange: a reward history with a known best/worst range."):
		memory = _build_memory(reward_history=[500.0] * 10, max_seen=500.0, min_seen=0.0)

	with step("Assert: the curve floors at 0.1 for the best-ever score."):
		assert curves.quadratic(500.0, memory) == pytest.approx(0.1)
