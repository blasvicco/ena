"""Tests for ENA's percentile-based return normalization."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.tools.reward_normalizer import normalize_reward


def test_warmup_scales_against_the_solved_return() -> None:
	"""Below 10 returns of history there are no reliable percentiles: a return is divided by the solved
	return, so solving reads as 1.0 (review #8: it used to be divided by the episode length)."""
	with step("Act/Assert: solving maps to 1, half of it to 0.5."):
		memory = SimpleNamespace(reward_history=[100.0, 200.0])
		assert normalize_reward(475.0, memory, return_scale=475.0) == pytest.approx(1.0)
		assert normalize_reward(237.5, memory, return_scale=475.0) == pytest.approx(0.5)


def test_with_history_it_rescales_between_the_5th_and_95th_percentiles_unclipped_above() -> None:
	"""With history, P5 -> 0 and P95 -> 1; a new record still reads above 1 (clipped at 5)."""
	with step("Arrange: 20 returns 0..475."):
		memory = SimpleNamespace(reward_history=[25.0 * index for index in range(20)])

	with step("Act/Assert: P5, P95 and a record."):
		low, high = 23.75, 451.25
		assert normalize_reward(low, memory, return_scale=475.0) == pytest.approx(0.0, abs=1e-6)
		assert normalize_reward(high, memory, return_scale=475.0) == pytest.approx(1.0, abs=1e-6)
		assert normalize_reward(10_000.0, memory, return_scale=475.0) == 5.0
