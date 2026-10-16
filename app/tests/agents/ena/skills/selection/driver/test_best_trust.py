"""Tests for the best-trust selection driver."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.skills.selection.driver.best_trust import Instance


def _recognizer(niche_id):
	"""Recognizer stub reporting a fixed current niche."""
	return SimpleNamespace(current_niche_id=niche_id)


def _population():
	"""Gladiators in niche 0 (trust 1) and niche 1 (trust 3); the non-gladiator has the highest trust of all."""
	gladiators = [{"trust": trust, "is_gladiator": True, "internal_env_id": niche_id} for niche_id, trust in enumerate((1.0, 3.0))]
	return gladiators + [{"trust": 9.0, "is_gladiator": False, "internal_env_id": -1}]


@pytest.mark.parametrize("niche_id", [0, 1, -1])
def test_plays_the_highest_trust_gladiator_whatever_the_niche(niche_id) -> None:
	"""The detector's niche is ignored, and non-gladiators are never played while a gladiator exists."""
	with step("Arrange: a driver evaluated on the population."):
		driver = Instance(exploration_rate=0.0)
		population = _population()
		driver.evaluate(population)

	with step("Act: two test episodes, the first scoring 0."):
		first = driver.select_individual(population, _recognizer(niche_id), memory=None, learn=False)
		driver.record_episode_result(0.0, memory=None)
		second = driver.select_individual(population, _recognizer(niche_id), memory=None, learn=False)

	with step("Assert: slot 1 both times."):
		assert (first, second) == (1, 1)
