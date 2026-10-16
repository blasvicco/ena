"""Tests for the selection skill facade across its drivers."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.skills.selection.main import SelectionController

DRIVERS = ["bpr", "niche_match", "best_trust"]


def _population(seed):
	"""Ten individuals with random trust, three of them gladiators of niches 0-2."""
	rng = np.random.default_rng(seed)
	population = [{"trust": float(trust), "is_gladiator": False, "internal_env_id": -1} for trust in rng.random(10)]
	for niche_id, index in enumerate((2, 5, 8)):
		population[index].update(is_gladiator=True, internal_env_id=niche_id)
	return population


def _training_picks(driver):
	"""200 seeded training selections over populations whose trust changes every episode."""
	controller = SelectionController(driver, exploration_rate=0.3, return_scale=475.0)
	np.random.seed(0)
	return [controller.select_individual(_population(episode), None, memory=None, learn=True) for episode in range(200)]


@pytest.mark.parametrize("driver", DRIVERS[1:])
def test_training_selection_is_identical_to_bpr(driver) -> None:
	"""Drivers differ only at test time: under the same seed, training picks and the random stream match BPR's exactly."""
	with step("Arrange/Act: the same seeded training selections under BPR and under the driver."):
		expected = _training_picks("bpr")
		expected_next_draw = np.random.rand()
		picks = _training_picks(driver)
		next_draw = np.random.rand()

	with step("Assert: identical picks, and the global random stream left in the same state."):
		assert picks == expected
		assert next_draw == expected_next_draw


@pytest.mark.parametrize("driver", DRIVERS)
def test_resolves_each_driver(driver) -> None:
	"""Every selection driver name resolves, evaluates a population and picks a gladiator at test time."""
	with step("Arrange: a controller evaluated on a population with gladiators."):
		controller = SelectionController(driver, exploration_rate=0.0, return_scale=475.0)
		population = _population(0)
		controller.evaluate(population)

	with step("Act: a test-time selection without a recognizer."):
		pick = controller.select_individual(population, None, memory=None, learn=False)

	with step("Assert: a gladiator plays."):
		assert population[pick]["is_gladiator"] is True


def test_an_unknown_driver_is_rejected() -> None:
	"""A misspelled driver name fails loudly."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError, match="Unknown selection driver"):
			SelectionController("nearest", exploration_rate=0.0)


@pytest.mark.parametrize("driver", DRIVERS)
def test_with_no_gladiators_it_plays_a_non_gladiator(driver) -> None:
	"""No archive: every driver falls back to a random non-gladiator at test time."""
	with step("Arrange: a population without gladiators."):
		controller = SelectionController(driver, exploration_rate=0.0, return_scale=475.0)
		population = [{**individual, "is_gladiator": False} for individual in _population(0)]
		controller.evaluate(population)

	with step("Act: a test-time selection."):
		pick = controller.select_individual(population, SimpleNamespace(current_niche_id=0), memory=None, learn=False)

	with step("Assert: a non-gladiator."):
		assert population[pick]["is_gladiator"] is False
