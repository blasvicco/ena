"""Tests for HallOfFame's archive-eviction population hygiene."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents.ena.hall_of_fame import HallOfFame

# pylint: disable=protected-access


def _brain(action_dim, state_dim):
	"""Build a minimal one-layer model, fast to construct."""
	return Sequential([Input(shape=(state_dim,)), Dense(action_dim)])


def _build_hall_of_fame(max_per_niche=5, max_gladiator_ratio=0.5):
	return HallOfFame(max_per_niche, max_gladiator_ratio, _brain, action_dim=2, state_dim=2)


def test_evict_clears_the_live_population_members_gladiator_tags() -> None:
	"""A record removed from the archive must not leave a 'zombie gladiator' behind —
	the corresponding live population slot must lose is_gladiator/internal_env_id/env_name."""
	with step("Arrange: a hall of fame with one archived record, and a matching live population slot."):
		hall_of_fame = _build_hall_of_fame()
		individual_id = "gladiator-1"
		hall_of_fame._records[individual_id] = {
			"individual_id": individual_id,
			"trust": 1.0,
			"internal_env_id": 0,
			"env_name": "Mars-Light",
		}
		population = [
			{
				"individual_id": individual_id,
				"is_gladiator": True,
				"internal_env_id": 0,
				"env_name": "Mars-Light",
			}
		]

	with step("Act: evict the record."):
		hall_of_fame._HallOfFame__evict(individual_id, population)

	with step("Assert: the archive entry is gone and the live population slot is no longer a gladiator."):
		assert individual_id not in hall_of_fame._records
		assert population[0]["is_gladiator"] is False
		assert population[0]["internal_env_id"] == -1
		assert population[0]["env_name"] == "Unknown"


def test_evict_leaves_other_population_members_untouched() -> None:
	"""Only the evicted individual's population slot is cleared — other gladiators keep their tags."""
	with step("Arrange: two archived records and two matching live population slots."):
		hall_of_fame = _build_hall_of_fame()
		hall_of_fame._records["gladiator-1"] = {"individual_id": "gladiator-1", "trust": 1.0}
		hall_of_fame._records["gladiator-2"] = {"individual_id": "gladiator-2", "trust": 2.0}
		population = [
			{"individual_id": "gladiator-1", "is_gladiator": True, "internal_env_id": 0, "env_name": "Mars-Light"},
			{"individual_id": "gladiator-2", "is_gladiator": True, "internal_env_id": 1, "env_name": "Jupiter-Heavy"},
		]

	with step("Act: evict only the first record."):
		hall_of_fame._HallOfFame__evict("gladiator-1", population)

	with step("Assert: the second population slot is untouched."):
		assert population[1]["is_gladiator"] is True
		assert population[1]["internal_env_id"] == 1
		assert population[1]["env_name"] == "Jupiter-Heavy"


def test_eco_niche_protection_evicts_the_weakest_and_clears_its_zombie_tags() -> None:
	"""Integration: __eco_niche_protection's eviction path must go through __evict (not a bare
	'del self._records[...]') so the replaced gladiator's population slot is cleared too."""
	with step("Arrange: one weak archived gladiator with a matching live population slot, and a winner from the same niche with higher trust."):
		hall_of_fame = _build_hall_of_fame()
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

		weak_id = "weak-gladiator"
		hall_of_fame._records[weak_id] = {
			"individual_id": weak_id,
			"model": _brain(2, 2),
			"trust": -1.0,
			"internal_env_id": 0,
			"env_name": "Mars-Light",
		}
		population = [
			{"individual_id": weak_id, "is_gladiator": True, "internal_env_id": 0, "env_name": "Mars-Light"},
		]
		winner = {"individual_id": "winner", "model": _brain(2, 2), "is_gladiator": False}

		class _FakeRecognizer:  # pylint: disable=too-few-public-methods
			"""Recognizer stub reporting a fixed current niche id."""

			current_niche_id = 0

	with step("Act: run eco_niche_protection with a winner that beats the weak gladiator's trust."):
		# pylint: disable-next=protected-access
		hall_of_fame._HallOfFame__eco_niche_protection(
			winner,
			current_trust=5.0,
			current_z_score=3.0,
			recognizer=_FakeRecognizer(),
			obs_normalizer=obs_normalizer,
			population=population,
		)

	with step("Assert: the weak gladiator's population slot lost its gladiator tags, and the winner was inducted in its place."):
		assert population[0]["is_gladiator"] is False
		assert population[0]["internal_env_id"] == -1
		assert weak_id not in hall_of_fame._records
		assert winner["is_gladiator"] is True
		assert winner["internal_env_id"] == 0
