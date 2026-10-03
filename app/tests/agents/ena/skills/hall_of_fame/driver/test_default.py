"""Tests for the default Hall of Fame driver's archive-eviction population hygiene."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents.ena.skills.hall_of_fame.driver.default import Instance

# pylint: disable=protected-access


def _brain(action_dim, state_dim):
	"""Build a minimal one-layer model, fast to construct."""
	return Sequential([Input(shape=(state_dim,)), Dense(action_dim)])


def _build_hall_of_fame(max_per_niche=5, max_gladiator_ratio=0.5):
	return Instance(max_per_niche, max_gladiator_ratio, _brain, action_dim=2, state_dim=2)


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
		hall_of_fame._Instance__evict(individual_id, population)

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
		hall_of_fame._Instance__evict("gladiator-1", population)

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
		hall_of_fame._Instance__eco_niche_protection(
			winner,
			current_trust=5.0,
			current_z_score=3.0,
			recognizer=_FakeRecognizer(),
			obs_normalizer=obs_normalizer,
			population=population,
			env_name="Mars-Light",
		)

	with step("Assert: the weak gladiator's population slot lost its gladiator tags, and the winner was inducted in its place."):
		assert population[0]["is_gladiator"] is False
		assert population[0]["internal_env_id"] == -1
		assert weak_id not in hall_of_fame._records
		assert winner["is_gladiator"] is True
		assert winner["internal_env_id"] == 0


# ---------------------------------------------------------------
# S8: a gladiator's env label is permanent; foreign-niche success clones it
# ---------------------------------------------------------------


class _NicheRecognizer:  # pylint: disable=too-few-public-methods
	"""Recognizer stub whose match_niche() resolves every winner to a fixed niche."""

	def __init__(self, niche_id):
		self.current_niche_id = niche_id

	def match_niche(self, _memory, _winner, _hall_of_fame, population=None):  # pylint: disable=unused-argument
		"""Resolve to the fixed niche."""
		return self.current_niche_id


def _archived_gladiator(hall_of_fame, individual_id="mars-gladiator", niche_id=0, env_name="Mars-Light", trust=1.0):
	"""Archive a gladiator and return its matching live population member."""
	model = _brain(2, 2)
	hall_of_fame._records[individual_id] = {
		"individual_id": individual_id,
		"model": _brain(2, 2),
		"trust": trust,
		"induction_z_score": 2.0,
		"internal_env_id": niche_id,
		"env_name": env_name,
		"is_gladiator": True,
	}
	hall_of_fame._records[individual_id]["model"].set_weights(model.get_weights())
	return {"individual_id": individual_id, "model": model, "trust": trust, "is_gladiator": True, "internal_env_id": niche_id, "env_name": env_name}


def _update(hall_of_fame, winner, population, *, niche_id, env_name, trust=5.0):  # pylint: disable=too-many-arguments
	"""Run update() for a winner resolved to niche_id while playing env_name, past the performance gate."""
	obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))
	memory = SimpleNamespace(plasticity_history=[0.1])
	hall_of_fame.update(winner, trust, population, memory, _NicheRecognizer(niche_id), obs_normalizer, env_name=env_name)


def test_gladiator_winning_in_a_foreign_niche_is_cloned_there_with_the_current_env_label() -> None:
	"""S8: the original keeps its niche and label; a clone joins the new niche, labeled with
	the environment it just succeeded in."""
	with step("Arrange: a Mars-Light gladiator (niche 0), live in the population."):
		hall_of_fame = _build_hall_of_fame()
		gladiator = _archived_gladiator(hall_of_fame)
		population = [gladiator, {"individual_id": "peer", "trust": 0.0}]

	with step("Act: the gladiator wins while playing Jupiter-Heavy, resolved to niche 1."):
		_update(hall_of_fame, gladiator, population, niche_id=1, env_name="Jupiter-Heavy")

	with step("Assert: original record and live member unchanged; one Jupiter-Heavy clone in niche 1 with the same weights."):
		original = hall_of_fame._records["mars-gladiator"]
		assert (original["internal_env_id"], original["env_name"]) == (0, "Mars-Light")
		assert (gladiator["internal_env_id"], gladiator["env_name"]) == (0, "Mars-Light")
		clones = [record for record in hall_of_fame._records.values() if record["internal_env_id"] == 1]
		assert len(clones) == 1
		assert clones[0]["env_name"] == "Jupiter-Heavy"
		assert clones[0]["individual_id"] != "mars-gladiator"
		for clone_weights, source_weights in zip(clones[0]["model"].get_weights(), gladiator["model"].get_weights()):
			assert np.allclose(clone_weights, source_weights)


def test_repeated_foreign_niche_wins_refresh_the_existing_clone_instead_of_duplicating_it() -> None:
	"""S8: the same gladiator winning in the same foreign niche again must not fill that niche
	with copies of itself — its existing clone is refreshed when the new trust is higher."""
	with step("Arrange: a Mars-Light gladiator that has already been cloned into niche 1 once."):
		hall_of_fame = _build_hall_of_fame()
		gladiator = _archived_gladiator(hall_of_fame)
		population = [gladiator, {"individual_id": "peer", "trust": 0.0}]
		_update(hall_of_fame, gladiator, population, niche_id=1, env_name="Jupiter-Heavy", trust=5.0)

	with step("Act: it wins in niche 1 again, with higher trust."):
		_update(hall_of_fame, gladiator, population, niche_id=1, env_name="Jupiter-Heavy", trust=7.0)

	with step("Assert: still exactly one clone in niche 1, carrying the higher trust."):
		clones = [record for record in hall_of_fame._records.values() if record["internal_env_id"] == 1]
		assert len(clones) == 1
		assert clones[0]["trust"] == 7.0


def test_same_niche_improvement_never_changes_the_gladiators_env_label() -> None:
	"""S8: refreshing a gladiator's own record keeps the label it was inducted with."""
	with step("Arrange: a Mars-Light gladiator (niche 0)."):
		hall_of_fame = _build_hall_of_fame()
		gladiator = _archived_gladiator(hall_of_fame, trust=1.0)
		population = [gladiator, {"individual_id": "peer", "trust": 0.0}]

	with step("Act: it improves in its own niche while the current environment reports another name."):
		_update(hall_of_fame, gladiator, population, niche_id=0, env_name="Earth-Standard", trust=5.0)

	with step("Assert: trust refreshed, label unchanged."):
		record = hall_of_fame._records["mars-gladiator"]
		assert record["trust"] == 5.0
		assert record["env_name"] == "Mars-Light"
		assert gladiator["env_name"] == "Mars-Light"


def test_newly_inducted_winner_is_labeled_with_the_current_env() -> None:
	"""S8: a first-time induction takes its env label from the environment it won in."""
	with step("Arrange: an empty hall of fame and a non-gladiator winner with no label."):
		hall_of_fame = _build_hall_of_fame()
		winner = {"individual_id": "mutant", "model": _brain(2, 2), "trust": 5.0, "is_gladiator": False}
		population = [winner, {"individual_id": "peer", "trust": 0.0}]

	with step("Act: it wins while playing Jupiter-Heavy, resolved to niche 1."):
		_update(hall_of_fame, winner, population, niche_id=1, env_name="Jupiter-Heavy")

	with step("Assert: archived and live member both labeled Jupiter-Heavy, in niche 1."):
		assert hall_of_fame._records["mutant"]["env_name"] == "Jupiter-Heavy"
		assert (winner["is_gladiator"], winner["internal_env_id"], winner["env_name"]) == (True, 1, "Jupiter-Heavy")
