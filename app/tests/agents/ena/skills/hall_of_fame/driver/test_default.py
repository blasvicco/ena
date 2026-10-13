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
from agents.ena.skills.hall_of_fame.driver.olympic import Instance as OlympicInstance

# pylint: disable=protected-access,duplicate-code


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


def test_records_in_niche_lists_only_that_niches_gladiators() -> None:
	"""The Olympics compete within one niche: the archive lists the records it holds for it."""
	with step("Arrange: records in niches 0, 0 and 1."):
		hall_of_fame = _build_hall_of_fame()
		for individual_id, niche_id in (("a", 0), ("b", 0), ("c", 1)):
			hall_of_fame._records[individual_id] = {"individual_id": individual_id, "internal_env_id": niche_id, "trust": 0.0}

	with step("Act/Assert: niche 0 holds a and b; an unknown niche holds nothing."):
		assert sorted(record["individual_id"] for record in hall_of_fame.records_in_niche(0)) == ["a", "b"]
		assert not hall_of_fame.records_in_niche(7)


def test_public_evict_removes_the_record_and_its_live_copys_gladiator_tags() -> None:
	"""The Olympics evict through the same path as the archive's own replacement, so no zombie gladiator is left."""
	with step("Arrange: one record and its live copy."):
		hall_of_fame = _build_hall_of_fame()
		hall_of_fame._records["a"] = {"individual_id": "a", "internal_env_id": 0, "trust": 0.0}
		population = [{"individual_id": "a", "is_gladiator": True, "internal_env_id": 0, "env_name": "Earth"}]

	with step("Act: evict it."):
		hall_of_fame.evict("a", population)

	with step("Assert: gone from the archive; the live copy is no longer a gladiator."):
		assert "a" not in hall_of_fame._records
		assert population[0]["is_gladiator"] is False


def test_inject_gladiators_can_be_limited_to_admitted_records() -> None:
	"""With the Olympics on, test time re-introduces only gladiators that competed; the flag travels with the copy."""
	with step("Arrange: a confirmed and a provisional record, a population of four."):
		hall_of_fame = _build_hall_of_fame(max_gladiator_ratio=1.0)
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))
		for individual_id, confirmed in (("confirmed", True), ("provisional", False)):
			hall_of_fame._records[individual_id] = {
				"individual_id": individual_id,
				"model": _brain(2, 2),
				"trust": 1.0,
				"internal_env_id": 0,
				"is_gladiator": True,
				"olympic_confirmed": confirmed,
			}
		population = [{"individual_id": f"p{index}", "model": _brain(2, 2), "is_gladiator": False} for index in range(4)]

	with step("Act: inject only the admitted ones."):
		hall_of_fame.inject_gladiators(population, 4, obs_normalizer, admit=lambda record: record["olympic_confirmed"])

	with step("Assert: the confirmed gladiator is in the population, flagged; the provisional one is not."):
		ids = [individual["individual_id"] for individual in population]
		assert "confirmed" in ids and "provisional" not in ids
		assert population[ids.index("confirmed")]["olympic_confirmed"] is True


class _NicheZero:  # pylint: disable=too-few-public-methods
	"""Recognizer stub: every winner belongs to niche 0."""

	current_niche_id = 0

	def match_niche(self, _memory, _winner, _hall_of_fame, population=None):  # pylint: disable=unused-argument
		"""Resolve to niche 0."""
		return 0


def _confirmed_archive(count=1, max_per_niche=5):
	"""An archive protecting confirmed gladiators, holding `count` confirmed records in niche 0 (with live copies)."""
	hall_of_fame = OlympicInstance(max_per_niche, 0.5, _brain, action_dim=2, state_dim=2, qualifier_seats=0)
	population = []
	for index in range(count):
		model = _brain(2, 2)
		model.set_weights([np.full_like(weights, 1.0 + index) for weights in model.get_weights()])
		individual_id = f"confirmed-{index}"
		hall_of_fame._records[individual_id] = {
			"individual_id": individual_id,
			"model": model,
			"trust": 0.5,
			"internal_env_id": 0,
			"env_name": "Earth",
			"is_gladiator": True,
			"olympic_confirmed": True,
			"last_mu": np.zeros(2),
			"last_var": np.ones(2),
		}
		population.append(
			{"individual_id": individual_id, "model": model, "trust": 0.5, "is_gladiator": True, "internal_env_id": 0, "olympic_confirmed": True}
		)
	return hall_of_fame, population


def _twin_of(record, individual_id="winner"):
	"""A live winner genetically identical to an archived record."""
	model = _brain(2, 2)
	model.set_weights(record["model"].get_weights())
	return {"individual_id": individual_id, "model": model, "is_gladiator": False, "olympic_confirmed": True}


def test_a_lucky_twin_never_replaces_a_confirmed_gladiator_but_joins_as_provisional() -> None:
	"""Only an Olympiad removes a confirmed gladiator: a similar winner with higher one-episode trust joins beside it, provisional."""
	with step("Arrange: one confirmed gladiator and a genetic twin winning with higher trust."):
		hall_of_fame, population = _confirmed_archive()
		winner = _twin_of(hall_of_fame._records["confirmed-0"])
		population.append(winner)
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: the genetic-diversity step."):
		handled = hall_of_fame._Instance__genetic_diversity(winner, 0.9, _NicheZero(), obs_normalizer, population, "Earth")

	with step("Assert: the confirmed record stays, untouched; the winner is archived, and its live copy is a provisional gladiator."):
		assert handled
		assert hall_of_fame._records["confirmed-0"]["trust"] == 0.5
		assert "winner" in hall_of_fame._records and not hall_of_fame._records["winner"].get("olympic_confirmed", False)
		assert winner["is_gladiator"] is True and winner["olympic_confirmed"] is False


def test_a_lucky_twin_is_dropped_when_the_niche_is_full() -> None:
	"""No free slot: the winner waits for a later chance instead of displacing a confirmed gladiator."""
	with step("Arrange: a full niche (1 slot) holding one confirmed gladiator, and its twin winning."):
		hall_of_fame, population = _confirmed_archive(max_per_niche=1)
		winner = _twin_of(hall_of_fame._records["confirmed-0"])
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: the genetic-diversity step."):
		hall_of_fame._Instance__genetic_diversity(winner, 0.9, _NicheZero(), obs_normalizer, population, "Earth")

	with step("Assert: the archive is unchanged."):
		assert list(hall_of_fame._records) == ["confirmed-0"]


def test_eco_niche_protection_evicts_only_provisional_gladiators() -> None:
	"""A full niche replaces its weakest provisional gladiator; confirmed ones are never the weakest candidate."""
	with step("Arrange: a full niche of two: a confirmed gladiator (trust 0.5) and a provisional one (0.8); a winner at 0.9."):
		hall_of_fame, population = _confirmed_archive(max_per_niche=2)
		hall_of_fame._records["provisional"] = {"individual_id": "provisional", "model": _brain(2, 2), "trust": 0.8, "internal_env_id": 0}
		winner = {"individual_id": "winner", "model": _brain(2, 2), "is_gladiator": False}
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: eco-niche protection."):
		hall_of_fame._Instance__eco_niche_protection(winner, 0.9, _NicheZero(), obs_normalizer, population, "Earth")

	with step("Assert: the provisional one was replaced, though the confirmed one has lower trust."):
		assert sorted(hall_of_fame._records) == ["confirmed-0", "winner"]


def test_eco_niche_protection_drops_the_winner_when_every_gladiator_is_confirmed() -> None:
	"""Nothing provisional to replace: the archive is unchanged."""
	with step("Arrange: a full niche of one confirmed gladiator."):
		hall_of_fame, population = _confirmed_archive(max_per_niche=1)
		winner = {"individual_id": "winner", "model": _brain(2, 2), "is_gladiator": False}
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: eco-niche protection with a far better one-episode trust."):
		hall_of_fame._Instance__eco_niche_protection(winner, 5.0, _NicheZero(), obs_normalizer, population, "Earth")

	with step("Assert: unchanged."):
		assert list(hall_of_fame._records) == ["confirmed-0"]


def test_a_confirmed_gladiator_winning_again_keeps_its_olympic_weights_and_trust() -> None:
	"""Its trust changes only at Olympiads; a one-episode win neither re-scores nor overwrites it."""
	with step("Arrange: a confirmed gladiator whose live copy wins with higher trust and other weights."):
		hall_of_fame, population = _confirmed_archive()
		live = population[0]
		live["model"] = _brain(2, 2)
		before = [weights.copy() for weights in hall_of_fame._records["confirmed-0"]["model"].get_weights()]
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: the continual-improvement step."):
		handled = hall_of_fame._Instance__continual_improvement(live, 0.9, _NicheZero(), obs_normalizer, "Earth")

	with step("Assert: handled; the record's trust and weights unchanged."):
		record = hall_of_fame._records["confirmed-0"]
		assert handled and record["trust"] == 0.5
		assert all(np.array_equal(a, b) for a, b in zip(before, record["model"].get_weights()))


def test_without_protection_a_twin_still_replaces_the_gladiator() -> None:
	"""The default (Olympics off) keeps the archive's original replacement, which the characterization test pins."""
	with step("Arrange: the same twin scenario without protection."):
		hall_of_fame = Instance(5, 0.5, _brain, action_dim=2, state_dim=2)
		protected, population = _confirmed_archive()
		hall_of_fame._records = protected._records
		winner = _twin_of(hall_of_fame._records["confirmed-0"])
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: the genetic-diversity step."):
		hall_of_fame._Instance__genetic_diversity(winner, 0.9, _NicheZero(), obs_normalizer, population, "Earth")

	with step("Assert: replaced."):
		assert list(hall_of_fame._records) == ["winner"]


def test_qualifier_seats_let_new_winners_in_beside_a_full_niche_of_confirmed_gladiators() -> None:
	"""A niche full of confirmed gladiators still admits provisional entrants into its qualifier seats; beyond them, the
	weakest provisional entrant is replaced, never a confirmed one."""
	with step("Arrange: one confirmed seat (full) and two qualifier seats."):
		hall_of_fame, population = _confirmed_archive(max_per_niche=1)
		hall_of_fame.qualifier_seats = 2
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))

	with step("Act: three new winners (trust 0.6, 0.7, 0.8) try to enter."):
		for index, trust in enumerate((0.6, 0.7, 0.8)):
			winner = {"individual_id": f"new-{index}", "model": _brain(2, 2), "trust": trust, "is_gladiator": False}
			hall_of_fame.update(
				winner, trust, population + [winner], SimpleNamespace(plasticity_history=[1.0]), _NicheZero(), obs_normalizer, "Earth"
			)

	with step("Assert: the confirmed one stays; the two seats hold the two best entrants."):
		assert sorted(hall_of_fame._records) == ["confirmed-0", "new-1", "new-2"]


def test_without_protection_qualifier_seats_do_not_change_the_capacity() -> None:
	"""Olympics off: the archive's capacity stays max_per_niche (CartPole's behavior)."""
	with step("Arrange: an unprotected archive with qualifier seats set, one slot, one record."):
		hall_of_fame = Instance(1, 0.5, _brain, action_dim=2, state_dim=2, qualifier_seats=2)
		hall_of_fame._records["old"] = {"individual_id": "old", "model": _brain(2, 2), "trust": 0.1, "internal_env_id": 0}
		obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))
		winner = {"individual_id": "new", "model": _brain(2, 2), "trust": 0.9, "is_gladiator": False}

	with step("Act: a better winner arrives."):
		hall_of_fame.update(winner, 0.9, [winner], SimpleNamespace(plasticity_history=[1.0]), _NicheZero(), obs_normalizer, "Earth")

	with step("Assert: it replaced the old record instead of taking an extra seat."):
		assert list(hall_of_fame._records) == ["new"]


def test_record_returns_one_archived_gladiator_or_none() -> None:
	"""Look-up by id (the Olympics re-sync a live copy from its record)."""
	with step("Arrange/Act/Assert: a known and an unknown id."):
		hall_of_fame, _population = _confirmed_archive()
		assert hall_of_fame.record("confirmed-0")["trust"] == 0.5
		assert hall_of_fame.record("nobody") is None
