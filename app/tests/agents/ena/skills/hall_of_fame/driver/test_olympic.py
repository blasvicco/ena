"""Tests for the olympic Hall of Fame driver: the default archive plus periodic Olympiads of its gladiators."""

# Libs imports
import numpy as np
import pytest
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents.ena.skills.hall_of_fame.driver.default import Instance as Default
from agents.ena.skills.hall_of_fame.driver.olympic import Instance as Olympic
from agents.ena.skills.hall_of_fame.main import HallOfFame

# pylint: disable=protected-access


def _brain(action_dim, state_dim):
	"""A minimal one-layer model."""
	return Sequential([Input(shape=(state_dim,)), Dense(action_dim)])


def _olympic(**kwargs):
	settings = {"max_per_niche": 3, "max_gladiator_ratio": 0.5, "brain": _brain, "action_dim": 2, "state_dim": 2, "olympics_interval": 1_000}
	return Olympic(**{**settings, **kwargs})


def _archive(hall_of_fame, ids, niche_id=0):
	"""One record per id in a niche, each with a live copy in the returned population."""
	population = []
	for individual_id in ids:
		hall_of_fame._records[individual_id] = {"individual_id": individual_id, "model": _brain(2, 2), "trust": 1.0, "internal_env_id": niche_id}
		population.append({"individual_id": individual_id, "is_gladiator": True, "internal_env_id": niche_id, "trust": 1.0})
	return population


def test_the_olympic_driver_is_selected_by_name_and_extends_the_default_archive() -> None:
	"""hall_of_fame_driver='olympic': the default archive, plus Olympics, protection of confirmed gladiators and qualifier seats."""
	with step("Act: the facade with the olympic driver."):
		hall_of_fame = HallOfFame("olympic", max_per_niche=3, max_gladiator_ratio=0.5, brain=_brain, action_dim=2, state_dim=2, qualifier_seats=2)

	with step("Assert: a default archive that holds Olympics and protects confirmed gladiators."):
		assert isinstance(hall_of_fame._instance, Default)
		assert hall_of_fame.holds_olympics and hall_of_fame._instance.protect_confirmed and hall_of_fame._instance.qualifier_seats == 2


@pytest.mark.parametrize("driver", ["default", "none"])
def test_archives_without_olympics_never_hold_one_and_admit_every_gladiator(driver) -> None:
	"""The default and none drivers: never due, nothing held, every gladiator usable at test (ENA as before the Olympics)."""
	with step(f"Act: the {driver} driver."):
		hall_of_fame = HallOfFame(driver, max_per_niche=3, max_gladiator_ratio=0.5, brain=_brain, action_dim=2, state_dim=2)

	with step("Assert: no Olympics."):
		assert not hall_of_fame.holds_olympics
		assert not hall_of_fame.olympics_due(10**9)
		assert hall_of_fame.admits({"is_gladiator": True})
		assert not hall_of_fame.hold_olympics(0, play=None, trust_of=None, eviction_bar=0.0, population=[], total_env_steps=0)


def test_an_olympiad_is_due_once_per_interval_of_environment_steps() -> None:
	"""Every olympics_interval steps, counted from the last one held (or found empty)."""
	with step("Arrange: an interval of 1,000 steps; an empty niche."):
		hall_of_fame = _olympic()

	with step("Act/Assert: due at 1,000; an empty Olympiad restarts the interval."):
		assert not hall_of_fame.olympics_due(999) and hall_of_fame.olympics_due(1_000)
		hall_of_fame.hold_olympics(0, play=None, trust_of=float, eviction_bar=0.0, population=[], total_env_steps=1_000)
		assert not hall_of_fame.olympics_due(1_999) and hall_of_fame.olympics_due(2_000)


def test_an_olympiad_evicts_below_the_bar_relegates_beyond_capacity_and_confirms_the_rest() -> None:
	"""Competitors play the shared seeds through ENA's callback; below the bar they are evicted (never the champion), beyond
	max_per_niche relegated; survivors take their Olympic trust and mean return, on the record and the live copy."""
	with step("Arrange: capacity 3, five gladiators with scripted returns, a bar of 100, trust = return / 100."):
		hall_of_fame = _olympic(olympics_episodes=2)
		population = _archive(hall_of_fame, ["a", "b", "c", "d", "e"])
		returns = {"a": [500, 500], "b": [400, 400], "c": [300, 300], "d": [200, 200], "e": [50, 50]}
		seeds_seen = []

		def play(records, seeds):
			seeds_seen.append(list(seeds))
			return [(returns[record["individual_id"]], [10, 10]) for record in records]

	with step("Act: hold the Olympiad."):
		rows = hall_of_fame.hold_olympics(
			0, play=play, trust_of=lambda value: value / 100, eviction_bar=1.0, population=population, total_env_steps=5
		)

	with step("Assert: e evicted (bar), d relegated (capacity 3); a, b, c confirmed with their Olympic trust and return."):
		assert sorted(record["individual_id"] for record in hall_of_fame.records_in_niche(0)) == ["a", "b", "c"]
		assert {row["individual_id"]: row["evicted"] for row in rows} == {"a": False, "b": False, "c": False, "d": True, "e": True}
		assert hall_of_fame.record("a")["olympic_confirmed"] and hall_of_fame.record("a")["trust"] == 5.0
		assert hall_of_fame.record("a")["olympic_return"] == 500.0 and population[0]["olympic_return"] == 500.0
		assert population[3]["is_gladiator"] is False and population[4]["is_gladiator"] is False
		assert len(seeds_seen) == 1 and len(seeds_seen[0]) == 2


def test_the_champion_survives_even_below_the_bar() -> None:
	"""A niche never empties: with every competitor below the bar, the best stays."""
	with step("Arrange/Act: two gladiators below a bar of 10."):
		hall_of_fame = _olympic()
		population = _archive(hall_of_fame, ["x", "y"])
		hall_of_fame.hold_olympics(
			0,
			play=lambda records, seeds: [([3.0], [1]) if r["individual_id"] == "x" else ([1.0], [1]) for r in records],
			trust_of=float,
			eviction_bar=10.0,
			population=population,
			total_env_steps=1,
		)

	with step("Assert: x stays."):
		assert [record["individual_id"] for record in hall_of_fame.records_in_niche(0)] == ["x"]


def test_only_confirmed_gladiators_are_admitted_at_test() -> None:
	"""Provisional status: a gladiator plays at test only once it has competed in and survived an Olympiad."""
	with step("Act/Assert: confirmed admitted, provisional not."):
		hall_of_fame = _olympic()
		assert hall_of_fame.admits({"olympic_confirmed": True}) and not hall_of_fame.admits({"is_gladiator": True})


def test_the_seeds_come_from_the_agents_seeded_generator() -> None:
	"""Reproducible: the same NumPy seed gives the same Olympiad seeds."""
	with step("Act: two Olympiads after the same np.random.seed."):
		drawn = []
		for _ in range(2):
			hall_of_fame = _olympic(olympics_episodes=3)
			population = _archive(hall_of_fame, ["g"])
			np.random.seed(5)
			hall_of_fame.hold_olympics(
				0,
				play=lambda records, seeds: drawn.append(list(seeds)) or [([1.0] * 3, [1] * 3) for _ in records],
				trust_of=float,
				eviction_bar=0.0,
				population=population,
				total_env_steps=1,
			)

	with step("Assert: identical seeds."):
		assert drawn[0] == drawn[1]


def test_without_competitors_nothing_is_played() -> None:
	"""An empty niche: no callback, no rows."""
	with step("Act/Assert: hold on an empty niche."):
		hall_of_fame = _olympic()
		assert not hall_of_fame.hold_olympics(7, play=None, trust_of=float, eviction_bar=0.0, population=[], total_env_steps=1)


def _induct(hall_of_fame, winner_mu):
	"""Induct a winner whose weights are aligned to winner_mu while the live normalizer has drifted to 5.0."""
	winner = {"individual_id": "w", "model": _brain(2, 2), "trust": 0.9, "last_mu": np.full(2, winner_mu), "last_var": np.full(2, 2.0)}
	normalizer = type("Normalizer", (), {"mean": np.full(2, 5.0), "var": np.full(2, 9.0)})()
	recognizer = type("Recognizer", (), {"current_niche_id": 0, "match_niche": lambda self, *args, **kwargs: 0})()
	hall_of_fame.update(winner, 0.9, [winner], type("Memory", (), {"plasticity_history": [1.0]})(), recognizer, normalizer, "Earth")
	return winner


def test_the_olympic_archive_stores_the_statistics_a_gladiators_weights_are_aligned_to() -> None:
	"""A record keeps the winner's own normalization (last_mu/last_var, set when its weights were last projected), not the
	normalizer's current statistics; the live copy keeps its own too. Otherwise the stored weights and statistics disagree."""
	with step("Act: induct a winner aligned to mean 1.0 while the normalizer reads 5.0."):
		hall_of_fame = _olympic()
		winner = _induct(hall_of_fame, winner_mu=1.0)

	with step("Assert: the record and the live copy carry 1.0 and variance 2.0."):
		assert np.allclose(hall_of_fame.record("w")["last_mu"], 1.0) and np.allclose(hall_of_fame.record("w")["last_var"], 2.0)
		assert np.allclose(winner["last_mu"], 1.0)


def test_the_default_archive_keeps_its_original_stamping() -> None:
	"""CartPole's behavior: the default archive stamps the normalizer's current statistics, as every reported run did."""
	with step("Act: the same induction with the default driver."):
		hall_of_fame = Default(3, 0.5, _brain, 2, 2)
		winner = _induct(hall_of_fame, winner_mu=1.0)

	with step("Assert: the normalizer's 5.0 on the record and the live copy."):
		assert np.allclose(hall_of_fame.record("w")["last_mu"], 5.0) and np.allclose(winner["last_mu"], 5.0)


def test_the_olympic_archive_marks_its_live_gladiators_as_provisional() -> None:
	"""As in CartPole, the live winner becomes a gladiator of its niche and keeps playing during training; the olympic
	driver marks it provisional until its first Olympiad (an Olympiad evicts mediocre ones, so they cannot anchor
	the search)."""
	with step("Act: induct a winner into the olympic archive."):
		hall_of_fame = _olympic()
		winner = _induct(hall_of_fame, winner_mu=1.0)

	with step("Assert: archived, and the live individual is a provisional gladiator of niche 0."):
		assert hall_of_fame.record("w") is not None and hall_of_fame.record("w")["internal_env_id"] == 0
		assert winner["is_gladiator"] is True and winner["internal_env_id"] == 0
		assert winner["olympic_confirmed"] is False


def test_the_default_archive_still_marks_its_live_gladiators() -> None:
	"""CartPole's behavior: the default archive tags the live winner as a gladiator (protected in the population)."""
	with step("Act/Assert: induct with the default driver."):
		winner = _induct(Default(3, 0.5, _brain, 2, 2), winner_mu=1.0)
		assert winner["is_gladiator"] is True and winner["internal_env_id"] == 0


def test_reinjected_olympic_records_are_gladiators_at_test() -> None:
	"""An archived record is a gladiator by definition: its re-injected copy must be one, or test time has none to play
	(the record itself carries no gladiator tag)."""
	with step("Arrange: a confirmed record, and a plain population."):
		hall_of_fame = _olympic()
		_induct(hall_of_fame, winner_mu=1.0)
		hall_of_fame.record("w")["olympic_confirmed"] = True
		population = [{"individual_id": f"p{index}", "model": _brain(2, 2), "is_gladiator": False} for index in range(4)]
		normalizer = type("Normalizer", (), {"mean": np.zeros(2), "var": np.ones(2)})()

	with step("Act: re-inject the admitted records."):
		hall_of_fame.inject_gladiators(population, 4, normalizer, admit=hall_of_fame.admits)

	with step("Assert: the copy is a gladiator of its niche."):
		copies = [individual for individual in population if individual["individual_id"] == "w"]
		assert copies and copies[0]["is_gladiator"] is True and copies[0]["internal_env_id"] == 0
