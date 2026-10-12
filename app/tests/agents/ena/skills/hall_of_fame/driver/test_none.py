"""Tests for the none Hall of Fame driver — the ablation-study disable switch."""

# General imports
from types import SimpleNamespace

# App imports
from helpers import step
from agents.ena.skills.hall_of_fame.driver.none import Instance


def test_none_driver_is_always_falsy() -> None:
	"""No gladiator is ever inducted, so the driver must always be falsy."""
	with step("Act/Assert: a freshly built none driver is falsy."):
		assert not Instance()


def test_none_driver_update_is_a_no_op() -> None:
	"""update() must never induct anything, regardless of how good the winner looks."""
	with step("Arrange: a none driver and a winner that would trivially pass any real gate."):
		hall_of_fame = Instance()
		winner = {"individual_id": "winner", "is_gladiator": False}
		memory = SimpleNamespace(plasticity_history=[1.0])

	with step("Act: call update() with an overwhelmingly good trust value."):
		hall_of_fame.update(winner, 1000.0, [{"trust": 0.0}], memory, recognizer=None, obs_normalizer=None)

	with step("Assert: the winner was never tagged as a gladiator, and the driver stays falsy."):
		assert winner["is_gladiator"] is False
		assert not hall_of_fame


def test_none_driver_inject_gladiators_is_a_no_op() -> None:
	"""inject_gladiators() must leave the population untouched — there's nothing to inject."""
	with step("Arrange: a none driver and a population slot with a known starting state."):
		hall_of_fame = Instance()
		population = [{"individual_id": "solo", "is_gladiator": False}]

	with step("Act: call inject_gladiators()."):
		hall_of_fame.inject_gladiators(population, pop_size=1, obs_normalizer=None)

	with step("Assert: the population slot is unchanged."):
		assert population[0] == {"individual_id": "solo", "is_gladiator": False}


def test_none_driver_has_no_records_to_compete_or_evict() -> None:
	"""The Olympics see an empty archive under the none driver."""
	with step("Act/Assert: no records in any niche; evict is a no-op."):
		hall_of_fame = Instance()
		assert not hall_of_fame.records_in_niche(0)
		hall_of_fame.evict("anyone", [])
