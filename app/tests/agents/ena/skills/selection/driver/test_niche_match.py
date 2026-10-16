"""Tests for the niche-match selection driver."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.skills.selection.driver.niche_match import Instance


def _recognizer(niche_id):
	"""Recognizer stub reporting a fixed current niche."""
	return SimpleNamespace(current_niche_id=niche_id)


def _population():
	"""Niche 0: slots 0 (trust 1) and 1 (trust 2); niche 1: slot 2 (trust 3, the highest overall); slot 3 is not a gladiator."""
	return [
		{"trust": 1.0, "is_gladiator": True, "internal_env_id": 0},
		{"trust": 2.0, "is_gladiator": True, "internal_env_id": 0},
		{"trust": 3.0, "is_gladiator": True, "internal_env_id": 1},
		{"trust": 9.0, "is_gladiator": False, "internal_env_id": -1},
	]


@pytest.mark.parametrize(
	"niche_id, expected",
	[
		(0, 1),  # niche 0's highest-trust gladiator, not the highest overall
		(1, 2),
		(-1, 2),  # no niche recognized: the highest-trust gladiator overall
		(7, 2),  # a niche without gladiators (novel physics): the same fallback
	],
)
def test_plays_the_recognized_niches_highest_trust_gladiator(niche_id, expected) -> None:
	"""The detector's current niche picks the gladiator; without a gladiator there, the highest trust overall plays."""
	with step("Arrange: a driver evaluated on the population."):
		driver = Instance(exploration_rate=0.0)
		population = _population()
		driver.evaluate(population)

	with step("Act: a test-time selection with the detector on the given niche."):
		pick = driver.select_individual(population, _recognizer(niche_id), memory=None, learn=False)

	with step("Assert: the expected slot."):
		assert pick == expected


def test_test_returns_do_not_change_the_pick() -> None:
	"""No belief: a gladiator failing in its own niche keeps playing while the detector stays on that niche."""
	with step("Arrange: the detector on niche 0."):
		driver = Instance(exploration_rate=0.0)
		population = _population()
		driver.evaluate(population)
		recognizer = _recognizer(0)

	with step("Act: five episodes, each scoring 0."):
		picks = []
		for _ in range(5):
			picks.append(driver.select_individual(population, recognizer, memory=None, learn=False))
			driver.record_episode_result(0.0, memory=None, recognizer=recognizer)

	with step("Assert: the same gladiator every time."):
		assert picks == [1] * 5


def test_follows_the_detector_when_the_niche_changes() -> None:
	"""The pick is read from the detector at every episode, not fixed at evaluate()."""
	with step("Arrange: a driver evaluated once."):
		driver = Instance(exploration_rate=0.0)
		population = _population()
		driver.evaluate(population)

	with step("Act: the detector moves from niche 0 to niche 1."):
		picks = [driver.select_individual(population, _recognizer(niche_id), memory=None, learn=False) for niche_id in (0, 1)]

	with step("Assert: each niche's own gladiator."):
		assert picks == [1, 2]
