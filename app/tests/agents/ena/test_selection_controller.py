"""Tests for HeuristicSelectionDriver's test-time selection cascade."""

# App imports
from helpers import step
from agents.ena.selection_controller import HeuristicSelectionDriver

# pylint: disable=protected-access


# pylint: disable=too-few-public-methods
class _FakeRecognizer:
	"""Recognizer stub with scriptable match()/closest_niche() results."""

	def __init__(self, match_result=(-1, -1.0), closest_result=(-1, -1.0)):
		self._match_result = match_result
		self._closest_result = closest_result
		self.match_calls = 0
		self.closest_niche_calls = 0

	def match(self, _memory):
		"""Return the scripted match() result, counting calls."""
		self.match_calls += 1
		return self._match_result

	def closest_niche(self, _memory):
		"""Return the scripted closest_niche() result, counting calls."""
		self.closest_niche_calls += 1
		return self._closest_result


def _population(gladiator_niches=()):
	"""Build a fake population: index niche_idx is a gladiator for each niche in gladiator_niches."""
	population = []
	for niche_idx in range(6):
		is_gladiator = niche_idx in gladiator_niches
		population.append(
			{
				"trust": float(niche_idx),
				"is_gladiator": is_gladiator,
				"internal_env_id": niche_idx if is_gladiator else -1,
			}
		)
	return population


def test_confident_recognizer_match_wins_outright() -> None:
	"""A confident match() result is trusted immediately — no need for the fallback tiers."""
	with step("Arrange: a recognizer that confidently matches niche 2, which has a gladiator."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		recognizer = _FakeRecognizer(match_result=(2, 0.95))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: niche 2's gladiator is chosen; the zero-shot fallback is never consulted."):
		assert chosen == 2
		assert recognizer.closest_niche_calls == 0


def test_zero_shot_transfer_picks_closest_niche_gladiator() -> None:
	"""With no confident match, zero-shot transfer picks the closest known niche's gladiator."""
	with step("Arrange: no confident match, but a closest-niche result pointing at niche 4."):
		driver = HeuristicSelectionDriver(zero_shot_transfer_rate=1.0)
		population = _population(gladiator_niches=(2, 4))
		recognizer = _FakeRecognizer(match_result=(-1, 0.5), closest_result=(4, 0.6))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: niche 4's gladiator is chosen via the zero-shot fallback."):
		assert chosen == 4
		assert recognizer.closest_niche_calls == 1


def test_zero_shot_transfer_skipped_falls_back_to_random_mutant() -> None:
	"""When the zero-shot roll misses, selection falls back to a random non-gladiator."""
	with step("Arrange: zero-shot transfer rate of 0.0 guarantees the roll always misses."):
		driver = HeuristicSelectionDriver(zero_shot_transfer_rate=0.0)
		population = _population(gladiator_niches=(2, 4))
		recognizer = _FakeRecognizer(match_result=(-1, 0.5))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: a random non-gladiator is chosen; the fallback niche was never even queried."):
		assert not population[chosen]["is_gladiator"]
		assert recognizer.closest_niche_calls == 0


def test_zero_shot_transfer_falls_back_to_random_gladiator_when_closest_niche_is_empty() -> None:
	"""If the closest niche currently has no gladiator (e.g. evicted), fall back to any gladiator
	rather than giving up on zero-shot transfer entirely."""
	with step("Arrange: the closest niche (99) has no gladiator, but others exist."):
		driver = HeuristicSelectionDriver(zero_shot_transfer_rate=1.0)
		population = _population(gladiator_niches=(2, 4))
		recognizer = _FakeRecognizer(match_result=(-1, 0.5), closest_result=(99, 0.6))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: some gladiator is still chosen, not a random mutant."):
		assert population[chosen]["is_gladiator"]


def test_no_gladiators_falls_back_to_random_non_gladiator() -> None:
	"""With no gladiators in the population at all, selection must fall back to a random mutant,
	regardless of the zero-shot transfer rate."""
	with step("Arrange: a population with no gladiators, and a high zero-shot transfer rate."):
		driver = HeuristicSelectionDriver(zero_shot_transfer_rate=1.0)
		population = _population(gladiator_niches=())
		recognizer = _FakeRecognizer(match_result=(-1, 0.5))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: a random non-gladiator is chosen."):
		assert not population[chosen]["is_gladiator"]
