"""Tests for HeuristicSelectionDriver's test-time selection cascade."""

# App imports
from helpers import step
from agents.ena.selection_controller import HeuristicSelectionDriver

# pylint: disable=protected-access


# pylint: disable=too-few-public-methods
class _FakeRecognizer:
	"""Recognizer stub with a scriptable match() result: fixed, or a sequence popped per call."""

	def __init__(self, match_result=(-1, -1.0), match_results=None):
		self._fixed_result = match_result
		self._match_results = list(match_results) if match_results is not None else None
		self.match_calls = 0

	def match(self, _memory):
		"""Return the next scripted match() result, counting calls."""
		self.match_calls += 1
		if self._match_results:
			return self._match_results.pop(0)
		return self._fixed_result


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


def test_evaluating_ignores_recognizer_match() -> None:
	"""S2: mid-evaluation, the queued candidate is used and match() is never even consulted —
	the EVALUATING queue must run to completion to gather a fresh score per niche."""
	with step("Arrange: a driver mid-evaluation with a queued (niche, gladiator) pair, and a "
			  "recognizer that would confidently (but wrongly, for this step) match niche 4."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		driver._state = driver.EVALUATING
		driver._eval_queue = [(2, 2)]
		recognizer = _FakeRecognizer(match_result=(4, 0.95))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: the queued candidate (niche 2) is chosen; match() was never called."):
		assert chosen == 2
		assert recognizer.match_calls == 0


def test_locked_trusts_a_confident_recognizer_match_over_the_locked_niche() -> None:
	"""S2: once LOCKED, a confident match() to a different, already-known niche is trusted —
	it can resolve to a niche switch without waiting for the next trigger_evaluation() cycle."""
	with step("Arrange: locked onto niche 2, but the recognizer confidently matches niche 4."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		driver._state = driver.LOCKED
		driver._locked_niche = 2
		recognizer = _FakeRecognizer(match_result=(4, 0.95))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: niche 4's gladiator is chosen, overriding the locked niche."):
		assert chosen == 4


def test_locked_falls_back_to_the_locked_niche_when_no_match() -> None:
	"""S2: once LOCKED, an unresolved match() (-1) must not disturb the current lock."""
	with step("Arrange: locked onto niche 2, and the recognizer has no confident match."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		driver._state = driver.LOCKED
		driver._locked_niche = 2
		recognizer = _FakeRecognizer(match_result=(-1, 0.5))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: the locked niche's gladiator is still chosen."):
		assert chosen == 2


def test_no_gladiators_falls_back_to_random_non_gladiator() -> None:
	"""With no gladiators in the population at all, selection must fall back to a random mutant."""
	with step("Arrange: a locked driver with no locked niche and a population with no gladiators."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=())
		recognizer = _FakeRecognizer(match_result=(-1, 0.5))

	with step("Act: select the test-time individual."):
		chosen = driver._select_test(population, recognizer, memory=None)

	with step("Assert: a non-gladiator is chosen."):
		assert not population[chosen]["is_gladiator"]


def test_evaluation_prefers_a_recognizer_confirmed_niche_over_a_higher_raw_score() -> None:
	"""S2b: a niche whose physics identity is independently confirmed by recognizer.match()
	during its own evaluation episode must win locking over a higher-scoring but unconfirmed
	niche — guards against a 'generalist' gladiator winning the raw-score contest under
	someone else's physics, which is what made the post-Stage-2 test run lock onto one wrong
	gladiator across every true environment."""
	with step("Arrange: a driver evaluating niches 2 and 4; the recognizer confirms niche 2's "
			  "own episode (match points back at 2) but not niche 4's (match points elsewhere)."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		driver._state = driver.EVALUATING
		driver._eval_queue = [(2, 2), (4, 4)]
		recognizer = _FakeRecognizer(match_results=[(2, 0.95), (7, 0.92)])

	with step("Act: run both evaluation episodes — niche 2 scores low, niche 4 scores much higher."):
		driver._select_test(population, recognizer, memory=None)
		driver.record_episode_result(50.0, memory=None, recognizer=recognizer)
		driver._select_test(population, recognizer, memory=None)
		driver.record_episode_result(500.0, memory=None, recognizer=recognizer)

	with step("Assert: the driver locks onto the confirmed niche (2), not the raw-score winner (4)."):
		assert driver._state == driver.LOCKED
		assert driver._locked_niche == 2


def test_evaluation_falls_back_to_raw_score_when_nothing_is_confirmed() -> None:
	"""S2b: if the recognizer confirms none of the evaluated niches, fall back to today's
	raw-score argmax rather than refusing to lock at all."""
	with step("Arrange: a driver evaluating niches 2 and 4; the recognizer confirms neither."):
		driver = HeuristicSelectionDriver()
		population = _population(gladiator_niches=(2, 4))
		driver._state = driver.EVALUATING
		driver._eval_queue = [(2, 2), (4, 4)]
		recognizer = _FakeRecognizer(match_results=[(7, 0.5), (9, 0.5)])

	with step("Act: run both evaluation episodes — niche 4 scores higher."):
		driver._select_test(population, recognizer, memory=None)
		driver.record_episode_result(50.0, memory=None, recognizer=recognizer)
		driver._select_test(population, recognizer, memory=None)
		driver.record_episode_result(500.0, memory=None, recognizer=recognizer)

	with step("Assert: the driver falls back to the raw-score winner (4)."):
		assert driver._state == driver.LOCKED
		assert driver._locked_niche == 4
