"""Tests for the Bayesian Policy Reuse selection driver."""

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.skills.selection.driver.bpr import Instance

RETURN_SCALE = 475.0  # CartPole-v1's solved threshold: utility 1 = solved


# pylint: disable=too-few-public-methods
class _FakeRecognizer:
	"""Recognizer stub with scriptable per-niche episode log-likelihoods."""

	def __init__(self, log_likelihoods=None):
		self.scores = log_likelihoods or {}

	def log_likelihoods(self, _memory):
		"""Return the scripted per-niche log-likelihoods."""
		return dict(self.scores)


def _population(gladiator_niches=(0, 1, 2)):
	"""Population whose slot n is niche n's (only) gladiator, plus two non-gladiators."""
	population = [{"trust": 1.0, "is_gladiator": True, "internal_env_id": niche_id} for niche_id in gladiator_niches]
	population += [{"trust": 0.0, "is_gladiator": False, "internal_env_id": -1} for _ in range(2)]
	return population


def _driver(**kwargs):
	driver = Instance(exploration_rate=0.0, return_scale=RETURN_SCALE, **kwargs)
	driver.evaluate(_population())
	return driver


def _episode(driver, score, recognizer=None):
	"""Select, then report the episode's score; return the population index that played."""
	played = driver.select_individual(_population(), recognizer, memory=None, learn=False)
	driver.record_episode_result(score, memory=None, recognizer=recognizer)
	return played


def test_evaluate_builds_a_uniform_belief_over_the_gladiator_niches() -> None:
	"""Types are exactly the niches with a gladiator in the population, starting equally likely."""
	with step("Arrange/Act: a driver evaluated on a population with gladiators in niches 0, 1, 2."):
		driver = _driver()

	with step("Assert: uniform belief over {0, 1, 2}."):
		assert driver.belief == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_dynamics_likelihoods_concentrate_the_belief_and_pick_that_niches_gladiator() -> None:
	"""With per-niche dynamics evidence favoring niche 2, one episode is enough to switch to niche 2's gladiator."""
	with step("Arrange: a recognizer whose dynamics models strongly favor niche 2."):
		driver = _driver()
		recognizer = _FakeRecognizer({0: -5000.0, 1: -4000.0, 2: 10.0})

	with step("Act: one episode (score irrelevant here), then the next selection."):
		_episode(driver, score=250.0, recognizer=recognizer)
		next_pick = driver.select_individual(_population(), recognizer, memory=None, learn=False)

	with step("Assert: belief on niche 2, and its gladiator (slot 2) plays next."):
		assert driver.belief[2] > 0.9
		assert next_pick == 2


def test_returns_alone_move_the_belief_away_from_a_failing_specialist() -> None:
	"""Without dynamics likelihoods (e.g."""
	# the none driver), a specialist scoring ~0 in what the belief thought was its own niche shifts the belief elsewhere and another gladiator gets
	# picked.
	with step("Arrange: a driver whose first pick is some niche's own gladiator."):
		driver = _driver()
		first = _episode(driver, score=0.0, recognizer=_FakeRecognizer())

	with step("Act: the next selection."):
		second = driver.select_individual(_population(), _FakeRecognizer(), memory=None, learn=False)

	with step("Assert: the failing niche lost belief mass, and a different gladiator plays."):
		assert driver.belief[first] < 1 / 3
		assert second != first


def test_switching_prior_lets_a_changed_environment_win_the_belief_back() -> None:
	"""After many episodes of certainty in niche 0, evidence for niche 1 must take over within a few episodes."""
	with step("Arrange: 50 episodes of dynamics evidence for niche 0."):
		driver = _driver(switch_rate=0.05)
		for _ in range(50):
			_episode(driver, score=500.0, recognizer=_FakeRecognizer({0: 0.0, 1: -30.0, 2: -30.0}))
		assert driver.belief[0] > 0.9

	with step("Act: the environment switches — evidence now favors niche 1."):
		picks = [_episode(driver, score=500.0, recognizer=_FakeRecognizer({0: -30.0, 1: 0.0, 2: -30.0})) for _ in range(3)]

	with step("Assert: niche 1's gladiator is playing by the third episode."):
		assert driver.belief[1] > 0.9
		assert picks[-1] == 1


def test_training_uses_epsilon_greedy_selection_by_trust() -> None:
	"""learn=True bypasses the belief entirely: highest trust wins (exploration off)."""
	with step("Arrange: a population where slot 3 (a non-gladiator) has the highest trust."):
		driver = _driver()
		population = _population()
		population[3]["trust"] = 9.0

	with step("Act/Assert: the training pick is slot 3."):
		assert driver.select_individual(population, None, memory=None, learn=True) == 3


def test_with_no_gladiators_it_plays_a_non_gladiator() -> None:
	"""No niches known yet: fall back to a random non-gladiator, and don't update anything."""
	with step("Arrange: a driver evaluated on a population without gladiators."):
		driver = Instance(exploration_rate=0.0, return_scale=RETURN_SCALE)
		population = _population(gladiator_niches=())
		driver.evaluate(population)

	with step("Act: select and report."):
		pick = driver.select_individual(population, None, memory=None, learn=False)
		driver.record_episode_result(100.0, memory=None, recognizer=None)

	with step("Assert: a non-gladiator, and an empty belief."):
		assert population[pick]["is_gladiator"] is False
		assert not driver.belief


def test_a_failing_own_niche_gladiator_is_abandoned_even_when_the_niche_is_certain() -> None:
	"""A gladiator failing in its own niche is abandoned even when the niche is certain."""
	# Regression: the dynamics evidence is sure the environment is niche 0, but niche 0's highest-trust
	# gladiator keeps scoring ~0 — another gladiator must get tried (observed: a Mars specialist scoring
	# 16/500 in Mars for a whole test phase, never replaced, while a pessimistic cross-niche prior of 0
	# made every alternative look worse untried).
	with step("Arrange: niche 0 has two gladiators (slot 1 has higher trust), niche 1 one; evidence says niche 0."):
		population = [
			{"individual_id": "mars-a", "trust": 1.0, "is_gladiator": True, "internal_env_id": 0},
			{"individual_id": "mars-b", "trust": 2.0, "is_gladiator": True, "internal_env_id": 0},
			{"individual_id": "jupiter", "trust": 1.0, "is_gladiator": True, "internal_env_id": 1},
		]
		driver = Instance(exploration_rate=0.0, return_scale=RETURN_SCALE)
		driver.evaluate(population)
		recognizer = _FakeRecognizer({0: 0.0, 1: -5000.0})

	with step("Act: slot 1 scores 0 whenever it plays; the others would score 500."):
		picks = []
		for _ in range(5):
			pick = driver.select_individual(population, recognizer, memory=None, learn=False)
			picks.append(pick)
			driver.record_episode_result(0.0 if pick == 1 else 500.0, memory=None, recognizer=recognizer)

	with step("Assert: starts with the higher-trust slot 1, abandons it, and settles on the working niche-0 gladiator."):
		assert picks[0] == 1
		assert picks[-1] == 0


def test_a_gladiator_solving_its_own_niche_strengthens_the_belief_in_that_niche() -> None:
	"""A gladiator solving its own niche (utility 1) strengthens the belief in that niche."""
	# Review #7/#8 regression: utility is return / solved return, so solving the environment (utility 1)
	# matches the own-niche prior and is evidence FOR the gladiator's niche. With the old step-count
	# scale (CartPole 1000), a perfect 475-500 read as utility ~0.5 — the cross-niche prior — and moved
	# the belief away from the right niche.
	with step("Arrange: three niches, uniform belief, no dynamics evidence."):
		driver = _driver()
		population = _population()

	with step("Act: the chosen gladiator scores exactly the solved return."):
		played = driver.select_individual(population, None, memory=None, learn=False)
		driver.record_episode_result(RETURN_SCALE, memory=None, recognizer=None)

	with step("Assert: its home niche gained belief."):
		home = population[played]["internal_env_id"]
		assert driver.belief[home] > 1 / 3


def _olympic_population():
	"""Niche 0 holds a champion (Olympic mean 3,000) with lower trust and a runner-up (2,000) with higher trust."""
	return [
		{"individual_id": "champion", "trust": 0.2, "is_gladiator": True, "internal_env_id": 0, "olympic_return": 3_000.0},
		{"individual_id": "runner-up", "trust": 0.9, "is_gladiator": True, "internal_env_id": 0, "olympic_return": 2_000.0},
		{"individual_id": "other", "trust": 0.0, "is_gladiator": False, "internal_env_id": -1},
	]


def test_a_confirmed_gladiators_prior_is_its_olympic_mean() -> None:
	"""A confirmed gladiator's own-niche prior is its Olympic mean return over the solved return."""
	# With an Olympic result, the own-niche prior utility is that mean return / the solved return, not 1: the champion plays
	# first, though its trust is lower, and a single episode at its level keeps it playing.
	with step("Arrange: HalfCheetah's solved return 4,800; the Olympic population."):
		driver = Instance(exploration_rate=0.0, return_scale=4_800.0)
		population = _olympic_population()
		driver.evaluate(population)

	with step("Act: three test episodes, each scoring the player's Olympic mean."):
		played = []
		for _ in range(3):
			index = driver.select_individual(population, None, memory=None, learn=False)
			played.append(index)
			driver.record_episode_result(population[index]["olympic_return"], memory=None)

	with step("Assert: the champion every time."):
		assert played == [0, 0, 0]


def test_without_an_olympic_result_the_own_niche_prior_stays_solved() -> None:
	"""No olympic_return (the Olympics off, CartPole): the prior is OWN_PRIOR and ties go to trust, as before."""
	with step("Arrange: the same population without Olympic results."):
		driver = Instance(exploration_rate=0.0, return_scale=4_800.0)
		population = [{key: value for key, value in individual.items() if key != "olympic_return"} for individual in _olympic_population()]
		driver.evaluate(population)

	with step("Act/Assert: the higher-trust runner-up plays first."):
		assert driver.select_individual(population, None, memory=None, learn=False) == 1


def test_the_foreign_niche_prior_is_a_share_of_the_gladiators_own() -> None:
	"""cross_prior scales the gladiator's own prior: 0.5 of 1 (no Olympic result) as before, 0.5 of its Olympic utility otherwise."""
	# pylint: disable=protected-access
	with step("Arrange: an Olympic gladiator of niche 0 and a plain one of niche 1."):
		driver = Instance(exploration_rate=0.0, return_scale=4_800.0, cross_prior=0.5)
		population = [
			{"individual_id": "olympian", "trust": 0.0, "is_gladiator": True, "internal_env_id": 0, "olympic_return": 2_400.0},
			{"individual_id": "plain", "trust": 0.0, "is_gladiator": True, "internal_env_id": 1},
		]
		driver.evaluate(population)
		olympian, plain = driver._policies  # pylint: disable=protected-access

	with step("Assert: own and foreign priors."):
		assert driver._mean_utility(0, olympian) == pytest.approx(0.5) and driver._mean_utility(1, olympian) == pytest.approx(
			0.25
		)  # pylint: disable=protected-access
		assert driver._mean_utility(1, plain) == pytest.approx(1.0) and driver._mean_utility(0, plain) == pytest.approx(
			0.5
		)  # pylint: disable=protected-access
