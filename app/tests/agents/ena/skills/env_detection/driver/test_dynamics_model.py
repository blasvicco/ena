"""Tests for the dynamics_model environment-detection driver (per-niche dynamics models + PM-CuSum)."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.skills.env_detection.main import EnvironmentRecognizer

STATE_DIM = 4


def _physics(seed):
	"""A random linear 'physics': next_state = state + state @ gain + onehot(action) @ push + noise."""
	rng = np.random.default_rng(seed)
	return 0.3 * rng.normal(size=(STATE_DIM, STATE_DIM)), 0.3 * rng.normal(size=(2, STATE_DIM))


PHYSICS_A = _physics(0)
PHYSICS_B = _physics(1)
PHYSICS_C = _physics(2)


def _play(recognizer, physics, steps, seed=0):
	"""Feed `steps` real transitions under `physics` through observe()."""
	gain, push = physics
	rng = np.random.default_rng(seed)
	for _ in range(steps):
		state = rng.normal(size=STATE_DIM)
		action = int(rng.integers(2))
		next_state = state + state @ gain + push[action] + 0.01 * rng.normal(size=STATE_DIM)
		recognizer.observe(state, action, next_state, memory=None)


def _episode(recognizer, physics, steps=200, seed=0):
	"""One episode: start_episode() then `steps` transitions."""
	recognizer.start_episode()
	_play(recognizer, physics, steps, seed)


def _recognizer():
	return EnvironmentRecognizer("dynamics_model", state_dim=STATE_DIM, action_dim=2, is_continuous=False)


def _induct(recognizer, individual=None):
	"""Late Identity Binding for a winner (a mutant unless given)."""
	return recognizer.match_niche(memory=None, current_individual=individual or {}, hall_of_fame=None)


def _two_niche_recognizer():
	"""A recognizer that has learned PHYSICS_A as niche 0 and PHYSICS_B as niche 1."""
	recognizer = _recognizer()
	_episode(recognizer, PHYSICS_A, seed=1)
	assert _induct(recognizer) == 0
	recognizer.reset()
	_episode(recognizer, PHYSICS_B, seed=2)
	assert _induct(recognizer) == 1
	return recognizer


def test_induction_provisions_one_niche_per_physics_and_recognizes_recurrence() -> None:
	"""A, then B, then A again: two niches, and the return to A resolves to A's niche."""
	with step("Arrange: a recognizer that has learned A (niche 0) and B (niche 1)."):
		recognizer = _two_niche_recognizer()

	with step("Act: A recurs."):
		recognizer.reset()
		_episode(recognizer, PHYSICS_A, seed=3)
		niche_id = _induct(recognizer)

	with step("Assert: back in niche 0, no third niche."):
		assert niche_id == 0
		assert recognizer.known_niche_ids == {0, 1}


def test_repeated_success_under_the_same_physics_never_fragments_niches() -> None:
	"""Many inductions under unchanged physics all resolve to the one niche."""
	with step("Arrange: niche 0 learned on A."):
		recognizer = _recognizer()
		_episode(recognizer, PHYSICS_A, seed=1)
		_induct(recognizer)

	with step("Act: 20 more episodes and inductions under A."):
		niche_ids = []
		for seed in range(20):
			_episode(recognizer, PHYSICS_A, seed=10 + seed)
			niche_ids.append(_induct(recognizer))

	with step("Assert: always niche 0."):
		assert set(niche_ids) == {0}
		assert recognizer.known_niche_ids == {0}


def test_detect_change_stays_quiet_under_unchanged_physics_and_fires_after_a_switch() -> None:
	"""PM-CuSum on the current niche's residuals: silent for many episodes, then fires once physics switch."""
	with step("Arrange: niche 0 learned on A."):
		recognizer = _recognizer()
		_episode(recognizer, PHYSICS_A, seed=1)
		_induct(recognizer)

	with step("Act: 30 more A episodes, then one B episode long enough to verify the alarm."):
		quiet = []
		for seed in range(30):
			_episode(recognizer, PHYSICS_A, seed=100 + seed)
			quiet.append(recognizer.detect_change(memory=None))
		_episode(recognizer, PHYSICS_B, steps=60, seed=200)
		fired = recognizer.detect_change(memory=None)

	with step("Assert: no false alarm, then a confirmed detection at the end of the B episode."):
		assert not any(quiet)
		assert fired is True


def test_reset_forgets_the_current_niche_and_start_episode_re_identifies_it() -> None:
	"""After a shift reset, the active niche is unknown until enough evidence re-identifies it."""
	with step("Arrange: a two-niche recognizer, just reset."):
		recognizer = _two_niche_recognizer()
		recognizer.reset()
		assert recognizer.current_niche_id == -1

	with step("Act: one episode under A, then the next episode starts."):
		_episode(recognizer, PHYSICS_A, seed=4)
		recognizer.start_episode()

	with step("Assert: re-identified as niche 0."):
		assert recognizer.current_niche_id == 0


@pytest.mark.parametrize("physics, expected", [(PHYSICS_A, 0), (PHYSICS_B, 1), (PHYSICS_C, -1)], ids=["A", "B", "novel"])
def test_match_resolves_known_physics_and_rejects_novel_ones(physics, expected) -> None:
	"""match() returns the niche whose model explains the transitions, or -1 for unseen physics."""
	with step("Arrange: a two-niche recognizer, reset, then one episode under the given physics."):
		recognizer = _two_niche_recognizer()
		recognizer.reset()
		_episode(recognizer, physics, seed=5)

	with step("Act: match."):
		niche_id, similarity = recognizer.match(memory=None)

	with step("Assert: the expected niche; similarity near 1 only when recognized."):
		assert niche_id == expected
		assert (similarity > 0.5) == (expected != -1)


def test_log_likelihoods_rank_the_true_niche_first() -> None:
	"""This episode's per-niche log-likelihoods put the physics' own niche on top (the BPR signal)."""
	with step("Arrange: a two-niche recognizer, then an episode under B."):
		recognizer = _two_niche_recognizer()
		_episode(recognizer, PHYSICS_B, seed=6)

	with step("Act: read the per-niche log-likelihoods."):
		scores = recognizer.log_likelihoods(memory=None)

	with step("Assert: both niches scored, niche 1 (B) far ahead."):
		assert set(scores) == {0, 1}
		assert scores[1] > scores[0] + 100.0


def test_gladiator_succeeding_under_foreign_physics_resolves_there_and_leaves_home_untouched() -> None:
	"""S8.1 parity: a niche-0 gladiator winning under B resolves to B's niche; niche 0's model is unchanged."""
	with step("Arrange: a two-niche recognizer and niche 0's current model weights."):
		recognizer = _two_niche_recognizer()
		home_model = recognizer._instance._models[0]  # pylint: disable=protected-access
		home_weights = home_model.weights.copy()
		recognizer.reset()
		_episode(recognizer, PHYSICS_B, seed=7)

	with step("Act: induct the niche-0 gladiator."):
		niche_id = _induct(recognizer, {"is_gladiator": True, "internal_env_id": 0})

	with step("Assert: resolved to niche 1; niche 0's model is exactly as before."):
		assert niche_id == 1
		assert np.allclose(home_model.weights, home_weights)


def test_an_immature_first_niche_keeps_growing_instead_of_being_duplicated() -> None:
	"""A niche provisioned from too few transitions absorbs its regime's later data rather than spawning twins."""
	with step("Arrange: the very first induction happens after only 10 transitions."):
		recognizer = _recognizer()
		_episode(recognizer, PHYSICS_A, steps=10, seed=1)
		assert _induct(recognizer) == 0

	with step("Act: more episodes and inductions under the same physics."):
		niche_ids = []
		for seed in range(5):
			_episode(recognizer, PHYSICS_A, seed=20 + seed)
			niche_ids.append(_induct(recognizer))

	with step("Assert: still one niche, now mature."):
		assert set(niche_ids) == {0}
		assert recognizer.known_niche_ids == {0}
		assert recognizer._instance._models[0].count >= 100  # pylint: disable=protected-access


def _force_one_alarm(recognizer):
	"""Make PM-CuSum raise exactly one alarm on the next residual it sees."""
	instance = recognizer._instance  # pylint: disable=protected-access
	real_update = instance._cusum.update  # pylint: disable=protected-access
	state = {"fired": False}

	def _update(residual):
		if not state["fired"]:
			state["fired"] = True
			return True
		return real_update(residual)

	instance._cusum.update = _update  # pylint: disable=protected-access


def test_an_alarm_the_current_niche_still_explains_is_learned_from_not_declared() -> None:
	"""Detect-then-verify: a CuSum alarm under unchanged physics (a coverage gap in the model) is
	dismissed, and the post-alarm transitions are folded into the niche's model."""
	with step("Arrange: niche 0 learned on A, and an alarm forced on the next step."):
		recognizer = _recognizer()
		_episode(recognizer, PHYSICS_A, seed=1)
		_induct(recognizer)
		model = recognizer._instance._models[0]  # pylint: disable=protected-access
		count_before = model.count
		_force_one_alarm(recognizer)

	with step("Act: an A episode, then the episode-end check."):
		_episode(recognizer, PHYSICS_A, steps=100, seed=2)
		fired = recognizer.detect_change(memory=None)

	with step("Assert: no shift, same niche, and the model learned from the episode."):
		assert fired is False
		assert recognizer.current_niche_id == 0
		assert model.count >= count_before + 100


def test_an_alarm_waits_for_enough_evidence_before_it_is_decided() -> None:
	"""With fewer than min_verify_samples post-alarm transitions, the decision is deferred — not dropped."""
	with step("Arrange: niche 0 learned on A, and an alarm forced on the next step."):
		recognizer = _recognizer()
		_episode(recognizer, PHYSICS_A, seed=1)
		_induct(recognizer)
		_force_one_alarm(recognizer)

	with step("Act: a 10-step B episode (too short to verify), then a longer one."):
		_episode(recognizer, PHYSICS_B, steps=10, seed=3)
		first = recognizer.detect_change(memory=None)
		_episode(recognizer, PHYSICS_B, steps=60, seed=4)
		second = recognizer.detect_change(memory=None)

	with step("Assert: undecided after the short episode, confirmed after the long one."):
		assert first is False
		assert second is True


def test_a_confirmed_shift_to_new_physics_fires_once_and_provisions_their_niche() -> None:
	"""Regression: after a real switch, the post-alarm transitions must seed the new regime. Previously
	reset() discarded them, the next induction re-attached the evidence-less regime to the ruled-out
	niche, whose model alarmed again — a shift declared on every single episode, and no new niche."""
	with step("Arrange: niche 0 learned on A over a few episodes."):
		recognizer = _recognizer()
		for seed in range(3):
			_episode(recognizer, PHYSICS_A, seed=seed)
			_induct(recognizer)

	with step("Act: 15 short B episodes, in ENAgent's per-episode order: shift check (+ reset), then induction."):
		fired, niche_ids = [], []
		for seed in range(15):
			_episode(recognizer, PHYSICS_B, steps=30, seed=50 + seed)
			fired.append(recognizer.detect_change(memory=None))
			if fired[-1]:
				recognizer.reset()
			niche_ids.append(_induct(recognizer))

	with step("Assert: one shift, one new niche, and every induction after the shift lands in it."):
		assert sum(fired) == 1
		assert recognizer.known_niche_ids == {0, 1}
		assert set(niche_ids[fired.index(True) :]) == {1}


def test_a_niche_recognizes_its_own_nonlinear_physics_on_states_it_saw_little_of() -> None:
	"""Identification uses the same tolerance-widened variance as detection: a linear model of nonlinear
	physics, fitted on a narrow state range, must still recognize those physics over a wider range
	(e.g. a test-time policy visiting other states) — and still reject genuinely different physics."""

	def _nonlinear(recognizer, gain, scale, seed):
		rng = np.random.default_rng(seed)
		recognizer.start_episode()
		for _ in range(300):
			state = scale * rng.normal(size=STATE_DIM)
			action = int(rng.integers(2))
			next_state = state + 0.2 * np.sin(state) @ gain + 0.01 * rng.normal(size=STATE_DIM)
			recognizer.observe(state, action, next_state, memory=None)

	with step("Arrange: niche 0 fitted on narrow states (std 0.3) under gain A."):
		recognizer = _recognizer()
		_nonlinear(recognizer, PHYSICS_A[0], scale=0.3, seed=1)
		_induct(recognizer)

	with step("Act: match wide states (std 1.0) under gain A, then under gain B."):
		recognizer.reset()
		_nonlinear(recognizer, PHYSICS_A[0], scale=1.0, seed=2)
		same, _ = recognizer.match(memory=None)
		recognizer.reset()
		_nonlinear(recognizer, PHYSICS_B[0], scale=1.0, seed=3)
		other, _ = recognizer.match(memory=None)

	with step("Assert: A recognized as niche 0; B rejected as novel."):
		assert same == 0
		assert other == -1
