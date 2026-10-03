"""Tests for the pluggable EnvironmentRecognizer strategy facade and its strategies."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.skills.env_detection.main import EnvironmentRecognizer

BASELINE = np.array([1.0, 0.0, 0.0, 0.0])
SHIFTED = np.array([0.0, 1.0, 0.0, 0.0])  # cosine similarity 0.0 vs BASELINE — well below threshold


def _script_detection_vectors(recognizer, vectors):
	"""Make compute_physical_vector return each vector in sequence, one per call.

	Patches the underlying driver instance, not the facade — detect_change() is
	implemented on the driver and calls its *own* compute_physical_vector internally
	(deliberately not compute_full_vector — see detect_change()'s docstring for why).
	"""
	iterator = iter(vectors)
	recognizer._instance.compute_physical_vector = lambda memory: next(iterator)  # noqa: ARG005  # pylint: disable=protected-access
	_script_fingerprint_ready(recognizer, True)


def _script_fingerprint_ready(recognizer, ready):
	"""Make the driver report its fingerprint as fully accumulated (or not)."""
	recognizer._instance._fingerprint_ready = lambda memory: ready  # noqa: ARG005  # pylint: disable=protected-access


# ---------------------------------------------------------------
# S5.1: facade
# ---------------------------------------------------------------


def test_facade_constructs_the_live_strategy_by_default() -> None:
	"""S5.1: 'live' is a valid, working strategy name — the default, behavior-preserving one."""
	with step("Act: build a recognizer with the 'live' strategy."):
		recognizer = EnvironmentRecognizer("live", state_dim=4)

	with step("Assert: it behaves like a recognizer with no known niches yet."):
		assert recognizer.known_niche_ids == set()
		assert recognizer.current_niche_id == -1


def test_facade_rejects_an_unknown_strategy_name() -> None:
	"""S5.1: an unrecognized strategy name must fail fast, not silently misconfigure."""
	with step("Act/Assert: constructing with a bogus strategy name raises ValueError."):
		with pytest.raises(ValueError):
			EnvironmentRecognizer("bogus", state_dim=4)


# ---------------------------------------------------------------
# S5.2: ConfirmationWindowRecognizer
# ---------------------------------------------------------------


def _make_confirmation_window_recognizer(confirmation_window=3):
	return EnvironmentRecognizer("confirmation_window", state_dim=4, confirmation_window=confirmation_window)


def test_confirmation_window_requires_consecutive_confirmations_before_firing() -> None:
	"""S5.2: a single noisy low-similarity reading must not fire detect_change on its own."""
	with step("Arrange: establish a baseline, then script fewer than the window's worth of shifts."):
		recognizer = _make_confirmation_window_recognizer(confirmation_window=3)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False  # establishes the baseline
		_script_detection_vectors(recognizer, [SHIFTED, SHIFTED])

	with step("Act/Assert: fewer than the confirmation window must not fire."):
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False

	with step("Act/Assert: the Nth consecutive low-similarity reading fires."):
		_script_detection_vectors(recognizer, [SHIFTED])
		assert recognizer.detect_change(memory=None) is True


def test_confirmation_window_resets_confirmation_streak_on_recovery() -> None:
	"""S5.2: a single high-similarity reading between low ones must reset the confirmation
	streak — this is what makes single-episode noise harmless instead of eventually
	accumulating to a fire."""
	with step("Arrange: establish baseline, then some low readings (not yet enough to fire)."):
		recognizer = _make_confirmation_window_recognizer(confirmation_window=3)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False
		_script_detection_vectors(recognizer, [SHIFTED, SHIFTED])
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False

	with step("Act: one high-similarity (baseline-like) reading interrupts the streak."):
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Assert: it takes a full new window of low readings to fire, not just one more."):
		_script_detection_vectors(recognizer, [SHIFTED, SHIFTED])
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False
		_script_detection_vectors(recognizer, [SHIFTED])
		assert recognizer.detect_change(memory=None) is True


def test_confirmation_window_still_fires_for_a_genuine_sustained_shift() -> None:
	"""S5.2: a real, sustained shift is still caught within one confirmation window, not lost forever."""
	with step("Arrange: establish a baseline."):
		recognizer = _make_confirmation_window_recognizer(confirmation_window=3)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Act: feed a sustained run of low-similarity readings."):
		results = []
		_script_detection_vectors(recognizer, [SHIFTED, SHIFTED, SHIFTED])
		for _ in range(3):
			results.append(recognizer.detect_change(memory=None))

	with step("Assert: it fires exactly once, on the 3rd (window-sized) consecutive reading."):
		assert results == [False, False, True]


def test_detect_change_ignores_performance_derivative_and_plasticity_oscillation() -> None:
	"""Stage 7: detect_change() must not fire from performance_derivative/recent_plasticity
	swinging on their own, with the underlying physics never changing. These are no longer
	part of the compared vector — this is specifically what breaks the feedback loop where a
	firing forces plasticity to 1.0 (clear_training_history's warmup path in the linear_ramp curve),
	which itself would otherwise look like a 'shift' on the very next reading, confirmed via
	a real run's live CSV: firings alternated in lockstep with plasticity swinging between
	its warmup value and its steady-state one, even while scores stayed at the maximum."""

	class _FakeMemory:  # pylint: disable=too-few-public-methods
		"""Memory stub with a perfectly constant physical signature but swinging distress terms."""

		class _Normalizer:  # pylint: disable=too-few-public-methods
			def __init__(self):
				self.mean = np.zeros(4)
				self.var = np.ones(4)
				self.count = self.max_count = 20

		def __init__(self, performance_derivative, recent_plasticity):
			self.short_term_normalizer = self._Normalizer()
			self.transition_normalizer = self._Normalizer()
			self.performance_derivative = performance_derivative
			self.recent_plasticity = recent_plasticity

	with step("Arrange: a live recognizer with an established baseline."):
		recognizer = EnvironmentRecognizer("live", state_dim=4)
		assert recognizer.detect_change(_FakeMemory(0.0, 1.0)) is False

	with step(
		"Act: swing performance_derivative/recent_plasticity wildly across many readings, "
		"with the underlying physics (normalizer stats) never changing."
	):
		results = [
			recognizer.detect_change(_FakeMemory(performance_derivative=perf_deriv, recent_plasticity=plasticity))
			for perf_deriv, plasticity in [(0.0, 0.1), (50.0, 1.0), (-50.0, 0.1), (0.0, 1.0), (0.0, 0.1)]
		]

	with step("Assert: detect_change() never fires — it's blind to distress-signal oscillation."):
		assert results == [False, False, False, False, False]


def test_live_strategy_has_no_confirmation_window_regression() -> None:
	"""S5.2: the default 'live' strategy must remain hysteresis-free — a single low-similarity
	reading still fires immediately, unlike 'confirmation_window'."""
	with step("Arrange: a 'live' recognizer with an established baseline."):
		recognizer = EnvironmentRecognizer("live", state_dim=4)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Act: a single shifted reading."):
		_script_detection_vectors(recognizer, [SHIFTED])
		result = recognizer.detect_change(memory=None)

	with step("Assert: it fires immediately, with no confirmation window."):
		assert result is True


def test_live_and_confirmation_window_reset_is_a_harmless_noop() -> None:
	"""'live'/'confirmation_window' source their physical vector from AgentMemory, already
	flushed by memory.reset_recognizer_statistics() — reset() must not error, and must not
	need memory to be reset (it has nothing of its own to flush)."""
	with step("Act/Assert: reset() runs without error for both strategies."):
		EnvironmentRecognizer("live", state_dim=4).reset()
		EnvironmentRecognizer("confirmation_window", state_dim=4).reset()


# ---------------------------------------------------------------
# S8.1: a gladiator never drags its home niche's prototype to foreign physics
# ---------------------------------------------------------------

HOME = np.array([1.0, 0.0, 0.0, 0.0])
OTHER = np.array([0.0, 1.0, 0.0, 0.0])
UNKNOWN = np.array([0.0, 0.0, 1.0, 0.0])


def _recognizer_with_prototypes(prototypes):
	"""A live recognizer seeded with the given {niche_id: prototype} map."""
	recognizer = EnvironmentRecognizer("live", state_dim=1, ema_alpha=0.5)
	instance = recognizer._instance  # pylint: disable=protected-access
	instance._physical_prototypes = {niche_id: vector.copy() for niche_id, vector in prototypes.items()}  # pylint: disable=protected-access
	instance._next_niche_id = max(prototypes) + 1  # pylint: disable=protected-access
	return recognizer


def _match_gladiator(recognizer, physical_vector, home_niche_id=0):
	"""Run match_niche() for a home_niche_id gladiator under the given physics."""
	recognizer._instance.compute_physical_vector = lambda memory: physical_vector  # pylint: disable=protected-access
	gladiator = {"is_gladiator": True, "internal_env_id": home_niche_id}
	hall_of_fame = type("_FakeHallOfFame", (), {"_records": {}})()
	return recognizer.match_niche(memory=None, current_individual=gladiator, hall_of_fame=hall_of_fame)


def test_gladiator_under_its_home_physics_resolves_home_and_refines_the_home_prototype() -> None:
	"""S8.1: a gladiator succeeding under physics that match its home prototype keeps its home
	niche, and the confirmed observation is blended into that prototype."""
	with step("Arrange: a recognizer whose home prototype (niche 0) is slightly off from the observed physics."):
		recognizer = _recognizer_with_prototypes({0: np.array([1.0, 0.1, 0.0, 0.0]), 1: OTHER})

	with step("Act: match a niche-0 gladiator under home physics."):
		niche_id = _match_gladiator(recognizer, HOME)

	with step("Assert: resolves to niche 0, whose prototype moved toward the observation."):
		assert niche_id == 0
		assert recognizer.current_niche_id == 0
		assert np.allclose(recognizer._instance._physical_prototypes[0], [1.0, 0.05, 0.0, 0.0])  # pylint: disable=protected-access


def test_gladiator_under_another_known_niches_physics_resolves_there_and_leaves_home_untouched() -> None:
	"""S8.1: a gladiator also winning under a different, known niche's physics must resolve to
	that niche (so the Hall of Fame can clone it there) — never overwrite its home prototype."""
	with step("Arrange: a recognizer with a home prototype (0) and a second, distinct known niche (1)."):
		recognizer = _recognizer_with_prototypes({0: HOME, 1: OTHER})

	with step("Act: match a niche-0 gladiator under niche 1's physics."):
		niche_id = _match_gladiator(recognizer, OTHER)

	with step("Assert: resolves to niche 1, and niche 0's prototype is exactly as it was."):
		assert niche_id == 1
		assert recognizer.current_niche_id == 1
		assert np.allclose(recognizer._instance._physical_prototypes[0], HOME)  # pylint: disable=protected-access


def test_gladiator_under_unknown_physics_provisions_a_new_niche_and_leaves_home_untouched() -> None:
	"""S8.1: physics matching no known prototype provisions a new niche for the clone, rather
	than being blended into the gladiator's home prototype."""
	with step("Arrange: a recognizer with two known niches."):
		recognizer = _recognizer_with_prototypes({0: HOME, 1: OTHER})

	with step("Act: match a niche-0 gladiator under physics resembling neither."):
		niche_id = _match_gladiator(recognizer, UNKNOWN)

	with step("Assert: a brand-new niche (2) holds the new physics; niche 0 is untouched."):
		assert niche_id == 2
		assert np.allclose(recognizer._instance._physical_prototypes[2], UNKNOWN)  # pylint: disable=protected-access
		assert np.allclose(recognizer._instance._physical_prototypes[0], HOME)  # pylint: disable=protected-access


# ---------------------------------------------------------------
# S11(a): no baseline from a not-yet-refilled fingerprint
# ---------------------------------------------------------------

IMMATURE = np.array([0.0, 0.0, 1.0, 0.0])  # read from a just-reset, near-empty normalizer
MATURE = np.array([0.0, 0.0, 0.0, 1.0])  # same physics once fully re-accumulated — dissimilar to IMMATURE


def _fired_live_recognizer():
	"""A live recognizer that has just fired on a genuine BASELINE -> SHIFTED switch."""
	recognizer = EnvironmentRecognizer("live", state_dim=4)
	_script_detection_vectors(recognizer, [BASELINE, SHIFTED])
	assert recognizer.detect_change(memory=None) is False
	assert recognizer.detect_change(memory=None) is True
	return recognizer


def test_detect_change_does_not_refire_as_a_just_reset_fingerprint_refills() -> None:
	"""S11(a): after a firing, the fingerprint is read from just-reset normalizers. Its drift
	while they refill is not a new shift — previously the baseline was set to that immature
	reading, so the refill itself looked like a shift, re-fired, reset again, and looped."""
	with step("Arrange: a recognizer that just fired on a real switch."):
		recognizer = _fired_live_recognizer()

	with step("Act: immature readings while the fingerprint refills, then the mature ones."):
		recognizer._instance.compute_physical_vector = lambda memory: IMMATURE  # pylint: disable=protected-access
		_script_fingerprint_ready(recognizer, False)
		refilling = [recognizer.detect_change(memory=None) for _ in range(3)]
		_script_detection_vectors(recognizer, [MATURE] * 10)
		refilled = [recognizer.detect_change(memory=None) for _ in range(10)]

	with step("Assert: nothing fires — neither during the refill nor once it's complete."):
		assert refilling == [False] * 3
		assert refilled == [False] * 10


def test_detect_change_still_catches_a_real_switch_after_the_fingerprint_refills() -> None:
	"""S11(a): waiting for the refill must not blind detection to the next genuine switch."""
	with step("Arrange: a recognizer that fired, waited out a refill, and re-established its baseline."):
		recognizer = _fired_live_recognizer()
		_script_fingerprint_ready(recognizer, False)
		assert recognizer.detect_change(memory=None) is False
		_script_detection_vectors(recognizer, [MATURE] * 7)
		assert [recognizer.detect_change(memory=None) for _ in range(7)] == [False] * 7  # baseline + cooldown + settle

	with step("Act: a genuine switch away from the re-established baseline."):
		_script_detection_vectors(recognizer, [BASELINE])
		result = recognizer.detect_change(memory=None)

	with step("Assert: it fires."):
		assert result is True
