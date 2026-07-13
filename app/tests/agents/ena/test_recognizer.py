"""Tests for EnvironmentRecognizer's shift-detection stability."""

# Libs imports
import numpy as np

# App imports
from helpers import step
from agents.ena.recognizer import EnvironmentRecognizer

BASELINE = np.array([1.0, 0.0, 0.0, 0.0])
SHIFTED = np.array([0.0, 1.0, 0.0, 0.0])  # cosine similarity 0.0 vs BASELINE — well below threshold


def _make_recognizer(confirmation_window=3):
	return EnvironmentRecognizer(state_dim=4, confirmation_window=confirmation_window)


def _script_full_vectors(recognizer, vectors):
	"""Make compute_full_vector return each vector in sequence, one per call."""
	iterator = iter(vectors)
	recognizer.compute_full_vector = lambda memory: next(iterator)  # noqa: ARG005


def _script_physical_vector(recognizer, vector):
	"""Make compute_physical_vector always return the given vector."""
	recognizer.compute_physical_vector = lambda memory: vector  # noqa: ARG005


def test_detect_change_requires_consecutive_confirmations_before_firing():
	"""A single noisy low-similarity reading must not fire detect_change on its own."""
	with step("Arrange: establish a baseline, then script fewer than the window's worth of shifts."):
		recognizer = _make_recognizer(confirmation_window=3)
		_script_full_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False  # establishes the baseline
		_script_full_vectors(recognizer, [SHIFTED, SHIFTED])

	with step("Act/Assert: fewer than the confirmation window must not fire."):
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False

	with step("Act/Assert: the Nth consecutive low-similarity reading fires."):
		_script_full_vectors(recognizer, [SHIFTED])
		assert recognizer.detect_change(memory=None) is True


def test_detect_change_resets_confirmation_streak_on_recovery():
	"""A single high-similarity reading between low ones must reset the confirmation streak —
	this is what makes single-episode noise harmless instead of eventually accumulating to a fire."""
	with step("Arrange: establish baseline, then some low readings (not yet enough to fire)."):
		recognizer = _make_recognizer(confirmation_window=3)
		_script_full_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False
		_script_full_vectors(recognizer, [SHIFTED, SHIFTED])
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False

	with step("Act: one high-similarity (baseline-like) reading interrupts the streak."):
		_script_full_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Assert: it takes a full new window of low readings to fire, not just one more."):
		_script_full_vectors(recognizer, [SHIFTED, SHIFTED])
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False
		_script_full_vectors(recognizer, [SHIFTED])
		assert recognizer.detect_change(memory=None) is True


def test_detect_change_still_fires_for_a_genuine_sustained_shift():
	"""A real, sustained shift is still caught within one confirmation window, not lost forever."""
	with step("Arrange: establish a baseline."):
		recognizer = _make_recognizer(confirmation_window=3)
		_script_full_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Act: feed a sustained run of low-similarity readings."):
		results = []
		_script_full_vectors(recognizer, [SHIFTED, SHIFTED, SHIFTED])
		for _ in range(3):
			results.append(recognizer.detect_change(memory=None))

	with step("Assert: it fires exactly once, on the 3rd (window-sized) consecutive reading."):
		assert results == [False, False, True]


def test_closest_niche_returns_none_when_no_prototypes_exist():
	"""With no known niches yet, closest_niche must not crash — it has nothing to compare to."""
	with step("Arrange: a fresh recognizer with no physical prototypes."):
		recognizer = _make_recognizer()
		_script_physical_vector(recognizer, BASELINE)

	with step("Act: ask for the closest niche."):
		niche_id, similarity = recognizer.closest_niche(memory=None)

	with step("Assert: -1 / -1.0 signal 'nothing to compare to' rather than a false match."):
		assert niche_id == -1
		assert similarity == -1.0


def test_closest_niche_returns_best_match_even_below_threshold():
	"""Unlike match(), closest_niche() returns the nearest prototype regardless of confidence —
	this is what powers best-effort zero-shot transfer when nothing confidently matches."""
	with step("Arrange: a recognizer with one registered prototype, far from the query vector."):
		recognizer = _make_recognizer()
		recognizer._physical_prototypes[7] = BASELINE  # pylint: disable=protected-access
		_script_physical_vector(recognizer, SHIFTED)  # cosine similarity 0.0 — well below threshold

	with step("Act: ask for the closest niche."):
		niche_id, similarity = recognizer.closest_niche(memory=None)

	with step("Assert: it still returns niche 7, unlike match() which would reject it."):
		assert niche_id == 7
		assert similarity < recognizer._similarity_threshold  # pylint: disable=protected-access
		matched_id, _ = recognizer.match(memory=None)
		assert matched_id == -1
