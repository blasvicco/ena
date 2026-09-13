"""Tests for the pluggable EnvironmentRecognizer strategy facade and its strategies."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.recognizer import EnvironmentRecognizer

BASELINE = np.array([1.0, 0.0, 0.0, 0.0])
SHIFTED = np.array([0.0, 1.0, 0.0, 0.0])  # cosine similarity 0.0 vs BASELINE — well below threshold


def _script_detection_vectors(recognizer, vectors):
	"""Make compute_physical_vector return each vector in sequence, one per call.

	Patches the underlying strategy instance, not the facade — detect_change() is
	implemented on the strategy and calls its *own* compute_physical_vector internally
	(deliberately not compute_full_vector — see detect_change()'s docstring for why).
	"""
	iterator = iter(vectors)
	recognizer.instance.compute_physical_vector = lambda memory: next(iterator)  # noqa: ARG005


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
	firing forces plasticity to 1.0 (clear_training_history's warmup path in __fuzzylogic),
	which itself would otherwise look like a 'shift' on the very next reading, confirmed via
	a real run's live CSV: firings alternated in lockstep with plasticity swinging between
	its warmup value and its steady-state one, even while scores stayed at the maximum."""

	class _FakeMemory:  # pylint: disable=too-few-public-methods
		"""Memory stub with a perfectly constant physical signature but swinging distress terms."""

		class _Normalizer:  # pylint: disable=too-few-public-methods
			def __init__(self):
				self.mean = np.zeros(4)
				self.var = np.ones(4)

		def __init__(self, performance_derivative, recent_plasticity):
			self.short_term_normalizer = self._Normalizer()
			self.transition_normalizer = self._Normalizer()
			self.performance_derivative = performance_derivative
			self.recent_plasticity = recent_plasticity

	with step("Arrange: a live recognizer with an established baseline."):
		recognizer = EnvironmentRecognizer("live", state_dim=4)
		assert recognizer.detect_change(_FakeMemory(0.0, 1.0)) is False

	with step("Act: swing performance_derivative/recent_plasticity wildly across many readings, "
			  "with the underlying physics (normalizer stats) never changing."):
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


# ---------------------------------------------------------------
# S5.3: NeutralProbeRecognizer
# ---------------------------------------------------------------


# pylint: disable=too-few-public-methods
class _ActionSpace:
	"""Minimal discrete action space stub."""

	def sample(self):
		"""Return the only available action."""
		return 0


class _NeverEndsEnv:
	"""Env stub with a fixed physics-dependent transition and a discrete action space,
	so probe() can run without a real gymnasium environment."""

	def __init__(self, state_dim, drift):
		self.unwrapped = self
		self._state_dim = state_dim
		self._drift = drift
		self._state = np.zeros(state_dim, dtype=np.float32)
		self.action_space = _ActionSpace()

	def reset(self):
		"""Reset to a zeroed observation."""
		self._state = np.zeros(self._state_dim, dtype=np.float32)
		return self._state, {}

	def step(self, _action):
		"""Advance the observation by a fixed, physics-like drift each step."""
		self._state = self._state + self._drift
		return self._state, 0.0, False, False, {}

	def close(self):
		"""No-op close."""


def _make_env_factory(drift):
	"""Build a make_env(**kwargs)-shaped factory for a fixed-drift _NeverEndsEnv."""

	def _factory(**_kwargs):
		return _NeverEndsEnv(4, drift=drift)

	return _factory


def test_neutral_probe_physical_vector_is_unaffected_by_the_live_rollout() -> None:
	"""S5.3: the probe-derived physical vector must not change no matter what the live
	rollout's short_term_normalizer/transition_normalizer currently look like — it's the
	whole point of decoupling the fingerprint from whichever policy is acting."""

	class _FakeMemory:
		"""Memory stub exposing wildly different live-rollout normalizer stats."""

		class _Normalizer:
			def __init__(self, mean, var):
				self.mean = mean
				self.var = var

		def __init__(self):
			self.short_term_normalizer = self._Normalizer(np.full(4, 999.0), np.full(4, 999.0))
			self.transition_normalizer = self._Normalizer(np.full(4, -999.0), np.full(4, 999.0))

	with step("Arrange: a neutral_probe recognizer, probed once against a fixed-physics env."):
		recognizer = EnvironmentRecognizer("neutral_probe", state_dim=4, probe_steps=5)
		make_env = _make_env_factory(drift=0.1)
		physics_env = _NeverEndsEnv(4, drift=0.1)
		recognizer.probe(make_env, physics_env, physics_keys=[])
		vector_before = recognizer.compute_physical_vector(memory=_FakeMemory())

	with step("Act: recompute the physical vector against a memory with wildly different live stats."):
		vector_after = recognizer.compute_physical_vector(memory=_FakeMemory())

	with step("Assert: the probe-derived vector is identical, ignoring the live rollout entirely."):
		assert np.allclose(vector_before, vector_after)


def test_neutral_probe_physical_vector_reflects_the_probe_envs_actual_physics() -> None:
	"""S5.3: two probe envs with different underlying dynamics must produce measurably
	different physical vectors — the probe is a real, working physics fingerprint."""
	with step("Arrange: two neutral_probe recognizers, probed against envs with different drift."):
		recognizer_a = EnvironmentRecognizer("neutral_probe", state_dim=4, probe_steps=5)
		recognizer_b = EnvironmentRecognizer("neutral_probe", state_dim=4, probe_steps=5)
		make_env_a = _make_env_factory(drift=0.1)
		make_env_b = _make_env_factory(drift=5.0)

	with step("Act: probe each and compute their physical vectors."):
		recognizer_a.probe(make_env_a, _NeverEndsEnv(4, drift=0.1), physics_keys=[])
		recognizer_b.probe(make_env_b, _NeverEndsEnv(4, drift=5.0), physics_keys=[])
		vector_a = recognizer_a.compute_physical_vector(memory=None)
		vector_b = recognizer_b.compute_physical_vector(memory=None)

	with step("Assert: the two probes produce measurably different vectors."):
		assert not np.allclose(vector_a, vector_b)


def test_neutral_probe_reset_flushes_the_probe_normalizers() -> None:
	"""A detected environment shift must flush the probe's own running normalizers, not just
	AgentMemory's — they live on this instance, so memory.reset_recognizer_statistics() alone
	never reached them, leaving every environment switch blending in the *previous*
	environment's physics for up to probe_steps*20 samples (~20 episodes) via EMA decay,
	right during the window niche identification most needs a clean reading."""
	with step("Arrange: a probed recognizer, and a never-probed one as the identity reference."):
		recognizer = EnvironmentRecognizer("neutral_probe", state_dim=4, probe_steps=5)
		fresh_recognizer = EnvironmentRecognizer("neutral_probe", state_dim=4, probe_steps=5)
		make_env = _make_env_factory(drift=5.0)
		recognizer.probe(make_env, _NeverEndsEnv(4, drift=5.0), physics_keys=[])
		assert not np.allclose(
			recognizer.compute_physical_vector(memory=None),
			fresh_recognizer.compute_physical_vector(memory=None),
		)  # sanity: probing actually changed something

	with step("Act: reset the probed recognizer."):
		recognizer.reset()

	with step("Assert: its physical vector matches a never-probed recognizer's identity state."):
		assert np.allclose(
			recognizer.compute_physical_vector(memory=None),
			fresh_recognizer.compute_physical_vector(memory=None),
		)


def test_live_and_confirmation_window_reset_is_a_harmless_noop() -> None:
	"""'live'/'confirmation_window' source their physical vector from AgentMemory, already
	flushed by memory.reset_recognizer_statistics() — reset() must not error, and must not
	need memory to be reset (it has nothing of its own to flush)."""
	with step("Act/Assert: reset() runs without error for both strategies."):
		EnvironmentRecognizer("live", state_dim=4).reset()
		EnvironmentRecognizer("confirmation_window", state_dim=4).reset()


# ---------------------------------------------------------------
# Composability: confirmation_window + neutral_probe together
# ---------------------------------------------------------------


def test_neutral_probe_defaults_to_immediate_fire_like_live() -> None:
	"""Composability: neutral_probe's own default (no explicit confirmation_window) must
	still fire immediately, same as the 'live' strategy's default — the two concerns
	(data source vs. confirmation window) are independent, and neither strategy secretly
	changes the other's default."""
	with step("Arrange: a neutral_probe recognizer with an established baseline, no confirmation_window given."):
		recognizer = EnvironmentRecognizer("neutral_probe", state_dim=4)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False

	with step("Act: a single shifted reading."):
		_script_detection_vectors(recognizer, [SHIFTED])
		result = recognizer.detect_change(memory=None)

	with step("Assert: it fires immediately, with no confirmation window."):
		assert result is True


def test_neutral_probe_composes_with_an_explicit_confirmation_window() -> None:
	"""Composability: neutral_probe's policy-independent fingerprint AND a confirmation
	window can be used together — this is the combination that fixes both niche
	fragmentation (neutral_probe) and spurious detect_change() firing (confirmation
	window) at the same time, instead of forcing a choice between the two strategies."""
	with step("Arrange: a neutral_probe recognizer with confirmation_window=3, baseline established."):
		recognizer = EnvironmentRecognizer("neutral_probe", state_dim=4, confirmation_window=3)
		_script_detection_vectors(recognizer, [BASELINE])
		assert recognizer.detect_change(memory=None) is False
		_script_detection_vectors(recognizer, [SHIFTED, SHIFTED])

	with step("Act/Assert: fewer than the confirmation window must not fire."):
		assert recognizer.detect_change(memory=None) is False
		assert recognizer.detect_change(memory=None) is False

	with step("Act/Assert: the 3rd consecutive low-similarity reading fires."):
		_script_detection_vectors(recognizer, [SHIFTED])
		assert recognizer.detect_change(memory=None) is True
