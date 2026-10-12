"""Tests for chirp_signals.py — the fixed excitations a probing detector plays."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.tools.chirp_signals import EXCITATIONS, excitation, to_discrete


@pytest.mark.parametrize("name", sorted(EXCITATIONS))
def test_every_excitation_has_the_requested_shape_and_stays_in_the_action_range(name) -> None:
	"""(steps, action_dim), within [-1, 1], deterministic."""
	with step(f"Act: a 64-step, 6-joint {name}."):
		first, second = excitation(name, 64, 6), excitation(name, 64, 6)

	with step("Assert: shape, range, reproducible."):
		assert first.shape == (64, 6)
		assert np.all(np.abs(first) <= 1.0)
		assert np.array_equal(first, second)


def test_an_impulse_is_a_short_kick_then_nothing() -> None:
	"""impulse_steps of amplitude, then zero."""
	with step("Act/Assert: a 3-step kick of 0.8."):
		signal = excitation("impulse", 16, 2, amplitude=0.8, impulse_steps=3)
		assert np.allclose(signal[:3], 0.8) and np.allclose(signal[3:], 0.0)


def test_a_linear_chirp_speeds_up() -> None:
	"""Its zero crossings get closer together as frequency rises from freq_start to freq_end."""
	with step("Act: a 256-step chirp from 0.01 to 0.2 cycles/step on one joint."):
		signal = excitation("chirp", 256, 1, freq_start=0.01, freq_end=0.2)[:, 0]

	with step("Assert: the gap between crossings shrinks from the first to the second half."):
		crossings = np.flatnonzero(np.diff(np.sign(signal)) != 0)
		gaps = np.diff(crossings)
		assert gaps[: len(gaps) // 2].mean() > 2 * gaps[len(gaps) // 2 :].mean()


def test_a_wavelet_train_plays_the_chosen_wavelet_at_dyadic_scales() -> None:
	"""Segments of length steps/2, steps/4, ... down to min_segment, each the wavelet's waveform; another wavelet gives another signal."""
	with step("Act: 64-step trains of db4 and of the Mexican hat."):
		db4 = excitation("wavelet_train", 64, 1, wavelet="db4", min_segment=8)[:, 0]
		mexh = excitation("wavelet_train", 64, 1, wavelet="mexh", min_segment=8)[:, 0]

	with step("Assert: dyadic segments 32 + 16 + 8 (+ 8 of zero padding); the shape follows the wavelet."):
		assert np.allclose(db4[56:], 0.0) and np.any(db4[:32] != 0) and np.any(db4[48:56] != 0)
		assert not np.allclose(db4, mexh)


def test_per_joint_phases_offset_the_sinusoidal_excitations() -> None:
	"""phase_spread spreads the joints' phases over [0, phase_spread]; 0 drives every joint identically."""
	with step("Act/Assert: spread vs. none."):
		assert not np.allclose(excitation("sine", 32, 3)[:, 0], excitation("sine", 32, 3)[:, 2])
		together = excitation("sine", 32, 3, phase_spread=0.0)
		assert np.allclose(together[:, 0], together[:, 2])


def test_unknown_excitation_is_rejected() -> None:
	"""A typo fails loudly."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError, match="excitation"):
			excitation("whistle", 16, 2)


def test_discrete_action_spaces_follow_the_sign_of_the_first_channel() -> None:
	"""CartPole-like actions: 1 where the excitation is positive, else 0."""
	with step("Act/Assert: a sign pattern to {0, 1}."):
		assert to_discrete(np.array([[0.5, 9.0], [-0.1, 9.0], [0.0, 9.0]])).tolist() == [1, 0, 0]
