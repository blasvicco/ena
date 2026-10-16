"""Fixed excitation signals a probing environment detector plays: the same actions in every environment."""

# Method: system identification with a known input signal (impulse, step, sine/chirp sweeps, pseudo-random
# binary sequences). Reference: Ljung, "System Identification: Theory for the User" (2nd ed., 1999).
# Wavelet train: the chosen wavelet's own waveform at dyadic scales (a multiresolution excitation), waveforms
# from PyWavelets (Lee et al., "PyWavelets: A Python package for wavelet analysis", JOSS 2019).

# General imports
import math

# Libs imports
import numpy as np
import pywt

WAVEFORM_LEVEL = 8  # PyWavelets' wavefun refinement: 2^8 points per support, resampled to each segment


def _phases(action_dim: int, phase_spread: float) -> np.ndarray:
	"""Per-joint phase offsets spread over [0, phase_spread]."""
	return np.linspace(0.0, phase_spread, action_dim)


def _impulse(steps, action_dim, amplitude=1.0, impulse_steps=5, **_params):
	signal = np.zeros((steps, action_dim))
	signal[:impulse_steps] = amplitude
	return signal


def _step(steps, action_dim, amplitude=0.5, **_params):
	return np.full((steps, action_dim), float(amplitude))


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _sine(steps, action_dim, amplitude=1.0, frequency=0.1, phase_spread=math.pi, **_params):
	time = np.arange(steps)[:, None]
	return amplitude * np.sin(2.0 * math.pi * frequency * time + _phases(action_dim, phase_spread))


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _chirp(steps, action_dim, amplitude=1.0, freq_start=0.01, freq_end=0.25, phase_spread=math.pi, **_params):
	"""Linear sweep from freq_start to freq_end cycles per step."""
	time = np.arange(steps)
	phase = 2.0 * math.pi * (freq_start * time + 0.5 * (freq_end - freq_start) / steps * time**2)
	return amplitude * np.sin(phase[:, None] + _phases(action_dim, phase_spread))


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _exponential_chirp(steps, action_dim, amplitude=1.0, freq_start=0.01, freq_end=0.25, phase_spread=math.pi, **_params):
	"""Geometric sweep from freq_start to freq_end: equal time per octave."""
	time, ratio = np.arange(steps), freq_end / freq_start
	phase = 2.0 * math.pi * freq_start * steps / math.log(ratio) * (ratio ** (time / steps) - 1.0)
	return amplitude * np.sin(phase[:, None] + _phases(action_dim, phase_spread))


def _prbs(steps, action_dim, amplitude=1.0, seed=0, **_params):
	"""A fixed pseudo-random binary sequence of +/- amplitude (its own seed, never the global RNG)."""
	return amplitude * np.where(np.random.default_rng(seed).integers(0, 2, size=(steps, action_dim)) == 1, 1.0, -1.0)


def _waveform(wavelet: str) -> np.ndarray:
	"""The wavelet function psi of a discrete or continuous PyWavelets wavelet, real part, peak 1."""
	if wavelet in pywt.wavelist(kind="continuous"):
		psi = pywt.ContinuousWavelet(wavelet).wavefun(level=WAVEFORM_LEVEL)[0]
	else:
		psi = pywt.Wavelet(wavelet).wavefun(level=WAVEFORM_LEVEL)[1]
	psi = np.real(np.asarray(psi, dtype=complex))
	return psi / max(float(np.max(np.abs(psi))), 1e-12)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _wavelet_train(steps, action_dim, amplitude=1.0, wavelet="haar", min_segment=4, **_params):
	"""The wavelet's waveform at dyadic scales, coarse to fine: segments of steps/2, steps/4, ... >= min_segment."""
	psi, segments, length = _waveform(wavelet), [], steps // 2
	while length >= min_segment:
		segments.append(np.interp(np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, len(psi)), psi))
		length //= 2
	signal = np.concatenate(segments) if segments else np.zeros(0)
	signal = np.pad(signal, (0, steps - len(signal)))
	return amplitude * np.repeat(signal[:, None], action_dim, axis=1)


EXCITATIONS = {
	"impulse": _impulse,
	"step": _step,
	"sine": _sine,
	"chirp": _chirp,
	"exponential_chirp": _exponential_chirp,
	"prbs": _prbs,
	"wavelet_train": _wavelet_train,
}


def excitation(name: str, steps: int, action_dim: int, **params: float | str) -> np.ndarray:
	"""(steps, action_dim) actions in [-1, 1] of the named excitation, with its parameters."""
	if name not in EXCITATIONS:
		raise ValueError(f"Unknown chirp excitation: {name!r}; expected one of {sorted(EXCITATIONS)}")
	return np.clip(EXCITATIONS[name](steps, action_dim, **params), -1.0, 1.0)


def to_discrete(signal: np.ndarray) -> np.ndarray:
	"""Discrete two-action spaces (CartPole): action 1 where the first channel is positive, else 0."""
	return (np.asarray(signal)[:, 0] > 0.0).astype(int)
