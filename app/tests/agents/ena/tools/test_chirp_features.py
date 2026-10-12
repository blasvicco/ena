"""Tests for chirp_features.py — features of a chirp's response (raw, or wavelet bands via PyWavelets)."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.tools.chirp_features import features

RESPONSE = np.random.default_rng(0).normal(size=(64, 3))  # 64 steps, 3 observation channels


def test_raw_features_are_the_flattened_response() -> None:
	"""No transform: every step of every channel."""
	with step("Act/Assert: 64 x 3 values."):
		assert np.array_equal(features(RESPONSE, kind="raw"), RESPONSE.ravel())


def test_the_default_is_a_dyadic_non_overlapping_dwt_with_one_energy_per_band() -> None:
	"""Default: Haar DWT, full depth (6 levels for 64 steps): 6 details + 1 approximation per channel."""
	with step("Act: default features."):
		values = features(RESPONSE)

	with step("Assert: 3 channels x 7 bands, log-energies."):
		assert values.shape == (21,)
		assert np.all(values >= 0.0)


def test_an_orthogonal_dwt_keeps_the_signal_energy_across_its_bands() -> None:
	"""Parseval: the bands' energies add up to the channel's energy (periodized orthogonal wavelets)."""
	with step("Act: db4 energies (undo the log)."):
		energies = np.expm1(features(RESPONSE, wavelet="db4", level=3)).reshape(3, -1)

	with step("Assert: per channel, the sum equals the signal energy."):
		assert np.allclose(energies.sum(axis=1), np.sum(RESPONSE**2, axis=0))


@pytest.mark.parametrize(
	"settings, bands",
	[
		({"transform": "dwt", "level": 3}, 4),
		({"transform": "swt", "level": 3}, 4),
		({"transform": "packet", "level": 3}, 8),
		({"transform": "cwt", "wavelet": "morl", "scales": [2, 4, 8, 16]}, 4),
	],
)
def test_each_transform_gives_its_number_of_bands(settings, bands) -> None:
	"""dwt: level details + approximation; swt (overlapping): the same bands, undecimated; packet: 2^level; cwt: one per scale."""
	with step(f"Act: {settings}."):
		values = features(RESPONSE, **settings)

	with step("Assert: bands per channel."):
		assert values.shape == (3 * bands,)


def test_coefficients_instead_of_energies() -> None:
	"""feature='coefficients' keeps every coefficient (an orthonormal DWT of the whole response)."""
	with step("Act/Assert: Haar DWT coefficients preserve the response's length and norm."):
		values = features(RESPONSE, feature="coefficients")
		assert values.shape == (64 * 3,)
		assert np.isclose(np.linalg.norm(values), np.linalg.norm(RESPONSE))


def test_a_continuous_wavelet_needs_the_cwt_and_unknown_settings_fail() -> None:
	"""Morlet has no DWT; unknown transforms or features are rejected."""
	with step("Act/Assert: ValueErrors."):
		with pytest.raises(ValueError):
			features(RESPONSE, wavelet="morl", transform="dwt")
		with pytest.raises(ValueError, match="transform"):
			features(RESPONSE, transform="fourier")
		with pytest.raises(ValueError, match="feature"):
			features(RESPONSE, feature="entropy")
