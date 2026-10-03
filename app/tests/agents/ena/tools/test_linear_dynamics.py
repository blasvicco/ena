"""Tests for the linear-Gaussian dynamics model tool."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.tools.linear_dynamics import LinearGaussianDynamics

TRUE_WEIGHTS = np.array([[1.0, 0.0], [0.5, -2.0], [0.0, 0.3]])


def _data(weights, count, noise, seed):
	"""Draw (features, targets) from targets = features @ weights + noise."""
	rng = np.random.default_rng(seed)
	features = rng.normal(size=(count, weights.shape[0]))
	targets = features @ weights + noise * rng.normal(size=(count, weights.shape[1]))
	return features, targets


def test_fit_recovers_the_true_weights_and_residual_variance() -> None:
	"""On linear data with known noise, the ridge fit recovers W and the per-output noise variance."""
	with step("Arrange: 5000 rows with noise std 0.1."):
		features, targets = _data(TRUE_WEIGHTS, 5000, noise=0.1, seed=0)

	with step("Act: fit."):
		model = LinearGaussianDynamics.fit(features, targets)

	with step("Assert: weights and variance match the generating process."):
		assert np.allclose(model.weights, TRUE_WEIGHTS, atol=0.01)
		assert np.allclose(model.variance, 0.01, rtol=0.1)
		assert model.count == 5000


def test_incremental_adds_match_a_single_batch_fit() -> None:
	"""Sufficient statistics make the fit order-independent: many small adds == one batch fit."""
	with step("Arrange: one dataset."):
		features, targets = _data(TRUE_WEIGHTS, 300, noise=0.1, seed=1)

	with step("Act: fit once in batch, and once row-chunk by row-chunk."):
		batch = LinearGaussianDynamics.fit(features, targets)
		incremental = LinearGaussianDynamics(3, 2)
		for start in range(0, 300, 7):
			incremental.add(features[start : start + 7], targets[start : start + 7])

	with step("Assert: identical weights and variance."):
		assert np.allclose(batch.weights, incremental.weights)
		assert np.allclose(batch.variance, incremental.variance)


def test_log_likelihood_prefers_data_from_the_models_own_dynamics() -> None:
	"""Held-out data from the same dynamics scores higher than data from different dynamics."""
	with step("Arrange: a model fitted on TRUE_WEIGHTS, held-out data from it and from other weights."):
		model = LinearGaussianDynamics.fit(*_data(TRUE_WEIGHTS, 2000, noise=0.1, seed=2))
		same = _data(TRUE_WEIGHTS, 500, noise=0.1, seed=3)
		other = _data(TRUE_WEIGHTS * 1.2, 500, noise=0.1, seed=3)

	with step("Act: mean per-row log-likelihood of each."):
		same_score = model.log_likelihood(*same).mean()
		other_score = model.log_likelihood(*other).mean()

	with step("Assert: same-dynamics data is clearly more likely."):
		assert same_score > other_score + 1.0


def test_standardized_residuals_are_unit_variance_on_same_dynamics_data() -> None:
	"""Standardized residuals of held-out same-dynamics data are ~N(0, I) — PM-CuSum's pre-change."""
	with step("Arrange: a fitted model and held-out data from the same dynamics."):
		model = LinearGaussianDynamics.fit(*_data(TRUE_WEIGHTS, 5000, noise=0.1, seed=4))
		held_out = _data(TRUE_WEIGHTS, 5000, noise=0.1, seed=5)

	with step("Act: standardize."):
		residuals = model.standardized_residuals(*held_out)

	with step("Assert: zero mean, unit variance per output."):
		assert np.allclose(residuals.mean(axis=0), 0.0, atol=0.05)
		assert np.allclose(residuals.var(axis=0), 1.0, atol=0.1)


@pytest.mark.parametrize("var_floor", [1e-6, 1e-3])
def test_variance_is_floored_on_noise_free_data(var_floor) -> None:
	"""Deterministic dynamics (exact fit) must not yield a zero variance and infinite likelihoods."""
	with step("Arrange/Act: fit noise-free data with the given floor."):
		model = LinearGaussianDynamics.fit(*_data(TRUE_WEIGHTS, 200, noise=0.0, seed=6), var_floor=var_floor)

	with step("Assert: the variance sits at the floor."):
		assert np.allclose(model.variance, var_floor)
