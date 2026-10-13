"""Linear-Gaussian dynamics model: predicts a state transition from [state, action, 1] by ridge least squares."""

# Libs imports
import numpy as np


# pylint: disable=too-many-instance-attributes
class LinearGaussianDynamics:
	"""Online ridge regression of targets on features, with a diagonal Gaussian residual model."""

	# Gradient-free: kept as sufficient statistics (X'X, X'Y, Y'Y) and solved in closed form on
	# demand, so adding data is O(features^2) and the fit is exact for everything seen so far.

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(self, feature_dim: int, output_dim: int, ridge: float = 1e-3, var_floor: float = 1e-6) -> None:
		self._ridge = ridge
		self._var_floor = var_floor
		self._xtx = np.zeros((feature_dim, feature_dim))
		self._xty = np.zeros((feature_dim, output_dim))
		self._yty = np.zeros(output_dim)
		self._count = 0
		self._weights = None
		self._variance = None

	@property
	def count(self) -> int:
		"""Number of transitions this model has been fitted on."""
		return self._count

	def add(self, features: np.ndarray, targets: np.ndarray) -> None:
		"""Accumulate a batch of (features, targets) rows into the fit."""
		features = np.atleast_2d(np.asarray(features, dtype=np.float64))
		targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
		if features.shape[0] == 0:
			return
		self._xtx += features.T @ features
		self._xty += features.T @ targets
		self._yty += np.sum(targets * targets, axis=0)
		self._count += features.shape[0]
		self._weights = None
		self._variance = None

	@property
	def weights(self) -> np.ndarray:
		"""Ridge solution W of targets ≈ features @ W."""
		if self._weights is None:
			regularized = self._xtx + self._ridge * np.eye(self._xtx.shape[0])
			self._weights = np.linalg.solve(regularized, self._xty)
		return self._weights

	@property
	def variance(self) -> np.ndarray:
		"""Per-output residual variance of the fit, floored at var_floor."""
		if self._variance is None:
			weights = self.weights
			# Residual sum of squares from the sufficient statistics: Y'Y - 2 W'X'Y + W'X'XW (diagonal).
			rss = self._yty - 2.0 * np.sum(weights * self._xty, axis=0) + np.sum(weights * (self._xtx @ weights), axis=0)
			self._variance = np.maximum(rss / max(self._count, 1), self._var_floor)
		return self._variance

	def standardized_residuals(self, features: np.ndarray, targets: np.ndarray, variance_scale: float = 1.0) -> np.ndarray:
		"""Residuals scaled by the model's residual std (times sqrt(variance_scale)) — ~N(0, I) on same-dynamics data at scale 1."""
		features = np.atleast_2d(features)
		targets = np.atleast_2d(targets)
		return (targets - features @ self.weights) / np.sqrt(self.variance * variance_scale)

	def log_likelihood(self, features: np.ndarray, targets: np.ndarray, variance_scale: float = 1.0) -> np.ndarray:
		"""Per-row log-density of targets given features, with the residual variance widened by variance_scale."""
		residuals = self.standardized_residuals(features, targets, variance_scale)
		return -0.5 * (np.sum(residuals * residuals, axis=1) + np.sum(np.log(2.0 * np.pi * self.variance * variance_scale)))

	@classmethod
	def fit(cls, features: np.ndarray, targets: np.ndarray, **kwargs) -> "LinearGaussianDynamics":
		"""Build a model fitted on exactly this batch."""
		features = np.atleast_2d(features)
		targets = np.atleast_2d(targets)
		model = cls(features.shape[1], targets.shape[1], **kwargs)
		model.add(features, targets)
		return model
