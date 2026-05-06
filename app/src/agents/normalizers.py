"""Shared Normalizers for Agents."""

# Libs imports
import numpy as np

class _IdentityNormalizer:
	"""No-op normalizer for discrete environments."""

	def __init__(self, shape):
		# Dummy attributes so USP metadata storage (last_mu/last_var)
		# doesn't crash when accessing normalizer.mean/var.
		self.mean = np.zeros(shape, dtype=np.float64)
		self.var = np.ones(shape, dtype=np.float64)

	def update(self, batch):
		"""No-op — discrete environments don't need normalization."""

	def normalize(self, _x_):
		"""Pass through unchanged."""
		return _x_


class _RunningNormalizer:
	"""Welford's online algorithm for running mean/variance normalization."""

	# Used to normalize observations (which vary wildly in scale)
	# to roughly [-clip, clip], stabilizing network inputs.

	def __init__(self, shape, clip=10.0, max_count=None):
		"""Initialize _RunningNormalizer."""
		self.shape = shape
		self.clip = clip
		self.max_count = max_count
		self.reset()

	def reset(self):
		"""Hard-reset running statistics to initial state."""
		self.mean = np.zeros(self.shape, dtype=np.float64)
		self.var = np.ones(self.shape, dtype=np.float64)
		self.count = 1e-4

	def update(self, batch):
		"""Update running statistics with a new batch of observations."""
		batch = np.asarray(batch, dtype=np.float64)
		if batch.shape[0] == 0:
			return

		batch_mean = batch.mean(axis=0)
		batch_var = batch.var(axis=0)
		batch_count = batch.shape[0]

		delta = batch_mean - self.mean
		total_count = self.count + batch_count

		if self.max_count is not None:
			total_count = min(total_count, self.max_count)

		# Compute the mixing rate alpha to handle max_count correctly
		alpha = batch_count / max(total_count, 1e-8)
		alpha = min(1.0, alpha)

		new_mean = self.mean + delta * alpha
		# Exponential moving average
		new_var = (1.0 - alpha) * self.var + alpha * batch_var + alpha * (1.0 - alpha) * np.square(delta)

		self.mean = new_mean
		self.var = new_var
		self.count = total_count

	def get_state(self):
		"""Returns the current state of the normalizer."""
		return {
			"mean": np.copy(self.mean),
			"var": np.copy(self.var),
			"count": self.count,
		}

	def set_state(self, state):
		"""Sets the state of the normalizer."""
		self.mean = np.copy(state["mean"])
		self.var = np.copy(state["var"])
		self.count = state["count"]

	def normalize(self, _x_):
		"""Normalize to zero mean, unit variance, clipped."""
		return np.clip(
			(_x_ - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip
		)
