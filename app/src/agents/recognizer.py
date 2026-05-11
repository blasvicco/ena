"""Unsupervised Environment Recognizer for EN Agent."""

# Replaces the fragile EnvironmentDetector with a recognizer that maintains
# a library of prototype context vectors. When the same physics recur later
# in training, the recognizer assigns the *same* stable niche ID rather than
# a new incremental one.

# Context vector = [obs_mean, log1p(obs_var), recent_scores]

# Libs imports
import numpy as np
from numpy.linalg import norm

# App imports
from .normalizers import _RunningNormalizer


class EnvironmentRecognizer:
	"""Identifies recurring environments from sensorimotor experience alone."""

	EMA_ALPHA = 0.10  # Faster prototype convergence → fewer spurious niches

	def __init__(self, state_dim, similarity_threshold=0.85):
		self._state_dim = state_dim
		self._similarity_threshold = similarity_threshold

		self._prototypes = {}
		self._next_niche_id = 0
		self._current_niche_id = -1
		self._short_term_normalizer = _RunningNormalizer(shape=state_dim, max_count=1000)

	@property
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._current_niche_id

	@property
	def num_niches(self) -> int:
		"""Number of learned prototypes."""
		return len(self._prototypes)

	def update_state(self, state):
		"""Track observation distribution purely for environment physics recognition."""
		self._short_term_normalizer.update(state[np.newaxis, :])

	def compute_context_vector(self):
		"""Build context exclusively from short-term physics statistics."""
		obs_mean = np.copy(self._short_term_normalizer.mean)
		obs_var = np.copy(self._short_term_normalizer.var)
		obs_var_log = np.log1p(obs_var)

		return np.concatenate([obs_mean, obs_var_log])

	def match(self):
		"""Return (niche_id, similarity) of the best matching prototype."""
		# Does **not** create a new prototype – safe for test‑time use.
		# Returns (-1, -1.0) if no match exceeds the similarity threshold.
		context = self.compute_context_vector()
		best_niche_id = -1
		best_similarity = -1.0
		for niche_id, prototype in self._prototypes.items():
			similarity = self._cosine_similarity(context, prototype)
			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		if best_similarity < self._similarity_threshold:
			return -1, best_similarity
		return best_niche_id, best_similarity

	def recognize(self):
		"""Compute context, match or create prototype, return stable niche_id."""
		context = self.compute_context_vector()

		best_niche_id = None
		best_similarity = -np.inf

		for niche_id, prototype in self._prototypes.items():
			similarity = self._cosine_similarity(context, prototype)
			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		if best_niche_id is not None and best_similarity > self._similarity_threshold:
			self._prototypes[best_niche_id] = (1 - self.EMA_ALPHA) * self._prototypes[
				best_niche_id
			] + self.EMA_ALPHA * context
			self._current_niche_id = best_niche_id
		else:
			new_id = self._next_niche_id
			self._prototypes[new_id] = context.copy()
			self._next_niche_id += 1
			self._current_niche_id = new_id

		return self._current_niche_id

	@staticmethod
	def _cosine_similarity(vec_a, vec_b):
		norm_a = norm(vec_a) + 1e-8
		norm_b = norm(vec_b) + 1e-8
		return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
