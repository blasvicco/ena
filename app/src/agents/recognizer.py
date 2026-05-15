"""Environment Recognizer Module for EN Agent."""

# Libs imports
import numpy as np
from numpy.linalg import norm


class EnvironmentRecognizer:
	"""Identifies recurring environments using a Detect -> Identify architecture."""

	EMA_ALPHA = 0.10  # Faster prototype convergence → fewer spurious niches

	def __init__(self, state_dim, similarity_threshold=0.85):
		self._state_dim = state_dim
		self._similarity_threshold = similarity_threshold

		self._physical_prototypes = {}
		self._current_full_baseline = None
		self._next_niche_id = 0
		self._current_niche_id = -1

		# Cooldown prevents rapid triggering during transitions
		self._cooldown = 0
		self.transition_gathering_episodes = 5

	@property
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._current_niche_id

	@property
	def num_niches(self) -> int:
		"""Number of learned prototypes."""
		return len(self._physical_prototypes)

	def update_state(self, state, memory):
		"""Track observation distribution purely for environment physics recognition."""
		memory.short_term_normalizer.update(state[np.newaxis, :])

	def compute_physical_vector(self, memory):
		"""Build context from strictly static state and transition dynamics."""
		obs_mean = np.copy(memory.short_term_normalizer.mean)
		obs_var = np.copy(memory.short_term_normalizer.var)
		obs_var_log = np.log1p(obs_var)

		trans_mean = np.copy(memory.transition_normalizer.mean)
		trans_var = np.copy(memory.transition_normalizer.var)
		trans_var_log = np.log1p(trans_var)

		return np.concatenate([obs_mean, obs_var_log, trans_mean, trans_var_log])

	def compute_full_vector(self, memory):
		"""Build context including distress signals for change detection."""
		physical = self.compute_physical_vector(memory)

		# We scale them so they contribute to angular shift without blowing up norms
		perf_deriv = np.array([memory.performance_derivative / 100.0], dtype=np.float32)
		plasticity = np.array([memory.recent_plasticity], dtype=np.float32)

		return np.concatenate([physical, perf_deriv, plasticity])

	def detect_change(self, memory):
		"""
		Check if the environment has significantly shifted from the baseline.
		Returns True if a shift is detected.
		"""
		full_vec = self.compute_full_vector(memory)

		if self._current_full_baseline is None:
			self._current_full_baseline = full_vec
			return False

		if self._cooldown > 0:
			self._cooldown -= 1
			# During cooldown, we aggressively update the baseline to track the new chaotic reality
			self._current_full_baseline = (1 - self.EMA_ALPHA) * self._current_full_baseline + self.EMA_ALPHA * full_vec
			return False

		similarity = self._cosine_similarity(full_vec, self._current_full_baseline)

		# If similarity drops significantly, we assume the environment changed
		if similarity < self._similarity_threshold:
			self._cooldown = self.transition_gathering_episodes
			self._current_full_baseline = full_vec  # Reset baseline abruptly
			return True

		# Otherwise smoothly track
		self._current_full_baseline = (1 - self.EMA_ALPHA) * self._current_full_baseline + self.EMA_ALPHA * full_vec
		return False

	def match(self, memory):
		"""Return (niche_id, similarity) of the best matching physical prototype (Test Mode)."""
		physical_vec = self.compute_physical_vector(memory)
		best_niche_id = -1
		best_similarity = -1.0
		for niche_id, prototype in self._physical_prototypes.items():
			similarity = self._cosine_similarity(physical_vec, prototype)
			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		if best_similarity < self._similarity_threshold:
			return -1, best_similarity
		return best_niche_id, best_similarity

	def match_niche(self, memory, current_individual, hall_of_fame):
		"""
		Late Identity Binding: Called only when a gladiator is being inducted.
		Uses gladiator identity first, then physical+genetic fallback for mutants.
		"""
		# 1. Did we succeed using an existing gladiator?
		if current_individual.get("is_gladiator"):
			best_niche_id = current_individual.get("internal_env_id")
			self._current_niche_id = best_niche_id
			# Update its physical prototype since it's confirmed
			physical_vec = self.compute_physical_vector(memory)
			self._physical_prototypes[best_niche_id] = (
				1 - self.EMA_ALPHA
			) * self._physical_prototypes[best_niche_id] + self.EMA_ALPHA * physical_vec
			return best_niche_id

		# 2. We succeeded using a mutant. Compare Physical Fingerprint against Prototypes
		physical_vec = self.compute_physical_vector(memory)
		best_niche_id = None
		best_similarity = -np.inf

		for niche_id, prototype in self._physical_prototypes.items():
			similarity = self._cosine_similarity(physical_vec, prototype)

			# Add genetic bonus if physically close but not quite there
			gladiators = [
				gladiator for gladiator in hall_of_fame._records.values()
				if gladiator.get("internal_env_id") == niche_id
			]
			if gladiators:
				max_gen_sim = max([
					self._calculate_genetic_similarity(current_individual, gladiator)
					for gladiator in gladiators
				])
				if max_gen_sim > 0.8:
					similarity += 0.5  # Massive bonus to avoid creating a new niche!

			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		# 3. Resolve the Match
		if best_niche_id is not None and best_similarity > self._similarity_threshold:
			# It matches physically (or with bonus). Assign to this niche!
			self._physical_prototypes[best_niche_id] = (
				1 - self.EMA_ALPHA
			) * self._physical_prototypes[best_niche_id] + self.EMA_ALPHA * physical_vec
			self._current_niche_id = best_niche_id
			return best_niche_id

		# 4. Provision a new niche
		new_id = self._next_niche_id
		self._physical_prototypes[new_id] = physical_vec.copy()
		self._next_niche_id += 1
		self._current_niche_id = new_id

		return self._current_niche_id

	@staticmethod
	def _cosine_similarity(vec_a, vec_b):
		norm_a = norm(vec_a) + 1e-8
		norm_b = norm(vec_b) + 1e-8
		return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

	@staticmethod
	def _calculate_genetic_similarity(individual_a, individual_b):
		"""Cosine Similarity between two individuals' weight vectors."""
		weights_a = individual_a["model"].get_weights()
		weights_b = individual_b["model"].get_weights()
		vector_a = np.concatenate([w.flatten() for w in weights_a])
		vector_b = np.concatenate([w.flatten() for w in weights_b])
		dot_product = np.dot(vector_a, vector_b)
		norm_a = norm(vector_a) + 1e-8
		norm_b = norm(vector_b) + 1e-8
		return float(dot_product / (norm_a * norm_b))
