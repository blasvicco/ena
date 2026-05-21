"""Environment Recognizer Module for EN Agent."""

# Libs imports
import numpy as np
from numpy.linalg import norm


class EnvironmentRecognizer:
	"""Identifies recurring environments using a Detect -> Identify architecture."""

	def __init__(self, state_dim, similarity_threshold=0.90, ema_alpha=0.10, genetic_threshold=0.80, genetic_bonus=0.50):
		self._state_dim = state_dim
		self._similarity_threshold = similarity_threshold
		self.ema_alpha = ema_alpha
		self._genetic_threshold = genetic_threshold
		self._genetic_bonus = genetic_bonus

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

	@property
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""
		return set(self._physical_prototypes.keys())

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
		live_reward = np.array([memory.live_reward_rate / 100.0], dtype=np.float32)

		return np.concatenate([physical, perf_deriv, plasticity, live_reward])

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
			self._current_full_baseline = (1 - self.ema_alpha) * self._current_full_baseline + self.ema_alpha * full_vec
			return False

		similarity = self._cosine_similarity(full_vec, self._current_full_baseline)

		# If similarity drops significantly, we assume the environment changed
		if similarity < self._similarity_threshold:
			self._cooldown = self.transition_gathering_episodes
			self._current_full_baseline = full_vec  # Reset baseline abruptly
			return True

		# Otherwise smoothly track
		self._current_full_baseline = (1 - self.ema_alpha) * self._current_full_baseline + self.ema_alpha * full_vec
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
			self._current_niche_id = -1
			return -1, best_similarity
		self._current_niche_id = best_niche_id
		return best_niche_id, best_similarity

	def match_niche(self, memory, current_individual, hall_of_fame, population=None):
		"""
		Late Identity Binding: Called only when a gladiator is being inducted.
		Uses gladiator identity first, then physical fallback for mutants.
		"""
		# 1. Did we succeed using an existing gladiator?
		if current_individual.get("is_gladiator"):
			home_niche_id = current_individual.get("internal_env_id")
			physical_vec = self.compute_physical_vector(memory)

			if home_niche_id in self._physical_prototypes:
				home_similarity = self._cosine_similarity(
					physical_vec, self._physical_prototypes[home_niche_id]
				)
				if home_similarity >= self._similarity_threshold:
					# Confirmed in home environment: update prototype and return
					self._current_niche_id = home_niche_id
					self._physical_prototypes[home_niche_id] = (
						1 - self.ema_alpha
					) * self._physical_prototypes[home_niche_id] + self.ema_alpha * physical_vec
					self.merge_similar_niches(hall_of_fame, population)
					return home_niche_id
				# Physics diverged from home niche — fall through to physical matching
				# This allows the HoF to fork a new niche for the current environment
			else:
				# First induction: create the initial prototype
				self._physical_prototypes[home_niche_id] = physical_vec.copy()
				self._current_niche_id = home_niche_id
				self.merge_similar_niches(hall_of_fame, population)
				return home_niche_id

		# 2. We succeeded using a mutant. Compare Physical Fingerprint against Prototypes
		physical_vec = self.compute_physical_vector(memory)
		best_niche_id = None
		best_similarity = -np.inf

		for niche_id, prototype in self._physical_prototypes.items():
			similarity = self._cosine_similarity(physical_vec, prototype)

			# Genetic similarity lineage bonus
			gladiators = [
				gladiator for gladiator in hall_of_fame._records.values()
				if gladiator.get("internal_env_id") == niche_id
			]
			if gladiators:
				max_gen_sim = max([
					self._calculate_genetic_similarity(current_individual, gladiator)
					for gladiator in gladiators
				])
				if max_gen_sim > self._genetic_threshold:
					# Continuous quadratic scaling: bonus grows smoothly from 0 to _genetic_bonus
					scaling = (max_gen_sim - self._genetic_threshold) / (1.0 - self._genetic_threshold + 1e-8)
					similarity += (scaling ** 2) * self._genetic_bonus

			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		# 3. Resolve the Match
		if best_niche_id is not None and best_similarity > self._similarity_threshold:
			# It matches physically. Assign to this niche!
			self._physical_prototypes[best_niche_id] = (
				1 - self.ema_alpha
			) * self._physical_prototypes[best_niche_id] + self.ema_alpha * physical_vec
			self._current_niche_id = best_niche_id
			self.merge_similar_niches(hall_of_fame, population)
			return best_niche_id

		# 4. Provision a new niche
		new_id = self._next_niche_id
		self._physical_prototypes[new_id] = physical_vec.copy()
		self._next_niche_id += 1
		self._current_niche_id = new_id
		self.merge_similar_niches(hall_of_fame, population)

		return self._current_niche_id

	def merge_similar_niches(self, hall_of_fame, population, merge_threshold=0.90):
		"""
		Find pairs of physical prototypes with similarity >= merge_threshold
		and merge the newer niche into the older one.
		"""
		niche_ids = sorted(list(self._physical_prototypes.keys()))
		to_merge = {}  # src_niche -> dst_niche

		for i in range(len(niche_ids)):
			niche_a = niche_ids[i]
			if niche_a in to_merge:
				continue
			for j in range(i + 1, len(niche_ids)):
				niche_b = niche_ids[j]
				if niche_b in to_merge:
					continue

				sim = self._cosine_similarity(
					self._physical_prototypes[niche_a],
					self._physical_prototypes[niche_b]
				)
				if sim >= merge_threshold:
					# Merge niche_b (newer/larger ID) into niche_a (older/smaller ID)
					to_merge[niche_b] = niche_a

		if not to_merge:
			return False

		# Apply merges
		for src, dst in to_merge.items():
			# 1. Blend prototypes (simple average)
			self._physical_prototypes[dst] = 0.5 * self._physical_prototypes[dst] + 0.5 * self._physical_prototypes[src]
			if src in self._physical_prototypes:
				del self._physical_prototypes[src]

			# 2. Update Hall of Fame records
			if hall_of_fame and hasattr(hall_of_fame, "_records"):
				for record in hall_of_fame._records.values():
					if record.get("internal_env_id") == src:
						record["internal_env_id"] = dst

			# 3. Update active population
			if population:
				for individual in population:
					if individual.get("internal_env_id") == src:
						individual["internal_env_id"] = dst

			# Update current niche ID if it was the merged one
			if self._current_niche_id == src:
				self._current_niche_id = dst

		return True

	@staticmethod
	def _cosine_similarity(vec_a, vec_b):
		"""Standard Cosine Similarity (scale-invariant)."""
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
