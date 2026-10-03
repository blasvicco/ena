"""Live-rollout environment-detection driver — sources its physics signature from whichever policy is currently acting."""

# Libs imports
import numpy as np
from numpy.linalg import norm

# App imports
from agents.ena.skills.env_detection.driver.abstract import ADriver
from agents.ena.tools.genetic_similarity import genetic_similarity


# pylint: disable=too-many-instance-attributes
class Instance(ADriver):
	"""Identifies recurring environments using a Detect -> Identify architecture."""

	# Sources its physics signature from the actively observed rollout (whichever policy is
	# currently acting) — the "live" (default) driver, preserving the original behavior.
	#
	# This base class owns the confirmation-window gating on detect_change(), parameterized,
	# so the confirmation_window driver is just this class with a non-trivial default.

	# pylint: disable=unused-argument,too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		state_dim,
		similarity_threshold=0.90,
		ema_alpha=0.10,
		genetic_threshold=0.80,
		genetic_bonus=0.50,
		confirmation_window=1,
		**_kwargs,
	):
		self._state_dim = state_dim
		self._similarity_threshold = similarity_threshold
		self.ema_alpha = ema_alpha
		self._genetic_threshold = genetic_threshold
		self._genetic_bonus = genetic_bonus

		self._physical_prototypes = {}
		self._current_baseline = None
		self._next_niche_id = 0
		self._current_niche_id = -1

		# Cooldown prevents rapid triggering during transitions
		self._cooldown = 0
		self.transition_gathering_episodes = 5

		# Confirmation window: require this many *consecutive* low-similarity readings
		# before firing detect_change() — default 1 preserves today's immediate-fire
		# behavior exactly. A single noisy reading shouldn't be enough to declare a shift.
		self._confirmation_window = confirmation_window
		self._pending_shift_streak = 0

	@property
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""
		return set(self._physical_prototypes.keys())

	def update(self, state, memory):
		"""Track observation distribution purely for environment physics recognition."""
		memory.short_term_normalizer.update(state[np.newaxis, :])

	def compute_physical_vector(self, memory):
		"""Build context from strictly static state and transition dynamics."""
		return normalizer_signature(memory.short_term_normalizer, memory.transition_normalizer)

	def compute_full_vector(self, memory):
		"""Build context including distress signals for change detection."""
		return with_distress_signals(self.compute_physical_vector(memory), memory)

	def detect_change(self, memory):
		"""Check if the environment has significantly shifted from the baseline."""
		# Requires confirmation_window consecutive low-similarity readings before firing
		# (default 1, i.e. immediate). Returns True only once the shift is confirmed.
		#
		# Compares on the physical vector alone (not compute_full_vector) — deliberately
		# excluding performance_derivative/recent_plasticity. recent_plasticity is itself
		# downstream of this method's own side effect (a firing clears reward_history, which
		# forces the linear_ramp curve's warmup path to return plasticity=1.0 for several episodes,
		# which is then fed right back into the next comparison) — a self-referential input
		# that was causing a feedback-loop oscillation of spurious firings, confirmed via the
		# live CSV: firings alternated every 5+confirmation_window episodes purely tracking
		# plasticity swinging between its warmup value and its steady-state one, even while
		# training scores stayed at the maximum the whole time.
		#
		# No reading counts until the fingerprint's normalizers have fully re-accumulated since
		# their last reset (_fingerprint_ready). A firing resets them, and a vector read from a
		# near-empty normalizer drifts as it refills: using it as the new baseline made that
		# refill look like another shift, which reset them again — a self-sustaining loop
		# (10-26 firings per training run vs. 2 real switches), each loop leaving a noisy
		# fingerprint that match_niche() provisioned as yet another niche.
		if not self._fingerprint_ready(memory):
			return False

		physical_vec = self.compute_physical_vector(memory)

		if self._current_baseline is None:
			self._current_baseline = physical_vec
			return False

		if self._cooldown > 0:
			self._cooldown -= 1
			# During cooldown, we aggressively update the baseline to track the new chaotic reality
			self._current_baseline = (1 - self.ema_alpha) * self._current_baseline + self.ema_alpha * physical_vec
			return False

		similarity = self._cosine_similarity(physical_vec, self._current_baseline)

		if similarity < self._similarity_threshold:
			self._pending_shift_streak += 1
			if self._pending_shift_streak >= self._confirmation_window:
				self._cooldown = self.transition_gathering_episodes
				# Re-established from the first ready reading of the new regime, not this one:
				# evaluate() is about to reset the normalizers it came from.
				self._current_baseline = None
				self._pending_shift_streak = 0
				return True
			# Not yet confirmed — baseline stays put so a genuine shift keeps registering
			# as low-similarity on the next reading instead of being chased away.
			return False

		# Otherwise smoothly track, and any high-similarity reading resets the streak —
		# a single noisy low reading in isolation must not accumulate toward a false fire.
		self._pending_shift_streak = 0
		self._current_baseline = (1 - self.ema_alpha) * self._current_baseline + self.ema_alpha * physical_vec
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
		"""Late Identity Binding: called only when a gladiator is being inducted."""
		# Uses gladiator identity first, then physical fallback for mutants.
		# 1. Did we succeed using an existing gladiator? Its home niche wins only if the
		# physics actually look like home. A gladiator that also succeeds under foreign
		# physics must not drag its home prototype there (that drift left no prototype
		# resembling the earlier training phases by test time) — it resolves to the niche
		# those physics belong to instead, so the Hall of Fame clones it there.
		if current_individual.get("is_gladiator"):
			home_niche_id = current_individual.get("internal_env_id")
			physical_vec = self.compute_physical_vector(memory)
			if home_niche_id not in self._physical_prototypes:
				# First induction: create the initial prototype
				self._physical_prototypes[home_niche_id] = physical_vec.copy()
				self._current_niche_id = home_niche_id
				return home_niche_id
			if self._cosine_similarity(physical_vec, self._physical_prototypes[home_niche_id]) > self._similarity_threshold:
				# Update the prototype with the confirmed observation
				self._physical_prototypes[home_niche_id] = (1 - self.ema_alpha) * self._physical_prototypes[
					home_niche_id
				] + self.ema_alpha * physical_vec
				self._current_niche_id = home_niche_id
				return home_niche_id
			# Foreign physics: resolve on physics alone (no genetic bonus — the gladiator's
			# own lineage would only pull it back home), never touching the home prototype.
			return self._resolve_physically(physical_vec, exclude_niche_id=home_niche_id)

		# 2. We succeeded using a mutant. Compare Physical Fingerprint against Prototypes
		physical_vec = self.compute_physical_vector(memory)
		best_niche_id = None
		best_similarity = -np.inf

		for niche_id, prototype in self._physical_prototypes.items():
			similarity = self._cosine_similarity(physical_vec, prototype)

			# Genetic similarity lineage bonus
			gladiators = [
				gladiator
				for gladiator in hall_of_fame._records.values()  # pylint: disable=protected-access
				if gladiator.get("internal_env_id") == niche_id
			]
			if gladiators:
				max_gen_sim = max(genetic_similarity(current_individual, gladiator) for gladiator in gladiators)
				if max_gen_sim > self._genetic_threshold:
					# Linear scaling: grows evenly from 0 to _genetic_bonus as gen_sim goes threshold→1
					scaling = (max_gen_sim - self._genetic_threshold) / (1.0 - self._genetic_threshold + 1e-8)
					similarity += scaling * self._genetic_bonus

			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		# 3. Resolve the Match
		return self._assign(physical_vec, best_niche_id, best_similarity)

	def _resolve_physically(self, physical_vec, exclude_niche_id):
		"""Assign physical_vec to its best-matching prototype (skipping exclude_niche_id), or a new niche."""
		best_niche_id = None
		best_similarity = -np.inf
		for niche_id, prototype in self._physical_prototypes.items():
			if niche_id == exclude_niche_id:
				continue
			similarity = self._cosine_similarity(physical_vec, prototype)
			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id
		return self._assign(physical_vec, best_niche_id, best_similarity)

	def _assign(self, physical_vec, best_niche_id, best_similarity):
		"""Blend physical_vec into best_niche_id if it clears the threshold, else provision a new niche."""
		if best_niche_id is not None and best_similarity > self._similarity_threshold:
			# It matches physically. Assign to this niche!
			self._physical_prototypes[best_niche_id] = (1 - self.ema_alpha) * self._physical_prototypes[best_niche_id] + self.ema_alpha * physical_vec
			self._current_niche_id = best_niche_id
			return best_niche_id

		# 4. Provision a new niche
		new_id = self._next_niche_id
		self._physical_prototypes[new_id] = physical_vec.copy()
		self._next_niche_id += 1
		self._current_niche_id = new_id

		return self._current_niche_id

	def start_episode(self):
		"""No-op: this driver keeps no per-episode state."""

	def observe(self, state, action, next_state, memory):
		"""No-op: this driver reads transitions through memory's normalizers instead."""

	def log_likelihoods(self, memory):
		"""No per-niche likelihood model — an empty dict."""
		return {}

	def _fingerprint_ready(self, memory):
		"""Whether the normalizers compute_physical_vector() reads have a full window since their last reset."""
		return _is_full(memory.short_term_normalizer) and _is_full(memory.transition_normalizer)

	def reset(self):
		"""No-op: this driver's physical vector comes from memory's own normalizers."""
		# Already flushed by memory.reset_recognizer_statistics().

	@staticmethod
	def _cosine_similarity(vec_a, vec_b):
		"""Standard Cosine Similarity (scale-invariant)."""
		norm_a = norm(vec_a) + 1e-8
		norm_b = norm(vec_b) + 1e-8
		return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def normalizer_signature(obs_normalizer, transition_normalizer) -> np.ndarray:
	"""Physics signature: observation and transition-delta means plus log(1 + variance)."""
	return np.concatenate(
		[
			np.copy(obs_normalizer.mean),
			np.log1p(np.copy(obs_normalizer.var)),
			np.copy(transition_normalizer.mean),
			np.log1p(np.copy(transition_normalizer.var)),
		]
	)


def with_distress_signals(physical: np.ndarray, memory) -> np.ndarray:
	"""Append the scaled performance derivative and recent plasticity to a physics signature."""
	# Scaled so they contribute to angular shift without blowing up norms.
	perf_deriv = np.array([memory.performance_derivative / 100.0], dtype=np.float32)
	plasticity = np.array([memory.recent_plasticity], dtype=np.float32)
	return np.concatenate([physical, perf_deriv, plasticity])


def _is_full(normalizer):
	"""Whether a running normalizer has accumulated its full max_count window since its last reset."""
	return normalizer.max_count is None or normalizer.count >= normalizer.max_count
