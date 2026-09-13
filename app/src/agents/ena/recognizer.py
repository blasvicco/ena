"""Environment Recognizer Module for EN Agent."""

# Libs imports
import numpy as np
from numpy.linalg import norm

# App imports
from ..tools.normalizers import _RunningNormalizer


class EnvironmentRecognizer:
	"""Facade that delegates environment-recognition logic to a pluggable strategy.

	Usage:
		recognizer = EnvironmentRecognizer("live", state_dim, similarity_threshold=0.90)
		niche_id, similarity = recognizer.match(memory)
	"""

	def __init__(self, strategy, state_dim, **kwargs):
		"""Instantiate the appropriate strategy based on the strategy name."""
		strategies = {
			"live": LiveRolloutRecognizer,
			"confirmation_window": ConfirmationWindowRecognizer,
			"neutral_probe": NeutralProbeRecognizer,
		}
		if strategy not in strategies:
			raise ValueError(f"Unknown recognizer strategy: '{strategy}'. Available: {list(strategies)}")
		self.instance = strategies[strategy](state_dim, **kwargs)

	# ---------------------------------------------------------------
	# Delegated public API
	# ---------------------------------------------------------------

	@property
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self.instance.current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""
		return self.instance.known_niche_ids

	def update_state(self, state, memory):
		"""Track observation distribution purely for environment physics recognition."""
		self.instance.update_state(state, memory)

	def compute_physical_vector(self, memory):
		"""Build context from strictly static state and transition dynamics."""
		return self.instance.compute_physical_vector(memory)

	def compute_full_vector(self, memory):
		"""Build context including distress signals for change detection."""
		return self.instance.compute_full_vector(memory)

	def detect_change(self, memory):
		"""Check if the environment has significantly shifted from the baseline."""
		return self.instance.detect_change(memory)

	def match(self, memory):
		"""Return (niche_id, similarity) of the best matching physical prototype (Test Mode)."""
		return self.instance.match(memory)

	def match_niche(self, memory, current_individual, hall_of_fame, population=None):
		"""Late Identity Binding: called only when a gladiator is being inducted."""
		return self.instance.match_niche(memory, current_individual, hall_of_fame, population=population)

	def probe(self, make_env, physics_env, physics_keys):
		"""Sample a policy-independent physics signature, once per episode. No-op unless
		the chosen strategy needs it (e.g. neutral_probe)."""
		self.instance.probe(make_env, physics_env, physics_keys)

	def reset(self):
		"""Flush any strategy-owned physical-fingerprint state on a detected environment
		shift. No-op unless the strategy keeps state outside of AgentMemory (e.g.
		neutral_probe's own probe normalizers) — memory.reset_recognizer_statistics()
		already handles memory-owned state (short_term_normalizer/transition_normalizer)."""
		self.instance.reset()


class LiveRolloutRecognizer:
	"""Identifies recurring environments using a Detect -> Identify architecture.

	Sources its physics signature from the actively observed rollout (whichever policy
	is currently acting) — the "live" (default) strategy, preserving the original behavior.

	confirmation_window and the data source (live rollout vs. neutral probe) are
	orthogonal: this base class owns the confirmation-window gating on detect_change(),
	so any subclass overriding compute_physical_vector() (e.g. NeutralProbeRecognizer)
	gets it for free too, instead of the two being mutually exclusive strategies.
	"""

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
		Check if the environment has significantly shifted from the baseline, requiring
		confirmation_window consecutive low-similarity readings before firing (default 1,
		i.e. immediate). Returns True only once the shift is confirmed.

		Compares on the physical vector alone (not compute_full_vector) — deliberately
		excluding performance_derivative/recent_plasticity. recent_plasticity is itself
		downstream of this method's own side effect (a firing clears reward_history, which
		forces __fuzzylogic's warmup path to return plasticity=1.0 for several episodes,
		which is then fed right back into the next comparison) — a self-referential input
		that was causing a feedback-loop oscillation of spurious firings, confirmed via the
		live CSV: firings alternated every 5+confirmation_window episodes purely tracking
		plasticity swinging between its warmup value and its steady-state one, even while
		training scores stayed at the maximum the whole time.
		"""
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
				self._current_baseline = physical_vec  # Reset baseline abruptly
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
		"""
		Late Identity Binding: Called only when a gladiator is being inducted.
		Uses gladiator identity first, then physical fallback for mutants.
		"""
		# 1. Did we succeed using an existing gladiator? Trust its registered home — always.
		if current_individual.get("is_gladiator"):
			home_niche_id = current_individual.get("internal_env_id")
			physical_vec = self.compute_physical_vector(memory)
			self._current_niche_id = home_niche_id
			if home_niche_id in self._physical_prototypes:
				# Update the prototype with the confirmed observation
				self._physical_prototypes[home_niche_id] = (1 - self.ema_alpha) * self._physical_prototypes[
					home_niche_id
				] + self.ema_alpha * physical_vec
			else:
				# First induction: create the initial prototype
				self._physical_prototypes[home_niche_id] = physical_vec.copy()
			return home_niche_id

		# 2. We succeeded using a mutant. Compare Physical Fingerprint against Prototypes
		physical_vec = self.compute_physical_vector(memory)
		best_niche_id = None
		best_similarity = -np.inf

		for niche_id, prototype in self._physical_prototypes.items():
			similarity = self._cosine_similarity(physical_vec, prototype)

			# Genetic similarity lineage bonus
			gladiators = [gladiator for gladiator in hall_of_fame._records.values() if gladiator.get("internal_env_id") == niche_id]
			if gladiators:
				max_gen_sim = max([self._calculate_genetic_similarity(current_individual, gladiator) for gladiator in gladiators])
				if max_gen_sim > self._genetic_threshold:
					# Linear scaling: grows evenly from 0 to _genetic_bonus as gen_sim goes threshold→1
					scaling = (max_gen_sim - self._genetic_threshold) / (1.0 - self._genetic_threshold + 1e-8)
					similarity += scaling * self._genetic_bonus

			if similarity > best_similarity:
				best_similarity = similarity
				best_niche_id = niche_id

		# 3. Resolve the Match
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

	def probe(self, make_env, physics_env, physics_keys):
		"""No-op: this strategy sources its physics signature from the live rollout."""

	def reset(self):
		"""No-op: this strategy's physical vector comes from memory's own normalizers,
		already flushed by memory.reset_recognizer_statistics()."""

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


class ConfirmationWindowRecognizer(LiveRolloutRecognizer):
	"""LiveRolloutRecognizer with a non-trivial confirmation window by default (3, instead
	of the base class's immediate-fire default of 1). The gating logic itself lives on the
	base class now so it composes with NeutralProbeRecognizer too — this subclass exists
	only to name the "live rollout + confirmation window" combination explicitly as its
	own strategy for experiment configs that want exactly that."""

	def __init__(self, state_dim, confirmation_window=3, **kwargs):
		super().__init__(state_dim, confirmation_window=confirmation_window, **kwargs)


class NeutralProbeRecognizer(LiveRolloutRecognizer):
	"""Sources the physical vector from a short neutral-action probe instead of the live
	rollout — decouples the physics fingerprint from whichever policy is currently acting,
	so a "generalist" individual's own rollout can no longer self-confirm the wrong niche.
	"""

	def __init__(self, state_dim, probe_steps=10, **kwargs):
		super().__init__(state_dim, **kwargs)
		self._probe_steps = probe_steps
		self._probe_obs_normalizer = _RunningNormalizer(shape=state_dim, max_count=probe_steps * 20)
		self._probe_transition_normalizer = _RunningNormalizer(shape=state_dim, max_count=probe_steps * 20)

	def probe(self, make_env, physics_env, physics_keys):
		"""Sample a few neutral (random-action) steps on a throwaway env built with the same
		physics as physics_env — never touches the actual scored episode's trajectory."""
		env_params = {key: getattr(physics_env.unwrapped, key) for key in physics_keys if hasattr(physics_env.unwrapped, key)}
		probe_env = make_env(**env_params)
		try:
			state, _info = probe_env.reset()
			state = np.asarray(state, dtype=np.float32)
			for _ in range(self._probe_steps):
				action = probe_env.action_space.sample()
				next_state, _reward, terminated, truncated, _info = probe_env.step(action)
				next_state = np.asarray(next_state, dtype=np.float32)
				self._probe_obs_normalizer.update(state[np.newaxis, :])
				self._probe_transition_normalizer.update((next_state - state)[np.newaxis, :])
				if terminated or truncated:
					state, _info = probe_env.reset()
					state = np.asarray(state, dtype=np.float32)
				else:
					state = next_state
		finally:
			probe_env.close()

	def compute_physical_vector(self, memory):
		"""Build context from the policy-independent probe normalizers, ignoring whatever
		the live rollout's short_term_normalizer/transition_normalizer currently look like."""
		obs_mean = np.copy(self._probe_obs_normalizer.mean)
		obs_var = np.copy(self._probe_obs_normalizer.var)
		obs_var_log = np.log1p(obs_var)

		trans_mean = np.copy(self._probe_transition_normalizer.mean)
		trans_var = np.copy(self._probe_transition_normalizer.var)
		trans_var_log = np.log1p(trans_var)

		return np.concatenate([obs_mean, obs_var_log, trans_mean, trans_var_log])

	def reset(self):
		"""Flush the probe's own running normalizers on a detected environment shift.

		These live on this instance, not AgentMemory, so memory.reset_recognizer_statistics()
		never reached them — meaning every environment switch previously left the probe
		blending in the *previous* environment's physics for up to probe_steps*20 samples
		(~20 episodes) via EMA decay, right during the window niche identification most
		needs a clean reading. This is what was producing spurious/duplicate "new niche"
		events shortly after a real switch that didn't match the final induced niche.
		"""
		self._probe_obs_normalizer.reset()
		self._probe_transition_normalizer.reset()
