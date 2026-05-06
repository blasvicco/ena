"""Hall of Fame — manages the archive of elite specialists."""
import numpy as np
from numpy.linalg import norm


class HallOfFame:
	"""
	Manages the archive of elite specialists (Gladiators).

	Responsibilities:
	- Z-score based induction with configurable threshold
	- Niche protection via cosine similarity (no strategic twins)
	- Ecological niche balancing (no single env dominates)
	- Memory decay so stale records become vulnerable
	"""

	SIMILARITY_THRESHOLD = 0.95
	MEMORY_DECAY_RATE = 0.999  # Softer decay: 99.9% retention per check (K7)

	def __init__(self, gladiator_amounts, brain, action_dim, state_dim):
		self.gladiator_amounts = gladiator_amounts
		self._records = {}
		self._brain = brain
		self._action_dim = action_dim
		self._state_dim = state_dim
		# UCB1 niche selection state — active during test phase only
		self._niche_map = {}    # {population_slot_idx: env_id}
		self._niche_stats = {}  # {env_id: {"total_reward": float, "tries": int}}

	# ------------------------------------------------------------------
	# Dict-like interface so ENAgent can use self.hall_of_fame naturally
	# ------------------------------------------------------------------

	def __bool__(self):
		return bool(self._records)

	def __len__(self):
		return len(self._records)

	def __contains__(self, key):
		return key in self._records

	def __getitem__(self, key):
		return self._records[key]

	def __setitem__(self, key, value):
		self._records[key] = value

	def __delitem__(self, key):
		del self._records[key]

	def values(self):
		"""Return HoF record values."""
		return self._records.values()

	def items(self):
		"""Return HoF record items."""
		return self._records.items()

	def keys(self):
		"""Return HoF record keys."""
		return self._records.keys()

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def inject_gladiators(self, population, pop_size, obs_normalizer, restore_trust=True):
		"""Re-introduces Gladiators into the active population with Niche Distribution.
		Also resets UCB1 niche stats for the new test phase.
		"""
		if not self._records:
			return

		# Group by Niche
		niches = {}
		for record in self._records.values():
			env_id = record.get("internal_env_id", -1)
			if env_id not in niches:
				niches[env_id] = []
			niches[env_id].append(record)

		# Sort each niche by Trust (best first)
		for env_id in niches:
			niches[env_id].sort(key=lambda r: r.get("trust", -99), reverse=True)

		unique_envs = list(niches.keys())
		num_niches = len(unique_envs)
		total_slots = self.gladiator_amounts
		base_budget = total_slots // num_niches if num_niches > 0 else total_slots

		selected_gladiators = []
		remaining_pool = []

		# Take from each niche according to budget
		for env_id in unique_envs:
			niche_glads = niches[env_id]
			selected_gladiators.extend(niche_glads[:base_budget])
			remaining_pool.extend(niche_glads[base_budget:])

		# Fill remaining slots with best global leftovers
		remaining_pool.sort(key=lambda r: r.get("trust", -99), reverse=True)
		needed = total_slots - len(selected_gladiators)
		if needed > 0:
			selected_gladiators.extend(remaining_pool[:needed])

		# Inject into population and build internal UCB1 niche maps directly
		self._niche_map.clear()
		self._niche_stats.clear()
		for index, gladiator in enumerate(selected_gladiators):
			if index >= pop_size or index >= total_slots:
				break

			target = population[index]
			target["age"] = 0
			target["individual_id"] = gladiator["individual_id"]
			target["model"].set_weights(gladiator["model"].get_weights())
			target["trust"] = gladiator["trust"] if restore_trust else 0.0
			target["is_gladiator"] = gladiator.get("is_gladiator", False)
			target["env_name"] = gladiator.get("env_name", "Unknown")

			env_id = gladiator.get("internal_env_id", -1)
			target["internal_env_id"] = env_id
			target["last_mu"] = np.copy(gladiator.get("last_mu", obs_normalizer.mean))
			target["last_var"] = np.copy(gladiator.get("last_var", obs_normalizer.var))

			# Setup UCB1 states
			self._niche_map[index] = env_id
			if env_id not in self._niche_stats:
				self._niche_stats[env_id] = {"total_reward": 0.0, "tries": 0}

	def record_episode_result(self, slot_idx, norm_score):
		"""Update UCB1 stats for the niche of the given population slot after a test episode."""
		env_id = self._niche_map.get(slot_idx)
		if env_id is not None and env_id in self._niche_stats:
			self._niche_stats[env_id]["total_reward"] += norm_score
			self._niche_stats[env_id]["tries"] += 1

	def select_best_gladiator_slot(self):
		"""Return the population slot index of the best gladiator to try next.

		Uses UCB1 at the niche level: every niche gets tried at least once before
		any is repeated. Returns None if no niche map is available (falls back to
		global trust argmax in ENAgent).
		"""
		if not self._niche_stats or not self._niche_map:
			return None
		T = sum(s["tries"] for s in self._niche_stats.values()) + 1
		C = 0.3
		best_niche, best_score = None, -np.inf
		for env_id, stats in self._niche_stats.items():
			if stats["tries"] == 0:
				score = np.inf  # Force exploration of every niche at least once
			else:
				avg = stats["total_reward"] / stats["tries"]
				score = avg + C * np.sqrt(np.log(T) / stats["tries"])
			if score > best_score:
				best_score, best_niche = score, env_id
		# Within the winning niche, return the slots so ENAgent picks max trust
		niche_slots = [i for i, eid in self._niche_map.items() if eid == best_niche]
		if not niche_slots:
			return None
		return niche_slots


	def update(self, winner, old_trust, population, plasticity_history, env_detector, obs_normalizer):
		"""
		Attempt to induct winner into the Hall of Fame.
		Called from ENAgent.train() only when failure_signal > 0.1.
		"""
		# --- STEP 1: MEMORY DECAY ---
		for item in self._records.values():
			item["induction_z_score"] = item.get("induction_z_score", 0) * self.MEMORY_DECAY_RATE

		# --- STEP 2: CALCULATE CONTEXT & METRICS ---
		current_trusts = [guy["trust"] for guy in population]
		avg_trust = np.mean(current_trusts)
		# K1: std_trust floor prevents Z-score explosions on uniform populations
		std_trust = max(np.std(current_trusts), 0.1)

		current_z_score = (old_trust - avg_trust) / std_trust

		current_plasticity = plasticity_history[-1] if plasticity_history else 1.0
		z_score_required = 1.5 * (1.0 - (current_plasticity * 0.5))

		# Early exit: not impressive enough relative to current peers
		if current_z_score < z_score_required and old_trust < 0.5:
			return

		individual_id = winner["individual_id"]

		# --- STEP 3: CHECK FOR EXISTING ID ---
		if individual_id in self._records:
			if old_trust > self._records[individual_id].get("trust", -99):
				self._records[individual_id]["model"].set_weights(winner["model"].get_weights())
				self._records[individual_id]["trust"] = old_trust
				self._records[individual_id]["induction_z_score"] = current_z_score
				self._records[individual_id]["env_name"] = winner.get("env_name", "Unknown")
				self._records[individual_id]["last_mu"] = np.copy(
					winner.get("last_mu", obs_normalizer.mean)
				)
				self._records[individual_id]["last_var"] = np.copy(
					winner.get("last_var", obs_normalizer.var)
				)
			return

		# --- STEP 4: NICHE PROTECTION (Genetic Similarity) ---
		for hof_id, hof_record in list(self._records.items()):
			similarity = self._calculate_genetic_similarity(winner, hof_record)
			if similarity > self.SIMILARITY_THRESHOLD:
				existing_trust = hof_record.get("trust", -99)
				if old_trust > existing_trust:
					del self._records[hof_id]
					self._save(winner, old_trust, current_z_score, env_detector, obs_normalizer)
				# Whether we replaced or not, twin found — exit
				return

		# --- STEP 5: OPEN SLOT INDUCTION ---
		if len(self._records) < self.gladiator_amounts:
			self._save(winner, old_trust, current_z_score, env_detector, obs_normalizer)
			return

		# --- STEP 6: ECOLOGICAL NICHE PROTECTION ---
		# Find the most over-populated internal niche
		env_counts = {}
		for record in self._records.values():
			env_id = record.get("internal_env_id", -1)
			env_counts[env_id] = env_counts.get(env_id, 0) + 1

		max_env_id = max(env_counts, key=env_counts.get)
		niche_size = env_counts[max_env_id]

		current_env_id = env_detector.current_env_id
		is_same_env = (current_env_id == max_env_id)

		# If there are multiple items in the dominant niche (or we belong to it)
		if (niche_size > 1) or is_same_env:
			niches = {}
			for r in self._records.values():
				eid = r.get("internal_env_id", -1)
				niches.setdefault(eid, []).append(r)

			if max_env_id in niches and niches[max_env_id]:
				weakest_record = min(niches[max_env_id], key=lambda r: r.get("trust", 999))
				weakest_trust = weakest_record.get("trust", 999)

				# Replace if from a new env, or if objectively better
				if (not is_same_env) or (old_trust > weakest_trust):
					del self._records[weakest_record["individual_id"]]
					self._save(winner, old_trust, current_z_score, env_detector, obs_normalizer)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _save(self, individual, trust, z_score, env_detector, obs_normalizer):
		"""Deep copy of a specialist into the Hall of Fame."""
		backup = {}
		backup["model"] = self._brain(self._action_dim, self._state_dim)
		backup["model"].set_weights(individual["model"].get_weights())
		backup["individual_id"] = individual["individual_id"]
		backup["trust"] = trust
		backup["induction_z_score"] = z_score
		backup["env_name"] = individual.get("env_name", "Unknown")
		backup["internal_env_id"] = env_detector.current_env_id
		backup["is_gladiator"] = True
		backup["last_mu"] = np.copy(individual.get("last_mu", obs_normalizer.mean))
		backup["last_var"] = np.copy(individual.get("last_var", obs_normalizer.var))
		self._records[individual["individual_id"]] = backup

		# Mark the live population member as a gladiator immediately.
		# Since dicts are passed by reference, this updates the active population slot directly.
		individual["is_gladiator"] = True

	def _calculate_genetic_similarity(self, individual_a, individual_b):
		"""
		Cosine Similarity between two individuals' weight vectors.
		Returns float in [-1.0, 1.0]. 1.0 = genetically identical.
		"""
		weights_a = individual_a["model"].get_weights()
		weights_b = individual_b["model"].get_weights()
		vector_a = np.concatenate([w.flatten() for w in weights_a])
		vector_b = np.concatenate([w.flatten() for w in weights_b])
		dot_product = np.dot(vector_a, vector_b)
		norm_a = norm(vector_a) + 1e-8
		norm_b = norm(vector_b) + 1e-8
		return dot_product / (norm_a * norm_b)
