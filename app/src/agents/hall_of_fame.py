"""Hall of Fame — manages the archive of elite specialists."""

# General imports
import os
import random
import uuid

# Libs imports
import numpy as np
from numpy.linalg import norm


class HallOfFame:
	"""
	Manages the archive of elite specialists (Gladiators).

	Responsibilities:
	- Z-score based induction with configurable threshold
	- Per-niche budget: each recognised niche holds up to max_per_niche gladiators
	- Niche protection via cosine similarity (no strategic twins)
	- Memory decay so stale records become vulnerable
	- Population injection: round-robin across niches, capped at max_gladiator_ratio * pop_size
	"""

	SIMILARITY_THRESHOLD = 0.95
	MEMORY_DECAY_RATE = 0.999  # Softer decay: 99.9% retention per check

	def __init__(self, max_per_niche, max_gladiator_ratio, brain, action_dim, state_dim):
		# max_per_niche:       max gladiators stored per recognised niche
		# max_gladiator_ratio: fraction of pop_size that can be gladiators at injection time
		self.max_per_niche = max_per_niche
		self.max_gladiator_ratio = max_gladiator_ratio
		self._records = {}
		self._brain = brain
		self._action_dim = action_dim
		self._state_dim = state_dim

	def __bool__(self):
		"""True when at least one gladiator has been inducted."""
		return bool(self._records)

	# pylint: disable=too-many-locals
	def inject_gladiators(
		self,
		population,
		pop_size,
		obs_normalizer,
		restore_trust=True,
	):
		"""
		Re-introduces Gladiators into the active population.

		Strategy: round-robin across niches (1 per niche per round), best-first within
		each niche, until floor(pop_size * max_gladiator_ratio) population slots are filled.
		"""
		if not self._records:
			return

		max_slots = int(pop_size * self.max_gladiator_ratio)
		if max_slots <= 0:
			return

		# Group by niche, randomly shuffle each niche so we try different gladiators
		niches = {}
		for individual in self._records.values():
			nid = individual.get("internal_env_id", -1)
			niches.setdefault(nid, []).append(individual)
		for niche in niches.values():
			random.shuffle(niche)

		# Round-robin: take 1 from each niche per round until max_slots filled
		selected = []
		niche_iters = {nid: iter(niche) for nid, niche in niches.items()}
		active_niches = list(niche_iters.keys())
		while len(selected) < max_slots and active_niches:
			exhausted = []
			for nid in active_niches:
				if len(selected) >= max_slots:
					break
				try:
					selected.append(next(niche_iters[nid]))
				except StopIteration:
					exhausted.append(nid)
			for nid in exhausted:
				active_niches.remove(nid)

		# Inject into the first N population slots
		for index, gladiator in enumerate(selected):
			if index >= pop_size:
				break
			target = population[index]
			target["age"] = 0
			target["individual_id"] = gladiator["individual_id"]
			target["model"].set_weights(gladiator["model"].get_weights())
			target["trust"] = gladiator["trust"] if restore_trust else 0.0
			target["is_gladiator"] = gladiator.get("is_gladiator", False)
			target["env_name"] = gladiator.get("env_name", "Unknown")
			target["internal_env_id"] = gladiator.get("internal_env_id", -1)
			target["last_mu"] = np.copy(gladiator.get("last_mu", obs_normalizer.mean))
			target["last_var"] = np.copy(gladiator.get("last_var", obs_normalizer.var))

	def save_niche_details(self, output_dir, agent_name):
		"""Save the niche details to a file in the output directory."""
		file_path = os.path.join(output_dir, f"niches_{agent_name}.txt")
		with open(file_path, "w", encoding="utf-8") as file:
			file.write(f"{'Niche ID':<10} | {'Env Name':<20} | {'Total Gladiators':<15}\n")
			file.write("-" * 51 + "\n")
			niches = {}
			for record in self._records.values():
				niche_id = record.get("internal_env_id", -1)
				env_name = record.get("env_name", "Unknown")
				if niche_id not in niches:
					niches[niche_id] = {"env_name": env_name, "count": 0}
				niches[niche_id]["count"] += 1
			if not niches:
				file.write("No gladiators inducted.\n")
			else:
				for niche_id, data in sorted(niches.items()):
					file.write(f"{niche_id:<10} | {data['env_name']:<20} | {data['count']:<15}\n")

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def update(
		self,
		winner,
		current_trust,
		population,
		memory,
		recognizer,
		obs_normalizer,
	):
		"""Attempt to induct winner into the Hall of Fame."""
		self.__memory_decay()

		current_z_score = self.__performance_gate(current_trust, population, memory.plasticity_history)
		if current_z_score is None:
			return

		# Late Identity Binding: Determine the niche ID for this successful individual
		recognizer.match_niche(memory, winner, self)

		if self.__continual_improvement(
			winner,
			current_trust,
			current_z_score,
			recognizer,
			obs_normalizer
		):
			return

		if self.__genetic_diversity(
			winner,
			current_trust,
			current_z_score,
			recognizer,
			obs_normalizer
		):
			return

		if self.__slot_induction(
			winner,
			current_trust,
			current_z_score,
			recognizer,
			obs_normalizer
		):
			return

		self.__eco_niche_protection(
			winner,
			current_trust,
			current_z_score,
			recognizer,
			obs_normalizer
		)

	# Private
	def __calculate_genetic_similarity(self, individual_a, individual_b):
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

	def __continual_improvement(self, winner, current_trust, current_z_score, recognizer, obs_normalizer):
		"""STEP 3: CHECK FOR EXISTING ID"""
		individual_id = winner["individual_id"]
		if individual_id not in self._records:
			return False

		existing = self._records[individual_id]
		current_env_id = recognizer.current_niche_id
		existing_env_id = existing.get("internal_env_id", -1)

		if current_env_id == existing_env_id:
			# Same niche: update in place if strictly better. env_id stays frozen.
			if current_trust > existing.get("trust", -99):
				existing["model"].set_weights(winner["model"].get_weights())
				existing["trust"] = current_trust
				existing["induction_z_score"] = current_z_score
				existing["env_name"] = winner.get("env_name", "Unknown")
				existing["last_mu"] = np.copy(winner.get("last_mu", obs_normalizer.mean))
				existing["last_var"] = np.copy(winner.get("last_var", obs_normalizer.var))
				winner["internal_env_id"] = current_env_id
			return True

		# Cross-niche: winner is already a gladiator from a different niche.
		# Its live weights have evolved since the original HoF snapshot —
		# fork it as a new entry for this niche if that niche has room.
		niche_count = sum(
			1 for individual in self._records.values()
			if individual.get("internal_env_id") == current_env_id
		)
		if niche_count < self.max_per_niche:
			clone_id = str(uuid.uuid4())
			clone = {
				"model": self._brain(self._action_dim, self._state_dim),
				"individual_id": clone_id,
				"trust": current_trust,
				"induction_z_score": current_z_score,
				"env_name": winner.get("env_name", "Unknown"),
				"internal_env_id": current_env_id,
				"is_gladiator": True,
				"last_mu": np.copy(winner.get("last_mu", obs_normalizer.mean)),
				"last_var": np.copy(winner.get("last_var", obs_normalizer.var)),
			}
			clone["model"].set_weights(winner["model"].get_weights())
			self._records[clone_id] = clone
		return True

	def __eco_niche_protection(self, winner, current_trust, current_z_score, recognizer, obs_normalizer):
		"""STEP 6: SAME-NICHE REPLACEMENT — evict weakest in winner's niche if winner is better."""
		current_niche_id = recognizer.current_niche_id
		niche_records = [
			individual for individual in self._records.values()
			if individual.get("internal_env_id") == current_niche_id
		]
		if not niche_records:
			return
		weakest = min(niche_records, key=lambda individual: individual.get("trust"))
		if current_trust > weakest.get("trust"):
			del self._records[weakest["individual_id"]]
			self.__save(winner, current_trust, current_z_score, recognizer, obs_normalizer)

	def __genetic_diversity(self, winner, current_trust, current_z_score, recognizer, obs_normalizer):
		"""STEP 4: NICHE PROTECTION (Genetic Similarity)"""
		current_env_id = recognizer.current_niche_id
		for hof_id, hof_record in list(self._records.items()):
			# Only check genetic similarity within the SAME niche
			if hof_record.get("internal_env_id", -1) != current_env_id:
				continue

			similarity = self.__calculate_genetic_similarity(winner, hof_record)
			if similarity > self.SIMILARITY_THRESHOLD:
				existing_trust = hof_record.get("trust", -99)
				if current_trust > existing_trust:
					del self._records[hof_id]
					self.__save(winner, current_trust, current_z_score, recognizer, obs_normalizer)
				# Whether we replaced or not, twin found — exit
				return True
		return False

	def __memory_decay(self):
		"""STEP 1: MEMORY DECAY"""
		for item in self._records.values():
			item["induction_z_score"] = (
				item.get("induction_z_score", 0) * self.MEMORY_DECAY_RATE
			)

	def __performance_gate(self, current_trust, population, plasticity_history):
		"""STEP 2: CALCULATE CONTEXT & METRICS"""
		current_trusts = [guy["trust"] for guy in population]
		avg_trust = np.mean(current_trusts)
		# std_trust floor prevents Z-score explosions on uniform populations
		std_trust = max(np.std(current_trusts), 0.1)

		current_z_score = (current_trust - avg_trust) / std_trust

		current_plasticity = plasticity_history[-1] if plasticity_history else 1.0
		z_score_required = 1.5 * (1.0 - (current_plasticity * 0.5))

		# Early exit: not impressive enough relative to current peers
		if current_z_score < z_score_required and current_trust < 0.5:
			return None
		return current_z_score

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __save(self, individual, trust, z_score, recognizer, obs_normalizer):
		"""Deep copy of a specialist into the Hall of Fame."""
		backup = {}
		backup["model"] = self._brain(self._action_dim, self._state_dim)
		backup["model"].set_weights(individual["model"].get_weights())
		backup["individual_id"] = individual["individual_id"]
		backup["trust"] = trust
		backup["induction_z_score"] = z_score
		backup["env_name"] = individual.get("env_name", "Unknown")
		backup["internal_env_id"] = recognizer.current_niche_id
		backup["is_gladiator"] = True
		backup["last_mu"] = np.copy(individual.get("last_mu", obs_normalizer.mean))
		backup["last_var"] = np.copy(individual.get("last_var", obs_normalizer.var))
		self._records[individual["individual_id"]] = backup

		# Mark the live population member as a gladiator immediately.
		# Since dicts are passed by reference, this updates the active population slot directly.
		individual["is_gladiator"] = True
		individual["internal_env_id"] = recognizer.current_niche_id
		individual["env_name"] = backup["env_name"]

	def __slot_induction(self, winner, current_trust, current_z_score, recognizer, obs_normalizer):
		"""STEP 5: OPEN SLOT INDUCTION — induct if winner's niche is below budget."""
		current_niche_id = recognizer.current_niche_id
		niche_count = sum(
			1 for individual in self._records.values()
			if individual.get("internal_env_id") == current_niche_id
		)
		if niche_count < self.max_per_niche:
			self.__save(winner, current_trust, current_z_score, recognizer, obs_normalizer)
			return True
		return False
