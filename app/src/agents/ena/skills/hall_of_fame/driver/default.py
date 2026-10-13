"""Default Hall of Fame driver — the real archive-and-induction pipeline."""

# General imports
import os
import random
import uuid

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.hall_of_fame.driver.abstract import ADriver
from agents.ena.tools.genetic_similarity import genetic_similarity


class Instance(ADriver):  # pylint: disable=too-many-instance-attributes
	"""Manages the archive of elite specialists (Gladiators)."""

	# Responsibilities:
	# - Z-score based induction with configurable threshold
	# - Per-niche budget: each recognised niche holds up to max_per_niche gladiators
	# - Niche protection via cosine similarity (no strategic twins)
	# - Retention: a record is displaced only by a strictly higher-trust candidate of its own niche
	# - Population injection: round-robin across niches, capped at max_gladiator_ratio * pop_size

	SIMILARITY_THRESHOLD = 0.95
	# Induction gate: z = (T - mean) / max(std, GATE_SPREAD_FLOOR) >= GATE_Z * (1 - GATE_PLASTICITY_RELIEF * P), or T >= GATE_TRUST.
	GATE_Z = 1.5
	GATE_PLASTICITY_RELIEF = 0.5
	GATE_SPREAD_FLOOR = 0.1
	GATE_TRUST = 0.5

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(self, max_per_niche, max_gladiator_ratio, brain, action_dim, state_dim, **_kwargs):
		# max_per_niche:       max gladiators stored per recognised niche
		# max_gladiator_ratio: fraction of pop_size that can be gladiators at injection time
		# protect_confirmed:   (the olympic driver) only an Olympiad removes or re-scores a gladiator that
		#                      competed (olympic_confirmed): a new winner's trust rests on one episode, a
		#                      confirmed gladiator's on its Olympic mean, so the archive's own replacement
		#                      steps would let a lucky winner displace it.
		# qualifier_seats:     (the olympic driver) extra seats per niche for provisional entrants; the
		#                      Olympics trim confirmed gladiators to max_per_niche, so these seats are
		#                      never held by one. Ignored without protection (capacity max_per_niche).
		# aligned_statistics: (the olympic driver) a record stores the statistics its weights are aligned to (the
		#                     winner's own last_mu/last_var), not the normalizer's current ones; the default keeps
		#                     stamping the current statistics, as every CartPole run did.
		self.protect_confirmed = False
		self.qualifier_seats = 0
		self.aligned_statistics = False
		# reinject_as_gladiators: (the olympic driver) a re-injected record's copy is a gladiator, since every record is
		#                        one; the default copies the record's own flag, as every CartPole run did.
		self.reinject_as_gladiators = False
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
		admit=None,
	):
		"""Re-introduces Gladiators (those admit(record) accepts, if given) into the active population."""
		# Strategy: round-robin across niches (1 per niche per round), best-first within
		# each niche, until floor(pop_size * max_gladiator_ratio) population slots are filled.
		records = [record for record in self._records.values() if admit is None or admit(record)]
		if not records:
			return

		max_slots = int(pop_size * self.max_gladiator_ratio)
		if max_slots <= 0:
			return

		# Group by niche, randomly shuffle each niche so we try different gladiators
		niches = {}
		for individual in records:
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
			target["is_gladiator"] = True if self.reinject_as_gladiators else gladiator.get("is_gladiator", False)
			target["env_name"] = gladiator.get("env_name", "Unknown")
			target["internal_env_id"] = gladiator.get("internal_env_id", -1)
			target["last_mu"] = np.copy(gladiator.get("last_mu", obs_normalizer.mean))
			target["last_var"] = np.copy(gladiator.get("last_var", obs_normalizer.var))
			target["olympic_confirmed"] = gladiator.get("olympic_confirmed", False)
			target["olympic_return"] = gladiator.get("olympic_return")

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
		env_name="Unknown",
	):
		"""Attempt to induct winner into the Hall of Fame."""
		# env_name is the environment the winner just succeeded in. It labels a new record
		# (first induction, or a cross-niche clone) and is never written onto an existing one —
		# a gladiator's label is fixed at induction.
		if not self.__passes_performance_gate(current_trust, population, memory.plasticity_history):
			return

		# Late Identity Binding: Determine the niche ID for this successful individual
		recognizer.match_niche(memory, winner, self, population=population)

		if self.__continual_improvement(winner, current_trust, recognizer, obs_normalizer, env_name):
			return

		if self.__genetic_diversity(
			winner,
			current_trust,
			recognizer,
			obs_normalizer,
			population,
			env_name,
		):
			return

		if self.__slot_induction(winner, current_trust, recognizer, obs_normalizer, env_name):
			return

		self.__eco_niche_protection(
			winner,
			current_trust,
			recognizer,
			obs_normalizer,
			population,
			env_name,
		)

	# Private
	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __continual_improvement(self, winner, current_trust, recognizer, obs_normalizer, env_name):
		"""STEP 3: CHECK FOR EXISTING ID"""
		individual_id = winner["individual_id"]
		if individual_id not in self._records:
			return False

		existing = self._records[individual_id]
		current_env_id = recognizer.current_niche_id
		existing_env_id = existing.get("internal_env_id", -1)

		if current_env_id == existing_env_id:
			# Same niche: update in place if strictly better. env_id and env_name stay frozen.
			if self.__protected(existing):
				return True
			if current_trust > existing.get("trust", -99):
				existing["model"].set_weights(winner["model"].get_weights())
				existing["trust"] = current_trust
				self.__stamp(existing, winner, obs_normalizer)
				winner["internal_env_id"] = current_env_id
				self.__stamp(winner, winner, obs_normalizer)
			return True

		# Cross-niche: winner is already a gladiator from a different niche.
		# The original keeps its niche and label; a clone joins this niche, labeled with
		# the environment it just succeeded in. If this niche already holds a genetic twin
		# (typically an earlier clone of this same gladiator), refresh that one instead, so
		# repeated foreign wins don't fill the niche with copies of one individual.
		niche_records = [individual for individual in self._records.values() if individual.get("internal_env_id") == current_env_id]
		for record in niche_records:
			if genetic_similarity(winner, record) > self.SIMILARITY_THRESHOLD:
				if current_trust > record.get("trust", -99) and not self.__protected(record):
					record["model"].set_weights(winner["model"].get_weights())
					record["trust"] = current_trust
					self.__stamp(record, winner, obs_normalizer)
				return True
		if len(niche_records) < self.__capacity():
			clone_id = str(uuid.uuid4())
			clone = {
				"model": self._brain(self._action_dim, self._state_dim),
				"individual_id": clone_id,
				"trust": current_trust,
				"env_name": env_name,
				"internal_env_id": current_env_id,
				"is_gladiator": True,
			}
			self.__stamp(clone, winner, obs_normalizer)
			clone["model"].set_weights(winner["model"].get_weights())
			self._records[clone_id] = clone
		return True

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __eco_niche_protection(self, winner, current_trust, recognizer, obs_normalizer, population, env_name):
		"""STEP 6: SAME-NICHE REPLACEMENT — evict weakest in winner's niche if winner is better."""
		current_niche_id = recognizer.current_niche_id
		niche_records = [
			individual
			for individual in self._records.values()
			if individual.get("internal_env_id") == current_niche_id and not self.__protected(individual)
		]
		if not niche_records:
			return
		weakest = min(niche_records, key=lambda individual: individual.get("trust"))
		if current_trust > weakest.get("trust"):
			self.__evict(weakest["individual_id"], population)
			self.__save(winner, current_trust, recognizer, obs_normalizer, env_name)

	def records_in_niche(self, niche_id):
		"""The archived records of one niche (the Olympics' competitors)."""
		return [record for record in self._records.values() if record.get("internal_env_id") == niche_id]

	def record(self, individual_id):
		"""One archived record, or None."""
		return self._records.get(individual_id)

	def evict(self, individual_id, population):
		"""Remove a record and clear its live copy's gladiator tags (the Olympics' eviction)."""
		self.__evict(individual_id, population)

	def __evict(self, individual_id, population):
		"""Remove a record from the archive and clear its live population member's gladiator tags."""
		# Without this, a population slot evicted from the archive keeps competing in
		# evaluate() and staying protected in __evolve_population() as a "zombie
		# gladiator" invisible to niches_*.txt.
		del self._records[individual_id]
		for individual in population:
			if individual.get("individual_id") == individual_id:
				individual["is_gladiator"] = False
				individual["internal_env_id"] = -1
				individual["env_name"] = "Unknown"
				if self.protect_confirmed:
					individual["olympic_confirmed"] = False
				break

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __genetic_diversity(self, winner, current_trust, recognizer, obs_normalizer, population, env_name):
		"""STEP 4: NICHE PROTECTION (Genetic Similarity)"""
		current_env_id = recognizer.current_niche_id
		for hof_id, hof_record in list(self._records.items()):
			# Only check genetic similarity within the SAME niche
			if hof_record.get("internal_env_id", -1) != current_env_id:
				continue

			similarity = genetic_similarity(winner, hof_record)
			if similarity > self.SIMILARITY_THRESHOLD:
				if self.__protected(hof_record):
					# The twin waits for the next Olympiad beside it, provisional, if the niche has room.
					self.__slot_induction(winner, current_trust, recognizer, obs_normalizer, env_name)
					return True
				existing_trust = hof_record.get("trust", -99)
				if current_trust > existing_trust:
					self.__evict(hof_id, population)
					self.__save(winner, current_trust, recognizer, obs_normalizer, env_name)
				# Whether we replaced or not, twin found — exit
				return True
		return False

	def __passes_performance_gate(self, current_trust, population, plasticity_history) -> bool:
		"""STEP 1: PERFORMANCE GATE — the winner's trust stands out (z-score, bar lowered by plasticity) or is high on its own."""
		current_trusts = [guy["trust"] for guy in population]
		avg_trust = np.mean(current_trusts)
		# std_trust floor prevents Z-score explosions on uniform populations
		std_trust = max(np.std(current_trusts), self.GATE_SPREAD_FLOOR)

		current_z_score = (current_trust - avg_trust) / std_trust

		current_plasticity = plasticity_history[-1] if plasticity_history else 1.0
		z_score_required = self.GATE_Z * (1.0 - (current_plasticity * self.GATE_PLASTICITY_RELIEF))

		return current_z_score >= z_score_required or current_trust >= self.GATE_TRUST

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __save(self, individual, trust, recognizer, obs_normalizer, env_name):
		"""Deep copy of a specialist into the Hall of Fame."""
		backup = {}
		backup["model"] = self._brain(self._action_dim, self._state_dim)
		backup["model"].set_weights(individual["model"].get_weights())
		backup["individual_id"] = individual["individual_id"]
		backup["trust"] = trust
		backup["env_name"] = env_name
		backup["internal_env_id"] = recognizer.current_niche_id
		self.__stamp(backup, individual, obs_normalizer)
		self._records[individual["individual_id"]] = backup

		# Mark the live population member as a gladiator immediately.
		# Since dicts are passed by reference, this updates the active population slot directly.
		individual["is_gladiator"] = True
		individual["internal_env_id"] = recognizer.current_niche_id
		individual["env_name"] = backup["env_name"]
		self.__stamp(individual, individual, obs_normalizer)
		if self.protect_confirmed:
			individual["olympic_confirmed"] = False  # a new record is provisional until its first Olympiad

	def __capacity(self) -> int:
		"""Records a niche may hold: max_per_niche, plus the qualifier seats when confirmed gladiators are protected."""
		return self.max_per_niche + (self.qualifier_seats if self.protect_confirmed else 0)

	def __stamp(self, target: dict, source: dict, obs_normalizer) -> None:
		"""Set target's normalization snapshot: source's own (aligned_statistics), else the normalizer's current one."""
		if self.aligned_statistics and source.get("last_mu") is not None:
			mean, var = source["last_mu"], source["last_var"]
		else:
			mean, var = obs_normalizer.mean, obs_normalizer.var
		target["last_mu"], target["last_var"] = np.copy(mean), np.copy(var)

	def __protected(self, record) -> bool:
		"""Whether only an Olympiad may remove or re-score this record."""
		return self.protect_confirmed and bool(record.get("olympic_confirmed", False))

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __slot_induction(self, winner, current_trust, recognizer, obs_normalizer, env_name):
		"""STEP 5: OPEN SLOT INDUCTION — induct if winner's niche is below budget."""
		current_niche_id = recognizer.current_niche_id
		niche_count = sum(1 for individual in self._records.values() if individual.get("internal_env_id") == current_niche_id)
		if niche_count < self.__capacity():
			self.__save(winner, current_trust, recognizer, obs_normalizer, env_name)
			return True
		return False
