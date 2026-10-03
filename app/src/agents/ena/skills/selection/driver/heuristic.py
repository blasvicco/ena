"""Deterministic heuristic selection driver: evaluation phases + trust-based selection."""

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.selection.driver.abstract import ADriver


# pylint: disable=too-many-instance-attributes
class Instance(ADriver):
	"""Deterministic heuristic driver: evaluation phases + trust-based selection."""

	# Training: epsilon-greedy trust selection (picks who plays the main episode).
	# Testing:  EVALUATING → LOCKED → DESPERATE state machine.

	EVALUATING = "evaluating"
	LOCKED = "locked"
	DESPERATE = "desperate"

	def __init__(self, exploration_rate=0.1, **_kwargs):
		self._exploration_rate = exploration_rate

		# Test-time state machine
		self._state = self.LOCKED
		self._eval_queue = []  # [(niche_id, gladiator_idx), ...]
		self._eval_scores = {}  # {niche_id: score}
		self._eval_confirmed = {}  # {niche_id: bool} — recognizer confirmed its own episode
		self._current_eval_niche = None
		self._current_eval_idx = None
		self._locked_niche = None
		self._locked_idx = None

	# ---------------------------------------------------------------
	# Public API
	# ---------------------------------------------------------------

	def reset_episode(self):
		"""Called at episode start. No per-episode state to reset."""

	def select_individual(self, population, recognizer, memory, learn):
		"""Returns population index of the selected individual."""
		if learn:
			return self._select_train(population)
		return self._select_test(population, recognizer, memory)

	def on_step(self, memory, env_detection, learn):
		"""No-op: heuristic driver does not record REINFORCE experiences mid-episode."""

	def on_learn_episode_end(self, memory, env_detection, env_max_steps):
		"""No-op: heuristic driver has nothing to backfill or update."""

	def record_episode_result(self, score, memory, recognizer=None):
		"""Called at episode end during testing. Returns True if env reset needed."""
		if self._state == self.EVALUATING:
			if self._current_eval_niche is not None:
				self._eval_scores[self._current_eval_niche] = score
				if recognizer is not None:
					matched_niche_id, _sim = recognizer.match(memory)
					self._eval_confirmed[self._current_eval_niche] = matched_niche_id == self._current_eval_niche

			if self._eval_queue:
				self._eval_queue.pop(0)

			self._current_eval_niche = None
			self._current_eval_idx = None

			if not self._eval_queue:
				if self._eval_scores:
					confirmed = [niche_id for niche_id in self._eval_scores if self._eval_confirmed.get(niche_id)]
					# Prefer a niche whose own physics identity was independently confirmed by
					# recognizer.match() during its evaluation episode — guards against a
					# "generalist" gladiator winning on raw score alone under the wrong physics.
					candidates = confirmed if confirmed else self._eval_scores
					best_niche = max(candidates, key=self._eval_scores.get)
					self._locked_niche = best_niche
					self._locked_idx = None  # resolved lazily in _select_test
					self._state = self.LOCKED
				else:
					self._state = self.DESPERATE
			return False

		if self._state == self.LOCKED:
			if len(memory.reward_history) >= 20:
				if memory.performance_derivative < -100:
					return True
			return False

		return False

	def evaluate(self, population):
		"""Build evaluation queue from population's gladiator niches."""
		niche_gladiators = {}
		for index, ind in enumerate(population):
			if ind.get("is_gladiator", False):
				niche_id = ind.get("internal_env_id", -1)
				if niche_id >= 0:
					if niche_id not in niche_gladiators:
						niche_gladiators[niche_id] = []
					niche_gladiators[niche_id].append(index)

		self._eval_queue = []
		for niche_id, indices in niche_gladiators.items():
			chosen = int(np.random.choice(indices))
			self._eval_queue.append((niche_id, chosen))

		np.random.shuffle(self._eval_queue)

		self._eval_scores = {}
		self._eval_confirmed = {}
		self._current_eval_niche = None
		self._current_eval_idx = None
		self._locked_niche = None
		self._locked_idx = None
		self._state = self.EVALUATING if self._eval_queue else self.DESPERATE

	def set_mode(self, memory, mode):
		"""Switch between train (mode=True) and test (mode=False)."""

	def record(self, memory, context_vector, niche_id, reward):
		"""No-op: heuristic driver does not record REINFORCE experiences."""

	def update(self, memory):
		"""No-op: heuristic driver does not do REINFORCE updates."""

	# ---------------------------------------------------------------
	# Private — Training
	# ---------------------------------------------------------------

	def _select_train(self, population):
		"""Training: epsilon-greedy trust selection."""
		if np.random.rand() < self._exploration_rate:
			non_gladiators = [index for index, ind in enumerate(population) if not ind.get("is_gladiator", False)]
			if non_gladiators:
				return int(np.random.choice(non_gladiators))

		trust_scores = [ind["trust"] for ind in population]
		return int(np.argmax(trust_scores))

	# ---------------------------------------------------------------
	# Private — Testing
	# ---------------------------------------------------------------

	# pylint: disable=too-many-return-statements
	def _select_test(self, population, recognizer, memory):
		"""Testing: state-machine selection."""
		# The EVALUATING queue always runs to completion first — it's the only way we
		# gather a fresh score per niche. recognizer.match() is only consulted once
		# LOCKED, where a confident match can resolve to a different, already-known
		# niche without waiting for the next evaluate() cycle.
		if self._state == self.EVALUATING:
			if self._current_eval_idx is not None:
				return self._current_eval_idx
			if self._eval_queue:
				niche_id, glad_idx = self._eval_queue[0]
				self._current_eval_niche = niche_id
				self._current_eval_idx = glad_idx
				return glad_idx
			self._state = self.DESPERATE
			return self._select_desperate(population)

		if self._state == self.LOCKED:
			rec_niche_id, _rec_sim = recognizer.match(memory)
			if rec_niche_id != -1:
				gladiator_idx = self._best_gladiator_in_niche(population, rec_niche_id)
				if gladiator_idx is not None:
					return gladiator_idx

			if self._locked_idx is not None:
				return self._locked_idx
			if self._locked_niche is not None:
				self._locked_idx = self._best_gladiator_in_niche(population, self._locked_niche)
				return self._locked_idx
			return self._best_gladiator_overall(population)

		return self._select_desperate(population)

	@staticmethod
	def _select_desperate(population):
		"""Pick a random non-gladiator."""
		non_gladiators = [index for index, ind in enumerate(population) if not ind.get("is_gladiator", False)]
		return int(np.random.choice(non_gladiators)) if non_gladiators else 0

	@staticmethod
	def _best_gladiator_in_niche(population, niche_id):
		"""Return index of the highest-trust gladiator for the given niche."""
		slots = [index for index, ind in enumerate(population) if ind.get("internal_env_id") == niche_id and ind.get("is_gladiator", False)]
		if slots:
			return max(slots, key=lambda index: population[index]["trust"])
		return Instance._best_gladiator_overall(population)

	@staticmethod
	def _best_gladiator_overall(population):
		"""Return index of the highest-trust gladiator in the population."""
		gladiators = [index for index, ind in enumerate(population) if ind.get("is_gladiator", False)]
		return max(gladiators, key=lambda index: population[index]["trust"]) if gladiators else 0
