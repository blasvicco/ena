"""Selection driver interface: picks which population member acts each episode."""

# General imports
from abc import ABC, abstractmethod

# Libs imports
import numpy as np

# App imports
from agents.ena.memory import AgentMemory
from agents.ena.skills.env_detection.main import EnvironmentRecognizer


class ADriver(ABC):
	"""Picks which population member acts each episode: epsilon-greedy by trust in training, the driver's own rule at test time."""

	# Training selection is shared by every driver, so drivers differ only at test time and a
	# seeded training run is identical whichever driver is configured.

	def __init__(self, exploration_rate: float = 0.1) -> None:
		self._exploration_rate = exploration_rate

	def select_individual(self, population: list, recognizer: EnvironmentRecognizer | None, memory: AgentMemory, learn: bool) -> int:
		"""Return the population index of the selected individual."""
		if learn:
			return self._select_train(population)
		return self._select_test(population, recognizer, memory)

	def record_episode_result(self, score: float, memory: AgentMemory, recognizer: EnvironmentRecognizer | None = None) -> None:
		"""Called at episode end during testing; drivers that learn from test returns override it."""

	@abstractmethod
	def evaluate(self, population: list) -> None:
		"""Rebuild the test-time candidates from the population's gladiators."""

	@abstractmethod
	def _select_test(self, population: list, recognizer, memory) -> int:
		"""Test time: the population index of the individual to play."""

	def _select_train(self, population: list) -> int:
		"""Training: epsilon-greedy — a random non-gladiator with probability exploration_rate, else the highest trust."""
		if np.random.rand() < self._exploration_rate:
			non_gladiators = [index for index, ind in enumerate(population) if not ind.get("is_gladiator", False)]
			if non_gladiators:
				return int(np.random.choice(non_gladiators))
		trust_scores = [ind["trust"] for ind in population]
		return int(np.argmax(trust_scores))

	@staticmethod
	def _select_desperate(population: list) -> int:
		"""Test time without any gladiator: a random non-gladiator."""
		non_gladiators = [index for index, ind in enumerate(population) if not ind.get("is_gladiator", False)]
		return int(np.random.choice(non_gladiators)) if non_gladiators else 0

	@staticmethod
	def _gladiators(population: list) -> list:
		"""[(population index, home niche, trust)] of every gladiator bound to a niche."""
		return [
			(index, ind["internal_env_id"], ind.get("trust", 0.0))
			for index, ind in enumerate(population)
			if ind.get("is_gladiator", False) and ind.get("internal_env_id", -1) >= 0
		]
