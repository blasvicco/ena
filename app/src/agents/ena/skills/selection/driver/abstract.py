"""Selection driver interface: picks which population member acts each episode."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):
	"""Picks which population member acts each episode, in training and at test time."""

	@abstractmethod
	def select_individual(self, population: list, recognizer, memory, learn: bool) -> int:
		"""Returns population index of the selected individual."""

	@abstractmethod
	def record_episode_result(self, score: float, memory, recognizer=None) -> None:
		"""Called at episode end during testing."""

	@abstractmethod
	def evaluate(self, population) -> None:
		"""Rebuild the test-time candidates from the population's gladiators."""
