"""Failure-signal driver interface: an episode's reward mapped to a factor in [0, 1]."""

# General imports
from abc import ABC, abstractmethod

# App imports
from agents.ena.memory import AgentMemory


class ADriver(ABC):  # pylint: disable=too-few-public-methods
	"""Map a reward to a factor in [0, 1]: Trust's decay rate, or Plasticity's search effort."""

	@abstractmethod
	def value(self, reward: float, memory: AgentMemory) -> float:
		"""Return the factor for the given reward."""
