"""Plasticity driver interface: computes the search-effort factor for a given reward."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):  # pylint: disable=too-few-public-methods
	"""Computes a plasticity factor (search-depth/mutation-intensity scale) from a reward."""

	@abstractmethod
	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
