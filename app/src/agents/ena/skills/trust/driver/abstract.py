"""Trust driver interface: computes the trust-decay factor for a given reward."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):  # pylint: disable=too-few-public-methods
	"""Computes a trust-decay factor (EMA blend weight) from a reward."""

	@abstractmethod
	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
