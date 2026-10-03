"""Mutation driver interface: perturbs an individual's weights."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):  # pylint: disable=too-few-public-methods
	"""Mutates a population individual's weights."""

	@abstractmethod
	def apply(self, parent: dict, target: dict, plasticity: float, default_trust: float) -> None:
		"""Mutate parent's weights into target, scaled by the given plasticity."""
