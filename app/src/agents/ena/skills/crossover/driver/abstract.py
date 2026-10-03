"""Crossover driver interface: mixes two parent individuals' weights into a target."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):  # pylint: disable=too-few-public-methods
	"""Crosses two parent individuals' weights into a target individual."""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	@abstractmethod
	def apply(self, parent_1: dict, parent_2: dict, target: dict, obs_normalizer, max_eval_steps: float, default_trust: float) -> None:
		"""Mix parent_1's and parent_2's weights into target."""
