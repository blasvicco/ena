"""Hall of Fame driver interface: archives elite specialists and re-introduces them."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):
	"""Archives elite specialists ("Gladiators") and re-introduces them into the population."""

	@abstractmethod
	def __bool__(self) -> bool:
		"""True when at least one gladiator has been inducted."""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	@abstractmethod
	def update(self, winner: dict, current_trust: float, population: list, memory, recognizer, obs_normalizer, env_name: str = "Unknown") -> None:
		"""Attempt to induct winner into the archive; env_name is the environment it just succeeded in."""

	@abstractmethod
	def inject_gladiators(self, population: list, pop_size: int, obs_normalizer, restore_trust: bool = True) -> None:
		"""Re-introduce archived gladiators into the active population."""

	@abstractmethod
	def save_niche_details(self, output_dir: str, agent_name: str) -> None:
		"""Write a niche/gladiator-count summary file."""
