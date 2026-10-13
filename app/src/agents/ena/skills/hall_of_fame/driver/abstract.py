"""Hall of Fame driver interface: archives elite specialists and re-introduces them."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):
	"""Archives elite specialists ("Gladiators") and re-introduces them into the population."""

	# Olympics (driver/olympic.py): periodic re-evaluation of the archive. Archives without them never hold one and
	# let every gladiator play at test, ENA's behavior before the Olympics.
	holds_olympics = False

	# pylint: disable=unused-argument
	def olympics_due(self, total_env_steps: int) -> bool:
		"""Whether an Olympiad is due (never, without the Olympics)."""
		return False

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def hold_olympics(self, niche_id: int, play, trust_of, eviction_bar: float, population: list, total_env_steps: int) -> list:
		"""Hold an Olympiad of a niche's gladiators; one row per competitor (none, without the Olympics)."""
		return []

	def admits(self, individual: dict) -> bool:
		"""Whether a gladiator may play at test (every one, without the Olympics)."""
		return True

	# pylint: enable=unused-argument

	@abstractmethod
	def __bool__(self) -> bool:
		"""True when at least one gladiator has been inducted."""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	@abstractmethod
	def update(self, winner: dict, current_trust: float, population: list, memory, recognizer, obs_normalizer, env_name: str = "Unknown") -> None:
		"""Attempt to induct winner into the archive; env_name is the environment it just succeeded in."""

	@abstractmethod
	def inject_gladiators(self, population: list, pop_size: int, obs_normalizer, restore_trust: bool = True, admit=None) -> None:
		"""Re-introduce archived gladiators (those admit(record) accepts, if given) into the active population."""

	@abstractmethod
	def records_in_niche(self, niche_id: int) -> list:
		"""The archived records of one niche."""

	@abstractmethod
	def record(self, individual_id: str):
		"""One archived record, or None."""

	@abstractmethod
	def evict(self, individual_id: str, population: list) -> None:
		"""Remove a record from the archive and clear its live copy's gladiator tags."""

	@abstractmethod
	def save_niche_details(self, output_dir: str, agent_name: str) -> None:
		"""Write a niche/gladiator-count summary file."""
