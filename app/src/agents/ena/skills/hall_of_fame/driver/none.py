"""No-op Hall of Fame driver — disables specialist archiving/recall entirely."""

# General imports
import os

# App imports
from agents.ena.skills.hall_of_fame.driver.abstract import ADriver


class Instance(ADriver):
	"""Archives nothing and injects nothing — every call is a no-op."""

	# Accepts and ignores whatever construction kwargs the default driver would need, so
	# ENAgent can build whichever hall_of_fame driver is configured without special-casing
	# the call site.

	def __init__(self, **_kwargs) -> None:
		pass

	def __bool__(self) -> bool:
		"""Never true — no gladiator is ever inducted."""
		return False

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def update(self, winner: dict, current_trust: float, population: list, memory, recognizer, obs_normalizer, env_name: str = "Unknown") -> None:
		"""No-op: this driver never inducts a winner."""

	def inject_gladiators(self, population: list, pop_size: int, obs_normalizer, restore_trust: bool = True, admit=None) -> None:
		"""No-op: there is nothing archived to re-introduce."""

	def records_in_niche(self, niche_id: int) -> list:
		"""Nothing is archived."""
		return []

	def record(self, individual_id: str):
		"""Nothing is archived."""
		return None

	def evict(self, individual_id: str, population: list) -> None:
		"""No-op: nothing is archived."""

	def save_niche_details(self, output_dir: str, agent_name: str) -> None:
		"""Write the same 'No gladiators inducted' summary the default driver writes when empty."""
		file_path = os.path.join(output_dir, f"niches_{agent_name}.txt")
		with open(file_path, "w", encoding="utf-8") as file:
			file.write(f"{'Niche ID':<10} | {'Env Name':<20} | {'Total Gladiators':<15}\n")
			file.write("-" * 51 + "\n")
			file.write("No gladiators inducted.\n")
