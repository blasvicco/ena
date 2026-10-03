"""Hall of Fame skill: archives elite specialists via a pluggable driver."""

# General imports
import importlib


class HallOfFame:
	"""Resolves and delegates to the configured Hall of Fame driver."""

	# Usage: HallOfFame("default", max_per_niche=5, max_gladiator_ratio=0.5, brain=brain, action_dim=2, state_dim=4)
	#        .update(winner, current_trust, population, memory, recognizer, obs_normalizer)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.hall_of_fame.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown hall_of_fame driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	def __bool__(self) -> bool:
		"""True when at least one gladiator has been inducted."""
		return bool(self._instance)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def update(self, winner: dict, current_trust: float, population: list, memory, recognizer, obs_normalizer, env_name: str = "Unknown") -> None:
		"""Attempt to induct winner into the archive; env_name is the environment it just succeeded in."""
		self._instance.update(winner, current_trust, population, memory, recognizer, obs_normalizer, env_name=env_name)

	def inject_gladiators(self, population: list, pop_size: int, obs_normalizer, restore_trust: bool = True) -> None:
		"""Re-introduce archived gladiators into the active population."""
		self._instance.inject_gladiators(population, pop_size, obs_normalizer, restore_trust=restore_trust)

	def save_niche_details(self, output_dir: str, agent_name: str) -> None:
		"""Write a niche/gladiator-count summary file."""
		self._instance.save_niche_details(output_dir, agent_name)
