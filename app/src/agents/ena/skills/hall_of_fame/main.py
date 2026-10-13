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

	@property
	def holds_olympics(self) -> bool:
		"""Whether the driver re-evaluates its gladiators in Olympiads (driver/olympic.py)."""
		return self._instance.holds_olympics

	def olympics_due(self, total_env_steps: int) -> bool:
		"""Whether an Olympiad is due."""
		return self._instance.olympics_due(total_env_steps)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def hold_olympics(self, niche_id: int, play, trust_of, eviction_bar: float, population: list, total_env_steps: int) -> list:
		"""Hold an Olympiad of a niche's gladiators (play/trust_of are the agent's); one row per competitor."""
		return self._instance.hold_olympics(niche_id, play, trust_of, eviction_bar, population, total_env_steps)

	def admits(self, individual: dict) -> bool:
		"""Whether a gladiator may play at test."""
		return self._instance.admits(individual)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def update(self, winner: dict, current_trust: float, population: list, memory, recognizer, obs_normalizer, env_name: str = "Unknown") -> None:
		"""Attempt to induct winner into the archive; env_name is the environment it just succeeded in."""
		self._instance.update(winner, current_trust, population, memory, recognizer, obs_normalizer, env_name=env_name)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def inject_gladiators(self, population: list, pop_size: int, obs_normalizer, restore_trust: bool = True, admit=None) -> None:
		"""Re-introduce archived gladiators (those admit(record) accepts, if given) into the active population."""
		self._instance.inject_gladiators(population, pop_size, obs_normalizer, restore_trust=restore_trust, admit=admit)

	def records_in_niche(self, niche_id: int) -> list:
		"""The archived records of one niche."""
		return self._instance.records_in_niche(niche_id)

	def record(self, individual_id: str):
		"""One archived record, or None."""
		return self._instance.record(individual_id)

	def evict(self, individual_id: str, population: list) -> None:
		"""Remove a record from the archive and clear its live copy's gladiator tags."""
		self._instance.evict(individual_id, population)

	def save_niche_details(self, output_dir: str, agent_name: str) -> None:
		"""Write a niche/gladiator-count summary file."""
		self._instance.save_niche_details(output_dir, agent_name)
