"""Selection skill: picks which population member acts each episode via a pluggable driver."""

# General imports
import importlib


class SelectionController:
	"""Resolves and delegates all selection logic to a pluggable driver."""

	# Usage: SelectionController("bpr", exploration_rate=0.2, return_scale=475.0).select_individual(population, recognizer, memory, learn=True)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.selection.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown selection driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	# ---------------------------------------------------------------
	# Delegated public API
	# ---------------------------------------------------------------

	def select_individual(self, population: list, recognizer, memory, learn: bool) -> int:
		"""Returns population index of the selected individual."""
		return self._instance.select_individual(population, recognizer, memory, learn)

	def on_episode_end(self, memory, learn: bool, env_detection) -> None:
		"""Test time only: let the driver learn from the episode's result."""
		if not learn:
			self.record_episode_result(memory.last_episode_score, memory, recognizer=env_detection)

	def record_episode_result(self, score: float, memory, recognizer=None) -> None:
		"""Called at episode end during testing."""
		self._instance.record_episode_result(score, memory, recognizer=recognizer)

	def evaluate(self, population: list) -> None:
		"""Rebuild the driver's test-time candidates from the population's gladiators."""
		self._instance.evaluate(population)
