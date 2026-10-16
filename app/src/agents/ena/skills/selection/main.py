"""Selection skill: picks which population member acts each episode via a pluggable driver."""

# App imports
from agents.ena.tools.driver_loader import load_driver
from agents.ena.memory import AgentMemory
from agents.ena.skills.env_detection.main import EnvironmentRecognizer


class SelectionController:
	"""Resolves and delegates all selection logic to a pluggable driver."""

	# Usage: SelectionController("bpr", exploration_rate=0.2, return_scale=475.0).select_individual(population, recognizer, memory, learn=True)

	def __init__(self, driver: str, **kwargs: object) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		self._instance = load_driver("selection", driver, **kwargs)

	# ---------------------------------------------------------------
	# Delegated public API
	# ---------------------------------------------------------------

	def select_individual(self, population: list, recognizer: EnvironmentRecognizer | None, memory: AgentMemory, learn: bool) -> int:
		"""Return the population index of the selected individual."""
		return self._instance.select_individual(population, recognizer, memory, learn)

	def on_episode_end(self, memory: AgentMemory, learn: bool, env_detection: EnvironmentRecognizer) -> None:
		"""Test time only: let the driver learn from the episode's result."""
		if not learn:
			self.record_episode_result(memory.last_episode_score, memory, recognizer=env_detection)

	def record_episode_result(self, score: float, memory: AgentMemory, recognizer: EnvironmentRecognizer | None = None) -> None:
		"""Called at episode end during testing."""
		self._instance.record_episode_result(score, memory, recognizer=recognizer)

	def evaluate(self, population: list) -> None:
		"""Rebuild the driver's test-time candidates from the population's gladiators."""
		self._instance.evaluate(population)
