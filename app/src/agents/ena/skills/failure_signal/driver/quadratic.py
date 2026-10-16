"""Failure-signal driver: the quadratic curve of tools/curves."""

# App imports
from agents.ena.memory import AgentMemory
from agents.ena.skills.failure_signal.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Quadratic: the squared normalized error against the run's best and worst scores."""

	def value(self, reward: float, memory: AgentMemory) -> float:
		"""Return the factor for the given reward."""
		return curves.quadratic(reward, memory)
