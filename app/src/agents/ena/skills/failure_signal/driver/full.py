"""Failure-signal driver: the full curve of tools/curves."""

# App imports
from agents.ena.memory import AgentMemory
from agents.ena.skills.failure_signal.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always one: trust tracks the latest reward, maximal search effort."""

	def value(self, reward: float, memory: AgentMemory) -> float:
		"""Return the factor for the given reward."""
		return curves.full(reward, memory)
