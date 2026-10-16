"""Failure-signal driver: the zero curve of tools/curves."""

# App imports
from agents.ena.memory import AgentMemory
from agents.ena.skills.failure_signal.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always zero: no trust update, minimal search effort."""

	def value(self, reward: float, memory: AgentMemory) -> float:
		"""Return the factor for the given reward."""
		return curves.zero(reward, memory)
