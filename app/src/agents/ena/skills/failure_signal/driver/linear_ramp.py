"""Failure-signal driver: the linear ramp curve of tools/curves."""

# App imports
from agents.ena.memory import AgentMemory
from agents.ena.skills.failure_signal.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Percentile-scaled linear ramp: a high reward gives a low value, rising linearly down to memory.min_seen."""

	def value(self, reward: float, memory: AgentMemory) -> float:
		"""Return the factor for the given reward."""
		return curves.linear_ramp(reward, memory)
