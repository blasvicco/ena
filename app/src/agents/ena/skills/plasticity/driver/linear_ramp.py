"""Plasticity driver: percentile-scaled linear ramp."""

# App imports
from agents.ena.skills.plasticity.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Linear ramp: high reward -> low plasticity, rising linearly down to memory.min_seen."""

	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
		return curves.linear_ramp(reward, memory)
