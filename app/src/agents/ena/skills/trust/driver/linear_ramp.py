"""Trust driver: percentile-scaled linear ramp."""

# App imports
from agents.ena.skills.trust.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Linear ramp: high reward -> low decay, rising linearly down to memory.min_seen."""

	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return curves.linear_ramp(reward, memory)
