"""Trust driver: aggressive quadratic curve using global benchmarks."""

# App imports
from agents.ena.skills.trust.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Aggressive decay: squared normalized error against the run's best/worst scores."""

	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return curves.quadratic(reward, memory)
