"""Plasticity driver: aggressive quadratic curve using global benchmarks."""

# App imports
from agents.ena.skills.plasticity.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Aggressive plasticity: squared normalized error against the run's best/worst scores."""

	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
		return curves.quadratic(reward, memory)
