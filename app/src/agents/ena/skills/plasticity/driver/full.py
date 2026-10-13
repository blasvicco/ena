"""Plasticity driver: always full (maximal search effort)."""

# App imports
from agents.ena.skills.plasticity.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always returns one — maximal search effort."""

	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
		return curves.full(reward, memory)
