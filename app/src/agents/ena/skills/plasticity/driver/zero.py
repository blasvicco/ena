"""Plasticity driver: always zero (no adaptive search effort)."""

# App imports
from agents.ena.skills.plasticity.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always returns zero — minimal search effort."""

	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
		return curves.zero(reward, memory)
