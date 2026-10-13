"""Trust driver: always zero decay (no trust update)."""

# App imports
from agents.ena.skills.trust.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always returns zero — trust never decays."""

	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return curves.zero(reward, memory)
