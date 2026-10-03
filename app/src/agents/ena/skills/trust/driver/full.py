"""Trust driver: always full decay (trust tracks the latest reward immediately)."""

# App imports
from agents.ena.skills.trust.driver.abstract import ADriver
from agents.ena.tools import curves


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Always returns one — trust fully tracks the latest reward."""

	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return curves.full(reward, memory)
