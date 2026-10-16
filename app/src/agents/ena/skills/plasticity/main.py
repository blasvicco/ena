"""Plasticity skill: computes a search-effort factor via a pluggable driver."""

# App imports
from agents.ena.tools.driver_loader import load_driver
from agents.ena.memory import AgentMemory


class Plasticity:  # pylint: disable=too-few-public-methods
	"""Resolves and delegates to the configured plasticity driver."""

	# Usage: Plasticity("linear_ramp").compute(reward, memory, learn=True)

	def __init__(self, driver: str, **kwargs: object) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		self._instance = load_driver(
			"plasticity",
			driver,
			packages=("agents.ena.skills.plasticity.driver", "agents.ena.skills.failure_signal.driver"),
			**kwargs,
		)

	def compute(self, reward: float, memory: AgentMemory, learn: bool) -> float:
		"""Return the plasticity factor for the given reward; forced to zero when not learning."""
		# No exploratory search effort during evaluation, regardless of which driver is
		# configured — independent of the configured driver, so gated here rather than
		# duplicated across every driver.
		if not learn:
			return 0.0
		return self._instance.value(reward, memory)
