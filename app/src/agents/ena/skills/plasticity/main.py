"""Plasticity skill: computes a search-effort factor via a pluggable driver."""

# General imports
import importlib


class Plasticity:  # pylint: disable=too-few-public-methods
	"""Resolves and delegates to the configured plasticity driver."""

	# Usage: Plasticity("linear_ramp").compute(reward, memory, learn=True)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.plasticity.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown plasticity driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	def compute(self, reward: float, memory, learn: bool) -> float:
		"""Return the plasticity factor for the given reward; forced to zero when not learning."""
		# No exploratory search effort during evaluation, regardless of which driver is
		# configured — independent of the configured driver, so gated here rather than
		# duplicated across every driver.
		if not learn:
			return 0.0
		return self._instance.compute(reward, memory)
