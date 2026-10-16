"""Mutation skill: perturbs an individual's weights via a pluggable driver."""

# App imports
from agents.ena.tools.driver_loader import load_driver


class Mutation:  # pylint: disable=too-few-public-methods
	"""Resolves and delegates to the configured mutation driver."""

	# Usage: Mutation("gaussian", mutation_rate=0.05, mutation_noise=0.1).apply(parent, target, plasticity, default_trust)

	def __init__(self, driver: str, **kwargs: object) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		self._instance = load_driver("mutation", driver, **kwargs)

	def apply(self, parent: dict, target: dict, plasticity: float, default_trust: float) -> None:
		"""Mutate parent's weights into target, scaled by the given plasticity."""
		self._instance.apply(parent, target, plasticity, default_trust)
