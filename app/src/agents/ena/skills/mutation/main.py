"""Mutation skill: perturbs an individual's weights via a pluggable driver."""

# General imports
import importlib


class Mutation:  # pylint: disable=too-few-public-methods
	"""Resolves and delegates to the configured mutation driver."""

	# Usage: Mutation("gaussian", mutation_rate=0.05, mutation_noise=0.1).apply(parent, target, plasticity, default_trust)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.mutation.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown mutation driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	def apply(self, parent: dict, target: dict, plasticity: float, default_trust: float) -> None:
		"""Mutate parent's weights into target, scaled by the given plasticity."""
		self._instance.apply(parent, target, plasticity, default_trust)
