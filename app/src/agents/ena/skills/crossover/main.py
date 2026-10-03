"""Crossover skill: mixes two parent individuals' weights via a pluggable driver."""

# General imports
import importlib


class Crossover:  # pylint: disable=too-few-public-methods
	"""Resolves and delegates to the configured crossover driver."""

	# Usage: Crossover("uniform_mask").apply(parent_1, parent_2, target, obs_normalizer, max_eval_steps, default_trust)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.crossover.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown crossover driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def apply(self, parent_1: dict, parent_2: dict, target: dict, obs_normalizer, max_eval_steps: float, default_trust: float) -> None:
		"""Mix parent_1's and parent_2's weights into target."""
		self._instance.apply(parent_1, parent_2, target, obs_normalizer, max_eval_steps, default_trust)
