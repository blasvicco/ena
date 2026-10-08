"""Trust skill: computes a trust-decay factor via a pluggable driver."""

# General imports
import importlib

# Libs imports
import numpy as np

# App imports
from agents.ena.tools.reward_normalizer import normalize_reward


class Trust:
	"""Resolves and delegates to the configured trust driver."""

	# Usage: Trust("linear_ramp").decay(reward, memory)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.trust.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown trust driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	def decay(self, reward: float, memory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return self._instance.decay(reward, memory)

	def reference(self, memory, return_scale: float) -> float:
		"""Return the reference trust for a freshly built or reset individual."""
		# Below 5 samples, -1.0 keeps a fresh/reset slot ranked above any evaluated mutant
		# (even one with a negative score) without creating an artificial ceiling at 0.0
		# in negative-reward environments. Independent of the configured driver — every
		# algorithm shares the same bootstrap reference.
		if len(memory.reward_history) < 5:
			return -1.0
		return normalize_reward(float(np.mean(memory.reward_history)), memory, return_scale)

	def reset_population(self, population: list, memory, return_scale: float, factor: float) -> None:
		"""Blend every non-gladiator's trust toward the reference trust, by the given factor."""
		# Gladiators are excluded: they're already protected from replacement in
		# __evolve_population, and an unrelated exploratory failure (e.g. the epsilon-greedy
		# random non-gladiator pick during training) shouldn't erode an
		# established gladiator's trust either.
		reference = self.reference(memory, return_scale)
		for individual in population:
			if individual.get("is_gladiator", False):
				continue
			individual["trust"] = (individual["trust"] * (1.0 - factor)) + (reference * factor)
