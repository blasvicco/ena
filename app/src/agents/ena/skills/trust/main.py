"""Trust skill: computes a trust-decay factor via a pluggable driver."""

# Libs imports
import numpy as np

# App imports
from agents.ena.tools.driver_loader import load_driver
from agents.ena.tools.reward_normalizer import normalize_reward
from agents.ena.memory import AgentMemory


REFERENCE_MIN_RETURNS = 5  # fewer training returns than this: no reference trust yet (reference() returns -1.0)


class Trust:
	"""Resolves and delegates to the configured trust driver."""

	# Usage: Trust("linear_ramp").decay(reward, memory)

	def __init__(self, driver: str, **kwargs: object) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		self._instance = load_driver("trust", driver, packages=("agents.ena.skills.failure_signal.driver",), **kwargs)

	def decay(self, reward: float, memory: AgentMemory) -> float:
		"""Return the trust-decay factor for the given reward."""
		return self._instance.value(reward, memory)

	def reference(self, memory: AgentMemory, return_scale: float) -> float:
		"""Return the reference trust for a freshly built or reset individual."""
		# Below REFERENCE_MIN_RETURNS samples, -1.0 keeps a fresh/reset slot ranked above any evaluated mutant
		# (even one with a negative score) without creating an artificial ceiling at 0.0
		# in negative-reward environments. Independent of the configured driver — every
		# algorithm shares the same bootstrap reference.
		if len(memory.reward_history) < REFERENCE_MIN_RETURNS:
			return -1.0
		return normalize_reward(float(np.mean(memory.reward_history)), memory, return_scale)

	def reset_population(self, population: list, memory: AgentMemory, return_scale: float, factor: float) -> None:
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
