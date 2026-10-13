"""Driver for flip-probability Gaussian-noise mutation."""

# General imports
import uuid

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.mutation.driver.abstract import ADriver


# Mutation-noise standard deviation at zero Plasticity (it grows linearly to mutation_noise at full Plasticity).
MIN_NOISE = 0.01


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Clones a parent's weights, flipping each with probability mutation_rate, adding Gaussian noise scaled by mutation_noise and plasticity."""

	def __init__(self, mutation_rate: float, mutation_noise: float) -> None:
		self._mutation_rate = mutation_rate
		self._mutation_noise = mutation_noise

	def apply(self, parent: dict, target: dict, plasticity: float, default_trust: float) -> None:
		"""Mutate parent's weights into target, scaled by the given plasticity."""
		target["model"].set_weights(parent["model"].get_weights())
		target["age"] = parent["age"]
		target["individual_id"] = str(uuid.uuid4())
		target["trust"] = default_trust
		target["is_gladiator"] = False
		target["internal_env_id"] = -1
		target["env_name"] = "Unknown"

		noise_mag = MIN_NOISE + ((self._mutation_noise - MIN_NOISE) * plasticity)

		# USP: Inherit the parent's alignment state. The child's weights are still
		# in the parent's space until explicitly projected during evaluation.
		target["last_mu"] = np.copy(parent["last_mu"])
		target["last_var"] = np.copy(parent["last_var"])

		weights = target["model"].get_weights()
		new_weights = []
		for weight in weights:
			mask = np.random.random(weight.shape) < self._mutation_rate
			noise = np.random.normal(0, noise_mag, weight.shape)
			weight[mask] += noise[mask]
			new_weights.append(weight)
		target["model"].set_weights(new_weights)
