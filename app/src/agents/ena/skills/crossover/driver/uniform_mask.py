"""Driver for per-parameter uniform-mask crossover."""

# General imports
import uuid

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.crossover.driver.abstract import ADriver
from agents.ena.tools.weight_projection import project_individual_weights
from agents.tools.normalizers import _RunningNormalizer


# Probability that a weight comes from the first parent.
PARENT_SHARE = 0.5


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Mixes two parents' weights per-parameter via a random 50/50 mask, after aligning both to the current normalizer."""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def apply(
		self, parent_1: dict, parent_2: dict, target: dict, obs_normalizer: _RunningNormalizer, max_eval_steps: float, default_trust: float
	) -> None:
		"""Mix parent_1's and parent_2's weights into target."""
		# USP: Align both parents to the current normalizer before mixing their weights
		project_individual_weights(parent_1, obs_normalizer, max_eval_steps)
		project_individual_weights(parent_2, obs_normalizer, max_eval_steps)

		weights_1 = parent_1["model"].get_weights()
		weights_2 = parent_2["model"].get_weights()

		new_weights = []
		for weight_1, weight_2 in zip(weights_1, weights_2):
			mask = np.random.random(weight_1.shape) < PARENT_SHARE
			new_weights.append(np.where(mask, weight_1, weight_2))

		target["model"].set_weights(new_weights)
		target["individual_id"] = str(uuid.uuid4())
		target["age"] = max(parent_1["age"], parent_2["age"])
		target["trust"] = default_trust
		target["is_gladiator"] = False
		target["internal_env_id"] = -1
		target["env_name"] = "Unknown"
		# Because parents were projected to current normalizer, the child inherits it
		target["last_mu"] = np.copy(obs_normalizer.mean)
		target["last_var"] = np.copy(obs_normalizer.var)
