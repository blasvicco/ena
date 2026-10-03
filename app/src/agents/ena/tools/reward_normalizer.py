"""Percentile-based reward normalization tool, shared across ENA's skills."""

# Libs imports
import numpy as np


def normalize_reward(reward: float, memory, env_max_steps: float) -> float:
	"""Percentile-based reward normalization. No [0, 1] clip, to preserve selection pressure."""
	if len(memory.reward_history) < 10:
		# Warmup: scale against env_max_steps so trust values are distinct
		return float(np.clip(reward / env_max_steps, -2.0, 5.0))
	low = float(np.percentile(memory.reward_history, 5))
	high = float(np.percentile(memory.reward_history, 95))

	# Linear scaling relative to historical percentiles, but ALLOW values > 1.0
	# to reward individuals that break the current records.
	norm = (reward - low) / (high - low + 1e-8)
	return float(np.clip(norm, -2.0, 5.0))
