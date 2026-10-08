"""Percentile-based reward normalization tool, shared across ENA's skills."""

# Libs imports
import numpy as np

# App imports
from agents.ena.tools.curves import WARMUP_RETURNS

NORMALIZED_RANGE = (-2.0, 5.0)  # clip; deliberately not [0, 1], so a record-breaking return still ranks higher
NORMALIZING_PERCENTILES = (5, 95)  # the history's low and high reference returns


def normalize_reward(reward: float, memory, return_scale: float) -> float:
	"""Percentile-based reward normalization. No [0, 1] clip, to preserve selection pressure."""
	if len(memory.reward_history) < WARMUP_RETURNS:
		# Warmup: no reliable percentiles yet; scale against the solved return
		return float(np.clip(reward / return_scale, *NORMALIZED_RANGE))
	low = float(np.percentile(memory.reward_history, NORMALIZING_PERCENTILES[0]))
	high = float(np.percentile(memory.reward_history, NORMALIZING_PERCENTILES[1]))

	# Linear scaling relative to historical percentiles, but ALLOW values > 1.0
	# to reward individuals that break the current records.
	norm = (reward - low) / (high - low + 1e-8)
	return float(np.clip(norm, *NORMALIZED_RANGE))
