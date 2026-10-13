"""Shared curve tools — the same four formulas back both the Trust and Plasticity skills."""

# Libs imports
import numpy as np

# Shared by the curves and Trust's reward normalizer (the paper reads them through tools/paper_config.py).
WARMUP_RETURNS = 10  # fewer returns than this in the history: no reliable percentiles yet, full signal
CONFIDENT_PERCENTILE = 90  # a return at or above this percentile of the history gives the minimum signal
MIN_SIGNAL = 0.1  # the lowest failure signal: a confident agent still searches a little


def zero(_reward: float, _memory) -> float:
	"""Always zero."""
	return 0.0


def full(_reward: float, _memory) -> float:
	"""Always one."""
	return 1.0


def linear_ramp(reward: float, memory) -> float:
	"""Percentile-scaled linear ramp: 0.1 at or above the 90th percentile of recent scores, rising linearly to 1.0 at the worst score seen this run."""
	# No hard floor: a previous version clamped anything below the 10th percentile to a flat
	# 1.0, so once the agent was consistently near-perfect (a very narrow percentile band), a
	# merely mediocre score and a catastrophic one both forced identical maximum-effort search
	# — no discrimination at all below that point. Interpolating all the way to memory.min_seen keeps
	# the value proportional across the whole observed range.
	if len(memory.reward_history) < WARMUP_RETURNS:
		return 1.0

	high_limit = np.percentile(memory.reward_history, CONFIDENT_PERCENTILE)
	floor = memory.min_seen

	if reward >= high_limit:
		return MIN_SIGNAL

	span = high_limit - floor
	if span < 1e-8:
		return 1.0

	ratio = np.clip((reward - floor) / span, 0.0, 1.0)
	return float(1.0 - (ratio * (1.0 - MIN_SIGNAL)))


def quadratic(reward: float, memory) -> float:
	"""Aggressive curve using global benchmarks."""
	if len(memory.reward_history) < WARMUP_RETURNS:
		return 1.0

	denominator = (memory.max_seen - memory.min_seen) + 1e-8
	normalized_error = (memory.max_seen - reward) / denominator
	return max(MIN_SIGNAL, min(1.0, normalized_error**2))
