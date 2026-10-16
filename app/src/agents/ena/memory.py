"""Agent Memory Module for EN Agent."""
# General imports
from collections import deque

# Libs imports
import numpy as np

# App imports
from ..tools.normalizers import _RunningNormalizer

# The return trend read by the fuzzy Plasticity controller (the paper reads these through tools/paper_config.py):
# the mean of the last TREND_RECENT returns minus the mean of the returns before them, up to TREND_WINDOW in all,
# and 0 until the history holds TREND_MIN_RETURNS returns.
TREND_RECENT = 5
TREND_WINDOW = 50
TREND_MIN_RETURNS = 20


# pylint: disable=too-many-instance-attributes
class AgentMemory:
	"""Centralized memory storage for all historical variables and statistics."""

	def __init__(self, history_size: int, state_dim: int, max_eval_steps: int = 500, obs_normalizer_max_count: int | None = None) -> None:
		# 1. Global Statistics
		self.max_seen = -np.inf
		self.min_seen = np.inf

		# The observation statistics average over the last max_count samples (default: ten evaluation
		# depths). A domain whose episodes are as long as that window passes a longer one, or the
		# statistics follow each episode's states.
		max_count = obs_normalizer_max_count if obs_normalizer_max_count is not None else max_eval_steps * 10
		self.obs_normalizer = _RunningNormalizer(shape=state_dim, max_count=max_count)

		# 2. Performance Histories
		self.reward_history = deque(maxlen=history_size)
		self.plasticity_history = []
		self.last_episode_score = 0
		self.current_episode_reward = 0
		self.steps_this_episode = 0

	def record_step(self, reward: float) -> None:
		"""Accumulate the running episode reward."""
		self.current_episode_reward += reward

	@property
	def performance_derivative(self) -> float:
		"""Return the derivative of the moving average of the reward history."""
		if len(self.reward_history) < TREND_MIN_RETURNS:
			return 0.0
		history_list = list(self.reward_history)
		baseline = float(np.mean(history_list[-TREND_WINDOW:-TREND_RECENT]))
		short_term = float(np.mean(history_list[-TREND_RECENT:]))
		return short_term - baseline

	def clear_training_history(self) -> None:
		"""Clear the return history on a training-time environment shift, so the new regime's returns don't mix with the old one's."""
		self.reward_history.clear()
		self.max_seen = -np.inf
		self.min_seen = np.inf
