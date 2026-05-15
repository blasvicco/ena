"""Agent Memory Module for EN Agent."""

import numpy as np
from collections import deque
from .normalizers import _RunningNormalizer

class AgentMemory:
	"""Centralized memory storage for all historical variables and statistics."""

	def __init__(self, history_size, state_dim, max_eval_steps=500):
		# 1. Global Statistics
		self.max_seen = -np.inf
		self.min_seen = np.inf
		self.episodes_seen = 0
		self.steps_seen = 0

		self.obs_normalizer = _RunningNormalizer(shape=state_dim, max_count=max_eval_steps * 10)

		# 2. Performance Histories
		self.reward_history = deque(maxlen=history_size)
		self.plasticity_history = []
		self.last_episode_score = 0
		self.current_episode_reward = 0

		# 3. Recognizer Statistics
		self.short_term_normalizer = _RunningNormalizer(shape=state_dim, max_count=1000)
		self.transition_normalizer = _RunningNormalizer(shape=state_dim, max_count=1000)

		# 4. Selection Controller Buffers (for REINFORCE)
		self.contexts = []
		self.actions = []
		self.rewards = []

	def update_transition(self, state, next_state):
		"""Computes and tracks the transition dynamics (physics signature)."""
		delta_s = next_state - state
		self.transition_normalizer.update(delta_s[np.newaxis, :])

	@property
	def performance_derivative(self):
		"""Returns the derivative of the moving average of reward history."""
		if len(self.reward_history) < 20:
			return 0.0
		history_list = list(self.reward_history)
		baseline = float(np.mean(history_list))
		short_term = float(np.mean(history_list[-5:]))
		return short_term - baseline

	@property
	def recent_plasticity(self):
		"""Returns the recent average plasticity."""
		if not self.plasticity_history:
			return 1.0
		return float(np.mean(self.plasticity_history[-5:]))

	def clear_training_history(self):
		"""Clears the training history when switching to test mode."""
		self.reward_history.clear()
		self.max_seen = -np.inf
		self.min_seen = np.inf

	def clear_selection_buffers(self):
		"""Clears the REINFORCE experience buffers."""
		self.contexts.clear()
		self.actions.clear()
		self.rewards.clear()

	def reset_recognizer_statistics(self):
		"""Flushes the short-term normalizers to cleanly adapt to a new environment."""
		self.short_term_normalizer.reset()
		self.transition_normalizer.reset()
