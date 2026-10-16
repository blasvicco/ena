"""Baseline Agent Abstract class"""

# General imports
from abc import abstractmethod

# Libs imports
import numpy as np
import gymnasium as gym

# App imports
from .abstract import AAgent


def transition(step_data: dict) -> tuple:
	"""(state, action, reward, next_state, terminal) for a replay buffer, from the study runner's step_data."""
	# Only a true termination is terminal. A time-limit truncation is not: the next state still has
	# value, so the target must bootstrap through it (treating it as terminal biases every value near
	# the limit, and in HalfCheetah every episode ends that way). No fallback to "done", which would
	# silently reintroduce that bias.
	return step_data["state"], step_data["action"], step_data["reward"], step_data["next_state"], float(step_data["terminated"])


class ABaseline(AAgent):
	"""Baseline Agent Abstract Class"""

	learn = True

	def set_learning(self, mode: bool, output_dir: str | None = None) -> None:
		"""Switch between training and evaluation modes (output_dir accepted for the study runner, unused)."""
		self.learn = mode

	@abstractmethod
	def act(self, env: gym.Env, state: np.ndarray) -> tuple:
		"""Action inference and rollout data collection."""

	@abstractmethod
	def train(self, **kwargs: object) -> None:
		"""Manual training loop."""
