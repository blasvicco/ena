"""Random-action baseline agent — the classic RL floor, no model, no learning."""

# General imports
from typing import Callable

# Libs imports
import gymnasium as gym
import numpy as np

# App imports
from .tools.abstract import AAgent


class RandomAgent(AAgent):
	"""Samples a uniformly random action every step; no model, no training."""

	def __init__(self, brain: Callable, make_env: Callable[..., gym.Env]) -> None:
		"""Initialize RandomAgent."""
		super().__init__(brain, make_env)
		self.learn = False

	def act(self, env: gym.Env, state: np.ndarray) -> tuple:
		"""Sample a uniformly random action and step the environment."""
		action = env.action_space.sample()
		next_state, reward, terminated, truncated, _ = env.step(action)
		return next_state, reward, terminated, truncated, action

	def train(self, **kwargs: object) -> None:
		"""No-op: nothing to train."""

	def set_learning(self, mode: bool, output_dir: str = None) -> None:
		"""No-op: no train/test distinction to switch between."""
