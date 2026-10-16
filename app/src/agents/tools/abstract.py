"""Agent Abstract Class"""

# General imports
from typing import Callable
from abc import ABC, abstractmethod

# Libs imports
import numpy as np
import gymnasium as gym


class AAgent(ABC):
	"""Abstract base class for all agents."""

	name = ""
	episodes = []
	brain = None
	make_env = None

	def __init__(self, brain: Callable, make_env: Callable[..., gym.Env]) -> None:
		"""Initialize AAgent"""
		self.brain = brain
		self.make_env = make_env

	def get_name(self) -> str:
		"""Get the agent name"""
		return self.name

	def get_episodes(self) -> list:
		"""Get the agent training episodes"""
		return self.episodes

	def set_name(self, name: str) -> None:
		"""Set the agent name"""
		self.name = name

	def set_episodes(self, episodes: list) -> None:
		"""Set the agent training episodes"""
		self.episodes = episodes

	@abstractmethod
	def act(self, env: gym.Env, state: np.ndarray) -> tuple:
		"""Pure inference for testing phase."""

	@abstractmethod
	def train(self, **kwargs: object) -> None:
		"""Train the agent."""

	@abstractmethod
	def set_learning(self, mode: bool, output_dir: str | None = None) -> None:
		"""Switch between training and evaluation modes."""
