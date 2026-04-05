"""Agent Abstract Class"""

# General imports
from abc import abstractmethod


class AAgent:
	"""Abstract base class for all agents."""
	name = ""
	episodes = []

	def __init__(self):
		"""Initialize AAgent"""

	def get_name(self):
		"""Get the agent name"""
		return self.name

	def get_episodes(self):
		"""Get the agent training episodes"""
		return self.episodes

	def set_name(self, name):
		"""Set the agent name"""
		self.name = name

	def set_episodes(self, episodes):
		"""Set the agent training episodes"""
		self.episodes = episodes

	@abstractmethod
	def act(self, env, state):
		"""Pure inference for testing phase."""

	@abstractmethod
	def train(self, **kwargs):
		"""Train the agent."""

	@abstractmethod
	def set_learning(self, mode):
		"""Switches between training and evaluation modes."""
