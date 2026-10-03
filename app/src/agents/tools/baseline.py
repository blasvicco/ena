"""Baseline Agent Abstract class"""

# General imports
from abc import abstractmethod

# App imports
from .abstract import AAgent


class ABaseline(AAgent):
	"""Baseline Agent Abstract Class"""

	learn = True

	def set_learning(self, mode, output_dir=None):
		"""Switches between training and evaluation modes (output_dir accepted for the study runner, unused)."""
		self.learn = mode

	@abstractmethod
	def act(self, env, state):
		"""Action inference and rollout data collection."""

	@abstractmethod
	def train(self, **kwargs):
		"""Manual training loop."""
