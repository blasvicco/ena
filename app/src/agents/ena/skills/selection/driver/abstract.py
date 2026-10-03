"""Selection driver interface: picks which population member acts each episode."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):
	"""Picks which population member acts each episode, and resolves niche identity at test time."""

	@abstractmethod
	def reset_episode(self) -> None:
		"""Called at episode start."""

	@abstractmethod
	def select_individual(self, population: list, recognizer, memory, learn: bool) -> int:
		"""Returns population index of the selected individual."""

	@abstractmethod
	def on_step(self, memory, env_detection, learn: bool) -> None:
		"""Record a mid-episode context sample if this driver is currently exploring."""

	@abstractmethod
	def on_learn_episode_end(self, memory, env_detection, env_max_steps: float) -> None:
		"""Backfill the episode reward into buffered contexts and update if the buffer is full."""

	@abstractmethod
	def record_episode_result(self, score: float, memory, recognizer=None) -> bool:
		"""Called at episode end during testing. Returns True if env reset needed."""

	@abstractmethod
	def evaluate(self, population) -> None:
		"""Build evaluation queue from population's gladiator niches."""

	@abstractmethod
	def set_mode(self, memory, mode) -> None:
		"""Switch between train (mode=True) and test (mode=False)."""

	@abstractmethod
	def record(self, memory, context_vector, niche_id, reward) -> None:
		"""Store a (context, niche, reward) experience for training."""

	@abstractmethod
	def update(self, memory) -> None:
		"""Run driver-specific update from buffered experiences."""
