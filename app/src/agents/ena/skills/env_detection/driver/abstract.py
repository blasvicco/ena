"""Environment-detection driver interface: unsupervised shift detection and niche ID."""

# General imports
from abc import ABC, abstractmethod


class ADriver(ABC):
	"""Detects environment shifts and identifies which recurring niche is currently active."""

	@property
	@abstractmethod
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""

	@property
	@abstractmethod
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""

	@abstractmethod
	def detect_change(self, memory, learn: bool = True) -> bool:
		"""Check if the environment has significantly shifted from the baseline."""

	@abstractmethod
	def match_niche(self, memory, current_individual: dict, hall_of_fame, population: list = None) -> int:
		"""Late Identity Binding: called only when a gladiator is being inducted."""

	@abstractmethod
	def start_episode(self) -> None:
		"""Called once at the start of every episode, before the first action."""

	@abstractmethod
	def observe(self, state, action, next_state, memory) -> None:
		"""Record one real (state, action, next_state) transition of the acting individual."""

	@abstractmethod
	def log_likelihoods(self, memory) -> dict:
		"""{niche_id: log-likelihood of this episode's transitions} — empty if the driver has no dynamics model."""

	@abstractmethod
	def reset(self) -> None:
		"""Flush the driver's evidence about the old regime on a detected environment shift."""
