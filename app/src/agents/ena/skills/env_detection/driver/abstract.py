"""Environment-detection driver interface: unsupervised shift detection and niche ID."""

# General imports
from abc import ABC, abstractmethod

# Libs imports
import numpy as np


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
	def update(self, state, memory) -> None:
		"""Track observation distribution purely for environment physics recognition."""

	@abstractmethod
	def compute_physical_vector(self, memory) -> np.ndarray:
		"""Build context from strictly static state and transition dynamics."""

	@abstractmethod
	def compute_full_vector(self, memory) -> np.ndarray:
		"""Build context including distress signals for change detection."""

	@abstractmethod
	def detect_change(self, memory) -> bool:
		"""Check if the environment has significantly shifted from the baseline."""

	@abstractmethod
	def match(self, memory) -> tuple:
		"""Return (niche_id, similarity) of the best matching physical prototype (Test Mode)."""

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
		"""Flush any driver-owned physical-fingerprint state on a detected environment shift."""
