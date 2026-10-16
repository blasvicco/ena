"""No-op environment-detection driver — collapses everything into a single universal niche (0)."""

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.env_detection.driver.abstract import ADriver
from agents.ena.memory import AgentMemory
from agents.ena.skills.hall_of_fame.main import HallOfFame


class Instance(ADriver):
	"""Never detects a shift and always resolves to niche 0 — the environment-detection off-switch."""

	# Combined with the hall_of_fame skill's none driver, this collapses the archive to pure
	# global elitism: gladiators get remembered, but never tied to which physics they were good for.

	def __init__(self, state_dim: int, **_kwargs) -> None:
		self._state_dim = state_dim

	@property
	def current_niche_id(self) -> int:
		"""Return niche 0: there is only ever one niche."""
		return 0

	@property
	def known_niche_ids(self) -> set:
		"""Return {0}: there is only ever one niche."""
		return {0}

	def detect_change(self, memory: AgentMemory, learn: bool = True) -> bool:
		"""Never detects a shift."""
		return False

	def match_niche(self, memory: AgentMemory, current_individual: dict, hall_of_fame: HallOfFame, population: list = None) -> int:
		"""Assign every individual to niche 0."""
		return 0

	def start_episode(self) -> None:
		"""No-op."""

	def observe(self, state: np.ndarray, action: int | np.ndarray, next_state: np.ndarray, memory: AgentMemory) -> None:
		"""No-op."""

	def log_likelihoods(self, memory: AgentMemory) -> dict:
		"""No per-niche likelihood model — an empty dict."""
		return {}

	def reset(self) -> None:
		"""No-op: this driver keeps no state to flush."""
