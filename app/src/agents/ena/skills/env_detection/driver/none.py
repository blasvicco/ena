"""No-op environment-detection driver — collapses everything into a single universal niche (0)."""

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.env_detection.driver.abstract import ADriver


class Instance(ADriver):
	"""Never detects a shift and always resolves to niche 0 — the environment-detection off-switch."""

	# Combined with the hall_of_fame skill's none driver, this collapses the archive to pure
	# global elitism: gladiators get remembered, but never tied to which physics they were good for.

	def __init__(self, state_dim, **_kwargs) -> None:
		self._state_dim = state_dim

	@property
	def current_niche_id(self) -> int:
		"""Always niche 0 — there is only ever one niche."""
		return 0

	@property
	def known_niche_ids(self) -> set:
		"""Always {0} — there is only ever one niche."""
		return {0}

	def update(self, state, memory) -> None:
		"""No-op: this driver tracks no physics signature."""

	def compute_physical_vector(self, memory) -> np.ndarray:
		"""A fixed-shape zero vector (4 * state_dim), matching the other drivers' shape so context_dim stays consistent."""
		return np.zeros(4 * self._state_dim, dtype=np.float32)

	def compute_full_vector(self, memory) -> np.ndarray:
		"""compute_physical_vector's shape plus the two distress scalars, matching the other drivers' compute_full_vector shape."""
		return np.zeros(4 * self._state_dim + 2, dtype=np.float32)

	def detect_change(self, memory) -> bool:
		"""Never detects a shift."""
		return False

	def match(self, memory) -> tuple:
		"""Always confidently resolves to niche 0."""
		return 0, 1.0

	def match_niche(self, memory, current_individual: dict, hall_of_fame, population: list = None) -> int:
		"""Always assigns to niche 0."""
		return 0

	def start_episode(self) -> None:
		"""No-op."""

	def observe(self, state, action, next_state, memory) -> None:
		"""No-op."""

	def log_likelihoods(self, memory) -> dict:
		"""No per-niche likelihood model — an empty dict."""
		return {}

	def reset(self) -> None:
		"""No-op: this driver keeps no state to flush."""
