"""Environment-detection skill: unsupervised shift detection and niche identification via a pluggable driver."""

# General imports
import importlib

# Libs imports
import numpy as np


class EnvironmentRecognizer:
	"""Resolves and delegates every call to the configured environment-detection driver."""

	# Usage: EnvironmentRecognizer("live", state_dim, similarity_threshold=0.90).match(memory) -> (niche_id, similarity)

	def __init__(self, driver: str, state_dim: int, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.env_detection.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown env_detection driver: {driver!r}") from error
		self._instance = module.Instance(state_dim, **kwargs)

	@property
	def current_niche_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._instance.current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""
		return self._instance.known_niche_ids

	def update(self, state, memory) -> None:
		"""Track observation distribution purely for environment physics recognition."""
		self._instance.update(state, memory)

	def compute_physical_vector(self, memory) -> np.ndarray:
		"""Build context from strictly static state and transition dynamics."""
		return self._instance.compute_physical_vector(memory)

	def compute_full_vector(self, memory) -> np.ndarray:
		"""Build context including distress signals for change detection."""
		return self._instance.compute_full_vector(memory)

	def detect_change(self, memory) -> bool:
		"""Check if the environment has significantly shifted from the baseline."""
		return self._instance.detect_change(memory)

	def evaluate(self, memory, learn: bool) -> bool:
		"""Check for an environment shift and, if found, flush stale physics/observation state."""
		shifted = self.detect_change(memory)
		if shifted:
			memory.reset_recognizer_statistics()
			self.reset()
			memory.obs_normalizer.reset()
			if learn:
				# Training-time shift: flush reward history so percentile-based normalization
				# (normalize_reward tool, linear_ramp curve, dynamic ref trust) doesn't mix
				# old- and new-phase scores for ~100 episodes after every switch.
				memory.clear_training_history()
		return shifted

	def match(self, memory) -> tuple:
		"""Return (niche_id, similarity) of the best matching physical prototype (Test Mode)."""
		return self._instance.match(memory)

	def match_niche(self, memory, current_individual: dict, hall_of_fame, population: list = None) -> int:
		"""Late Identity Binding: called only when a gladiator is being inducted."""
		return self._instance.match_niche(memory, current_individual, hall_of_fame, population=population)

	def start_episode(self) -> None:
		"""Called once at the start of every episode, before the first action."""
		self._instance.start_episode()

	def observe(self, state, action, next_state, memory) -> None:
		"""Record one real (state, action, next_state) transition of the acting individual."""
		self._instance.observe(state, action, next_state, memory)

	def log_likelihoods(self, memory) -> dict:
		"""{niche_id: log-likelihood of this episode's transitions} — empty if the driver has no dynamics model."""
		return self._instance.log_likelihoods(memory)

	def reset(self) -> None:
		"""Flush any driver-owned physical-fingerprint state on a detected environment shift."""
		# No-op unless the driver keeps state outside of AgentMemory (e.g. dynamics_model's own
		# transition buffers) — memory.reset_recognizer_statistics() already handles
		# memory-owned state (short_term_normalizer/transition_normalizer).
		self._instance.reset()
