"""Environment-detection skill: unsupervised shift detection and niche identification via a pluggable driver."""

# General imports
import importlib


class EnvironmentRecognizer:
	"""Resolves and delegates every call to the configured environment-detection driver."""

	# Usage: EnvironmentRecognizer("dynamics_model", state_dim, action_dim=2).log_likelihoods(memory) -> {niche_id: log-likelihood}

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
	def chirp_actions(self):
		"""The driver's fixed chirp actions, or None if it does not chirp."""
		return self._instance.chirp_actions

	def observe_chirp(self, response) -> None:
		"""Hand the driver a chirp's observations (start observation first)."""
		self._instance.observe_chirp(response)

	@property
	def known_niche_ids(self) -> set:
		"""Set of niche IDs that have physical prototypes."""
		return self._instance.known_niche_ids

	def detect_change(self, memory, learn: bool = True) -> bool:
		"""Check if the environment has significantly shifted from the baseline (the driver learns only when learn)."""
		return self._instance.detect_change(memory, learn)

	def evaluate(self, memory, learn: bool) -> bool:
		"""Check for an environment shift and, if found, flush stale physics/observation state."""
		shifted = self.detect_change(memory, learn)
		if shifted:
			self.reset()
			memory.obs_normalizer.reset()
			if learn:
				# Training-time shift: flush reward history so percentile-based normalization
				# (normalize_reward tool, linear_ramp curve, dynamic ref trust) doesn't mix
				# old- and new-phase scores for ~100 episodes after every switch.
				memory.clear_training_history()
		return shifted

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
		"""Flush the driver's evidence about the old regime on a detected environment shift."""
		# dynamics_model drops the old regime's transition buffers (keeping a confirmed shift's
		# post-alarm evidence); none has nothing to flush.
		self._instance.reset()
