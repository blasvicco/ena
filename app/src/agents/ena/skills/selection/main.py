"""Selection skill: picks which population member acts each episode via a pluggable driver."""

# General imports
import importlib


class SelectionController:
	"""Resolves and delegates all selection logic to a pluggable driver."""

	# Usage: SelectionController("heuristic", exploration_rate=0.15).select_individual(population, recognizer, memory, learn=True)

	def __init__(self, driver: str, **kwargs) -> None:
		"""Instantiate the appropriate driver based on the driver name."""
		try:
			module = importlib.import_module(f"agents.ena.skills.selection.driver.{driver}")
		except ModuleNotFoundError as error:
			raise ValueError(f"Unknown selection driver: {driver!r}") from error
		self._instance = module.Instance(**kwargs)

	# ---------------------------------------------------------------
	# Delegated public API
	# ---------------------------------------------------------------

	def reset_episode(self) -> None:
		"""Called at episode start."""
		self._instance.reset_episode()

	def select_individual(self, population: list, recognizer, memory, learn: bool) -> int:
		"""Returns population index of the selected individual."""
		return self._instance.select_individual(population, recognizer, memory, learn)

	def on_step(self, memory, env_detection, learn: bool) -> None:
		"""Delegate mid-episode bookkeeping to the active driver."""
		self._instance.on_step(memory, env_detection, learn)

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def on_episode_end(self, memory, learn: bool, population: list, env_detection, env_max_steps: float) -> bool:
		"""Finalize the episode; returns True if test mode forced a re-evaluation reset."""
		if learn:
			self._instance.on_learn_episode_end(memory, env_detection, env_max_steps)
			return False

		memory.reward_history.append(memory.last_episode_score)
		memory.max_seen = max(memory.max_seen, memory.last_episode_score)
		memory.min_seen = min(memory.min_seen, memory.last_episode_score)

		force_reset = self.record_episode_result(memory.last_episode_score, memory, recognizer=env_detection)
		if force_reset:
			memory.reward_history.clear()
			self.evaluate(population)
		return force_reset

	def record_episode_result(self, score: float, memory, recognizer=None) -> bool:
		"""Called at episode end during testing. Returns True if env reset needed."""
		return self._instance.record_episode_result(score, memory, recognizer=recognizer)

	def evaluate(self, population: list) -> None:
		"""Build evaluation queue from population's gladiator niches."""
		self._instance.evaluate(population)

	def set_mode(self, memory, mode: bool) -> None:
		"""Switch between train (mode=True) and test (mode=False)."""
		self._instance.set_mode(memory, mode)

	def record(self, memory, context_vector, niche_id: int, reward: float) -> None:
		"""Store a (context, niche, reward) experience for training."""
		self._instance.record(memory, context_vector, niche_id, reward)

	def update(self, memory) -> None:
		"""Run driver-specific update from buffered experiences."""
		self._instance.update(memory)
