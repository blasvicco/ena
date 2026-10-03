"""Per-environment early stopping for training: periodic frozen evaluation, stop when solved or on a plateau."""


class StoppingRule:  # pylint: disable=too-many-instance-attributes
	"""Decides, from periodic evaluation means, when an agent has solved (or stalled on) the current environment."""

	# One rule, applied identically to every agent so comparisons stay fair:
	# - a check falls due every check_every env-steps consumed in the phase (by the agent's own
	#   accounting — ENA's includes its candidate-evaluation steps), whatever the episode length;
	# - each check is eval_episodes frozen episodes on a separate copy of the environment, with the
	#   policy the agent would use at test time; those steps are reported, never counted as training;
	# - "solved" once the mean reaches solved_score (CartPole-v1's official threshold is 475);
	# - "plateau" once the best mean of the last plateau_checks checks hasn't beaten the best before
	#   them by at least plateau_min_gain;
	# - the caller enforces max_steps per environment ("cap").
	# With stop_when_solved=False / stop_on_plateau=False (the fixed-budget protocol) the outcomes
	# are still reported but don't end the environment: training continues to max_steps, and no
	# further checks fall due in an environment once it has been solved (only the first solve is
	# needed, and checks cost evaluation time).

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		check_every: int = 20_000,
		eval_episodes: int = 5,
		solved_score: float = 475.0,
		plateau_checks: int = 10,
		plateau_min_gain: float = 10.0,
		max_steps: int = 3_000_000,
		stop_when_solved: bool = True,
		stop_on_plateau: bool = True,
	) -> None:
		for name, value in (
			("check_every", check_every),
			("eval_episodes", eval_episodes),
			("plateau_checks", plateau_checks),
			("max_steps", max_steps),
		):
			if value <= 0:
				raise ValueError(f"{name} must be positive, got {value}")
		self.check_every = check_every
		self.eval_episodes = eval_episodes
		self.solved_score = solved_score
		self.plateau_checks = plateau_checks
		self.plateau_min_gain = plateau_min_gain
		self.max_steps = max_steps
		self.stop_when_solved = stop_when_solved
		self.stop_on_plateau = stop_on_plateau
		self._means = []
		self._next_check = check_every
		self._solved = False

	def start_phase(self) -> None:
		"""Forget the previous environment's checks; the first check falls due after check_every steps."""
		self._means = []
		self._next_check = self.check_every
		self._solved = False

	def due(self, phase_steps: int) -> bool:
		"""Whether a check is due after phase_steps env-steps in this environment."""
		if self._solved and not self.stop_when_solved:
			return False
		return phase_steps >= self._next_check

	def record(self, phase_steps: int, mean_score: float) -> str:
		"""Record one check's evaluation mean; return "solved", "plateau" or "continue"."""
		self._means.append(float(mean_score))
		while self._next_check <= phase_steps:
			self._next_check += self.check_every
		if mean_score >= self.solved_score:
			self._solved = True
			return "solved"
		if len(self._means) > self.plateau_checks:
			recent_best = max(self._means[-self.plateau_checks :])
			earlier_best = max(self._means[: -self.plateau_checks])
			if recent_best < earlier_best + self.plateau_min_gain:
				return "plateau"
		return "continue"
