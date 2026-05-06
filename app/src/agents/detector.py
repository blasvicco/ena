"""Environment change detector for EN Agent."""

from collections import deque


class EnvironmentDetector:
	"""
	Detects environment changes by tracking the mathematical derivative of the
	agent's performance history.

	Designed to be used as a standalone helper — ENAgent only needs to call
	`detect()` at decision points.
	"""

	def __init__(
		self,
		window_size: int,
		cooldown_window: int,
		performance_derivative_threshold: float = -0.1,
		short_window_size: int = None,
	):
		"""
		Args:
			window_size:     Number of episodes for the long-term baseline.
			cooldown_window: Episodes to skip after a detection event.
			performance_derivative_threshold: The negative threshold required to trigger.
		"""
		self._window_size = window_size
		self._cooldown_window = cooldown_window
		self._performance_derivative_threshold = performance_derivative_threshold

		# We use two buffers to calculate a robust "Moving Average Derivative"
		# 1. Long history: The baseline we are comparing against.
		self._history: deque = deque(maxlen=window_size)
		# 2. Short history: The current trend (smoothed latest performance).
		self._short_window = short_window_size if short_window_size is not None else max(1, window_size // 5)
		self._short_history: deque = deque(maxlen=self._short_window)

		self._cooldown: int = 0
		self._current_env_id: int = 0
		self._transition_detected: bool = False
		self._change_history: list = []  # [(episode_count, new_env_id), ...]
		self._episode_count: int = 0

	# ------------------------------------------------------------------
	# Public interface
	# ------------------------------------------------------------------

	@property
	def current_env_id(self) -> int:
		"""The last confirmed environment ID."""
		return self._current_env_id

	@property
	def transition_detected(self) -> bool:
		"""
		Returns True if a transition was confirmed in the last call to detect().
		Resets to False after being read.
		"""
		detected = self._transition_detected
		self._transition_detected = False
		return detected

	@property
	def change_history(self) -> list:
		"""List of (episode_count, new_env_id) tuples — used for plotting."""
		return self._change_history

	def detect(self, current_performance: float) -> int:
		"""
		Determine if a transition occurred by comparing the short-term trend
		to the long-term baseline.
		"""
		self._episode_count += 1
		if self._cooldown > 0:
			self._cooldown -= 1
			return self._current_env_id

		# Update both buffers
		self._history.append(current_performance)
		self._short_history.append(current_performance)

		# 1. Wait for enough history to establish a reliable baseline
		if len(self._history) < self._window_size:
			return self._current_env_id

		# 2. Calculate "Derivative of the Moving Average"
		# We compare the current smoothed trend against the historical baseline.
		historical_baseline = sum(self._history) / len(self._history)
		current_trend = sum(self._short_history) / len(self._short_history)

		performance_derivative = current_trend - historical_baseline

		# 3. Trigger on sustained drop relative to baseline
		if performance_derivative < self._performance_derivative_threshold:
			self._current_env_id += 1
			self._change_history.append((self._episode_count, self._current_env_id))
			self._transition_detected = True
			self._cooldown = self._cooldown_window

			# Reset history with the current performance to establish an immediate
			# new baseline. This prevents the detector from being "blind" during
			# the window-filling period and ensures the cooldown is the only
			# thing limiting re-triggering.
			self._history = deque([current_performance] * self._window_size, maxlen=self._window_size)
			self._short_history = deque([current_performance] * self._short_window, maxlen=self._short_window)

		return self._current_env_id
