"""Predictive-Mixture CuSum: quickest detection of a change away from a known pre-change distribution."""

# Algorithm: Predictive-Mixture CuSum (PM-CuSum)
# Reference: Halme, Poor & Koivunen, "Adaptive Sequential Change Detection using Mixtures of Predictive Distributions" (2026)
# https://arxiv.org/abs/2606.05072

# General imports
import math
from collections import deque

# Libs imports
import numpy as np


class PMCuSum:
	"""CuSum over a Fixed-Share mixture of sliding-window predictive densities, against a N(0, I) pre-change."""

	# Observations are expected to be standardized so the pre-change distribution is N(0, I)
	# (e.g. a dynamics model's standardized residuals). The post-change distribution is unknown:
	# each window w in {2, 4, ..., 2^ceil(log2 b)} predicts the next observation from its last w
	# samples, and the predictions are mixed with Fixed Share weights driven by their predictive
	# likelihood. S_n = max(S_{n-1}, 0) + log(p_hat_n(x_n) / q(x_n)); a change is declared once
	# S_n > b. b = log(gamma) guarantees an average run length to false alarm of at least gamma.
	#
	# Deviation from the paper's Gaussian example (known unit variance, estimated mean): here each
	# window also estimates a per-dimension variance, shrunk toward 1 with variance_prior pseudo-
	# samples and never below 1. Physics changes in a regression residual show up as a variance
	# increase at least as much as a mean shift (a wrong gain on a zero-mean state leaves the
	# residual mean ~0), which a mean-only predictor would miss. The variance is one-sided because
	# a wrong dynamics model can only fit *worse*: residuals narrower than the pre-change (e.g. the
	# exactly-predictable kinematic dimensions of deterministic physics) must not read as a change.

	def __init__(self, dim: int, threshold: float, variance_prior: float = 2.0) -> None:
		self._dim = dim
		self._threshold = threshold
		self._variance_prior = variance_prior
		self._windows = [2**power for power in range(1, max(1, math.ceil(math.log2(max(threshold, 2.0)))) + 1)]
		self._history = deque(maxlen=self._windows[-1])
		self._log_weights = np.full(len(self._windows), -math.log(len(self._windows)))
		self._statistic = 0.0

	@property
	def statistic(self) -> float:
		"""Current CuSum statistic S_n."""
		return self._statistic

	@property
	def windows(self) -> list:
		"""Sliding-window lengths of the predictive mixture."""
		return list(self._windows)

	def reset(self) -> None:
		"""Forget all history and restart the statistic from zero."""
		self._history.clear()
		self._log_weights = np.full(len(self._windows), -math.log(len(self._windows)))
		self._statistic = 0.0

	def update(self, observation: np.ndarray) -> bool:
		"""Feed one observation; return True once the statistic exceeds the threshold."""
		observation = np.asarray(observation, dtype=np.float64).ravel()
		log_q = self._log_normal(observation, np.zeros(self._dim), np.ones(self._dim))
		window_log_p = np.array([self._window_log_density(observation, window) for window in self._windows])

		# Mixture predictive density, then the CuSum recursion.
		log_p_hat = _logsumexp(self._log_weights + window_log_p)
		previous_positive = max(self._statistic, 0.0)
		self._statistic = previous_positive + (log_p_hat - log_q)

		# Fixed Share: Bayesian update by each window's predictive likelihood, then share mass
		# uniformly with the paper's adaptive rate alpha = 1 / (1 + exp(S^+)).
		posterior = self._log_weights + window_log_p
		posterior -= _logsumexp(posterior)
		alpha = 1.0 / (1.0 + math.exp(min(previous_positive, 700.0)))
		mixed = (1.0 - alpha) * np.exp(posterior) + alpha / len(self._windows)
		self._log_weights = np.log(mixed)

		self._history.append(observation)
		return self._statistic > self._threshold

	def _window_log_density(self, observation: np.ndarray, window: int) -> float:
		"""Log-density of observation under the predictor fitted on the last `window` samples."""
		if not self._history:
			return self._log_normal(observation, np.zeros(self._dim), np.ones(self._dim))
		samples = np.array(list(self._history)[-window:])
		count = samples.shape[0]
		mean = samples.mean(axis=0)
		spread = np.sum((samples - mean) ** 2, axis=0)
		variance = np.maximum((spread + self._variance_prior) / (count + self._variance_prior), 1.0)
		return self._log_normal(observation, mean, variance)

	@staticmethod
	def _log_normal(observation: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
		"""Log-density of a diagonal Gaussian."""
		return float(-0.5 * np.sum((observation - mean) ** 2 / variance + np.log(2.0 * np.pi * variance)))


def _logsumexp(values: np.ndarray) -> float:
	"""Numerically stable log(sum(exp(values)))."""
	peak = float(np.max(values))
	return peak + math.log(float(np.sum(np.exp(values - peak))))
