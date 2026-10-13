"""Environment-detection driver: one linear-Gaussian dynamics model per niche, PM-CuSum shift detection."""

# Algorithm: per-context dynamics models, identified by predictive likelihood (RL-CD lineage)
# Reference: da Silva, Basso, Bazzan & Engel, "Dealing with Non-Stationary Environments using Context Detection" (2006)
# https://doi.org/10.1145/1143844.1143872
# Shift detection: Predictive-Mixture CuSum — Halme, Poor & Koivunen (2026), https://arxiv.org/abs/2606.05072

# General imports
from collections import deque
import math

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.env_detection.driver.abstract import ADriver
from agents.ena.tools.linear_dynamics import LinearGaussianDynamics
from agents.ena.tools.pm_cusum import PMCuSum


# pylint: disable=too-many-instance-attributes
class Instance(ADriver):
	"""Scores niches by how well their own dynamics model predicts the real transitions."""

	# Each niche owns a LinearGaussianDynamics model of next_state - state given [state, action].
	# Conditioning on the action makes the evidence policy-independent without a probe env, and
	# every real step of every episode counts as evidence.
	#
	# - Detection: PM-CuSum on the current niche's standardized residuals, widened by
	#   residual_tolerance so the linear model's own error on nonlinear physics isn't read as a
	#   change (calibrated on CartPole: tolerance 3 + threshold log(1e8) gave zero false alarms over
	#   ~37k own-env steps; distinct physics are caught within 2-3 steps). An alarm is only a
	#   candidate: detect_change() confirms it once min_verify_samples post-alarm transitions exist,
	#   by the identification test below. If the alarm's niche still explains them, the alarm meant
	#   the model hadn't covered those states yet (early in training, the model is fitted on short
	#   failing episodes and a better policy visits new states) — they're folded into the model and
	#   CuSum restarts, instead of declaring a shift.
	# - Identification: mean per-step log-likelihood of the recent transitions under each niche.
	#   A niche is recognized only if it explains them within novelty_margin nats/step of a model
	#   freshly fitted on those same transitions; otherwise they are novel physics. Every
	#   likelihood here uses the same residual_tolerance-widened variance as detection, so the two
	#   agree: otherwise identification is residual_tolerance^2 times stricter, and a niche's model
	#   rejects its own physics on states it saw little of in training (observed at test time).
	#   Nearly identical physics (CartPole Earth vs. Neptune) then share a niche.
	# - A confirmed shift carries its post-alarm transitions into the new regime's evidence, so the
	#   new physics can be identified (or provisioned) right away instead of being re-attached,
	#   evidence-less, to the niche that was just ruled out — which looped alarm -> shift -> reset.
	# - Late Identity Binding (match_niche): the winner's recent transitions resolve to the
	#   recognized niche (and refine its model), or provision a new niche fitted on them. A
	#   gladiator is resolved the same way, so foreign-physics success never overwrites its home.

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		state_dim,
		action_dim=1,
		is_continuous=False,
		cusum_false_alarm_steps=1e8,
		residual_tolerance=3.0,
		novelty_margin=2.0,
		min_samples=100,
		min_verify_samples=50,
		recent_size=500,
		var_floor=1e-6,
		**_kwargs,
	):
		self._state_dim = state_dim
		self._action_dim = action_dim
		self._is_continuous = is_continuous
		self._residual_tolerance = residual_tolerance
		self._novelty_margin = novelty_margin
		self._min_samples = min_samples
		self._min_verify_samples = min_verify_samples
		self._var_floor = var_floor

		self._models = {}
		self._next_niche_id = 0
		self._current_niche_id = -1

		self._recent = deque(maxlen=recent_size)  # transitions since the last reset, for identification
		self._pending = deque(maxlen=recent_size)  # transitions not yet absorbed into any niche's model
		self._episode = []  # this episode's transitions, for log_likelihoods()
		self._cusum = PMCuSum(state_dim, threshold=math.log(cusum_false_alarm_steps))
		self._alarm_niche_id = None  # niche PM-CuSum raised an alarm against, pending verification
		self._alarm_evidence = []  # transitions since that alarm
		self._carried_evidence = []  # a confirmed shift's post-alarm transitions, seeding the next regime at reset()
		self._variance_scale = residual_tolerance**2

	@property
	def current_niche_id(self) -> int:
		"""Niche whose dynamics currently explain the transitions, or -1 if none (yet)."""
		return self._current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Niches that own a dynamics model."""
		return set(self._models.keys())

	def start_episode(self) -> None:
		"""Clear the per-episode buffer and re-identify the active niche from recent evidence."""
		self._episode = []
		current = self._models.get(self._current_niche_id)
		if current is not None and current.count < self._min_samples:
			return  # still growing (see match_niche): it can't be judged against the mature niches yet
		if self._alarm_niche_id is not None:
			return  # the recent buffer may straddle a change until the pending alarm is decided
		if len(self._recent) >= self._min_verify_samples:
			self._current_niche_id = self._identify()

	def observe(self, state, action, next_state, memory) -> None:
		"""Record one transition and feed its residual under the current niche's model to PM-CuSum."""
		features = self._features(state, action)
		target = np.asarray(next_state, dtype=np.float64) - np.asarray(state, dtype=np.float64)
		transition = (features, target)
		self._recent.append(transition)
		self._pending.append(transition)
		self._episode.append(transition)

		if self._alarm_niche_id is not None:
			self._alarm_evidence.append(transition)
			return
		model = self._models.get(self._current_niche_id)
		if model is None or model.count < self._min_samples:
			return
		residual = model.standardized_residuals(features, target)[0] / self._residual_tolerance
		if self._cusum.update(residual):
			self._alarm_niche_id = self._current_niche_id
			self._alarm_evidence = [transition]

	def detect_change(self, memory, learn: bool = True) -> bool:
		"""Confirm or dismiss a pending PM-CuSum alarm once enough post-alarm evidence exists (models learn only in training)."""
		if self._alarm_niche_id is None or len(self._alarm_evidence) < self._min_verify_samples:
			return False
		alarm_niche_id = self._alarm_niche_id
		explained = self._gap(alarm_niche_id, self._alarm_evidence) <= self._novelty_margin
		evidence = self._alarm_evidence
		self._alarm_niche_id = None
		self._alarm_evidence = []
		if explained:
			# False alarm: a coverage gap in the model, not new physics. Learn from it (training only:
			# the test phase is frozen) and restart.
			if learn:
				self._absorb_pending(alarm_niche_id)
			self._cusum.reset()
			return False
		self._carried_evidence = evidence
		return True

	def match_niche(self, memory, current_individual: dict, hall_of_fame, population: list = None) -> int:
		"""Late Identity Binding: resolve the winner's recent transitions to a niche, refining or creating its model."""
		held = self._held_niche()
		if held is not None:
			return held
		niche_id = self._identify()
		if niche_id == -1:
			return self._provision()
		self._absorb_pending(niche_id)
		self._current_niche_id = niche_id
		return niche_id

	def _held_niche(self):
		"""The niche to resolve to without judging novelty right now, or None if the evidence can be judged."""
		if not self._models:
			return self._provision()
		if self._alarm_niche_id is not None:
			# A pending alarm means the recent buffer may straddle a change: identifying (or
			# provisioning) on it would fit a model to two regimes' mix. Wait for the verdict.
			return self._alarm_niche_id
		current = self._models.get(self._current_niche_id)
		if current is not None and current.count < self._min_samples:
			# The current niche was provisioned from too little data to be judged yet: keep
			# growing it with this regime's transitions instead of comparing against it.
			self._absorb_pending(self._current_niche_id)
			return self._current_niche_id
		if len(self._recent) < self._min_verify_samples:
			# Too little evidence to judge novelty: stay in the current niche without learning from
			# it; with no current niche (e.g. right after a test-time reset), start a new one, which
			# keeps growing — rather than attach to a niche these physics may just have ruled out.
			return self._current_niche_id if current is not None else self._provision()
		return None

	def log_likelihoods(self, memory) -> dict:
		"""{niche_id: log-likelihood of this episode's transitions} for every niche with a usable model."""
		if not self._episode:
			return {}
		features, targets = self._stack(self._episode)
		return {
			niche_id: float(np.sum(model.log_likelihood(features, targets, self._variance_scale)))
			for niche_id, model in self._models.items()
			if model.count >= self._min_samples
		}

	def reset(self) -> None:
		"""On a detected shift: drop the old regime's evidence (keeping the post-alarm transitions) and restart detection."""
		self._recent.clear()
		self._pending.clear()
		self._recent.extend(self._carried_evidence)
		self._pending.extend(self._carried_evidence)
		self._carried_evidence = []
		self._cusum.reset()
		self._alarm_niche_id = None
		self._alarm_evidence = []
		self._current_niche_id = -1

	# ---------------------------------------------------------------
	# Private
	# ---------------------------------------------------------------

	def _features(self, state, action) -> np.ndarray:
		"""[state, action encoding]: one-hot for discrete actions (which also serves as the intercept), [action, 1] for continuous."""
		state = np.asarray(state, dtype=np.float64).ravel()
		if self._is_continuous:
			return np.concatenate([state, np.asarray(action, dtype=np.float64).ravel(), [1.0]])
		return np.concatenate([state, np.eye(self._action_dim)[int(action)]])

	@staticmethod
	def _stack(transitions) -> tuple:
		"""Stack (features, target) pairs into two matrices."""
		return np.array([features for features, _ in transitions]), np.array([target for _, target in transitions])

	def _mean_log_likelihoods(self, transitions) -> dict:
		"""{niche_id: mean per-step log-likelihood of the transitions}, over mature models."""
		features, targets = self._stack(transitions)
		return {
			niche_id: float(np.mean(model.log_likelihood(features, targets, self._variance_scale)))
			for niche_id, model in self._models.items()
			if model.count >= self._min_samples
		}

	def _best_niche_and_gap(self) -> tuple:
		"""Most likely mature niche for the recent transitions, and how many nats/step a fresh fit beats it by."""
		scores = self._mean_log_likelihoods(self._recent)
		if not scores:
			return -1, math.inf
		best_niche_id = max(scores, key=scores.get)
		return best_niche_id, self._gap(best_niche_id, self._recent)

	def _gap(self, niche_id, transitions) -> float:
		"""How many nats/step a model freshly fitted on the transitions beats niche_id's model on them."""
		features, targets = self._stack(transitions)
		fresh = LinearGaussianDynamics.fit(features, targets, var_floor=self._var_floor)
		niche_score = float(np.mean(self._models[niche_id].log_likelihood(features, targets, self._variance_scale)))
		return max(float(np.mean(fresh.log_likelihood(features, targets, self._variance_scale))) - niche_score, 0.0)

	def _identify(self) -> int:
		"""The recognized niche for the recent transitions, or -1 if they're novel."""
		best_niche_id, gap = self._best_niche_and_gap()
		return best_niche_id if gap <= self._novelty_margin else -1

	def _absorb_pending(self, niche_id) -> None:
		"""Refine niche_id's model with the transitions no model has learned from yet."""
		if self._pending:
			self._models[niche_id].add(*self._stack(self._pending))
			self._pending.clear()

	def _provision(self) -> int:
		"""Create a niche whose model is fitted on the recent transitions."""
		niche_id = self._next_niche_id
		self._next_niche_id += 1
		model = LinearGaussianDynamics(self._feature_dim(), self._state_dim, var_floor=self._var_floor)
		if self._recent:
			model.add(*self._stack(self._recent))
		self._models[niche_id] = model
		self._pending.clear()
		self._current_niche_id = niche_id
		return niche_id

	def _feature_dim(self) -> int:
		"""Length of the feature vector _features() builds."""
		return self._state_dim + (self._action_dim + 1 if self._is_continuous else self._action_dim)
