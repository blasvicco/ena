"""Environment-detection driver: a fixed probing excitation, its response's features compared per niche."""

# Method: system identification with a known excitation (Ljung, "System Identification: Theory for the User",
# 1999), whose response's (wavelet) features detect a change of the physics, as in structural damage detection
# and machine fault diagnosis (Kijewski & Kareem 2003; Peng & Chu 2004). Probing the environment to infer its
# dynamics in RL: Zhou, Pinto & Gupta, "Environment Probing Interaction Policies" (ICLR 2019), who learn the
# chirp; here it is fixed, so the response cannot depend on the agent.

# General imports
import copy
from collections import deque

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.env_detection.driver.abstract import ADriver
from agents.ena.tools.chirp_features import features
from agents.ena.tools.chirp_signals import excitation, to_discrete

DEFAULTS = {
	"steps": 64,  # chirp length (a power of 2 for the dyadic transforms)
	"excitation": "wavelet_train",  # tools/chirp_signals.EXCITATIONS (chosen on the detection ablation)
	"excitation_params": {"wavelet": "mexh"},  # the Mexican hat; e.g. {"freq_start": 0.01, "freq_end": 0.25} for a linear chirp
	"features": {"kind": "wavelet", "transform": "packet", "level": 3},  # tools/chirp_features.features: Haar wavelet-packet energies
	"relative": True,  # response = observations minus the chirp's start observation
	"window": 4,  # chirps whose mean is tested against the current niche
	"min_chirps": 8,  # a niche is judged only once it holds this many chirps
	"margin": 3.0,  # change/novelty threshold on the window statistic (about 1 under the same physics)
	"var_floor": 1e-6,
}


def _merge(defaults: dict, overrides: dict) -> dict:
	"""defaults with the given settings replacing them key by key; an unknown key is an error.

	A given dict (features, excitation_params) replaces the default dict whole, never merges into it: a configuration is
	then fully determined by what it passes (its missing keys fall back to the feature or signal function's defaults),
	so results stay reproducible when DEFAULTS change."""
	unknown = set(overrides) - set(defaults)
	if unknown:
		raise ValueError(f"Unknown chirp settings: {sorted(unknown)}; expected {sorted(defaults)}")
	merged = copy.deepcopy(defaults)
	merged.update(copy.deepcopy(overrides))
	return merged


# pylint: disable=too-many-instance-attributes
class Instance(ADriver):
	"""Niches are the response signatures of a fixed chirp; a change is a window of chirps the current niche cannot explain."""

	# The chirp is the same sequence of actions in every environment, from the environment's own reset
	# distribution, so its response differs only through the physics (and reset noise): detection no longer
	# depends on what the acting individual does (dynamics_model's weakness in HalfCheetah, where each new
	# behavior visits new contact states). ENAgent plays it at every episode end (observe_chirp), in the real
	# environment, counted in its step budget.
	#
	# Each niche keeps the mean of its chirps' feature vectors; the per-feature spread is pooled over niches
	# (it comes from the reset noise, shared by every environment). The window statistic is the RMS over
	# features of (window mean - niche mean) / (pooled std * sqrt(1/window + 1/niche count)): about 1 under the
	# same physics. Above `margin` the current niche no longer explains the window: a change is confirmed, and
	# the most recent half of the window identifies the new physics (a known niche within the margin, else a
	# new niche in training; at test time, no niche is ever created).

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(self, state_dim, action_dim=1, is_continuous=False, chirp=None, **_kwargs):
		self._settings = _merge(DEFAULTS, chirp or {})
		self._state_dim = state_dim
		signal = excitation(self._settings["excitation"], self._settings["steps"], action_dim, **self._settings["excitation_params"])
		self._actions = signal if is_continuous else to_discrete(signal)
		self._niches = {}  # niche id -> {"sum", "sumsq", "count"} of its chirps' features
		self._next_niche_id = 0
		self._current_niche_id = -1
		self._window = deque(maxlen=self._settings["window"])
		self._carried = []  # a confirmed change's recent chirps: the new physics' first evidence
		self._pending = None  # this episode's chirp features, until detect_change
		self._last = None  # the latest chirp features (log_likelihoods)

	@property
	def chirp_actions(self):
		"""The fixed chirp: (steps, action_dim) actions (continuous) or (steps,) action indices (discrete)."""
		return self._actions

	@property
	def current_niche_id(self) -> int:
		"""Niche the chirps identified, or -1 if none (yet)."""
		return self._current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Niches with chirp statistics."""
		return set(self._niches)

	def observe_chirp(self, response) -> None:
		"""One chirp's observations (start observation first) -> its feature vector, decided at detect_change."""
		response = np.asarray(response, dtype=np.float64)
		if self._settings["relative"]:
			response = response[1:] - response[0]
		self._pending = self._last = features(response, **self._settings["features"])

	def detect_change(self, memory, learn: bool = True) -> bool:
		"""Test this episode's chirp (with the window) against the current niche; True on a confirmed change."""
		if self._pending is None:
			return False
		chirp, self._pending = self._pending, None
		if self._current_niche_id == -1:
			evidence, self._carried = self._carried + [chirp], []
			self.__identify(evidence, learn)
			self._window.clear()
			return False
		niche = self._niches[self._current_niche_id]
		if niche["count"] < self._settings["min_chirps"] or self.__pooled_variance() is None:
			if learn:
				self.__absorb(self._current_niche_id, [chirp])
			return False
		self._window.append(chirp)
		if len(self._window) < self._window.maxlen:
			return False
		if self.__statistic(list(self._window), self._current_niche_id) <= self._settings["margin"]:
			if learn:
				self.__absorb(self._current_niche_id, [chirp])
			return False
		recent = list(self._window)[-max(1, len(self._window) // 2) :]
		self._carried = recent
		self._current_niche_id = -1
		self._window.clear()
		return True

	def match_niche(self, memory, current_individual: dict, hall_of_fame, population: list = None) -> int:
		"""Late Identity Binding: the chirps' niche (identified from a pending change's evidence, or provisioned if none)."""
		if self._current_niche_id == -1 and self._carried:
			evidence, self._carried = self._carried, []
			self.__identify(evidence, learn=True)
		if self._current_niche_id == -1:
			self._current_niche_id = self.__provision([])
		return self._current_niche_id

	def start_episode(self) -> None:
		"""Nothing per episode: the evidence is the chirp."""

	def observe(self, state, action, next_state, memory) -> None:
		"""The acting individual's transitions are not used."""

	def log_likelihoods(self, memory) -> dict:
		"""{niche_id: Gaussian log-likelihood of the latest chirp} over judged niches (BPR's dynamics evidence)."""
		variance = self.__pooled_variance()
		if self._last is None or variance is None:
			return {}
		return {
			niche_id: float(-0.5 * np.sum((self._last - stats["sum"] / stats["count"]) ** 2 / variance))
			for niche_id, stats in self._niches.items()
			if stats["count"] >= self._settings["min_chirps"]
		}

	def reset(self) -> None:
		"""On a detected shift (or entering test mode): no current niche until the next chirps identify it."""
		self._window.clear()
		self._pending = None
		self._current_niche_id = -1

	# ---------------------------------------------------------------
	# Private
	# ---------------------------------------------------------------

	def __identify(self, evidence: list, learn: bool) -> None:
		"""The judged niche explaining the evidence within the margin; else a new niche (training) or none (test)."""
		judged = [niche_id for niche_id, stats in self._niches.items() if stats["count"] >= self._settings["min_chirps"]]
		if judged and self.__pooled_variance() is not None:
			scores = {niche_id: self.__statistic(evidence, niche_id) for niche_id in judged}
			best = min(scores, key=scores.get)
			if scores[best] <= self._settings["margin"]:
				self._current_niche_id = best
				if learn:
					self.__absorb(best, evidence)
				return
		if not learn:
			self._current_niche_id = -1
			return
		immature = [niche_id for niche_id, stats in self._niches.items() if stats["count"] < self._settings["min_chirps"]]
		if immature:
			# A niche still growing (provisioned before its chirps arrived) takes the evidence.
			self._current_niche_id = immature[-1]
			self.__absorb(immature[-1], evidence)
			return
		self._current_niche_id = self.__provision(evidence)

	def __provision(self, evidence: list) -> int:
		"""A new niche holding the evidence's chirps."""
		niche_id = self._next_niche_id
		self._next_niche_id += 1
		self._niches[niche_id] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
		self.__absorb(niche_id, evidence)
		return niche_id

	def __absorb(self, niche_id: int, chirps: list) -> None:
		"""Add chirps to a niche's statistics."""
		stats = self._niches[niche_id]
		for chirp in chirps:
			stats["sum"] = stats["sum"] + chirp
			stats["sumsq"] = stats["sumsq"] + chirp * chirp
			stats["count"] += 1

	def __pooled_variance(self):
		"""Per-feature within-niche variance pooled over niches with at least two chirps, or None."""
		squares, degrees = 0.0, 0
		for stats in self._niches.values():
			if stats["count"] >= 2:
				squares = squares + stats["sumsq"] - stats["sum"] ** 2 / stats["count"]
				degrees += stats["count"] - 1
		if degrees < 2:
			return None
		return np.maximum(squares / degrees, self._settings["var_floor"])

	def __statistic(self, chirps: list, niche_id: int) -> float:
		"""RMS over features of the window mean's deviation from the niche mean, in standard errors (about 1 if the same)."""
		stats = self._niches[niche_id]
		standard_error = np.sqrt(self.__pooled_variance() * (1.0 / len(chirps) + 1.0 / stats["count"]))
		deviation = (np.mean(chirps, axis=0) - stats["sum"] / stats["count"]) / standard_error
		return float(np.sqrt(np.mean(deviation**2)))
