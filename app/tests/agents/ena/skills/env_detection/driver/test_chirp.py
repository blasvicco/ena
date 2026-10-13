"""Tests for the chirp environment-detection driver: a fixed excitation's response, compared per niche."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.skills.env_detection.main import EnvironmentRecognizer

STATE_DIM, STEPS = 2, 16
SETTINGS = {"steps": STEPS, "features": {"kind": "raw"}, "window": 3, "min_chirps": 5, "margin": 3.0}
PHYSICS = {"A": 0.0, "B": 1.0, "C": -1.0}  # each physics shifts the response by this much


def _detector(learn_settings=None, continuous=True, action_dim=3):
	return EnvironmentRecognizer("chirp", STATE_DIM, action_dim=action_dim, is_continuous=continuous, chirp={**SETTINGS, **(learn_settings or {})})


def _response(physics, rng):
	"""Observations of one chirp: the start observation, then STEPS responses (offset by the physics, plus reset noise)."""
	return np.vstack([np.zeros((1, STATE_DIM)), PHYSICS[physics] + 0.1 * rng.normal(size=(STEPS, STATE_DIM))])


def _episode(detector, physics, rng, learn=True):
	"""One episode end as ENA runs it: the chirp's response, then the change check."""
	detector.observe_chirp(_response(physics, rng))
	shifted = detector.evaluate(memory=_Memory(), learn=learn)
	return shifted


class _Memory:  # pylint: disable=too-few-public-methods
	"""What the facade touches on a shift."""

	def __init__(self):
		self.obs_normalizer = type("Normalizer", (), {"reset": lambda self: None})()

	def clear_training_history(self):
		"""Nothing to clear."""


def test_the_chirp_is_the_configured_excitation_for_the_action_space() -> None:
	"""Continuous: (steps, action_dim) torques; discrete: one action per step; other drivers do not chirp."""
	with step("Act: a continuous and a discrete chirp driver, and a dynamics_model one."):
		continuous = _detector().chirp_actions
		discrete = _detector(continuous=False, action_dim=2).chirp_actions
		passive = EnvironmentRecognizer("dynamics_model", STATE_DIM).chirp_actions

	with step("Assert: shapes and types."):
		assert continuous.shape == (STEPS, 3)
		assert discrete.shape == (STEPS,) and set(discrete) <= {0, 1}
		assert passive is None


def test_the_same_physics_stays_one_quiet_niche() -> None:
	"""40 episodes of physics A: no change, a single niche."""
	with step("Act: 40 chirps of A."):
		detector, rng = _detector(), np.random.default_rng(0)
		shifts = [_episode(detector, "A", rng) for _ in range(40)]

	with step("Assert: quiet; one niche."):
		assert not any(shifts)
		assert detector.known_niche_ids == {0} and detector.current_niche_id == 0


def test_a_switch_is_confirmed_within_the_window_and_a_recurrence_is_recognized() -> None:
	"""A -> B: one change within `window` episodes and a new niche; B -> A: a change back to A's niche, nothing new."""
	with step("Arrange: 20 episodes of A."):
		detector, rng = _detector(), np.random.default_rng(1)
		for _ in range(20):
			_episode(detector, "A", rng)

	with step("Act: 20 episodes of B, then 20 of A again."):
		to_b = [_episode(detector, "B", rng) for _ in range(20)]
		niche_b = detector.current_niche_id
		back = [_episode(detector, "A", rng) for _ in range(20)]

	with step("Assert: one change each way, early; B is a new niche; A is recognized."):
		assert sum(to_b) == 1 and to_b.index(True) < SETTINGS["window"]
		assert niche_b not in (0, -1)
		assert sum(back) == 1 and back.index(True) < SETTINGS["window"]
		assert detector.current_niche_id == 0 and detector.known_niche_ids == {0, niche_b}


def test_test_time_never_creates_a_niche() -> None:
	"""Frozen: a novel physics is detected but resolves to no niche (-1); a known one to its niche."""
	with step("Arrange: niches for A and B learned in training."):
		detector, rng = _detector(), np.random.default_rng(2)
		for physics in ("A", "B"):
			for _ in range(15):
				_episode(detector, physics, rng)
		known = set(detector.known_niche_ids)
		detector.reset()

	with step("Act: test episodes of C (novel), then of A."):
		for _ in range(6):
			_episode(detector, "C", rng, learn=False)
		novel = detector.current_niche_id
		for _ in range(6):
			_episode(detector, "A", rng, learn=False)

	with step("Assert: no new niche; C unknown; A recognized."):
		assert detector.known_niche_ids == known
		assert novel == -1 and detector.current_niche_id == 0


def test_log_likelihoods_rank_the_true_niche_first() -> None:
	"""BPR's dynamics evidence: the latest chirp is likeliest under its own physics' niche."""
	with step("Arrange: niches for A and B; then a chirp of A."):
		detector, rng = _detector(), np.random.default_rng(3)
		for physics in ("A", "B"):
			for _ in range(15):
				_episode(detector, physics, rng)
		niche_a = 0
		detector.observe_chirp(_response("A", rng))

	with step("Assert: A's niche has the highest log-likelihood."):
		likelihoods = detector.log_likelihoods(memory=None)
		assert max(likelihoods, key=likelihoods.get) == niche_a


def test_match_niche_binds_gladiators_to_the_current_niche() -> None:
	"""Late identity binding resolves to the niche the chirps identified (one is provisioned before any chirp)."""
	with step("Act/Assert: before any chirp, a niche is provisioned; after A's chirps, A's niche."):
		detector, rng = _detector(), np.random.default_rng(4)
		assert detector.match_niche(None, None, None) == 0
		for _ in range(8):
			_episode(detector, "A", rng)
		assert detector.match_niche(None, None, None) == 0


def test_unknown_chirp_settings_are_rejected() -> None:
	"""A typo in the settings fails at construction."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError, match="chirp"):
			_detector({"windwo": 3})


def test_given_features_and_excitation_parameters_replace_the_defaults_whole() -> None:
	"""A configuration is fully determined by what it passes: features {'kind': 'wavelet'} is the feature function's own default
	(a Haar DWT), never merged with the driver's default packets; likewise excitation parameters."""
	# pylint: disable=protected-access
	with step("Act: a chirp driver given plain wavelet features and empty excitation parameters for a linear chirp."):
		detector = EnvironmentRecognizer(
			"chirp",
			STATE_DIM,
			action_dim=3,
			is_continuous=True,
			chirp={"features": {"kind": "wavelet"}, "excitation": "chirp", "excitation_params": {}},
		)._instance  # pylint: disable=protected-access

	with step("Assert: exactly what was given."):
		assert detector._settings["features"] == {"kind": "wavelet"}  # pylint: disable=protected-access
		assert detector._settings["excitation_params"] == {}  # pylint: disable=protected-access
