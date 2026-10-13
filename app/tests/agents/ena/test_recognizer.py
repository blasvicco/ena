"""Tests for the pluggable EnvironmentRecognizer facade (driver-specific behavior lives in skills/env_detection/driver/)."""

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.skills.env_detection.main import EnvironmentRecognizer


def test_facade_constructs_the_named_driver() -> None:
	"""The facade resolves a driver by name; a fresh dynamics_model knows no niche yet."""
	with step("Act: build a recognizer with the dynamics_model driver."):
		recognizer = EnvironmentRecognizer("dynamics_model", state_dim=4, action_dim=2)

	with step("Assert: it behaves like a recognizer with no known niches yet."):
		assert recognizer.known_niche_ids == set()
		assert recognizer.current_niche_id == -1


def test_facade_rejects_an_unknown_strategy_name() -> None:
	"""An unrecognized driver name must fail fast, not silently misconfigure."""
	with step("Act/Assert: constructing with a bogus driver name raises ValueError."):
		with pytest.raises(ValueError):
			EnvironmentRecognizer("bogus", state_dim=4)
