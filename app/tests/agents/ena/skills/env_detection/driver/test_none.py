"""Tests for the none env_detection driver — the ablation-study disable switch."""

# App imports
from helpers import step
from agents.ena.skills.env_detection.driver.none import Instance


def test_none_driver_always_resolves_to_niche_zero() -> None:
	"""There is only ever one niche when detection is disabled."""
	with step("Act/Assert: current_niche_id and known_niche_ids both report niche 0 only."):
		recognizer = Instance(state_dim=4)
		assert recognizer.current_niche_id == 0
		assert recognizer.known_niche_ids == {0}


def test_none_driver_never_detects_a_change() -> None:
	"""detect_change() must always return False, regardless of memory state."""
	with step("Act/Assert: detect_change() returns False."):
		recognizer = Instance(state_dim=4)
		assert recognizer.detect_change(memory=None) is False


def test_none_driver_match_niche_always_resolves_niche_zero() -> None:
	"""match_niche() resolves every winner to niche 0."""
	with step("Act/Assert: match_niche() reports niche 0."):
		recognizer = Instance(state_dim=4)
		assert recognizer.match_niche(memory=None, current_individual={}, hall_of_fame=None) == 0
