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


def test_none_driver_match_always_confirms_niche_zero() -> None:
	"""match() and match_niche() both resolve confidently to niche 0."""
	with step("Act/Assert: match() and match_niche() both report niche 0."):
		recognizer = Instance(state_dim=4)
		assert recognizer.match(memory=None) == (0, 1.0)
		assert recognizer.match_niche(memory=None, current_individual={}, hall_of_fame=None) == 0


def test_none_driver_vector_shapes_match_the_live_driver() -> None:
	"""context_dim must stay consistent regardless of which driver is chosen."""
	with step("Act/Assert: physical/full vector shapes match 4*state_dim and 4*state_dim+2."):
		recognizer = Instance(state_dim=4)
		assert recognizer.compute_physical_vector(memory=None).shape == (16,)
		assert recognizer.compute_full_vector(memory=None).shape == (18,)
