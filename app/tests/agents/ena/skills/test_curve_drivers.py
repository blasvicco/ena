"""Tests for the Plasticity and Trust skills' curve drivers after the linear_ramp rename."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
from agents.ena.skills.plasticity.main import Plasticity
from agents.ena.skills.trust.main import Trust
from agents.ena.tools import curves

MEMORY = SimpleNamespace(reward_history=[float(score) for score in range(0, 500, 25)], min_seen=0.0, max_seen=475.0, performance_derivative=0.0)


def _evaluate(skill_class, driver, reward):
	"""The driver's output for one reward, through the skill's public method."""
	skill = skill_class(driver)
	return skill.compute(reward, MEMORY, learn=True) if skill_class is Plasticity else skill.decay(reward, MEMORY)


@pytest.mark.parametrize("skill_class", [Plasticity, Trust])
@pytest.mark.parametrize("reward", [0.0, 150.0, 300.0, 480.0])
def test_linear_ramp_driver_is_the_linear_ramp_curve(skill_class, reward) -> None:
	"""Both skills' linear_ramp driver returns exactly the shared curve's value."""
	with step("Act/Assert: driver output equals curves.linear_ramp."):
		assert _evaluate(skill_class, "linear_ramp", reward) == pytest.approx(curves.linear_ramp(reward, MEMORY))


def test_trust_has_no_fuzzylogic_driver() -> None:
	"""The old name must not silently keep selecting the linear ramp (it was never fuzzy logic)."""
	with step("Act/Assert: Trust has no driver by that name."):
		with pytest.raises(ValueError):
			Trust("fuzzylogic")


def test_plasticity_fuzzylogic_is_not_the_linear_ramp() -> None:
	"""Plasticity's fuzzylogic is the real (Mamdani) controller, not the renamed ramp under its old name."""
	with step("Act/Assert: its output differs from the ramp for a mid-range score."):
		assert _evaluate(Plasticity, "fuzzylogic", 300.0) != pytest.approx(curves.linear_ramp(300.0, MEMORY))
