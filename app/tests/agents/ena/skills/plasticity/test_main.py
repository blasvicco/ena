"""Tests for the Plasticity facade's learn-mode gate."""

# General imports
from types import SimpleNamespace

# App imports
from helpers import step
from agents.ena.skills.plasticity.main import Plasticity


def test_compute_forces_zero_when_not_learning() -> None:
	"""The Plasticity facade returns zero when not learning, whatever the configured driver."""
	# Test mode must never trigger adaptive search effort, regardless of the configured
	# plasticity driver — the facade's own gate, not the curve tool (see test_curves.py).
	with step("Arrange: a high-scoring reward history that would otherwise yield low plasticity."):
		plasticity = Plasticity("linear_ramp")
		memory = SimpleNamespace(reward_history=[500.0] * 10, min_seen=0.0)

	with step("Assert: plasticity is forced to 0.0 when learn=False."):
		assert plasticity.compute(500.0, memory, learn=False) == 0.0


def test_compute_delegates_to_the_driver_when_learning() -> None:
	"""When learning, the facade must not itself compute a value — it delegates to the driver."""
	with step("Arrange: the zero driver, which always returns 0.0 regardless of input."):
		plasticity = Plasticity("full")
		memory = SimpleNamespace(reward_history=[], min_seen=0.0)

	with step("Assert: the driver's own value (1.0 for 'full') is returned when learn=True."):
		assert plasticity.compute(0.0, memory, learn=True) == 1.0
