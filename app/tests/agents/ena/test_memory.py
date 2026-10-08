"""Tests for AgentMemory's history-clearing helpers."""

# App imports
from helpers import step
from agents.ena.memory import AgentMemory


def test_clear_training_history_resets_reward_state() -> None:
	"""Called on a training-time environment shift: the new regime's returns must not mix with the old one's."""
	with step("Arrange: build a memory instance and seed it with reward/step history."):
		memory = AgentMemory(history_size=10, state_dim=4)
		memory.reward_history.extend([1.0, 2.0, 3.0])
		memory.max_seen = 5.0
		memory.min_seen = -5.0

	with step("Act: clear the training history."):
		memory.clear_training_history()

	with step("Assert: reward history and min/max bounds are reset."):
		assert not list(memory.reward_history)
		assert memory.max_seen == float("-inf")
		assert memory.min_seen == float("inf")
