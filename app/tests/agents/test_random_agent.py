"""Tests for RandomAgent's action sampling and no-op training/mode-switching."""

# Libs imports
import gymnasium as gym
import numpy as np

# App imports
from helpers import step
from agents.random import RandomAgent


# pylint: disable=too-few-public-methods
class _FakeEnv:
	"""Minimal env stub: discrete action space, terminates on the first step."""

	def __init__(self) -> None:
		self.action_space = gym.spaces.Discrete(2)

	def step(self, action):
		"""Terminate on the first step, echoing the action taken via reward."""
		return np.zeros(4, dtype=np.float32), 1.0, True, False, {"action_taken": action}


def test_act_samples_a_valid_action_and_steps_the_env() -> None:
	"""act() must sample from the env's own action space and step it."""
	with step("Arrange: a RandomAgent and a fake discrete-action env."):
		agent = RandomAgent(brain=None, make_env=_FakeEnv)
		env = _FakeEnv()
		state = np.zeros(4, dtype=np.float32)

	with step("Act: call act()."):
		next_state, reward, terminated, truncated, action = agent.act(env, state)

	with step("Assert: the sampled action is valid, and the step result is passed through."):
		assert env.action_space.contains(action)
		assert reward == 1.0
		assert terminated is True
		assert truncated is False
		assert next_state.shape == (4,)


def test_train_and_set_learning_are_safe_no_ops() -> None:
	"""train()/set_learning() must not raise, and learn stays False regardless."""
	with step("Arrange: a RandomAgent."):
		agent = RandomAgent(brain=None, make_env=_FakeEnv)

	with step("Act: call train() and set_learning(True)."):
		agent.train(env=None, step_data={})
		agent.set_learning(True)

	with step("Assert: learn stays False — RandomAgent has no train/test distinction."):
		assert agent.learn is False
