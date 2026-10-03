"""Tests for the SAC-Discrete baseline agent."""

# Libs imports
import numpy as np
import pytest
from gymnasium import spaces
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents import SACDiscreteAgent


def _brain(output_dim, input_dim):
	"""Small MLP factory, fast to build."""
	return Sequential([Input(shape=(input_dim,)), Dense(16, activation="relu"), Dense(output_dim)])


class _BanditEnv:
	"""One-step episodes; action 1 pays 1, action 0 pays 0 — the optimal policy is 'always 1'."""

	def __init__(self):
		self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
		self.action_space = spaces.Discrete(2)
		self.unwrapped = self

	def reset(self, seed=None):  # pylint: disable=unused-argument
		"""A fixed observation."""
		return np.array([1.0, 0.0], dtype=np.float32), {}

	def step(self, action):
		"""Pay 1 for action 1, then terminate."""
		return np.array([1.0, 0.0], dtype=np.float32), float(action == 1), True, False, {}

	def close(self):
		"""No-op."""


def _agent(**overrides):
	kwargs = {"brain": _brain, "make_env": _BanditEnv, "learning_starts": 64, "batch_size": 32, "train_freq": 1, "learning_rate": 3e-3}
	kwargs.update(overrides)
	return SACDiscreteAgent(**kwargs)


def _train(agent, env, steps):
	"""Run `steps` one-step episodes through act()+train()."""
	for _ in range(steps):
		state, _ = env.reset()
		next_state, reward, terminated, truncated, action = agent.act(env, state)
		agent.train(
			step_data={
				"state": state,
				"action": action,
				"reward": reward,
				"next_state": next_state,
				"done": terminated or truncated,
				"terminated": terminated,
			}
		)


def test_act_returns_a_valid_discrete_action() -> None:
	"""Actions are integer indices into the discrete action space."""
	with step("Arrange/Act: one step in training and one in evaluation mode."):
		agent, env = _agent(), _BanditEnv()
		state, _ = env.reset()
		training_action = agent.act(env, state)[4]
		agent.set_learning(False, output_dir=None)
		eval_action = agent.act(env, state)[4]

	with step("Assert: both are valid actions."):
		assert training_action in (0, 1)
		assert eval_action in (0, 1)


def test_learns_the_optimal_action_on_a_two_armed_bandit() -> None:
	"""After training, the greedy (evaluation) policy picks the paying action."""
	with step("Arrange: a fresh agent."):
		np.random.seed(0)
		agent, env = _agent(), _BanditEnv()

	with step("Act: 600 training steps, then evaluate greedily."):
		_train(agent, env, 600)
		agent.set_learning(False)
		state, _ = env.reset()
		probabilities = agent._policy(state[np.newaxis]).numpy()[0]  # pylint: disable=protected-access
		q_values = agent.critic_1(state[np.newaxis]).numpy()[0]
		greedy_action = agent.act(env, state)[4]

	with step("Assert: greedy picks action 1, the policy prefers it, and Q(s,1) - Q(s,0) ~ the 1.0 reward gap."):
		# The stochastic policy stays near-uniform by design: the paper's target entropy, 0.98 of
		# log|A|, is almost maximal for two actions — the preference shows in Q and the argmax.
		assert greedy_action == 1
		assert probabilities[1] > probabilities[0]
		assert q_values[1] - q_values[0] == pytest.approx(1.0, abs=0.25)


def test_no_update_runs_before_learning_starts() -> None:
	"""The networks are untouched until learning_starts transitions are stored."""
	with step("Arrange: an agent that starts learning at 100 transitions."):
		agent, env = _agent(learning_starts=100), _BanditEnv()
		before = [weights.copy() for weights in agent.actor.get_weights()]

	with step("Act: 99 transitions."):
		_train(agent, env, 99)

	with step("Assert: actor weights unchanged."):
		for old, new in zip(before, agent.actor.get_weights()):
			assert np.array_equal(old, new)


@pytest.mark.parametrize("mode", [True, False])
def test_set_learning_accepts_the_runner_output_dir_keyword(mode) -> None:
	"""The study runner calls set_learning(mode, output_dir=...) on every agent."""
	with step("Act/Assert: no TypeError, and the mode is applied."):
		agent = _agent()
		agent.set_learning(mode, output_dir="/tmp")
		assert agent.learn is mode
