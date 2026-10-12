"""Tests for SAC's fused update: the gradient steps of one update in a single compiled call, same algorithm."""

# Libs imports
import keras
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents import SACAgent

# pylint: disable=protected-access


def _brain(output_dim, input_dim):
	"""Small MLP factory."""
	return Sequential([Input(shape=(input_dim,)), Dense(8, activation="tanh"), Dense(output_dim)])


class _BoxEnv:
	"""Fixed observation, one continuous action, one-step episodes."""

	def __init__(self):
		self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
		self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
		self.unwrapped = self

	def reset(self, seed=None):  # pylint: disable=unused-argument
		"""A fixed observation."""
		return np.array([0.5, -0.5], dtype=np.float32), {}

	def step(self, action):
		"""Terminate immediately."""
		return np.array([0.5, -0.5], dtype=np.float32), float(np.asarray(action).sum()), True, False, {}

	def close(self):
		"""No-op."""


def _batches(agent, count, seed=0):
	"""count stacked (states, actions, rewards, next_states, dones) batches of the agent's batch size."""
	rng = np.random.default_rng(seed)
	size, state_dim, action_dim = agent.batch_size, agent.state_dim, agent.action_dim
	return (
		rng.normal(size=(count, size, state_dim)).astype(np.float32),
		rng.uniform(-1, 1, size=(count, size, action_dim)).astype(np.float32),
		rng.normal(size=(count, size, 1)).astype(np.float32),
		rng.normal(size=(count, size, state_dim)).astype(np.float32),
		np.zeros((count, size, 1), dtype=np.float32),
	)


def _noise(agent, count, seed=1):
	"""Fixed reparameterization noise for each step: (next-state noise, current-state noise)."""
	rng = np.random.default_rng(seed)
	shape = (count, agent.batch_size, agent.action_dim)
	return rng.normal(size=shape).astype(np.float32), rng.normal(size=shape).astype(np.float32)


def _weights(agent):
	return [weights.copy() for network in (agent.actor, agent.critic_1, agent.critic_2, agent.target_critic_1) for weights in network.get_weights()]


def test_a_fused_update_equals_the_same_steps_taken_one_by_one() -> None:
	"""Same batches, same noise: one fused call of k steps leaves every network and alpha where k single steps do."""
	with step("Arrange: two identically initialized agents, 4 batches and their noise."):
		keras.utils.set_random_seed(0)
		single = SACAgent(brain=_brain, make_env=_BoxEnv, batch_size=8)
		fused = SACAgent(brain=_brain, make_env=_BoxEnv, batch_size=8)
		for source, target in ((single.actor, fused.actor), (single.critic_1, fused.critic_1), (single.critic_2, fused.critic_2)):
			target.set_weights(source.get_weights())
		fused.target_critic_1.set_weights(single.target_critic_1.get_weights())
		fused.target_critic_2.set_weights(single.target_critic_2.get_weights())
		batches, (next_noise, current_noise) = _batches(single, 4), _noise(single, 4)

	with step("Act: 4 single steps vs one fused call."):
		for index in range(4):
			single._sac_step(*(tf.constant(array[index]) for array in batches), tf.constant(next_noise[index]), tf.constant(current_noise[index]))
		fused._sac_steps(*(tf.constant(array) for array in batches), tf.constant(next_noise), tf.constant(current_noise))

	with step("Assert: same weights and temperature."):
		for left, right in zip(_weights(single), _weights(fused)):
			assert np.allclose(left, right, atol=1e-5)
		assert np.isclose(float(single.log_alpha.numpy()), float(fused.log_alpha.numpy()), atol=1e-6)


def test_fused_training_runs_one_compiled_call_per_update(monkeypatch) -> None:
	"""With fused_updates, train() samples the update's batches and calls the fused step once, not gradient_steps times."""
	with step("Arrange: a fused agent past warm-up with a full buffer; count the calls."):
		agent = SACAgent(brain=_brain, make_env=_BoxEnv, batch_size=2, update_frequency=4, fused_updates=True)
		calls = []
		monkeypatch.setattr(agent, "_sac_steps", lambda *args: calls.append(args[0].shape[0]))
		state = np.zeros(agent.state_dim, dtype=np.float32)
		step_data = {
			"state": state,
			"action": np.zeros(agent.action_dim, dtype=np.float32),
			"reward": 0.0,
			"next_state": state,
			"terminated": False,
			"done": False,
		}

	with step("Act: 8 training steps."):
		for _ in range(8):
			agent.train(step_data=step_data)

	with step("Assert: two updates, each one call of 4 stacked batches."):
		assert calls == [4, 4]


def test_the_default_keeps_the_single_step_update() -> None:
	"""Without the arguments SAC updates one step per call, as before; XLA is off."""
	with step("Act/Assert: defaults."):
		agent = SACAgent(brain=_brain, make_env=_BoxEnv)
		assert agent.fused_updates is False and agent.jit_compile is False
