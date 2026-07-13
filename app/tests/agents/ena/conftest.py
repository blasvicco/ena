"""Shared fixtures for ENAgent-level tests."""

# Libs imports
import gymnasium as gym
import pytest
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from agents.ena.ena import ENAgent


def _make_env(**_kwargs):
	"""Build a real CartPole env — cheap enough for fast tests."""
	return gym.make("CartPole-v1")


def _brain(action_dim, state_dim):
	"""Build a minimal one-layer model, fast to construct and mutate."""
	return Sequential([Input(shape=(state_dim,)), Dense(action_dim)])


@pytest.fixture
def build_agent():
	"""Factory fixture: build_agent(**overrides) -> a small, fast ENAgent."""

	def _build(**overrides):
		kwargs = {
			"brain": _brain,
			"make_env": _make_env,
			"pop_size": 4,
			"max_threads": 1,
			"history_size": 20,
			"max_eval_steps": 20,
			"recognizer_ema_alpha": 0.5,
		}
		kwargs.update(overrides)
		return ENAgent(**kwargs)

	return _build


@pytest.fixture
def agent(build_agent):  # pylint: disable=redefined-outer-name
	"""A small ENAgent instance, fresh per test, default configuration."""
	return build_agent()
