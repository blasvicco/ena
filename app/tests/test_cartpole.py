"""Tests for cartpole.py's play() per-episode history recording."""

# Libs imports
import numpy as np

# App imports
from helpers import step
from cartpole import play


# pylint: disable=too-few-public-methods
class _FakeUnwrapped:
	"""Minimal stand-in for env.unwrapped, exposing only what play() reads."""

	name = "Fake-Env"


class _FakeEnv:
	"""A minimal env stub: one step per episode, always terminates immediately."""

	def __init__(self):
		self.unwrapped = _FakeUnwrapped()

	def reset(self):
		"""Return a zeroed observation."""
		return np.zeros(4, dtype=np.float32), {}

	def step(self, _action):
		"""Terminate on the first step, awarding a fixed reward."""
		return np.zeros(4, dtype=np.float32), 1.0, True, False, {}


class _FakeAgent:
	"""A minimal agent stub exercising exactly the attributes play() reads."""

	def __init__(self):
		self.learn = False
		self.best_individual_idx = 0
		self.current_env_id = 7
		self.last_env_shift_detected = True
		self.population = [{"is_gladiator": True, "internal_env_id": 3, "env_name": "Mars-Light"}]

	def act(self, _env, state):
		"""Return a step tuple that always ends the episode."""
		return state, 1.0, True, False, 0

	def train(self, **_kwargs):
		"""Must not be called when agent.learn is False."""
		raise AssertionError("train() must not be called when agent.learn is False")


def test_play_records_env_shift_and_internal_env_id_columns() -> None:
	"""C1/C2: play() must record the acting individual's own niche id and the
	real detect_change() flag, not just the recognizer's global current_env_id."""
	with step("Arrange: build a fake agent/env pair with a scripted single-step episode."):
		agent = _FakeAgent()

	with step("Act: play one episode."):
		history = play(agent, [_FakeEnv()], [1])

	with step("Assert: the recorded row reflects the individual's own niche and detect_change()."):
		row = history.iloc[0]
		assert bool(row["env_shift_detected"]) is True
		assert row["internal_env_id"] == 3
		assert row["detected_env_id"] == 7
		assert bool(row["is_gladiator"]) is True
		assert row["gladiator_env"] == "Mars-Light"
