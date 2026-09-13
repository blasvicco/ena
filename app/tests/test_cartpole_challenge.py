"""Tests for cartpole_challenge.py's play() per-episode history recording."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
import pandas as pd
import pytest

# App imports
from helpers import step
import cartpole_challenge
from cartpole_challenge import make_eta_reporter, play


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


def test_play_without_live_log_path_creates_no_file(tmp_path) -> None:
	"""S6: omitting live_log_path must leave behavior exactly as before — no file created."""
	with step("Arrange: a fake agent/env pair, and a directory that must stay empty."):
		agent = _FakeAgent()

	with step("Act: play two episodes with no live_log_path."):
		play(agent, [_FakeEnv()], [2])

	with step("Assert: nothing was written to disk."):
		assert not list(tmp_path.iterdir())


def test_play_with_live_log_path_mirrors_the_in_memory_history(tmp_path) -> None:
	"""S6: with live_log_path set, the CSV on disk must match the in-memory history exactly,
	row for row — this is what lets a separate process watch convergence live."""
	with step("Arrange: a fake agent/env pair and a live_log_path under a temp directory."):
		agent = _FakeAgent()
		live_log_path = tmp_path / "live_training.csv"

	with step("Act: play three episodes with live_log_path set."):
		history = play(agent, [_FakeEnv()], [3], live_log_path=str(live_log_path))

	with step("Assert: the CSV has one row per episode, matching the in-memory history."):
		assert live_log_path.exists()
		logged = pd.read_csv(live_log_path)
		assert len(logged) == len(history) == 3
		assert logged["episode"].tolist() == history["episode"].tolist()
		assert logged["score"].tolist() == history["score"].tolist()
		assert logged["detected_env_id"].tolist() == history["detected_env_id"].tolist()


def test_play_calls_on_episode_with_running_count_and_total() -> None:
	"""S6 (ETA): play() must call on_episode(episode_num, total_episodes) after every
	episode — this is the hook make_eta_reporter's live ETA is built on."""
	with step("Arrange: a fake agent/env pair and a recorder for on_episode calls."):
		agent = _FakeAgent()
		calls = []

	with step("Act: play three episodes across two 'environments' (2 + 1)."):
		play(agent, [_FakeEnv(), _FakeEnv()], [2, 1], on_episode=lambda num, total: calls.append((num, total)))

	with step("Assert: on_episode was called once per episode, with a running count and a fixed total."):
		assert calls == [(1, 3), (2, 3), (3, 3)]


def test_make_eta_reporter_estimates_and_reports_eta(monkeypatch) -> None:
	"""ETA is estimated from the average time per episode elapsed so far this phase, and
	posted as a status string with an explicit, unambiguous unit suffix (e.g. '30s',
	'6m 22s', '1h 55m 31s') rather than a bare H:MM:SS that could be misread."""
	with step("Arrange: a fake queue, and time.time() scripted to advance by 10s."):
		posted = []
		queue = SimpleNamespace(put=posted.append)
		times = iter([100.0, 110.0])
		monkeypatch.setattr(cartpole_challenge.time, "time", lambda: next(times))
		start_time = cartpole_challenge.time.time()  # consumes the first scripted value (100.0)
		reporter = make_eta_reporter(
			experiment_id=0, queue=queue, label="Training", agent_name="Agent-X", start_time=start_time
		)

	with step("Act: report at episode 5 of 20 (10s for 5 episodes -> 2s/episode -> 30s remaining)."):
		reporter(5, 20)

	with step("Assert: the posted status carries the correctly-computed, unit-labeled ETA."):
		assert len(posted) == 1
		assert posted[0]["status"] == "Training Agent-X (ETA 30s)"


@pytest.mark.parametrize(
	("seconds", "expected"),
	[(0, "0s"), (42, "42s"), (382, "6m 22s"), (6931, "1h 55m 31s")],
)
def test_format_duration_includes_explicit_units(seconds, expected) -> None:
	"""ETA must be unambiguous — explicit h/m/s units, not a bare H:MM:SS."""
	with step("Act/Assert: format_duration renders the expected unit-labeled string."):
		assert cartpole_challenge.format_duration(seconds) == expected


def test_make_eta_reporter_skips_at_phase_boundaries() -> None:
	"""No rate is known yet at episode 0, and there's nothing left to estimate once the
	phase's last episode has finished — both must be silently skipped, not divide by zero."""
	with step("Arrange: a fake queue and a reporter."):
		posted = []
		queue = SimpleNamespace(put=posted.append)
		reporter = make_eta_reporter(0, queue, "Training", "Agent-X", start_time=0.0)

	with step("Act: call at episode 0 and at the final episode of a 10-episode phase."):
		reporter(0, 10)
		reporter(10, 10)

	with step("Assert: nothing was posted."):
		assert not posted
