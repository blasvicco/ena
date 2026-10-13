"""Tests for the shared study runner's play() (cartpole_ena_ablation.py, reused by every CartPole study)."""

# General imports
from types import SimpleNamespace

# Libs imports
import numpy as np
import pandas as pd
import pytest

# App imports
from helpers import step
import cartpole_ena_ablation
from cartpole_ena_ablation import make_eta_reporter, play
from tools.early_stopping import StoppingRule


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
		monkeypatch.setattr(cartpole_ena_ablation.time, "time", lambda: next(times))
		start_time = cartpole_ena_ablation.time.time()  # consumes the first scripted value (100.0)
		reporter = make_eta_reporter(experiment_id=0, queue=queue, label="Training", agent_name="Agent-X", start_time=start_time)

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
		assert cartpole_ena_ablation.format_duration(seconds) == expected


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


class _StepsEnv(_FakeEnv):
	"""Episodes of exactly `length` steps."""

	def __init__(self, length):
		super().__init__()
		self._length = length
		self._step = 0

	def reset(self):
		"""Restart the step count."""
		self._step = 0
		return super().reset()

	def step(self, _action):
		"""End the episode after `length` steps."""
		self._step += 1
		return np.zeros(4, dtype=np.float32), 1.0, self._step >= self._length, False, {}


class _StepsAgent:
	"""An agent that plays whatever the env does, optionally reporting extra (evaluation) env-steps."""

	def __init__(self, extra_steps_per_episode=0):
		self.learn = False
		self._extra = extra_steps_per_episode
		self._played = 0
		self._episodes = 0

	def act(self, env, _state):
		"""Step the env."""
		self._played += 1
		next_state, reward, terminated, truncated, _ = env.step(0)
		if terminated:
			self._episodes += 1
		return next_state, reward, terminated, truncated, 0

	@property
	def total_env_steps(self):
		"""Played steps plus a fixed number of extra steps per finished episode."""
		return self._played + self._extra * self._episodes


def test_play_with_a_step_budget_plays_each_env_until_the_budget_is_consumed() -> None:
	"""S16: steps_per_env replaces episode counts — whole episodes until each env's budget is reached."""
	with step("Act: 10-step episodes, budgets of 25 and 40 steps."):
		history = play(_StepsAgent(), [_StepsEnv(10), _StepsEnv(10)], [999, 999], steps_per_env=[25, 40])

	with step("Assert: 3 episodes (30 steps) then 4 (40 steps); env_steps is cumulative."):
		assert history["env_steps"].tolist() == [10, 20, 30, 40, 50, 60, 70]


def test_play_env_steps_uses_the_agents_own_total_including_evaluation_steps() -> None:
	"""S16: an agent reporting total_env_steps (ENA: played + candidate evaluations) is budgeted on that total."""
	with step("Act: 10-step episodes plus 15 evaluation steps each, one env with a 50-step budget."):
		history = play(_StepsAgent(extra_steps_per_episode=15), [_StepsEnv(10)], [999], steps_per_env=[50])

	with step("Assert: each episode consumes 25 steps, so 2 episodes exhaust the budget."):
		assert history["env_steps"].tolist() == [25, 50]


def test_play_without_a_budget_records_played_steps_per_episode() -> None:
	"""Episode-count mode still fills env_steps from the steps actually played."""
	with step("Act/Assert: two 10-step episodes."):
		history = play(_StepsAgent(), [_StepsEnv(10)], [2])
		assert history["env_steps"].tolist() == [10, 20]


class _EvalAgent(_StepsAgent):
	"""A _StepsAgent whose evaluation scores are scripted (one per check), recording learn flags seen in act()."""

	def __init__(self, eval_scores):
		super().__init__()
		self.learn = True
		self._eval_scores = list(eval_scores)
		self.trained = 0

	def train(self, **_kwargs):
		"""Count training calls."""
		self.trained += 1

	def evaluation_episode(self, _env):
		"""Return the next scripted score, as a 10-step episode."""
		score = self._eval_scores[0] if len(self._eval_scores) == 1 else self._eval_scores.pop(0)
		return score, 10


def test_play_with_a_stopping_rule_ends_each_environment_when_solved() -> None:
	"""Checks every 20 steps on 10-step episodes; solved at the second check of env 1, first check of env 2."""
	with step("Arrange: scripted evaluation means 300, 480 (env 1), then 490 (env 2)."):
		rule = StoppingRule(check_every=20, eval_episodes=1, solved_score=475.0, max_steps=10_000)
		agent = _EvalAgent([300.0, 480.0, 490.0])

	with step("Act: play two environments under the rule."):
		history = play(agent, [_StepsEnv(10), _StepsEnv(10)], [0, 0], stopping=rule, eval_environments=[_StepsEnv(10), _StepsEnv(10)])

	with step("Assert: 4 episodes (40 steps) then 2 (20 steps); evaluation steps don't count as training steps."):
		assert history["env_steps"].tolist() == [10, 20, 30, 40, 50, 60]
		assert history.attrs["phases"] == [
			{"env_id": "Fake-Env", "status": "solved", "steps": 40, "solve_steps": 40, "eval_steps": 20},
			{"env_id": "Fake-Env", "status": "solved", "steps": 20, "solve_steps": 20, "eval_steps": 10},
		]
		assert [row["eval_mean"] for row in history.attrs["evaluations"]] == [300.0, 480.0, 490.0]


def test_play_with_a_stopping_rule_stops_at_the_cap() -> None:
	"""An agent that never solves or plateaus stops at max_steps."""
	with step("Act: steadily improving but never solving, cap 50 steps."):
		rule = StoppingRule(check_every=20, eval_episodes=1, solved_score=1e9, max_steps=50, plateau_checks=100)
		agent = _EvalAgent([100.0])
		agent._eval_scores = [100.0 + 50.0 * index for index in range(10)]  # pylint: disable=protected-access
		history = play(agent, [_StepsEnv(10)], [0], stopping=rule, eval_environments=[_StepsEnv(10)])

	with step("Assert: stopped at 50 steps with status cap."):
		assert history.attrs["phases"][0]["status"] == "cap"
		assert history.attrs["phases"][0]["steps"] == 50


def test_evaluate_agent_plays_frozen_without_training_and_restores_the_learn_flag() -> None:
	"""Agents without evaluation_episode are evaluated with learn=False through act(); train() is never called."""
	with step("Arrange: a learning stub without an evaluation_episode method."):
		agent = _StepsAgent()
		agent.learn = True
		seen = []
		original_act = agent.act

		def _recording_act(env, state):
			seen.append(agent.learn)
			return original_act(env, state)

		agent.act = _recording_act

	with step("Act: evaluate 3 episodes of 10 steps."):
		mean, steps = cartpole_ena_ablation.evaluate_agent(agent, _StepsEnv(10), episodes=3)

	with step("Assert: frozen during evaluation, learning again afterwards; 30 evaluation steps."):
		assert set(seen) == {False}
		assert agent.learn is True
		assert (mean, steps) == (10.0, 30)


def test_play_with_a_fixed_budget_records_the_first_solve_and_trains_to_the_budget() -> None:
	"""Option 3: solving at 40 steps is recorded, but the environment keeps training to its 60-step budget."""
	with step("Arrange: a non-stopping rule, scripted means 300 then 480, budget 60 steps."):
		rule = StoppingRule(check_every=20, eval_episodes=1, solved_score=475.0, max_steps=60, stop_when_solved=False, stop_on_plateau=False)
		agent = _EvalAgent([300.0, 480.0])

	with step("Act: play one environment."):
		history = play(agent, [_StepsEnv(10)], [0], stopping=rule, eval_environments=[_StepsEnv(10)])

	with step("Assert: 60 training steps, first solve at 40, two checks only (none after solving)."):
		assert history["env_steps"].tolist()[-1] == 60
		assert history.attrs["phases"] == [{"env_id": "Fake-Env", "status": "solved", "steps": 60, "solve_steps": 40, "eval_steps": 20}]
		assert len(history.attrs["evaluations"]) == 2


def test_play_with_a_fixed_budget_never_solved_reports_the_budget() -> None:
	"""An environment never solved within its budget is reported unsolved, at the budget."""
	with step("Act: means stay at 300, budget 40."):
		rule = StoppingRule(check_every=20, eval_episodes=1, max_steps=40, stop_when_solved=False, stop_on_plateau=False)
		history = play(_EvalAgent([300.0]), [_StepsEnv(10)], [0], stopping=rule, eval_environments=[_StepsEnv(10)])

	with step("Assert: status cap, no solve steps."):
		assert history.attrs["phases"][0]["status"] == "cap"
		assert history.attrs["phases"][0]["solve_steps"] is None
		assert history.attrs["phases"][0]["steps"] == 40


def _random_returns(env_seed, episodes=5):
	"""Returns of a Random agent on a fresh CartPole world, with or without environment seeding."""
	from agents import RandomAgent  # pylint: disable=import-outside-toplevel

	agent = RandomAgent(brain=None, make_env=cartpole_ena_ablation.make_env)
	agent.set_learning(False)
	history = play(agent, [cartpole_ena_ablation.make_env(name="Earth")], [episodes], env_seed=env_seed)
	return history["score"].tolist()


def test_play_with_an_env_seed_is_reproducible_and_seed_specific() -> None:
	"""Seeded environments: the same seed replays the same episodes (initial states and, for Random, its
	actions); a different seed gives different ones."""
	with step("Act: two runs with seed 7 on fresh worlds and agents, one with seed 8."):
		first, second, other = _random_returns(7), _random_returns(7), _random_returns(8)

	with step("Assert: identical for the same seed, different for another."):
		assert first == second
		assert first != other


def test_training_and_testing_seed_the_environments_per_replicate_and_equally_for_every_agent(monkeypatch, tmp_path) -> None:
	"""Every agent of a replicate meets the same seeded environment stream (common random numbers);
	replicates differ; the test phase uses its own seed."""
	with step("Arrange: capture play()'s env_seed for two agents, in two replicates."):
		seeds = []

		def _fake_play(agent, _worlds, _episodes, env_seed=None, **_kwargs):
			seeds.append((agent.get_name(), env_seed))
			return pd.DataFrame({"env_id": ["Mars"], "env_steps": [1], "score": [1.0]})

		monkeypatch.setattr(cartpole_ena_ablation, "play", _fake_play)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)
		agents = [
			SimpleNamespace(get_name=lambda: "A", get_episodes=lambda: [1] * 5, set_learning=lambda *a, **k: None),
			SimpleNamespace(get_name=lambda: "B", get_episodes=lambda: [1] * 5, set_learning=lambda *a, **k: None),
		]

	with step("Act: train and test replicate 0, then train replicate 1."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(agents, 0, str(tmp_path), output_file, None, ["W"] * 3)
			cartpole_ena_ablation.agent_testing(agents, 0, str(tmp_path), output_file, None, ["W"] * 3, test_worlds=["W"] * 5)
			cartpole_ena_ablation.agent_training(agents, 1, str(tmp_path), output_file, None, ["W"] * 3)

	with step("Assert: same seed for both agents within a phase; distinct across phases and replicates."):
		(train_a, train_b, test_a, test_b, next_a, next_b) = [seed for _name, seed in seeds]
		assert train_a is not None
		assert train_a == train_b
		assert test_a == test_b
		assert len({train_a, test_a, next_a}) == 3
		assert next_a == next_b


class _CandidateAgent(_StepsAgent):
	"""A steps agent that also spends a growing number of candidate-evaluation steps per episode."""

	@property
	def candidate_env_steps(self):
		"""Episode n spends 10 * n candidate steps (cumulative counter, like ENAgent's)."""
		return sum(10 * episode for episode in range(1, self._episodes + 1))


def test_play_logs_the_actual_candidate_steps_spent_in_each_episode() -> None:
	"""The compute column is what the agent actually spent per episode (the delta of its counter),
	not an upper bound from its search-effort settings."""
	with step("Act: three 10-step episodes."):
		history = play(_CandidateAgent(), [_StepsEnv(10)], [3])

	with step("Assert: 10, 20, 30 candidate steps; none logged for an agent without the counter."):
		assert history["candidate_env_steps"].tolist() == [10, 20, 30]
		assert play(_StepsAgent(), [_StepsEnv(10)], [1])["candidate_env_steps"].isna().all()
