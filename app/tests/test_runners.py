"""Cross-runner tests — behavior every domain runner must share (they stay structurally parallel)."""

# General imports
import ast
import importlib
import os

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step

RUNNERS = ["cartpole_ena_ablation"]  # cartpole_challenge and half_cheetah reuse cartpole_ena_ablation.agent_testing


class _FakeAgent:
	"""Just enough of the AAgent surface for agent_testing() to drive."""

	def __init__(self, name):
		self._name = name

	def get_name(self):
		"""Agent display name."""
		return self._name

	def get_episodes(self):
		"""One episode per test world."""
		return [1, 1, 1, 1, 1]

	def set_learning(self, mode, output_dir=None):
		"""No-op."""


@pytest.mark.parametrize("runner_name", RUNNERS)
def test_agent_testing_live_logs_each_agent_to_its_own_testing_csv(runner_name, monkeypatch, tmp_path) -> None:
	"""The test phase is live-logged to each agent's own live_testing CSV."""
	# S8: the test phase must be live-logged too (live_testing_{agent_name}.csv), so test-time
	# selection (re-evaluation firings, match() results) can be diagnosed per episode.
	with step("Arrange: stub play/make_env/metrics/plots and capture play()'s live_log_path."):
		runner = importlib.import_module(runner_name)
		captured = {}

		def _fake_play(agent, _environments, _episodes_per_env, live_log_path=None, on_episode=None, **_kwargs):  # pylint: disable=unused-argument
			captured[agent.get_name()] = live_log_path
			return pd.DataFrame([{"episode": 1, "score": 1.0, "env_id": "Earth"}])

		monkeypatch.setattr(runner, "play", _fake_play)
		monkeypatch.setattr(runner, "make_env", lambda *args, **kwargs: None)
		for helper in ("calculate_paper_metrics", "plot_academic_comparison"):
			monkeypatch.setattr(runner, helper, lambda *args, **kwargs: None)
		agents = [_FakeAgent("Agent-A"), _FakeAgent("Agent-B")]

	with step("Act: run the testing phase."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			runner.agent_testing(agents, 0, str(tmp_path), output_file, None, [None, None, None])

	with step("Assert: each agent was played with its own live_testing CSV in the output dir."):
		assert captured == {
			"Agent-A": str(tmp_path / "live_testing_Agent-A.csv"),
			"Agent-B": str(tmp_path / "live_testing_Agent-B.csv"),
		}


STUDY_SCRIPTS = [
	"cartpole_ena_ablation",
	"cartpole_ena_compute",
	"cartpole_ena_plasticity_controller",
	"cartpole_ena_selection",
	"cartpole_challenge",
	"half_cheetah",
]


@pytest.mark.parametrize("script", STUDY_SCRIPTS)
def test_every_study_script_passes_replicate_to_run_study(script) -> None:
	"""Regression: --replicate was parsed but not forwarded, so "only exp_K" silently ran the whole study."""
	with step("Arrange: the script's source."):
		path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", f"{script}.py")
		with open(path, encoding="utf-8") as source:
			tree = ast.parse(source.read())

	with step("Act: every run_study(...) call's keyword arguments."):
		calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_study"]

	with step("Assert: at least one call, and each passes replicate."):
		assert calls
		assert all("replicate" in {keyword.arg for keyword in call.keywords} for call in calls)
