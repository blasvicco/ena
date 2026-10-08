"""Tests for replot_replicate.py — redrawing a finished replicate's training.png/test.png from its live CSVs."""

# Libs imports
import pandas as pd

# App imports
from helpers import step
from tools import replot_replicate
from tools.replot_replicate import agent_order, known_niches, replot

OUTPUTS_TXT = """Experiment 1 Log
==============================

  --- Training Performance vs Time ---
  Random: 0.75 seconds, 32,262 env-steps
  ENA-HoF: 315.33 seconds, 115,576 env-steps
  ENA-Combined: 2664.63 seconds, 5,576,493 env-steps

Training:
"""


def _history(niches, shifts=None):
	"""A live CSV's columns for a 4-episode, two-environment run."""
	return pd.DataFrame(
		{
			"env_id": ["Mars", "Mars", "Earth", "Earth"],
			"episode": range(1, 5),
			"score": [10.0, 20.0, 30.0, 40.0],
			"detected_env_id": niches,
			"env_shift_detected": shifts or [False] * 4,
		}
	)


def test_agent_order_is_the_order_the_replicate_trained_its_agents() -> None:
	"""The chart panels keep the run's own order, read from outputs.txt's training-time block."""
	with step("Act/Assert: Random, ENA-HoF, ENA-Combined."):
		assert agent_order(OUTPUTS_TXT) == ["Random", "ENA-HoF", "ENA-Combined"]


def test_known_niches_are_those_resolved_in_training_or_none_without_env_detection() -> None:
	"""A test chart treats training's niches as already known; agents without a recognizer get None."""
	with step("Act/Assert: resolved training niches, ignoring -1; None for the always-0 none driver."):
		assert known_niches(_history([-1, 0, 2, 2], [False, False, True, False])) == {0, 2}
		assert known_niches(_history([0, 0, 0, 0])) is None


def test_replot_redraws_both_charts_for_every_agent_in_run_order(tmp_path, monkeypatch) -> None:
	"""Both charts are rewritten from the CSVs, panels in run order, test chart seeded with training's niches."""
	with step("Arrange: a replicate directory with outputs.txt and each agent's training and test CSVs; record the plot calls."):
		(tmp_path / "outputs.txt").write_text(OUTPUTS_TXT, encoding="utf-8")
		for agent in ("Random", "ENA-HoF", "ENA-Combined"):
			niches = [-1, 0, 1, 1] if agent == "ENA-Combined" else [0, 0, 0, 0]
			for phase in ("training", "testing"):
				_history(niches).to_csv(tmp_path / f"live_{phase}_{agent}.csv", index=False)
		calls = []
		monkeypatch.setattr(replot_replicate, "plot_academic_comparison", lambda histories, names, **kwargs: calls.append((list(names), kwargs)))

	with step("Act: replot."):
		replot(str(tmp_path))

	with step("Assert: training.png then test.png, same agent order; only the test chart gets known niches."):
		assert [kwargs["save_path"] for _names, kwargs in calls] == [str(tmp_path / "training.png"), str(tmp_path / "test.png")]
		assert all(names == ["Random", "ENA-HoF", "ENA-Combined"] for names, _kwargs in calls)
		assert "known_niche_ids_by_history" not in calls[0][1]
		assert calls[1][1]["known_niche_ids_by_history"] == [None, None, {0, 1}]


def test_replot_draws_training_on_the_requested_axis_and_testing_by_episode(tmp_path, monkeypatch) -> None:
	"""Equal-step-budget studies redraw training on env steps; the test chart stays by episode (equal test episodes)."""
	with step("Arrange: a three-agent replicate; record the plot calls."):
		(tmp_path / "outputs.txt").write_text(OUTPUTS_TXT, encoding="utf-8")
		for agent in ("Random", "ENA-HoF", "ENA-Combined"):
			for phase in ("training", "testing"):
				_history([0, 0, 0, 0]).to_csv(tmp_path / f"live_{phase}_{agent}.csv", index=False)
		calls = []
		monkeypatch.setattr(replot_replicate, "plot_academic_comparison", lambda histories, names, **kwargs: calls.append(kwargs))

	with step("Act: replot with the env-step training axis."):
		replot(str(tmp_path), training_x_axis="env_steps")

	with step("Assert: training on env_steps, testing on the default axis."):
		assert calls[0]["x_axis"] == "env_steps"
		assert "x_axis" not in calls[1]
