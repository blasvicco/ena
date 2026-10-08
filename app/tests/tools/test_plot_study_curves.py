"""Tests for plot_study_curves.py's cross-replicate aggregation."""

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.plot_study_curves import discover_agents, load_replicates, phase_boundaries, plot, summarize


def _write_replicate(study, exp_name, agent, scores, envs, *, phase="testing"):  # pylint: disable=too-many-arguments
	"""Write one replicate's live CSV with the given per-episode scores and env names."""
	exp_dir = study / exp_name
	exp_dir.mkdir(parents=True, exist_ok=True)
	frame = pd.DataFrame({"episode": range(1, len(scores) + 1), "score": scores, "env_id": envs})
	frame.to_csv(exp_dir / f"live_{phase}_{agent}.csv", index=False)


def test_load_and_summarize_pool_replicates_per_episode(tmp_path) -> None:
	"""Each episode's mean and quartiles are taken across replicates (window 1 = unsmoothed)."""
	with step("Arrange: three replicates of a 2-episode run."):
		for name, scores in (("exp_1", [0, 100]), ("exp_2", [100, 200]), ("exp_3", [200, 300])):
			_write_replicate(tmp_path, name, "Agent-X", scores, ["Mars", "Mars"])

	with step("Act: load and summarize without smoothing."):
		matrix = load_replicates(str(tmp_path), "testing", "Agent-X")
		summary = summarize(matrix, window=1)

	with step("Assert: one column per replicate; per-episode mean and quartiles."):
		assert matrix.shape == (2, 3)
		assert summary["mean"].tolist() == [100.0, 200.0]
		assert summary["low"].tolist() == [50.0, 150.0]
		assert summary["high"].tolist() == [150.0, 250.0]


def test_phase_boundaries_mark_each_contiguous_environment() -> None:
	"""One (first episode, env) pair per contiguous phase, in order — a recurring env is a new phase."""
	with step("Act/Assert: Mars, Jupiter, Mars."):
		env_ids = pd.Series(["Mars", "Mars", "Jupiter", "Mars"], index=[1, 2, 3, 4])
		assert phase_boundaries(env_ids) == [(1, "Mars"), (3, "Jupiter"), (4, "Mars")]


def test_discover_agents_lists_random_first(tmp_path) -> None:
	"""Agents are found from exp_1's CSVs, with the Random floor on top."""
	with step("Arrange: three agents in exp_1."):
		for agent in ("ENA-Combined", "Random", "ENA-HoF"):
			_write_replicate(tmp_path, "exp_1", agent, [1], ["Mars"])

	with step("Act/Assert: Random first, the rest alphabetical."):
		assert discover_agents(str(tmp_path), "testing") == ["Random", "ENA-Combined", "ENA-HoF"]


def test_plot_writes_an_image_and_rejects_a_missing_agent(tmp_path) -> None:
	"""The figure is saved; an agent with no replicates is an error, not an empty panel."""
	with step("Arrange: two replicates of one agent over two phases."):
		for name in ("exp_1", "exp_2"):
			_write_replicate(tmp_path, name, "Agent-X", [10, 20, 30, 40], ["Mars", "Mars", "Jupiter", "Jupiter"])

	with step("Act: plot it."):
		path = plot(str(tmp_path), "testing", ["Agent-X"], str(tmp_path / "curves.png"))

	with step("Assert: a non-empty image exists; a missing agent raises."):
		assert (tmp_path / "curves.png").stat().st_size > 0 and path.endswith("curves.png")
		with pytest.raises(FileNotFoundError):
			plot(str(tmp_path), "testing", ["Agent-Missing"], str(tmp_path / "other.png"))


@pytest.mark.parametrize(
	"kwargs, mastery, ylim", [({}, 475.0, (-10, 560)), ({"mastery_score": 4800.0, "ylim": (-1000, 12000)}, 4800.0, (-1000, 12000))]
)
def test_plot_uses_the_domains_mastery_bar_and_range(tmp_path, monkeypatch, kwargs, mastery, ylim) -> None:
	"""CartPole's 475 bar and range by default; another domain (HalfCheetah) passes its own."""
	with step("Arrange: one replicate, and keep the figure instead of closing it."):
		_write_replicate(tmp_path, "exp_1", "Agent-X", [10, 20], ["Mars", "Mars"])
		figures = []
		monkeypatch.setattr("tools.plot_study_curves.plt.close", figures.append)

	with step("Act: plot it."):
		plot(str(tmp_path), "testing", ["Agent-X"], str(tmp_path / "curves.png"), **kwargs)

	with step("Assert: the axis range and the dotted mastery line."):
		axis = figures[0].axes[0]
		assert axis.get_ylim() == ylim
		assert [line.get_ydata()[0] for line in axis.get_lines() if line.get_linestyle() == ":"] == [mastery]
