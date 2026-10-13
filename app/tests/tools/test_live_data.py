"""Tests for the live dashboard's data helpers (downsampling, discovery)."""

# Libs imports
import numpy as np
import pandas as pd

# App imports
from helpers import step
from tools.live_data import binned_scores, discover_studies, list_agents, list_replicates


def test_binned_scores_caps_the_point_count_and_keeps_the_mean() -> None:
	"""365k episodes (Random under a 7.5M-step budget) must reach the browser as at most max_points points."""
	with step("Arrange: 365,000 episodes with cumulative env_steps."):
		frame = pd.DataFrame({"episode": np.arange(1, 365_001), "score": np.full(365_000, 20.0), "env_steps": np.arange(1, 365_001) * 20})

	with step("Act: bin to 1,000 points."):
		binned = binned_scores(frame, max_points=1_000)

	with step("Assert: at most 1,000 points, mean preserved, x on env_steps."):
		assert len(binned) <= 1_000
		assert np.allclose(binned["score"], 20.0)
		assert binned.index.name == "env_steps"
		assert binned.index.max() <= 365_000 * 20


def test_binned_scores_leaves_short_histories_alone_and_falls_back_to_episodes() -> None:
	"""Below max_points nothing is aggregated; without env_steps the x axis is the episode."""
	with step("Act/Assert: 50 episodes, no env_steps column."):
		frame = pd.DataFrame({"episode": np.arange(1, 51), "score": np.arange(50, dtype=float)})
		binned = binned_scores(frame, max_points=1_000)
		assert len(binned) == 50
		assert binned.index.name == "episode"
		assert binned["score"].tolist() == list(np.arange(50, dtype=float))


def test_discovery_lists_studies_replicates_and_agents(tmp_path) -> None:
	"""Studies are *_outputs dirs with exp_* replicates; agents come from live_training_*.csv names."""
	with step("Arrange: a study with exp_2 and exp_10 (numeric order), and an unrelated dir."):
		for exp in ("exp_2", "exp_10"):
			(tmp_path / "x_outputs" / exp).mkdir(parents=True)
			for agent in ("PPO", "ENA-Combined"):
				(tmp_path / "x_outputs" / exp / f"live_training_{agent}.csv").write_text("episode,score\n1,1\n", encoding="utf-8")
		(tmp_path / "notes").mkdir()

	with step("Act/Assert: one study; replicates in numeric order; agents sorted."):
		assert discover_studies(str(tmp_path)) == [str(tmp_path / "x_outputs")]
		assert list_replicates(str(tmp_path / "x_outputs")) == ["exp_2", "exp_10"]
		assert list_agents(str(tmp_path / "x_outputs" / "exp_2")) == ["ENA-Combined", "PPO"]
