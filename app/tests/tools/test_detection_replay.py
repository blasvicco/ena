"""Tests for detection_replay.py — replaying a recorded run's detector calls under other settings."""

# Libs imports
import numpy as np
import pandas as pd

# App imports
from helpers import step
import half_cheetah
from tools.detection_replay import check_replay, replay, summarize
from tools.half_cheetah_probe import probe


def test_replay_with_the_recorded_settings_reproduces_a_real_probe_run(tmp_path, monkeypatch) -> None:
	"""Record a short live ENA run, replay it with the same settings: the confirmed changes match episode by episode."""
	with step("Arrange: a tiny ENA probe (small population, short phases) that records its detector."):
		monkeypatch.setattr(half_cheetah, "CHECK_EVERY", 4_000)
		monkeypatch.setattr(half_cheetah, "EVAL_EPISODES", 1)
		probe(
			"ENA-Combined",
			["Mars-Slippery", "Jupiter-Rough"],
			6_000,
			1,
			str(tmp_path),
			ena_overrides={"pop_size": 4, "max_eval_steps": 20},
			record_detection=True,
		)
		recording = dict(np.load(tmp_path / "detection.npz"))

	with step("Act: replay with the study's detection settings."):
		replayed = replay(recording, recording["states"].shape[1], recording["actions"].shape[1], half_cheetah.ENA_DETECTION)

	with step("Assert: one row per training episode, the recorded changes reproduced, a summary per environment."):
		training = pd.read_csv(tmp_path / "live_training.csv")
		assert len(replayed) == len(training)
		assert check_replay(recording, replayed)
		assert list(summarize(replayed, training["env_id"]).index) == ["Mars-Slippery", "Jupiter-Rough", "all"]
