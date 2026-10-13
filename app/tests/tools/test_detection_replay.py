"""Tests for detection_replay.py — replaying a recorded run's detector calls under other settings."""

# Libs imports
import numpy as np
import pandas as pd

# App imports
from helpers import step
import half_cheetah
from tools import detection_replay
from tools.detection_replay import EVALUATE, OBSERVE, START, check_replay, replay, summarize
from tools.half_cheetah_probe import probe


def test_replay_with_the_recorded_settings_reproduces_a_real_chirp_run(tmp_path) -> None:
	"""Record a short live ENA run, replay it with the same settings: the confirmed changes match episode by episode."""
	with step("Arrange: a tiny ENA chirp (small population, short phases) that records its detector."):
		probe(
			"ENA-Combined",
			["Earth-Standard", "Heavy-Rough"],
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
		assert list(summarize(replayed, training["env_id"]).index) == ["Earth-Standard", "Heavy-Rough", "all"]


def test_a_recording_can_be_replayed_through_another_detector_driver(monkeypatch) -> None:
	"""--strategy replays the same evidence through another driver, with its own settings."""
	with step("Arrange: a synthetic recording of 3 training episodes of 50 transitions; capture the driver built."):
		rng = np.random.default_rng(0)
		kinds, learn = [], []
		for _ in range(3):
			kinds += [START] + [OBSERVE] * 50 + [EVALUATE]
			learn += [False] + [False] * 50 + [True]
		recording = {
			"kinds": np.array(kinds),
			"learn": np.array(learn),
			"outcomes": np.zeros(len(kinds)),
			"states": rng.normal(size=(150, 3)),
			"actions": rng.normal(size=(150, 2)),
			"next_states": rng.normal(size=(150, 3)),
		}
		built = []
		original = detection_replay.EnvironmentRecognizer
		monkeypatch.setattr(
			detection_replay,
			"EnvironmentRecognizer",
			lambda driver, *args, **kwargs: built.append((driver, kwargs)) or original(driver, *args, **kwargs),
		)

	with step("Act: replay through the none driver with a maturity setting."):
		replayed = replay(recording, 3, 2, {"recognizer_min_samples": 300}, strategy="none")

	with step("Assert: that driver with its settings; one row per training episode."):
		assert built[0][0] == "none" and built[0][1]["min_samples"] == 300
		assert len(replayed) == 3


def test_the_recorder_keeps_chirps_and_replay_feeds_them_to_the_driver(tmp_path) -> None:
	"""A probing detector's observe_chirp calls are recorded (event + observations) and replayed in order."""
	with step("Arrange: a recorder on a chirp detector; two chirps recorded and saved."):
		from agents.ena.skills.env_detection.main import EnvironmentRecognizer  # pylint: disable=import-outside-toplevel

		detector = EnvironmentRecognizer("chirp", 2, action_dim=1, is_continuous=True, chirp={"steps": 4, "features": {"kind": "raw"}})
		recorder = detection_replay.Recorder(detector)
		responses = [np.full((5, 2), float(index)) for index in range(2)]
		for response in responses:
			recorder.start_episode()
			detector.observe_chirp(response)
			detector.evaluate(memory=_memory(), learn=True)
		recorder.save(str(tmp_path / "detection.npz"))
		recording = dict(np.load(tmp_path / "detection.npz"))

	with step("Act: replay through a fresh chirp driver."):
		replayed = replay(recording, 2, 1, {"recognizer_chirp": {"steps": 4, "features": {"kind": "raw"}}}, strategy="chirp")

	with step("Assert: the chirps were stored, and replay produced one row per episode."):
		assert np.array_equal(recording["chirps"], np.array(responses))
		assert len(replayed) == 2


def _memory():
	"""What the facade touches on a shift."""
	return type(
		"Memory", (), {"obs_normalizer": type("Normalizer", (), {"reset": lambda self: None})(), "clear_training_history": lambda self: None}
	)()
