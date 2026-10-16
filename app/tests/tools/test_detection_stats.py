"""Tests for detection_stats.py — per-phase environment-recognition numbers from the live logs."""

# General imports
import math

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.detection_stats import phase_detection, niches_per_env, replicate_detection, study_detection, summarize, write_detection_report

NICHES_TXT = "\n".join(
	[
		"Niche ID   | Env Name             | Total Gladiators",
		"---------------------------------------------------",
		"0          | Mars-Light           | 5",
		"1          | Mars-Light           | 5",
		"2          | Earth-Standard       | 4",
		"",
	]
)


def _history(envs, niches, shifts):
	"""A live CSV's detection columns."""
	return pd.DataFrame(
		{
			"env_id": envs,
			"episode": range(1, len(envs) + 1),
			"score": [100.0] * len(envs),
			"detected_env_id": niches,
			"env_shift_detected": shifts,
		}
	)


def _write_replicate(exp_dir, agent, training, testing, niches_txt=NICHES_TXT):
	"""A replicate directory with the agent's logs and niche report."""
	exp_dir.mkdir(parents=True, exist_ok=True)
	training.to_csv(exp_dir / f"live_training_{agent}.csv", index=False)
	testing.to_csv(exp_dir / f"live_testing_{agent}.csv", index=False)
	(exp_dir / f"niches_{agent}.txt").write_text(niches_txt, encoding="utf-8")


TRAINING = _history(["Mars"] * 3 + ["Earth"] * 4, [-1, 0, 0, 0, -1, 2, 2], [False, False, False, False, True, False, False])
TESTING = _history(["Mars"] * 2 + ["Brown"] * 4, [0, 0, -1, 2, -1, -1], [False, False, True, False, True, False])


def test_phase_detection_reports_each_contiguous_phase() -> None:
	"""Per phase: unrecognized share, confirmed changes, episodes into the phase of the first one, distinct niches."""
	with step("Act: a two-phase test history."):
		phases = phase_detection(TESTING)

	with step("Assert: Mars fully recognized and quiet; Brown 75% unrecognized, 2 changes, the first at its first episode."):
		assert phases == [
			{"env_id": "Mars", "episodes": 2, "unrecognized": 0.0, "changes": 0, "first_change": None, "niches": 1},
			{"env_id": "Brown", "episodes": 4, "unrecognized": 75.0, "changes": 2, "first_change": 0, "niches": 1},
		]


def test_niches_per_env_counts_the_hall_of_fames_niches_by_environment() -> None:
	"""Fragmentation: Mars spread over two niches, Earth one."""
	with step("Act/Assert: counts from the niche report."):
		assert niches_per_env(NICHES_TXT) == {"Mars-Light": 2, "Earth-Standard": 1}


def test_replicate_detection_returns_one_row_per_phase_and_metric(tmp_path) -> None:
	"""Long format: (phase, env, metric, value); training niches come from the niche report, by env name."""
	with step("Arrange: one replicate for ENA-Combined, niche report naming the training envs."):
		niches_txt = NICHES_TXT.replace("Mars-Light", "Mars").replace("Earth-Standard", "Earth")
		_write_replicate(tmp_path / "exp_1", "ENA-Combined", TRAINING, TESTING, niches_txt=niches_txt)

	with step("Act: read it."):
		rows = replicate_detection(str(tmp_path / "exp_1"), "ENA-Combined")

	with step("Assert: training Earth: 1 change, 1 episode in, 1 niche; Mars: 2 niches; test Brown: 75% unrecognized."):
		frame = pd.DataFrame(rows).set_index(["phase", "env_id", "metric"])["value"]
		assert frame[("train", "Earth", "changes")] == 1
		assert frame[("train", "Earth", "first_change")] == 1
		assert frame[("train", "Earth", "niches")] == 1
		assert frame[("train", "Mars", "niches")] == 2
		assert math.isnan(frame[("train", "Mars", "first_change")])
		assert frame[("test", "Brown", "unrecognized")] == 75.0
		assert ("test", "Brown", "niches") not in frame.index


def test_study_detection_keeps_the_agents_with_env_detection(tmp_path) -> None:
	"""Agents without a recognizer (constant niche id, never fires) are left out."""
	with step("Arrange: two replicates; ENA-Combined detects, ENA-HoF doesn't."):
		for name in ("exp_1", "exp_2"):
			_write_replicate(tmp_path / name, "ENA-Combined", TRAINING, TESTING)
			quiet = TESTING.assign(detected_env_id=0, env_shift_detected=False)
			_write_replicate(tmp_path / name, "ENA-HoF", quiet, quiet)

	with step("Act: collect the study."):
		frame = study_detection(str(tmp_path))

	with step("Assert: only ENA-Combined, from exp_1 and exp_2."):
		assert set(frame["agent"]) == {"ENA-Combined"}
		assert set(frame["exp"]) == {"exp_1", "exp_2"}


def test_summarize_gives_mean_std_and_replicate_count_per_agent_phase_env_metric(tmp_path) -> None:
	"""Across replicates; a first change that never happened (NaN) is left out of its mean and counted out of n."""
	with step("Arrange: two replicates differing in the test Brown phase."):
		_write_replicate(tmp_path / "exp_1", "ENA-Combined", TRAINING, TESTING)
		calm = TESTING.assign(env_shift_detected=False)
		_write_replicate(tmp_path / "exp_2", "ENA-Combined", TRAINING, calm)

	with step("Act: summarize."):
		summary = summarize(study_detection(str(tmp_path))).set_index(["agent", "phase", "env_id", "metric"])

	with step("Assert: Brown changes average 1 (2 and 0); its first change exists in one replicate only."):
		changes = summary.loc[("ENA-Combined", "test", "Brown", "changes")]
		assert changes["mean"] == pytest.approx(1.0)
		assert changes["n"] == 2
		first = summary.loc[("ENA-Combined", "test", "Brown", "first_change")]
		assert first["mean"] == 0
		assert first["n"] == 1


def test_write_detection_report_lists_every_agent_phase_and_env(tmp_path) -> None:
	"""The merging step's text report, next to aggregate_summary.txt."""
	with step("Arrange: one replicate."):
		_write_replicate(tmp_path / "study" / "exp_1", "ENA-Combined", TRAINING, TESTING)

	with step("Act: write the report."):
		written = write_detection_report(str(tmp_path / "study"))

	with step("Assert: the report names the agent, both phases and the Brown row."):
		report = (tmp_path / "study" / "detection_summary.txt").read_text(encoding="utf-8")
		assert written.endswith("detection_summary.txt")
		assert "ENA-Combined" in report and "Training" in report and "Test" in report and "Brown" in report
		assert not (tmp_path / "empty" / "detection_summary.txt").exists()
