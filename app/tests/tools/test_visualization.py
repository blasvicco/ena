"""Tests for visualization.py's console output: a run can be followed from a redirected log."""

# General imports
import queue

# App imports
from helpers import step
from tools.visualization import ProgressReporter, vprint


def test_vprint_writes_through_to_its_file(tmp_path) -> None:
	"""A replicate's outputs.txt is readable while the run goes: every vprint to a file is flushed at once."""
	with step("Act: print one line to an open file, and read the file before closing it."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output:
			vprint("Hardware: test", file=output)
			content = (tmp_path / "outputs.txt").read_text(encoding="utf-8")

	with step("Assert: the line is already on disk."):
		assert content == "Hardware: test\n"


def test_progress_reporter_prints_plain_status_lines_when_not_on_a_terminal(capsys) -> None:
	"""Redirected to a log (no terminal), the live bar never reaches the file, so each status change is also printed
	as one plain line (at most one per replicate per min_interval seconds; the final one always)."""
	with step("Arrange: two statuses of replicate 1, then its end."):
		updates = queue.Queue()
		for status in ("Training PPO (ETA 1h 2m)", "Testing PPO"):
			updates.put({"exp_id": 0, "status": status, "progress": 0, "done": False})
		updates.put({"exp_id": 0, "status": None, "progress": 100, "done": True})

	with step("Act: run the reporter (not a terminal under pytest), with no throttling."):
		ProgressReporter(updates, 1, min_interval=0.0).run()

	with step("Assert: one plain line per status, and the end."):
		lines = capsys.readouterr().out
		assert "Experiment 1: Training PPO (ETA 1h 2m)" in lines
		assert "Experiment 1: Testing PPO" in lines
		assert "Experiment 1: Done" in lines
