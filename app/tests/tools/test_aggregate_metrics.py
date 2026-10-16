"""Tests for aggregate_metrics.py: parsing a study's outputs.txt files and the two-agent significance comparison."""

# General imports
import io

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.aggregate_metrics import PHASES, aggregate, compute_significance, write_comparison_report
from tools.metrics import calculate_adaptation, calculate_compute_cost, calculate_paper_metrics


def _write_replicate(exp_dir, histories: dict) -> None:
	"""Write a replicate's outputs.txt with the runner's own summary writers: {agent: (training, testing)} histories."""
	exp_dir.mkdir(parents=True, exist_ok=True)
	names, trainings, testings = list(histories), [pair[0] for pair in histories.values()], [pair[1] for pair in histories.values()]
	times = {name: 10.0 for name in names}
	buffer = io.StringIO()
	print("Experiment 1 Log\n" + "=" * 30 + "\n\n  --- Training Performance vs Time ---", file=buffer)
	for name in names:
		print(f"  {name}: 10.00 seconds, 1,000 env-steps", file=buffer)
	calculate_paper_metrics(trainings, names, times, file=buffer)
	calculate_compute_cost(trainings, names, file=buffer)
	calculate_adaptation(trainings, names, file=buffer)
	print("\n  --- Experiment 1 Test Results ---", file=buffer)
	calculate_paper_metrics(testings, names, times, file=buffer)
	(exp_dir / "outputs.txt").write_text(buffer.getvalue(), encoding="utf-8")


def _histories(score: float, phase: dict) -> tuple:
	"""A training history (with its search effort and phase outcome) and a testing history, both scoring `score`."""
	training = pd.DataFrame(
		{"env_id": ["Mars"] * 2, "score": [score] * 2, "eval_step_cap": [50.0] * 2, "num_to_eval": [10.0] * 2, "candidate_env_steps": [100] * 2}
	)
	training.attrs["phases"] = [phase]
	return training, pd.DataFrame({"env_id": ["Mars"] * 2, "score": [score] * 2})


SOLVED = {"env_id": "Mars", "status": "solved", "steps": 2_500_000, "solve_steps": 120_000, "eval_steps": 5_000}
CAPPED = {"env_id": "Mars", "status": "cap", "steps": 3_000_000, "solve_steps": None, "eval_steps": 60_000}


def _all_results(phase_frames: dict) -> dict:
	"""An aggregate-style {phase: [frames]} with every phase present (the given frames, else one placeholder row)."""
	placeholder = pd.DataFrame({"Agent": ["Agent-A"], "Reliability (%)": [0.0], "Exp": ["exp_1"]})
	return {phase: phase_frames.get(phase, [placeholder]) for phase in PHASES}


def test_compute_significance_flags_a_clear_gap_as_significant() -> None:
	"""A large, well-separated gap between two low-variance samples must be flagged significant."""
	with step("Arrange: two samples with a large, clean gap and modest spread."):
		values_a = pd.Series([9.0, 10.0, 11.0, 9.5, 10.5, 10.0, 9.8, 10.2, 9.9, 10.1])
		values_b = pd.Series([49.0, 50.0, 51.0, 49.5, 50.5, 50.0, 49.8, 50.2, 49.9, 50.1])

	with step("Act: compute significance between them."):
		result = compute_significance(values_a, values_b)

	with step("Assert: the ~40-point gap is flagged significant."):
		assert result["diff"] == pytest.approx(40.0, abs=0.5)
		assert result["significant"] is True


def test_compute_significance_does_not_flag_overlapping_samples() -> None:
	"""Two samples whose spread comfortably overlaps must not be flagged significant."""
	with step("Arrange: two samples with a small mean gap relative to their spread."):
		values_a = pd.Series([10.0, 30.0, 50.0, 20.0, 40.0, 15.0, 35.0, 25.0])
		values_b = pd.Series([15.0, 35.0, 55.0, 25.0, 45.0, 20.0, 40.0, 30.0])

	with step("Act: compute significance between them."):
		result = compute_significance(values_a, values_b)

	with step("Assert: the noisy, overlapping samples are not flagged significant."):
		assert result["significant"] is False


def test_write_comparison_report_includes_both_agents_and_a_diff_line(tmp_path) -> None:
	"""The report must name both agents and print a diff line for a shared metric."""
	with step("Arrange: two replicates of two agents in the training phase."):
		all_results = _all_results(
			{
				"Training": [
					pd.DataFrame({"Agent": ["Agent-A", "Agent-B"], "Reliability (%)": [20.0, 60.0], "Exp": ["exp_1", "exp_1"]}),
					pd.DataFrame({"Agent": ["Agent-A", "Agent-B"], "Reliability (%)": [22.0, 58.0], "Exp": ["exp_2", "exp_2"]}),
				]
			}
		)
		output_path = tmp_path / "compare_Agent-A_vs_Agent-B.txt"

	with step("Act: write the comparison report."):
		write_comparison_report(all_results, "Agent-A", "Agent-B", str(output_path))

	with step("Assert: the report names both agents and reports the Reliability diff."):
		content = output_path.read_text(encoding="utf-8")
		assert "Agent-B vs Agent-A" in content
		assert "Agent-A=21.00" in content
		assert "Agent-B=59.00" in content


def test_write_comparison_report_skips_a_phase_missing_either_agent(tmp_path) -> None:
	"""A phase without one of the two agents (e.g. compute cost, which only ENA logs) is skipped, not raised on."""
	with step("Arrange: every phase holds only Agent-A."):
		all_results = _all_results({})
		output_path = tmp_path / "compare.txt"

	with step("Act: write the comparison report."):
		write_comparison_report(all_results, "Agent-A", "Agent-B", str(output_path))

	with step("Assert: no phase section is written."):
		assert "Phase ---" not in output_path.read_text(encoding="utf-8")


def test_aggregate_summarizes_every_replicate_of_a_study(tmp_path) -> None:
	"""Every replicate's tables are parsed and summarized in the study's aggregate_summary.txt."""
	with step("Arrange: two replicates of one agent, solving (500) in one and failing (100) in the other."):
		_write_replicate(tmp_path / "study" / "exp_1", {"Agent-X": _histories(500.0, SOLVED)})
		_write_replicate(tmp_path / "study" / "exp_2", {"Agent-X": _histories(100.0, CAPPED)})

	with step("Act: aggregate."):
		aggregate(str(tmp_path / "study"))

	with step("Assert: both replicates counted; test reliability averages 100% and 0%."):
		summary = (tmp_path / "study" / "aggregate_summary.txt").read_text(encoding="utf-8")
		assert "Across 2 Experiments" in summary
		assert "Reliability (%).....: Mean =    50.00" in summary.split("--- Test Phase ---")[1]


def test_adaptation_table_round_trips_through_the_aggregator(tmp_path) -> None:
	"""calculate_adaptation's table in outputs.txt is parsed into an Adaptation phase with numeric columns."""
	with step("Arrange: one replicate with a solved agent and a capped one."):
		_write_replicate(tmp_path / "study" / "exp_1", {"Agent-X": _histories(500.0, SOLVED), "Agent-Y": _histories(100.0, CAPPED)})

	with step("Act: aggregate."):
		aggregate(str(tmp_path / "study"))

	with step("Assert: steps-to-solve (not the phase's training steps) when solved; the phase's steps when not."):
		summary = (tmp_path / "study" / "aggregate_summary.txt").read_text(encoding="utf-8")
		adaptation = summary.split("--- Adaptation Phase ---")[1]
		assert "Mean = 120000.00" in adaptation.split("Agent: Agent-Y")[0]
		assert "Mean = 3000000.00" in adaptation
		assert "Mars:solved" in adaptation
		assert "Mean = 2500000.00" in adaptation  # Agent-X's Train-Env-Steps: the whole phase, not the solve point
