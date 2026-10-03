"""Tests for aggregate_metrics.py's two-agent significance comparison."""

# General imports
import io

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.aggregate_metrics import aggregate, compute_significance, write_comparison_report
from tools.metrics import calculate_adaptation


def _write_minimal_outputs_file(exp_dir, agent_name: str, train_reliability: float, test_reliability: float) -> None:
	"""Write a minimal but validly-formatted outputs.txt for one experiment replicate."""
	exp_dir.mkdir(parents=True, exist_ok=True)
	content = (
		"Experiment 1 Log\n"
		"==============================\n\n"
		"  --- Training Performance vs Time ---\n"
		f"  {agent_name}: 10.00 seconds\n\n"
		"Training:\n\n"
		"--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---\n"
		"   Agent Mars-Light Avg Reliability (%) Stability Gap (↓) Efficiency (Score/s)\n"
		f"  {agent_name}          100.00           {train_reliability:.1f}%              10.00               500.00\n\n"
		"  --- Experiment 1 Test Results ---\n\n"
		"--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---\n"
		"   Agent Mars-Light Avg Reliability (%) Stability Gap (↓) Efficiency (Score/s)\n"
		f"  {agent_name}          110.00           {test_reliability:.1f}%              12.00               600.00\n"
	)
	(exp_dir / "outputs.txt").write_text(content, encoding="utf-8")


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
	with step("Arrange: a minimal all_results dict with one phase and two agents."):
		all_results = {
			"Training": [
				pd.DataFrame({"Agent": ["Agent-A", "Agent-B"], "Reliability (%)": [20.0, 60.0], "Exp": ["exp_1", "exp_1"]}),
				pd.DataFrame({"Agent": ["Agent-A", "Agent-B"], "Reliability (%)": [22.0, 58.0], "Exp": ["exp_2", "exp_2"]}),
			],
			"Compute Cost": [],
			"Test": [],
		}
		output_path = tmp_path / "compare_Agent-A_vs_Agent-B.txt"

	with step("Act: write the comparison report."):
		write_comparison_report(all_results, "Agent-A", "Agent-B", str(output_path))

	with step("Assert: the report names both agents and reports the Reliability diff."):
		content = output_path.read_text(encoding="utf-8")
		assert "Agent-B vs Agent-A" in content
		assert "Reliability (%)" in content
		assert "Agent-A=21.00" in content
		assert "Agent-B=59.00" in content


def test_write_comparison_report_skips_a_phase_missing_either_agent(tmp_path) -> None:
	"""A phase where one of the two agents never appears must be skipped, not raise."""
	with step("Arrange: a Test phase containing only Agent-A."):
		all_results = {
			"Training": [],
			"Compute Cost": [],
			"Test": [pd.DataFrame({"Agent": ["Agent-A"], "Reliability (%)": [30.0], "Exp": ["exp_1"]})],
		}
		output_path = tmp_path / "compare.txt"

	with step("Act: write the comparison report."):
		write_comparison_report(all_results, "Agent-A", "Agent-B", str(output_path))

	with step("Assert: the report is written but the Test phase section is absent."):
		content = output_path.read_text(encoding="utf-8")
		assert "--- Test Phase ---" not in content


def test_aggregate_pools_replicates_across_multiple_output_dirs(tmp_path) -> None:
	"""Two separate studies' directories, each with one replicate of the same agent, must pool
	into a single 2-sample summary — this is what lets an already-run config (e.g. a prior
	study's identical row) be reused instead of re-run in a newer study's own directory."""
	with step("Arrange: one replicate in each of two separate output directories."):
		study_a = tmp_path / "study_a"
		study_b = tmp_path / "study_b"
		_write_minimal_outputs_file(study_a / "exp_1", "Agent-X", train_reliability=50.0, test_reliability=55.0)
		_write_minimal_outputs_file(study_b / "exp_1", "Agent-X", train_reliability=60.0, test_reliability=65.0)

	with step("Act: aggregate across both directories."):
		aggregate([str(study_a), str(study_b)])

	with step("Assert: the summary (written to the first directory) reflects both replicates."):
		content = (study_a / "aggregate_summary.txt").read_text(encoding="utf-8")
		assert "Averages and Standard Deviations Across 2 Experiments" in content
		# Mean of 50.0 and 60.0 — proof both directories' rows actually made it into one pool.
		assert "Mean =    55.00" in content


def test_aggregate_excludes_an_agent_from_one_output_dir_only(tmp_path) -> None:
	"""--exclude DIR:AGENT drops a superseded row from one study (e.g. the original study's
	ENA-Combined, re-run with a new env-detection driver) while keeping that agent from the others."""
	with step("Arrange: Agent-X in an old and a new study; the old one's is superseded."):
		old_study = tmp_path / "old_study"
		new_study = tmp_path / "new_study"
		_write_minimal_outputs_file(old_study / "exp_1", "Agent-X", train_reliability=10.0, test_reliability=10.0)
		_write_minimal_outputs_file(new_study / "exp_1", "Agent-X", train_reliability=90.0, test_reliability=90.0)

	with step("Act: pool both, excluding the old study's Agent-X."):
		aggregate([str(new_study), str(old_study)], exclude=[f"{old_study}:Agent-X"])

	with step("Assert: only the new study's replicate is in the pool."):
		content = (new_study / "aggregate_summary.txt").read_text(encoding="utf-8")
		assert "Mean =    90.00" in content
		assert "Mean =    50.00" not in content


def test_aggregate_rejects_a_malformed_exclude_entry(tmp_path) -> None:
	"""An --exclude entry without DIR:AGENT form is a usage error, not a silent no-op."""
	with step("Arrange: one study."):
		study = tmp_path / "study"
		_write_minimal_outputs_file(study / "exp_1", "Agent-X", train_reliability=10.0, test_reliability=10.0)

	with step("Act/Assert: a colon-less entry raises ValueError."):
		with pytest.raises(ValueError):
			aggregate([str(study)], exclude=["Agent-X"])


def test_adaptation_table_round_trips_through_the_aggregator(tmp_path) -> None:
	"""S16: calculate_adaptation's table in outputs.txt is parsed into an Adaptation phase with numeric columns."""
	with step("Arrange: a replicate whose train block carries an adaptation table for two agents."):
		frame = pd.DataFrame({"env_id": ["Mars"], "env_steps": [0], "score": [0.0]})
		frame.attrs["phases"] = [{"env_id": "Mars", "status": "solved", "steps": 2_500_000, "solve_steps": 120_000, "eval_steps": 5_000}]
		other = pd.DataFrame({"env_id": ["Mars"], "env_steps": [0], "score": [0.0]})
		other.attrs["phases"] = [{"env_id": "Mars", "status": "cap", "steps": 3_000_000, "solve_steps": None, "eval_steps": 60_000}]
		exp_dir = tmp_path / "study" / "exp_1"
		_write_minimal_outputs_file(exp_dir, "Agent-X", train_reliability=50.0, test_reliability=55.0)
		content = (exp_dir / "outputs.txt").read_text(encoding="utf-8")
		buffer = io.StringIO()
		calculate_adaptation([frame, other], ["Agent-X", "Agent-Y"], file=buffer)
		marker = "  --- Experiment 1 Test Results ---"
		(exp_dir / "outputs.txt").write_text(content.replace(marker, buffer.getvalue().strip("\n") + "\n\n" + marker), encoding="utf-8")

	with step("Act: aggregate."):
		aggregate([str(tmp_path / "study")])

	with step("Assert: steps-to-solve (not the phase's training steps) when solved; the phase's steps when not."):
		summary = (tmp_path / "study" / "aggregate_summary.txt").read_text(encoding="utf-8")
		adaptation = summary.split("--- Adaptation Phase ---")[1]
		assert "Mean = 120000.00" in adaptation.split("Agent: Agent-Y")[0]
		assert "Mean = 3000000.00" in adaptation
		assert "Mars:solved" in adaptation
		assert "Mean = 2500000.00" in adaptation  # Agent-X's Train-Env-Steps: the whole phase, not the solve point
