"""Tests for cheetah_results.py, paper_results.py's HalfCheetah entries: test return, retention of the environments left behind, compute."""

# Libs imports
import math

import pandas as pd
import pytest

# App imports
from helpers import step
from tools.cheetah_results import (
	end_of_phase,
	hardware_entries,
	retention_comparisons,
	retention_entries,
	retention_figure,
	retention_rows,
	retention_table,
)

TRAIN = ["Mars", "Heavy", "Earth"]
LEFT = ["Mars", "Heavy"]


def _replicate(exp_dir, training: dict, tests: dict, *, complete: bool = True, hardware: str = None) -> None:
	"""One replicate's files: SAC's training episodes (50,000 env-steps each, cumulative env_steps) and test episodes per
	environment, and outputs.txt."""
	exp_dir.mkdir(parents=True, exist_ok=True)
	rows, steps = [], 0
	for env, values in training.items():
		for value in values:
			steps += 50_000
			rows.append({"env_id": env, "env_steps": steps, "score": value})
	pd.DataFrame(rows).to_csv(exp_dir / "live_training_SAC.csv", index=False)
	rows = [{"env_id": env, "score": value} for env, values in tests.items() for value in values]
	pd.DataFrame(rows).to_csv(exp_dir / "live_testing_SAC.csv", index=False)
	text = "Experiment Log\n" + (f"  Hardware: {hardware}\n" if hardware else "") + ("  --- Experiment 1 Test Results ---\n" if complete else "")
	outputs = exp_dir / "outputs.txt"
	outputs.write_text(outputs.read_text() + text if outputs.exists() else text)


def _rows(agent_values: dict) -> pd.DataFrame:
	"""Retention rows from {agent: [(exp, env, end, test), ...]}."""
	rows = []
	for agent, values in agent_values.items():
		for exp, env, end, test in values:
			rows.append({"Exp": exp, "Agent": agent, "Env": env, "end": end, "test": test, "retention": 100 * test / end, "drop": end - test})
	return pd.DataFrame(rows)


def test_end_of_phase_is_the_mean_training_return_over_the_phase_last_100k_steps() -> None:
	"""The episodes that end within the last 100k env-steps of the environment's phase, whatever their length (ENA's
	episodes are spread by its candidate evaluations); other environments' episodes never count."""
	with step("Arrange: Mars episodes ending at 100k, 900k, 950k and 1M; one Earth episode after."):
		training = pd.DataFrame(
			{"env_id": ["Mars"] * 4 + ["Earth"], "env_steps": [100_000, 900_000, 950_000, 1_000_000, 1_050_000], "score": [1.0, 2.0, 3.0, 5.0, 99.0]}
		)

	with step("Act/Assert: Mars's last 100k are the episodes ending after 900k: (3 + 5) / 2."):
		assert end_of_phase(training, "Mars") == 4.0
		assert end_of_phase(training, "Earth") == 99.0
		assert math.isnan(end_of_phase(training, "Heavy"))


def test_retention_rows_compare_each_test_with_the_training_return_at_the_end_of_the_phase(tmp_path) -> None:
	"""End of phase = mean training return over the phase's last 100k env-steps (the agents learn, noise included, until
	the end); test = mean test return there; retention = test / end of phase (%); drop = end - test. Unfinished
	replicates are skipped."""
	with step("Arrange: two finished replicates and one still running."):
		_replicate(
			tmp_path / "exp_1",
			{"Mars": [0, 900, 1000, 1000], "Heavy": [500, 600, 700], "Earth": [10, 20, 30]},
			{"Mars": [500, 600], "Heavy": [700], "Earth": [20]},
		)
		_replicate(
			tmp_path / "exp_2",
			{"Mars": [2000, 2000, 2000], "Heavy": [-50, -60, -70], "Earth": [5, 5, 5]},
			{"Mars": [1000], "Heavy": [-10], "Earth": [5]},
		)
		_replicate(tmp_path / "exp_3", {"Mars": [1, 1, 1]}, {"Mars": [1]}, complete=False)

	with step("Act: the retention rows."):
		rows = retention_rows(str(tmp_path), ["SAC"], TRAIN).set_index(["Exp", "Env"])

	with step("Assert: one row per finished replicate and training environment, with the right values."):
		assert set(rows.index.get_level_values("Exp")) == {"exp_1", "exp_2"} and len(rows) == 6
		mars = rows.loc[("exp_1", "Mars")]
		assert (mars["end"], mars["test"], mars["drop"]) == (1000.0, 550.0, 450.0)
		assert mars["retention"] == pytest.approx(55.0)
		assert rows.loc[("exp_2", "Mars"), "retention"] == pytest.approx(50.0)

	with step("Assert: no retention where the phase ended at or below zero (nothing learned to retain)."):
		assert math.isnan(rows.loc[("exp_2", "Heavy"), "retention"])


def test_retention_entries_give_per_environment_values_and_the_mean_over_the_environments_left_behind() -> None:
	"""Per agent and environment: end of phase and retention (mean, std); 'left' averages each replicate's retention over
	the environments left behind, then over replicates. The test return per environment is the summary's
	(<study>/<agent>/test/<env>), the same mean from outputs.txt, so it is not repeated."""
	with step("Arrange: two replicates of one agent."):
		rows = _rows({"SAC": [("e1", "Mars", 1000, 500), ("e1", "Heavy", 1000, 700), ("e2", "Mars", 1000, 300), ("e2", "Heavy", 1000, 900)]})

	with step("Act: the entries."):
		entries = retention_entries("cheetah", rows, LEFT)

	with step("Assert: per environment, and the left-behind mean (60 and 60 -> 60.0, std 0)."):
		assert entries["cheetah/SAC/retention/Mars/mean"] == "40.0"
		assert entries["cheetah/SAC/end/Heavy/mean"] == "1,000"
		assert not any("/test/" in key for key in entries)
		assert entries["cheetah/SAC/retention/left/mean"] == "60.0"
		assert entries["cheetah/SAC/retention/left/std"] == "0.0"


def test_retention_comparisons_test_the_agent_against_each_baseline_as_one_holm_family() -> None:
	"""ENA vs each baseline on the left-behind retention: Welch p, Holm across the baselines, difference with its interval,
	effect size and the replicates needed for 80% power."""
	with step("Arrange: ENA retains about 90%, PPO about 20%, SAC about 85%, over four replicates."):
		values = {"ENA": [90, 92, 88, 91], "PPO": [20, 25, 15, 22], "SAC": [80, 90, 85, 86]}
		rows = _rows({agent: [(f"e{index}", "Mars", 100, value) for index, value in enumerate(series)] for agent, series in values.items()})

	with step("Act: the comparisons."):
		entries = retention_comparisons("cheetah", rows, ["Mars"], "ENA", ["PPO", "SAC"])

	with step("Assert: PPO's gap is significant after Holm, SAC's is not; the difference is ENA minus the baseline."):
		assert entries["cheetah/PPO-vs-ENA/retention/left/diff"] == "+69.8"
		assert entries["cheetah/PPO-vs-ENA/retention/left/pholm"].startswith("$p")
		assert float(entries["cheetah/PPO-vs-ENA/retention/left/pholm"].strip("$p=<")) < 0.05
		assert float(entries["cheetah/SAC-vs-ENA/retention/left/pholm"].strip("$p=<")) > 0.05
		assert {"ci/low", "ci/high", "d", "nreq", "p"} <= {key.split("/left/")[1] for key in entries if key.startswith("cheetah/SAC-vs-ENA/")}


def test_hardware_entries_read_the_runner_s_hardware_line(tmp_path) -> None:
	"""The machine and thread counts every agent of the study ran with, so wall-clock times are comparable."""
	with step("Arrange: two finished replicates on the same machine."):
		for exp in ("exp_1", "exp_2"):
			_replicate(tmp_path / exp, {"Mars": [1]}, {"Mars": [1]}, hardware="Linux x86_64, 8 cores; TensorFlow threads 8; ENA threads 1")

	with step("Act/Assert: one hardware macro."):
		assert hardware_entries("cheetah", str(tmp_path)) == {"cheetah/hardware": "Linux x86_64, 8 cores; TensorFlow threads 8; ENA threads 1"}


def test_hardware_entries_list_every_distinct_machine(tmp_path) -> None:
	"""Replicates run on different machines are never reported as one."""
	with step("Arrange: two machines."):
		_replicate(tmp_path / "exp_1", {"Mars": [1]}, {"Mars": [1]}, hardware="A")
		_replicate(tmp_path / "exp_2", {"Mars": [1]}, {"Mars": [1]}, hardware="B")

	with step("Act/Assert: both."):
		assert hardware_entries("cheetah", str(tmp_path)) == {"cheetah/hardware": "A; B"}


def test_retention_table_has_one_row_per_agent_with_end_test_retention_and_time() -> None:
	"""Per environment left behind: end of phase, test and retention; then the mean retention and the training time."""
	with step("Arrange: two agents, one replicate each."):
		rows = _rows({"SAC": [("e1", "Mars", 1000, 500), ("e1", "Heavy", 2000, 1000)], "ENA": [("e1", "Mars", 800, 720), ("e1", "Heavy", 900, 900)]})

	with step("Act: the table."):
		table = retention_table(rows, LEFT, ["SAC", "ENA"], {"SAC": "1,200.0", "ENA": "40.0"}, "Retention", "tab:cheetah_retention")

	with step("Assert: the rows' values."):
		assert "\\label{tab:cheetah_retention}" in table
		assert "SAC & 1,000 & 500 & 50.0 & 2,000 & 1,000 & 50.0 & 50.0 & 1,200.0 \\\\" in table
		assert "ENA & 800 & 720 & 90.0 & 900 & 900 & 100.0 & 95.0 & 40.0 \\\\" in table


def test_retention_figure_writes_one_box_per_agent(tmp_path) -> None:
	"""A box per agent of each replicate's left-behind retention."""
	with step("Arrange: two agents."):
		rows = _rows({"SAC": [("e1", "Mars", 100, 50), ("e2", "Mars", 100, 60)], "ENA": [("e1", "Mars", 100, 90), ("e2", "Mars", 100, 95)]})

	with step("Act: the figure."):
		path = retention_figure(rows, ["Mars"], ["SAC", "ENA"], str(tmp_path / "retention.pdf"))

	with step("Assert: written."):
		assert path.endswith("retention.pdf") and (tmp_path / "retention.pdf").stat().st_size > 0


def test_the_analysis_follows_the_study_s_worlds_and_order() -> None:
	"""The analysis's environment lists are the runner's: training order, test order, and the environments left behind."""
	# pylint: disable=import-outside-toplevel
	import half_cheetah
	from tools.cheetah_results import CHEETAH_ENVS, CHEETAH_LEFT_BEHIND, CHEETAH_TRAIN_ENVS

	with step("Act: the runner's worlds."):
		training = half_cheetah.make_training_worlds()
		test = half_cheetah.make_test_worlds(training)

	with step("Assert: the same names in the same order; every training world but the last is left behind."):
		assert CHEETAH_TRAIN_ENVS == [world.unwrapped.name for world in training]
		assert CHEETAH_ENVS == [world.unwrapped.name for world in test]
		assert CHEETAH_LEFT_BEHIND == ["Earth-Standard", "Heavy-Rough"]
