"""Tests for half_cheetah_probe.py — the HalfCheetah design probes (forgetting, learning, health)."""

# General imports
import json

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
import half_cheetah
from tools.half_cheetah_probe import forgetting, parse_overrides, probe, summarize


def test_forgetting_reports_the_drop_below_the_end_of_phase_return_for_environments_left_behind() -> None:
	"""The last training environment is not 'left behind'; earlier ones are compared with their own phase's end."""
	with step("Act: Mars ended at 4000 and tests at 1000; Earth (last) is skipped."):
		result = forgetting({"Mars": 4000.0, "Earth": 5000.0}, {"Mars": 1000.0, "Earth": 5100.0})

	with step("Assert: a 75% drop on Mars, nothing for Earth."):
		assert result == {"Mars": {"end_of_phase": 4000.0, "test": 1000.0, "drop_percent": 75.0}}


def test_summarize_takes_the_last_and_best_check_per_phase_and_the_test_mean_per_environment() -> None:
	"""End of phase = the last evaluation check in that environment."""
	with step("Arrange: two checks per training environment, a test of two environments."):
		evaluations = pd.DataFrame({"env_id": ["Mars", "Mars", "Earth", "Earth"], "eval_mean": [100.0, 300.0, 500.0, 400.0]})
		testing = pd.DataFrame({"env_id": ["Mars", "Mars", "Earth", "Earth"], "score": [50.0, 70.0, 380.0, 420.0]})

	with step("Act: summarize."):
		summary = summarize(evaluations, testing, ["Mars", "Earth"])

	with step("Assert: end of phase 300 / 400, best 300 / 500, test means 60 / 400, Mars drop 80%."):
		assert summary["end_of_phase"] == {"Mars": 300.0, "Earth": 400.0}
		assert summary["best_check"] == {"Mars": 300.0, "Earth": 500.0}
		assert summary["test"]["Mars"]["mean"] == 60.0
		assert summary["forgetting"]["Mars"]["drop_percent"] == pytest.approx(80.0)


def test_probe_runs_end_to_end_and_writes_every_output(tmp_path, monkeypatch) -> None:
	"""A tiny real run: training with checks, frozen test on all five environments, trace and summary."""
	with step("Arrange: checks every 1,000 steps, so a 2,000-step phase has checks."):
		monkeypatch.setattr(half_cheetah, "CHECK_EVERY", 1_000)
		monkeypatch.setattr(half_cheetah, "EVAL_EPISODES", 1)

	with step("Act: PPO, one environment, 2,000 steps, one test episode per environment."):
		summary = probe("PPO", ["Earth-Standard"], 2_000, 1, str(tmp_path), trace_every=500)

	with step("Assert: outputs exist; the test covers the five environments; the trace has PPO's learning rate."):
		for name in ("live_training.csv", "live_evaluation.csv", "live_testing.csv", "trace.csv", "summary.json"):
			assert (tmp_path / name).exists(), name
		assert len(summary["test"]) == 5
		assert "learning_rate" in pd.read_csv(tmp_path / "trace.csv").columns
		assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["agent"] == "PPO"


def test_parse_overrides_casts_numbers_and_keeps_text() -> None:
	"""Calibration overrides from the command line."""
	with step("Act/Assert: int, float and text values."):
		assert parse_overrides(["pop_size=50", "mutation_noise=0.2", "selection_driver=bpr"]) == {
			"pop_size": 50,
			"mutation_noise": 0.2,
			"selection_driver": "bpr",
		}


def test_ena_overrides_reach_the_agent() -> None:
	"""build_agent passes calibration overrides to ENA and keeps every other study setting."""
	with step("Act: ENA with a smaller population and shallower evaluation."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1, pop_size=50, max_eval_steps=100)

	with step("Assert: overridden values, the rest from ENA_SETTINGS."):
		assert (agent.pop_size, agent.max_eval_steps) == (50, 100)
		assert agent._skills["env_detection"]._instance._min_verify_samples == 1_000  # pylint: disable=protected-access


def test_ena_network_size_is_a_calibration_option() -> None:
	"""brain_hidden sets ENA's hidden units per layer (default 8)."""
	with step("Act: ENA with 32-unit hidden layers."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1, brain_hidden=32, pop_size=4)

	with step("Assert: 32 units in each hidden layer."):
		model = agent.population[0]["model"]
		assert [layer.units for layer in model.layers] == [32, 32, 6]
