"""Tests for benchmark_half_cheetah.py's throughput measurement and study cost model."""

# Libs imports
import pytest

# App imports
from helpers import step
from tools.benchmark_half_cheetah import estimate_study, measure_agent, replicate_seconds


def test_replicate_seconds_counts_training_and_the_frozen_test() -> None:
	"""Training: budget x environments at the training rate (no evaluation checks); the test: 1000-step episodes at the
	inference rate."""
	with step("Arrange: 0.01 s per training step, 0.001 s per inference step."):
		measurement = {"train_seconds_per_step": 0.01, "eval_seconds_per_step": 0.001}

	with step("Act: 3 environments x 100k steps, 2 test episodes on each of 5 worlds."):
		seconds = replicate_seconds(measurement, budget_per_env=100_000, n_envs=3, test_episodes=[2] * 5, episode_steps=1000)

	with step("Assert: 3,000 s training + 10 s test."):
		assert seconds == pytest.approx(3_000 + 10)


def test_estimate_study_runs_replicates_in_waves_of_workers() -> None:
	"""Agents run one after another inside a replicate; replicates run `workers` at a time."""
	with step("Act: 3 h + 1 h per replicate, 50 replicates, 20 workers of 2 threads, $3/h."):
		estimate = estimate_study({"SAC": 3 * 3600, "PPO": 3600}, replicates=50, workers=20, threads_per_worker=2, price_per_hour=3.0)

	with step("Assert: 4 h per replicate, 3 waves = 12 h, $36, 40 vCPUs, 50 x 4 h x 2 threads = 400 core-hours."):
		assert estimate["replicate_hours"] == pytest.approx(4.0)
		assert estimate["waves"] == 3
		assert estimate["wall_hours"] == pytest.approx(12.0)
		assert estimate["cost"] == pytest.approx(36.0)
		assert estimate["vcpus_needed"] == 40
		assert estimate["core_hours"] == pytest.approx(400.0)
		assert estimate["bottleneck_agent"] == "SAC"


def test_measure_agent_reports_rates_and_memory_for_a_real_world() -> None:
	"""A short real measurement: positive rates, steps actually consumed, memory reported."""
	with step("Act: measure PPO for 2,000 training steps and 1 evaluation episode."):
		measurement = measure_agent("PPO", train_steps=2_000, eval_episodes=1)

	with step("Assert: the shape of the result."):
		assert measurement["agent"] == "PPO"
		assert measurement["train_steps_measured"] >= 2_000
		assert measurement["train_seconds_per_step"] > 0
		assert measurement["eval_seconds_per_step"] > 0
		assert measurement["rss_mb"] > 0
