"""Tests for sac_discrete_entropy_check.py — the measurement behind SAC-Discrete's target-entropy choice."""

# Libs imports
import pandas as pd

# App imports
from helpers import step
from tools.sac_discrete_entropy_check import run_one, summarize


def test_summarize_takes_per_seed_start_peak_and_end_then_the_median_across_seeds() -> None:
	"""Per ratio: medians across seeds of alpha's first, peak and last value, of the peak critic value; seed count."""
	with step("Arrange: ratio 0.98 diverges in both seeds, 0.5 stays bounded."):
		rows = []
		for seed, scale in ((0, 1.0), (1, 3.0)):
			rows += [
				{"ratio": 0.98, "seed": seed, "env_steps": steps, "alpha": alpha * scale, "value": value}
				for steps, alpha, value in ((0, 1.0, 10.0), (100, 500.0, 900.0), (200, 400.0, 800.0))
			]
			rows += [
				{"ratio": 0.5, "seed": seed, "env_steps": steps, "alpha": alpha, "value": 20.0} for steps, alpha in ((0, 1.0), (100, 2.0), (200, 1.5))
			]

	with step("Act: summarize."):
		summary = summarize(pd.DataFrame(rows)).set_index("ratio")

	with step("Assert: 0.98's median peak alpha is (500 + 1500) / 2; 0.5 peaks at 2; two seeds; first row is the larger ratio."):
		assert summary.loc[0.98, "alpha_max"] == 1000.0
		assert summary.loc[0.98, "alpha_start"] == 2.0
		assert summary.loc[0.98, "value_max"] == 900.0
		assert summary.loc[0.5, "alpha_max"] == 2.0
		assert summary.loc[0.5, "seeds"] == 2
		assert list(summarize(pd.DataFrame(rows))["ratio"]) == [0.98, 0.5]


def test_run_one_trains_the_challenges_sac_discrete_and_logs_alpha_and_value() -> None:
	"""A short real run: a trace starting at the first episode, positive temperatures, the ratio and seed recorded."""
	with step("Act: 600 steps of ratio 0.5, logging every 200."):
		trace = run_one(0.5, seed=0, steps=600, log_every=200)

	with step("Assert: several entries, increasing steps, alpha > 0."):
		assert len(trace) >= 2
		assert [row["env_steps"] for row in trace] == sorted(row["env_steps"] for row in trace)
		assert all(row["alpha"] > 0 and row["ratio"] == 0.5 and row["seed"] == 0 for row in trace)
