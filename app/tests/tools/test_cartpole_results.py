"""Tests for cartpole_results.py: reliability on the training environments, and steps to first solve."""

# Libs imports
import pandas as pd

# App imports
from helpers import step
from tools.cartpole_results import TRAINING_RELIABILITY, add_training_reliability, first_solve_entries
from tools.metrics import reliability


def test_reliability_is_the_share_of_episodes_reaching_the_threshold() -> None:
	"""Reliability counts an episode at exactly the threshold as a success."""
	with step("Act/Assert: two of four episodes reach 475."):
		assert reliability(pd.Series([475.0, 500.0, 474.9, 10.0]), 475.0) == 50.0


def test_add_training_reliability_counts_only_the_test_episodes_in_the_training_environments(tmp_path) -> None:
	"""The column leaves out the unseen environments' episodes, per agent and replicate."""
	with step("Arrange: one replicate; ENA succeeds in one of two Mars episodes and fails Neptune, which is unseen."):
		(tmp_path / "exp_1").mkdir()
		testing = pd.DataFrame({"env_id": ["Mars", "Mars", "Neptune", "Neptune"], "score": [500.0, 100.0, 0.0, 0.0]})
		testing.to_csv(tmp_path / "exp_1" / "live_testing_ENA.csv", index=False)
		results = {"Test": [pd.DataFrame([{"Agent": "ENA", "Exp": "exp_1", "Reliability (%)": "25.0%"}])]}

	with step("Act: add the column."):
		add_training_reliability(results, str(tmp_path), ["Mars"], 475.0)

	with step("Assert: 50% on Mars alone."):
		assert results["Test"][0].loc[0, TRAINING_RELIABILITY] == 50.0


def test_first_solve_entries_give_quartiles_unsolved_counts_and_a_rank_test_of_the_last_agent_against_each_other() -> None:
	"""Medians and quartiles per agent, the replicates that never solved, and the last agent vs each baseline."""
	with step("Arrange: PPO solves Mars in 10-50 steps; ENA in 100-500, and one replicate never solves it."):
		steps = {"PPO": [10, 20, 30, 40, 50], "ENA": [100, 200, 300, 400, 500]}
		solved = {"PPO": [1, 1, 1, 1, 1], "ENA": [1, 1, 1, 1, 0]}
		series = {(agent, "Mars:steps"): pd.Series(values) for agent, values in steps.items()}
		series.update({(agent, "Mars:solved"): pd.Series(values) for agent, values in solved.items()})

	with step("Act: the entries."):
		entries = first_solve_entries("challenge", lambda agent, metric: series[(agent, metric)], ["PPO", "ENA"], ["Mars"])

	with step("Assert: quartiles and unsolved counts, and a rank test only of ENA against PPO."):
		assert (entries["challenge/ENA/adapt/Mars:steps/q1"], entries["challenge/ENA/adapt/Mars:steps/median"]) == ("200", "300")
		assert entries["challenge/ENA/adapt/Mars:steps/q3"] == "400"
		assert (entries["challenge/ENA/adapt/Mars:steps/unsolved"], entries["challenge/PPO/adapt/Mars:steps/unsolved"]) == ("1", "0")
		assert entries["challenge/PPO-vs-ENA/adapt/Mars:steps/p"] == "$p=0.008$"  # fully separated 5 vs 5: exact 2/252
		assert not any("ENA-vs-" in key for key in entries)
