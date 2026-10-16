"""Tests for the selection study's paper numbers (training identity, gladiators played, paired counts)."""

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.selection_results import SELECTION_AGENTS, SELECTION_REFERENCE, paired_count_entries, played_entries, training_identity_entries


def _replicate(root, exp_dir, agents, testing_ids=(1,), complete=True):
	"""Write one replicate's outputs.txt and, per agent, its training and testing logs."""
	folder = root / exp_dir
	folder.mkdir(parents=True)
	(folder / "outputs.txt").write_text("--- Test Results ---\n" if complete else "header\n", encoding="utf-8")
	for agent in agents:
		(folder / f"live_training_{agent}.csv").write_text("episode,score\n1,10\n", encoding="utf-8")
		pd.DataFrame({"specialist_id": list(testing_ids)}).to_csv(folder / f"live_testing_{agent}.csv", index=False)


def test_training_identity_counts_byte_identical_logs_over_complete_replicates(tmp_path) -> None:
	"""exp_1 identical for both variants, exp_2 differs for one; the incomplete exp_3 is not checked."""
	with step("Arrange: the reference study and the selection study."):
		reference, study = tmp_path / "ablation", tmp_path / "selection"
		for exp_dir in ("exp_1", "exp_2", "exp_3"):
			_replicate(reference, exp_dir, [SELECTION_REFERENCE])
		_replicate(study, "exp_1", SELECTION_AGENTS)
		_replicate(study, "exp_2", SELECTION_AGENTS[:1])
		(study / "exp_2" / f"live_training_{SELECTION_AGENTS[1]}.csv").write_text("episode,score\n1,11\n", encoding="utf-8")
		_replicate(study, "exp_3", SELECTION_AGENTS, complete=False)

	with step("Act: the identity entries."):
		entries = training_identity_entries(str(study), str(reference))

	with step("Assert: 3 of 4 checked logs identical."):
		assert entries == {"selection/identity/identical": "3", "selection/identity/checked": "4"}


def test_played_entries_average_distinct_individuals_per_test(tmp_path) -> None:
	"""The reference is read from its own study, the variants from the selection study."""
	with step("Arrange: two replicates; the reference plays 3 then 5 individuals, each variant 1."):
		reference, study = tmp_path / "ablation", tmp_path / "selection"
		_replicate(reference, "exp_1", [SELECTION_REFERENCE], testing_ids=(1, 2, 3, 3))
		_replicate(reference, "exp_2", [SELECTION_REFERENCE], testing_ids=(1, 2, 3, 4, 5))
		for exp_dir in ("exp_1", "exp_2"):
			_replicate(study, exp_dir, SELECTION_AGENTS, testing_ids=(7, 7))

	with step("Act: the played entries."):
		entries = played_entries(str(study), str(reference))

	with step("Assert: means of 4.0 and 1.0."):
		assert entries[f"selection/{SELECTION_REFERENCE}/played/mean"] == "4.0"
		assert all(entries[f"selection/{agent}/played/mean"] == "1.0" for agent in SELECTION_AGENTS)


@pytest.mark.parametrize(
	"values_b, expected",
	[
		([1.0, 2.0, 3.0], ("3", "0", "3")),
		([3.0, 4.0, 5.0], ("0", "3", "3")),
		([2.0, None, 5.0], ("0", "1", "2")),  # a missing replicate is not a pair; a tie counts in neither
	],
)
def test_paired_count_entries_compare_matching_replicates(values_b, expected) -> None:
	"""Counts where b is lower and higher than a over the replicates both have; ties count in neither."""
	with step("Arrange: a = 2, 3, 4 on exp_1..exp_3."):
		index = ["exp_1", "exp_2", "exp_3"]
		values_a = pd.Series([2.0, 3.0, 4.0], index=index)

	with step("Act: the paired counts."):
		entries = paired_count_entries("key", values_a, pd.Series(values_b, index=index, dtype=float))

	with step("Assert: lower, higher, pairs."):
		assert (entries["key/lower"], entries["key/higher"], entries["key/pairs"]) == expected
