"""Tests for the selection study's paper numbers (training identity, gladiators played, paired counts, the Trust audit)."""

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.selection_results import (
	SELECTION_AGENTS,
	SELECTION_REFERENCE,
	paired_count_entries,
	played_entries,
	training_identity_entries,
	trust_audit_entries,
)


def _replicate(root, exp_dir, agents, testing_ids=(1,)):
	"""Write one replicate's training and testing logs, per agent."""
	folder = root / exp_dir
	folder.mkdir(parents=True)
	for agent in agents:
		(folder / f"live_training_{agent}.csv").write_text("episode,score\n1,10\n", encoding="utf-8")
		pd.DataFrame({"specialist_id": list(testing_ids)}).to_csv(folder / f"live_testing_{agent}.csv", index=False)


def test_training_identity_counts_byte_identical_logs(tmp_path) -> None:
	"""exp_1 is identical for both variants, exp_2 differs for one."""
	with step("Arrange: the reference study and the selection study."):
		reference, study = tmp_path / "ablation", tmp_path / "selection"
		for exp_dir in ("exp_1", "exp_2"):
			_replicate(reference, exp_dir, [SELECTION_REFERENCE])
		_replicate(study, "exp_1", SELECTION_AGENTS)
		_replicate(study, "exp_2", SELECTION_AGENTS[:1])
		(study / "exp_2" / f"live_training_{SELECTION_AGENTS[1]}.csv").write_text("episode,score\n1,11\n", encoding="utf-8")

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


def _test_log(phases: list) -> pd.DataFrame:
	"""A live_testing log: per (environment, [(population index, home niche, label, score)]) phase, its episodes."""
	rows = [
		{"env_id": env, "specialist_id": index, "is_gladiator": True, "internal_env_id": niche, "gladiator_env": label, "score": score}
		for env, episodes in phases
		for index, niche, label, score in episodes
	]
	return pd.DataFrame(rows)


def _audit_replicate(root, exp_dir, agent, phases) -> None:
	"""Write one driver's test log of one replicate."""
	folder = root / exp_dir
	folder.mkdir(parents=True, exist_ok=True)
	_test_log(phases).to_csv(folder / f"live_testing_{agent}.csv", index=False)


def test_trust_audit_compares_the_drivers_where_bpr_settles_on_another_gladiator_of_the_same_niche(tmp_path) -> None:
	"""Mars: BPR settles on gladiator 2 of niche 0, niche match plays its top-Trust 1; Earth: both play 3; a mismatched replicate is excluded."""
	with step("Arrange: exp_1 audited; exp_2's logs give population index 1 two different niches."):
		reference, study = tmp_path / "ablation", tmp_path / "selection"
		bpr = [("Mars", [(1, 0, "Mars", 100.0), (2, 0, "Mars", 500.0), (2, 0, "Mars", 500.0)]), ("Earth", [(3, 1, "Earth", 500.0)] * 2)]
		matched = [("Mars", [(1, 0, "Mars", 100.0), (1, 0, "Mars", 500.0), (1, 0, "Mars", 100.0)]), ("Earth", [(3, 1, "Earth", 500.0)] * 2)]
		_audit_replicate(reference, "exp_1", SELECTION_REFERENCE, bpr)
		_audit_replicate(study, "exp_1", SELECTION_AGENTS[0], matched)
		_audit_replicate(reference, "exp_2", SELECTION_REFERENCE, bpr)
		_audit_replicate(study, "exp_2", SELECTION_AGENTS[0], [("Mars", [(1, 1, "Earth", 500.0)]), ("Earth", [(3, 1, "Earth", 500.0)])])

	with step("Act: the audit over the training environments Mars and Earth."):
		entries = trust_audit_entries(str(study), str(reference), ["Mars", "Earth"], 475.0)

	with step("Assert: 1 of 2 phases audited; BPR 66.7% against 33.3%; exp_2 excluded."):
		assert (entries["selection/audit/phases"], entries["selection/audit/of"], entries["selection/audit/excluded"]) == ("1", "2", "1")
		assert (entries["selection/audit/bpr"], entries["selection/audit/nm"]) == ("66.7", "33.3")
		assert (entries["selection/audit/bprhigher"], entries["selection/audit/nmhigher"]) == ("1", "0")
