"""Selection study (cartpole_ena_selection.py) numbers the paper quotes beyond the shared tables: the training-identity
check against the ablation's ENA-Combined, the gladiators each driver plays at test, and paired replicate counts."""

# General imports
import filecmp
import os

# Libs imports
import pandas as pd

# The BPR reference is the ablation's own ENA-Combined (same seeds, identical training): docs/PLAN_ena_selection.md.
SELECTION_REFERENCE = "ENA-Combined"
SELECTION_AGENTS = ["ENA-Combined-NicheMatch", "ENA-Combined-BestTrust"]
# The planned family (Holm over 2): Q1 BPR vs. detector-only retrieval; Q2 detector-only vs. highest-trust retrieval.
SELECTION_FAMILY = [(SELECTION_REFERENCE, SELECTION_AGENTS[0]), (SELECTION_AGENTS[0], SELECTION_AGENTS[1])]


def complete_replicates(study_dir: str) -> list:
	"""The study's exp_* directories whose outputs.txt holds the test results, sorted by replicate number."""
	complete = []
	for exp_dir in os.listdir(study_dir):
		path = os.path.join(study_dir, exp_dir, "outputs.txt")
		if exp_dir.startswith("exp_") and os.path.exists(path):
			with open(path, encoding="utf-8") as outputs:
				if "Test Results ---" in outputs.read():
					complete.append(exp_dir)
	return sorted(complete, key=lambda name: int(name.split("_")[1]))


def training_identity_entries(study_dir: str, reference_dir: str) -> dict:
	"""How many (replicate, variant) training logs are byte-identical to the reference's same replicate, of those checked."""
	identical = checked = 0
	for exp_dir in complete_replicates(study_dir):
		reference = os.path.join(reference_dir, exp_dir, f"live_training_{SELECTION_REFERENCE}.csv")
		for agent in SELECTION_AGENTS:
			checked += 1
			path = os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv")
			identical += os.path.exists(reference) and os.path.exists(path) and filecmp.cmp(reference, path, shallow=False)
	return {"selection/identity/identical": str(identical), "selection/identity/checked": str(checked)}


def played_entries(study_dir: str, reference_dir: str) -> dict:
	"""Mean number of distinct individuals each driver plays over a replicate's test, over the study's complete replicates."""
	sources = {SELECTION_REFERENCE: reference_dir, **{agent: study_dir for agent in SELECTION_AGENTS}}
	entries = {}
	replicates = complete_replicates(study_dir)
	for agent, directory in sources.items():
		counts = []
		for exp_dir in replicates:
			path = os.path.join(directory, exp_dir, f"live_testing_{agent}.csv")
			if os.path.exists(path):
				counts.append(pd.read_csv(path, usecols=["specialist_id"])["specialist_id"].nunique())
		if counts:
			entries[f"selection/{agent}/played/mean"] = f"{sum(counts) / len(counts):.1f}"
	return entries


def paired_count_entries(key: str, values_a: pd.Series, values_b: pd.Series) -> dict:
	"""Over the replicates both have (Series indexed by replicate): how many b is lower, higher than a, and the pairs."""
	pairs = pd.concat({"a": values_a, "b": values_b}, axis=1).dropna()
	return {
		f"{key}/lower": str(int((pairs["b"] < pairs["a"]).sum())),
		f"{key}/higher": str(int((pairs["b"] > pairs["a"]).sum())),
		f"{key}/pairs": str(len(pairs)),
	}
