"""Selection study numbers beyond the shared tables: training identity, individuals played, paired counts, the Trust audit."""
# Selection study (cartpole_ena_selection.py) numbers the paper quotes beyond the shared tables: the training-identity
# check against the ablation's ENA-Combined, the gladiators each driver plays at test, paired replicate counts, and an
# audit of the stored Trust ranking: where Bayesian policy reuse settles on another gladiator of the niche whose
# highest-Trust gladiator niche match plays, and how each driver does there.

# General imports
import filecmp
import os

# Libs imports
import pandas as pd

# App imports
from tools.metrics import reliability
from tools.replicates import get_experiment_directories

# The BPR reference is the ablation's own ENA-Combined (same seeds, identical training).
SELECTION_REFERENCE = "ENA-Combined"
SELECTION_AGENTS = ["ENA-Combined-NicheMatch", "ENA-Combined-BestTrust"]
# The planned family (Holm over 2): Q1 BPR vs. detector-only retrieval; Q2 detector-only vs. highest-trust retrieval.
SELECTION_FAMILY = [(SELECTION_REFERENCE, SELECTION_AGENTS[0]), (SELECTION_AGENTS[0], SELECTION_AGENTS[1])]
# Exploratory, outside the family: the archive with each simpler driver against no archive at all, the ablation's
# ENA-Plasticity of the same replicates (same seeds).
SELECTION_NO_ARCHIVE = "ENA-Plasticity"
SELECTION_DESCRIPTIVE = [(SELECTION_NO_ARCHIVE, agent) for agent in SELECTION_AGENTS]


def training_identity_entries(study_dir: str, reference_dir: str) -> dict:
	"""How many (replicate, variant) training logs are byte-identical to the reference's same replicate, of those compared."""
	identical = checked = 0
	for exp_dir in get_experiment_directories(study_dir):
		reference = os.path.join(reference_dir, exp_dir, f"live_training_{SELECTION_REFERENCE}.csv")
		for agent in SELECTION_AGENTS:
			checked += 1
			path = os.path.join(study_dir, exp_dir, f"live_training_{agent}.csv")
			identical += filecmp.cmp(reference, path, shallow=False)
	return {"selection/identity/identical": str(identical), "selection/identity/checked": str(checked)}


def played_entries(study_dir: str, reference_dir: str) -> dict:
	"""Mean number of distinct individuals each driver plays over a replicate's test, over the study's replicates."""
	sources = {SELECTION_REFERENCE: reference_dir, **{agent: study_dir for agent in SELECTION_AGENTS}}
	entries = {}
	replicates = get_experiment_directories(study_dir)
	for agent, directory in sources.items():
		counts = []
		for exp_dir in replicates:
			path = os.path.join(directory, exp_dir, f"live_testing_{agent}.csv")
			counts.append(pd.read_csv(path, usecols=["specialist_id"])["specialist_id"].nunique())
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


def _gladiator_layout(testing: pd.DataFrame) -> pd.DataFrame | None:
	"""Each gladiator's home niche and label by population index in a test log; None if an index maps to more than one."""
	gladiators = testing[testing["is_gladiator"].astype(bool)]
	if (gladiators.groupby("specialist_id")[["internal_env_id", "gladiator_env"]].nunique() > 1).any().any():
		return None
	return gladiators.drop_duplicates("specialist_id").set_index("specialist_id")[["internal_env_id", "gladiator_env"]].sort_index()


def _same_layout(testing_a: pd.DataFrame, testing_b: pd.DataFrame) -> bool:
	"""Whether two drivers' test logs of one replicate give every population index they share the same gladiator."""
	layout_a, layout_b = _gladiator_layout(testing_a), _gladiator_layout(testing_b)
	if layout_a is None or layout_b is None:
		return False
	shared = layout_a.index.intersection(layout_b.index)
	return layout_a.loc[shared].equals(layout_b.loc[shared])


def _mostly_played(phase: pd.DataFrame) -> pd.Series:
	"""The log row of the individual a test phase plays most often (its population index, gladiator flag and home niche)."""
	return phase[phase["specialist_id"] == phase["specialist_id"].mode().iloc[0]].iloc[0]


# pylint: disable=too-many-locals
def trust_audit_entries(study_dir: str, reference_dir: str, train_envs: list, threshold: float) -> dict:
	"""Test phases where BPR mostly plays another gladiator of the niche whose highest-Trust gladiator niche match plays."""
	# Per replicate and training environment, both drivers meet the same seeded test episodes with the same archive (the
	# training logs are byte-identical). Niche match plays the highest stored Trust gladiator of a niche; a phase counts
	# when BPR mostly plays another gladiator of that same home niche. Over those phases: each driver's mean test
	# reliability and how many phases each is the more reliable in. A replicate whose logs disagree on which gladiator a
	# population index holds is excluded (and counted).
	niche_match = SELECTION_AGENTS[0]
	audited, phases, excluded = [], 0, 0
	for exp_dir in get_experiment_directories(study_dir):
		bpr = pd.read_csv(os.path.join(reference_dir, exp_dir, f"live_testing_{SELECTION_REFERENCE}.csv"))
		matched = pd.read_csv(os.path.join(study_dir, exp_dir, f"live_testing_{niche_match}.csv"))
		if not _same_layout(bpr, matched):
			excluded += 1
			continue
		for env in train_envs:
			phase_bpr, phase_matched = bpr[bpr["env_id"] == env], matched[matched["env_id"] == env]
			phases += 1
			played_bpr, played_matched = _mostly_played(phase_bpr), _mostly_played(phase_matched)
			gladiators = bool(played_bpr["is_gladiator"]) and bool(played_matched["is_gladiator"])
			if (
				gladiators
				and played_bpr["specialist_id"] != played_matched["specialist_id"]
				and played_bpr["internal_env_id"] == played_matched["internal_env_id"]
			):
				audited.append((reliability(phase_bpr["score"], threshold), reliability(phase_matched["score"], threshold)))
	frame = pd.DataFrame(audited, columns=["bpr", "nm"])
	return {
		"selection/audit/phases": str(len(frame)),
		"selection/audit/of": str(phases),
		"selection/audit/excluded": str(excluded),
		"selection/audit/bpr": f"{frame['bpr'].mean():.1f}" if len(frame) else "n/a",
		"selection/audit/nm": f"{frame['nm'].mean():.1f}" if len(frame) else "n/a",
		"selection/audit/bprhigher": str(int((frame["bpr"] > frame["nm"]).sum())),
		"selection/audit/nmhigher": str(int((frame["nm"] > frame["bpr"]).sum())),
	}
