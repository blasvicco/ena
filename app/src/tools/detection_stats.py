#!/usr/bin/env python
"""Environment-recognition numbers per phase, from a study's live logs (the paper's appendix on what the detector achieves)."""
# Usage: python src/tools/detection_stats.py --output-dir cartpole_ena_ablation_outputs

# General imports
import argparse
import glob
import math
import os
import sys

# Libs imports
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from tools.metrics import uses_env_detection
from tools.replicates import get_experiment_directories

# What each number means; written at the top of every detection_summary.txt.
RECOGNITION_LEGEND = "\n".join(
	[
		"For every agent with a real recognizer and every complete replicate, per contiguous environment phase:",
		"  - unrecognized: % of episodes whose transitions matched no known niche (detected_env_id -1);",
		"  - changes: confirmed changes (detect_change() firing). A phase follows a real switch, except the",
		"    run's very first, so at most one is a detection and the rest are false alarms;",
		"  - first_change: episodes into the phase of the first confirmation (none: NaN);",
		"  - niches (training only): niches the Hall of Fame holds for the environment (fragmentation if > 1).",
	]
)

PHASES = {"train": ("training", "Training"), "test": ("testing", "Test")}
METRICS = ("unrecognized", "changes", "first_change", "niches")


def phase_detection(history: pd.DataFrame) -> list:
	"""One dict per contiguous environment phase of a live log."""
	phase_ids = history["env_id"].ne(history["env_id"].shift()).cumsum()
	phases = []
	for _phase, phase in history.groupby(phase_ids, sort=True):
		fired = phase["env_shift_detected"].astype(bool).to_numpy()
		niches = phase["detected_env_id"]
		phases.append(
			{
				"env_id": phase["env_id"].iloc[0],
				"episodes": len(phase),
				"unrecognized": float((niches == -1).mean() * 100),
				"changes": int(fired.sum()),
				"first_change": int(fired.argmax()) if fired.any() else None,
				"niches": int(niches[niches != -1].nunique()),
			}
		)
	return phases


def niches_per_env(niches_text: str) -> dict:
	"""{environment: number of Hall of Fame niches labelled with it}, from a niches_<agent>.txt report."""
	counts = {}
	for line in niches_text.splitlines():
		cells = [cell.strip() for cell in line.split("|")]
		if len(cells) == 3 and cells[0].isdigit():
			counts[cells[1]] = counts.get(cells[1], 0) + 1
	return counts


def replicate_detection(exp_dir: str, agent: str) -> list:
	"""Long-format rows (phase, env_id, metric, value) for one agent in one replicate."""
	rows = []
	with open(os.path.join(exp_dir, f"niches_{agent}.txt"), encoding="utf-8") as report:
		niche_counts = niches_per_env(report.read())
	for phase_key, (file_phase, _label) in PHASES.items():
		history = pd.read_csv(os.path.join(exp_dir, f"live_{file_phase}_{agent}.csv"))
		for phase in phase_detection(history):
			values = {
				"unrecognized": phase["unrecognized"],
				"changes": phase["changes"],
				"first_change": math.nan if phase["first_change"] is None else phase["first_change"],
			}
			if phase_key == "train":
				values["niches"] = niche_counts.get(phase["env_id"], 0)
			rows += [{"phase": phase_key, "env_id": phase["env_id"], "metric": metric, "value": value} for metric, value in values.items()]
	return rows


def study_detection(study_dir: str) -> pd.DataFrame:
	"""Every replicate's rows, for every agent whose training log shows a real recognizer."""
	rows = []
	for exp_dir in (os.path.join(study_dir, name) for name in get_experiment_directories(study_dir)):
		for path in sorted(glob.glob(os.path.join(exp_dir, "live_training_*.csv"))):
			agent = os.path.basename(path)[len("live_training_") : -len(".csv")]
			if not uses_env_detection(pd.read_csv(path)):
				continue
			rows += [{"agent": agent, "exp": os.path.basename(exp_dir), **row} for row in replicate_detection(exp_dir, agent)]
	return pd.DataFrame(rows, columns=["agent", "exp", "phase", "env_id", "metric", "value"])


def summarize(detection: pd.DataFrame) -> pd.DataFrame:
	"""Mean, std and replicate count of each (agent, phase, env_id, metric); NaN values (no change) are not counted."""
	keys = ["agent", "phase", "env_id", "metric"]
	grouped = detection.dropna(subset=["value"]).groupby(keys, sort=False)["value"]
	return grouped.agg(mean="mean", std="std", n="count").reset_index()


def write_detection_report(study_dir: str) -> str:
	"""Write detection_summary.txt into the study directory and return its path."""
	detection = study_detection(study_dir)
	summary = summarize(detection)
	path = os.path.join(study_dir, "detection_summary.txt")
	with open(path, "w", encoding="utf-8") as report:
		report.write(f"Environment recognition across {detection['exp'].nunique()} replicates (mean ± std; n = replicates with a value)\n")
		report.write("=" * 60 + "\n")
		report.write(RECOGNITION_LEGEND + "\n")
		for agent, by_agent in summary.groupby("agent", sort=False):
			for phase_key, (_file_phase, label) in PHASES.items():
				rows = by_agent[by_agent["phase"] == phase_key]
				if rows.empty:
					continue
				report.write(f"\n--- {agent}: {label} ---\n")
				for env_id, by_env in rows.groupby("env_id", sort=False):
					by_metric = {row.metric: row for row in by_env.itertuples()}
					cells = [
						f"{metric} {row.mean:.1f} ± {0.0 if math.isnan(row.std) else row.std:.1f} (n={row.n})"
						for metric, row in ((metric, by_metric.get(metric)) for metric in METRICS)
						if row is not None
					]
					report.write(f"  {env_id:<16} " + " | ".join(cells) + "\n")
	return path


def main() -> None:
	"""Write a study's detection_summary.txt."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--output-dir", required=True, help="Study directory holding the exp_* replicates.")
	print(f"Wrote {write_detection_report(parser.parse_args().output_dir)}")


if __name__ == "__main__":
	main()
