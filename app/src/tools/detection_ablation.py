#!/usr/bin/env python
"""The chirp detector's ablation on the detection benchmark: excitation x response features, plus its baselines."""
# The chirp detector's ablation on the detection benchmark: excitation x response features, over benchmark seeds,
# plus the baselines that motivate it (the transition-model detector under changing and under fixed behavior).
#
# Every run is tools/detection_benchmark.run() on HalfCheetah (Earth -> Heavy -> Mars, policies changing every
# 4 episodes): each benchmark seed draws other policies and resets. Writes, into --output-dir:
#   - grid.csv: one row per (excitation, features, seed) with the score and whether it meets the target;
#   - confirmation.csv: one row per (baseline, seed) on the confirmation seeds (the study's chirp is a baseline);
#   - cartpole.csv: CartPole's detector on CartPole's benchmark, confirmation seeds (the benchmark's sanity check).
# tools/paper_results.py turns them into the paper's table and macros.
#
#         python src/tools/detection_ablation.py --output-dir outputs/detection_ablation_outputs --workers 4

# General imports
import argparse
import itertools
import multiprocessing
import os
import sys

# Libs imports
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from tools.detection_benchmark import meets_target, run

# Three disjoint seed sets, each with one role:
#   - SELECTION_SEEDS (0-4): first screening;
#   - SEEDS (5-34): the ablation grid the paper tabulates; the study's configuration is chosen from it;
#   - CONFIRMATION_SEEDS (35-64): the chosen configuration and the baselines, never used for any choice: the
#     pass rate and the comparisons the paper reports.
# 30 seeds: a configuration passing all of them has a two-sided 95% Clopper-Pearson interval of [0.884, 1].
SELECTION_SEEDS = range(5)
# The configuration screening first chose (seeds 0-4), replaced after the grid: reported for transparency.
SCREENED = ("Wavelet train (db4)", "DWT Haar")
SEEDS = range(5, 35)
CONFIRMATION_SEEDS = range(35, 65)
# Excitations (tools/chirp_signals.py): label -> (name, parameters).
EXCITATIONS = {
	"Linear chirp": ("chirp", {}),
	"Exponential chirp": ("exponential_chirp", {}),
	"Sine": ("sine", {}),
	"Impulse": ("impulse", {}),
	"Wavelet train (Haar)": ("wavelet_train", {"wavelet": "haar"}),
	"Wavelet train (db4)": ("wavelet_train", {"wavelet": "db4"}),
	"Wavelet train (Mexican hat)": ("wavelet_train", {"wavelet": "mexh"}),
}
# Response features (tools/chirp_features.py): label -> settings. Wavelet ones keep one energy per channel and band.
FEATURES = {
	"Raw": {"kind": "raw"},
	"DWT Haar": {"kind": "wavelet"},
	"DWT db4": {"kind": "wavelet", "wavelet": "db4", "level": 3},
	"SWT Haar": {"kind": "wavelet", "transform": "swt"},
	"Packets Haar": {"kind": "wavelet", "transform": "packet", "level": 3},
}
# The transition-model detector with the settings calibrated for HalfCheetah before the chirp replaced it
# (calibration): verification window 1,000, margin 3, recent window 3,000, maturity 1,000.
DYNAMICS_MODEL_CHEETAH = {
	"recognizer_residual_tolerance": 3.0,
	"recognizer_novelty_margin": 3.0,
	"recognizer_cusum_false_alarm_steps": 1e8,
	"recognizer_min_verify_samples": 1_000,
	"recognizer_recent_size": 3_000,
	"recognizer_min_samples": 1_000,
}
# label -> (driver, settings, policy family).
BASELINES = {
	"Transition model, changing behavior": ("dynamics_model", DYNAMICS_MODEL_CHEETAH, "drifting"),
	"Transition model, one fixed gait": ("dynamics_model", DYNAMICS_MODEL_CHEETAH, "fixed"),
	"Chirp (study), changing behavior": ("chirp", {}, "drifting"),
}


def _row(result: dict) -> dict:
	"""The benchmark score's fields kept per run."""
	return {
		"passes": meets_target(result),
		"changes": result["changes"],
		"inside": result["inside"],
		"niches": result["niches"],
		"min_purity": min(result["purity"].values()),
		"max_delay": max(result["delays"]) if min(result["delays"]) >= 0 else -1,
		"distinct": result["distinct"],
	}


def _grid_task(task: tuple) -> dict:
	"""One (domain, excitation, features, seed, chirp steps) run."""
	domain, excitation, features, seed, chirp_steps = task
	name, params = EXCITATIONS[excitation]
	chirp = {"excitation": name, "excitation_params": params, "features": FEATURES[features]}
	if chirp_steps:
		chirp["steps"] = chirp_steps
	return {"excitation": excitation, "features": features, "seed": seed, **_row(run(domain, "chirp", {"recognizer_chirp": chirp}, seed=seed))}


def _baseline_task(task: tuple) -> dict:
	"""One (domain, baseline, seed) run."""
	domain, baseline, seed = task
	driver, settings, family = BASELINES[baseline]
	return {"baseline": baseline, "seed": seed, **_row(run(domain, driver, dict(settings), seed=seed, policy_family=family))}


def _map(function, tasks: list, workers: int) -> list:
	"""Run tasks in order, in `workers` fresh processes (spawned: TensorFlow is imported by the domain modules)."""
	if workers <= 1:
		return [function(task) for task in tasks]
	with multiprocessing.get_context("spawn").Pool(workers) as pool:
		return pool.map(function, tasks)


def run_grid(domain: str = "half_cheetah", seeds: list | range = SEEDS, workers: int = 1, chirp_steps: int = None) -> pd.DataFrame:
	"""Every excitation x features x seed."""
	tasks = [(domain, excitation, features, seed, chirp_steps) for excitation, features, seed in itertools.product(EXCITATIONS, FEATURES, seeds)]
	return pd.DataFrame(_map(_grid_task, tasks, workers))


def run_baselines(domain: str = "half_cheetah", seeds: list | range = CONFIRMATION_SEEDS, workers: int = 1) -> pd.DataFrame:
	"""Every baseline x seed (by default the confirmation seeds)."""
	return pd.DataFrame(_map(_baseline_task, [(domain, baseline, seed) for baseline in BASELINES for seed in seeds], workers))


def study_labels(chirp_settings: dict) -> tuple:
	"""The (excitation, features) labels of a chirp driver's settings (the study's choice), or None where absent."""
	excitation = next(
		(
			label
			for label, (name, params) in EXCITATIONS.items()
			if name == chirp_settings["excitation"] and params == chirp_settings["excitation_params"]
		),
		None,
	)
	features = next((label for label, settings in FEATURES.items() if settings == chirp_settings["features"]), None)
	return excitation, features


def _cartpole_task(seed: int) -> dict:
	"""CartPole's transition-model detector (its study settings) on CartPole's benchmark, one seed."""
	return {"seed": seed, **_row(run("cartpole", seed=seed))}


def run_cartpole_check(seeds: list | range = CONFIRMATION_SEEDS, workers: int = 1) -> pd.DataFrame:
	"""The benchmark's sanity check: the detector known to work (CartPole's) on its own domain."""
	return pd.DataFrame(_map(_cartpole_task, list(seeds), workers))


def pass_table(rows: pd.DataFrame) -> pd.DataFrame:
	"""Count the seeds meeting the target per excitation (rows, in grid order) and feature type (columns, in grid order)."""
	table = rows.pivot_table(index="excitation", columns="features", values="passes", aggfunc="sum")
	return table.loc[_in_order(list(table.index), list(EXCITATIONS)), _in_order(list(table.columns), list(FEATURES))].astype(
		"Int64"
	)  # an absent combination stays missing (shown as --), never 0


def _in_order(labels: list, known: list) -> list:
	"""labels in the grid's order, any unknown ones last."""
	return [label for label in known if label in labels] + [label for label in labels if label not in known]


def main() -> None:
	"""Run the ablation grid and/or the confirmation runs and write grid.csv and confirmation.csv."""
	# Run the grid (ablation seeds) and/or the confirmation (the baselines, the study's chirp among them, on the
	# confirmation seeds); write grid.csv and confirmation.csv.
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--workers", type=int, default=1)
	parser.add_argument("--stage", choices=["grid", "confirmation", "cartpole", "all"], default="all")
	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	if args.stage in ("confirmation", "all"):
		confirmation = run_baselines(workers=args.workers)
		confirmation.to_csv(os.path.join(args.output_dir, "confirmation.csv"), index=False)
		print(confirmation.to_string(index=False))
	if args.stage in ("cartpole", "all"):
		cartpole = run_cartpole_check(workers=args.workers)
		cartpole.to_csv(os.path.join(args.output_dir, "cartpole.csv"), index=False)
		print(f"CartPole: {int(cartpole['passes'].sum())}/{len(cartpole)} seeds meet the target")
	if args.stage in ("grid", "all"):
		grid = run_grid(workers=args.workers)
		grid.to_csv(os.path.join(args.output_dir, "grid.csv"), index=False)
		print(pass_table(grid).to_string())


if __name__ == "__main__":
	main()
