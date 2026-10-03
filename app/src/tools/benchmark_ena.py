#!/usr/bin/env python
"""Fast synthetic benchmark for ENA's niche/gladiator distribution.

Exercises the real ENAgent/HallOfFame/EnvironmentRecognizer control flow — via the CartPole studies' shared
play() loop (cartpole_ena_performance.py), unmodified — with SyntheticCartPoleEnv standing in for real CartPole physics,
so a candidate fix's effect on niche fragmentation and gladiator-count skew across true
environments can be measured in minutes instead of the hours a full cartpole_ena_performance.py experiment
takes. Same governing physics equations and score semantics as the real environment, so the
resulting niches_*.txt-style distribution is directly comparable in kind, if not in exact
magnitude, to a real run's.

A single seed is not reproducible even with random/np.random/tf.random all fixed — Keras/TF's
own internal op-level parallelism (independent of --max-threads) introduces floating-point
non-associativity that compounds over hundreds of episodes of evolutionary dynamics into
completely different outcomes. So this runs several seeds and reports mean±std per
environment, the same way the project's own tools/aggregate_metrics.py and plot_variance.py
treat a single full experiment run as statistically insufficient on its own.

Usage:
	python tools/benchmark_ena.py --save benchmark_outputs/before.json
	# ... apply a candidate fix ...
	python tools/benchmark_ena.py --compare benchmark_outputs/before.json --save benchmark_outputs/after.json
"""

# General imports
import argparse
import json
import os
import random
import sys
import time

# Libs imports
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from agents.ena.ena import ENAgent
from cartpole_ena_performance import ena_brain, play
from tools.synthetic_cartpole import make_synthetic_env

OUTPUT_DIR = "benchmark_outputs"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
EMPTY_BUCKET = {"niche_count": 0, "gladiator_count": 0}

PHASES = [
	{"gravity": 3.7, "masspole": 0.05, "length": 0.1, "name": "Mars-Light"},
	{"gravity": 24.8, "masspole": 0.5, "length": 1.5, "name": "Jupiter-Heavy"},
	{"gravity": 9.8, "masspole": 0.1, "length": 0.5, "name": "Earth-Standard"},
]


def _build_agent(
	pop_size: int,
	max_threads: int,
	recognizer_threshold: float = 0.90,
	recognizer_ema_alpha: float = 0.05,
) -> ENAgent:
	"""Build an ENAgent close to cartpole_ena_performance.py's ENA-Combined configuration. The
	recognizer_* params default to the studies' current values but are overridable so this
	tool can compare a candidate recognizer tuning against baseline without editing the file."""
	agent = ENAgent(
		brain=ena_brain,
		controller_episode_rate=0.15,
		make_env=make_synthetic_env,
		max_per_niche=5,
		max_gladiator_ratio=0.5,
		max_threads=max_threads,
		mutation_rate=0.05,
		physics=["gravity", "masspole", "length"],
		plasticity_algorithm="linear_ramp",
		pop_size=pop_size,
		recognizer_threshold=recognizer_threshold,
		recognizer_ema_alpha=recognizer_ema_alpha,
		recognizer_strategy="dynamics_model",
		selection_driver="heuristic",
		trust_algorithm="linear_ramp",
	)
	agent.set_name("ENA-Benchmark")
	return agent


def _parse_niches_file(niches_path: str) -> dict:
	"""Aggregate a niches_*.txt file's per-niche rows into per-true-env totals."""
	per_env = {}
	with open(niches_path, encoding="utf-8") as niches_file:
		for line in niches_file.readlines()[2:]:
			columns = [column.strip() for column in line.split("|")]
			if len(columns) != 3:
				continue
			env_name, count = columns[1], int(columns[2])
			bucket = per_env.setdefault(env_name, {"niche_count": 0, "gladiator_count": 0})
			bucket["niche_count"] += 1
			bucket["gladiator_count"] += count
	return per_env


def _run_single_seed(pop_size: int, episodes_per_phase: int, max_threads: int, seed: int, recognizer_overrides: dict) -> dict:
	"""Train one seed against SyntheticCartPoleEnv and report the resulting niche/gladiator
	distribution per true environment, plus timing and per-phase average score."""
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	agent = _build_agent(pop_size, max_threads, **recognizer_overrides)
	environments = [make_synthetic_env(**phase) for phase in PHASES]

	start = time.time()
	history = play(agent, environments, [episodes_per_phase] * len(PHASES))
	elapsed = time.time() - start

	agent.set_learning(False)
	niches_path = os.path.join(OUTPUT_DIR, f"niches_{agent.get_name()}_seed{seed}.txt")
	agent._skills["hall_of_fame"].save_niche_details(OUTPUT_DIR, f"{agent.get_name()}_seed{seed}")  # pylint: disable=protected-access

	late_half = history.groupby("env_id", sort=False).tail(episodes_per_phase // 2)

	return {
		"seed": seed,
		"wall_time_seconds": round(elapsed, 2),
		"per_env": _parse_niches_file(niches_path),
		"avg_score_by_env": history.groupby("env_id")["score"].mean().round(1).to_dict(),
		"late_avg_score_by_env": late_half.groupby("env_id")["score"].mean().round(1).to_dict(),
	}


def _aggregate_runs(pop_size: int, episodes_per_phase: int, runs: list) -> dict:
	"""Aggregate several _run_single_seed() results into a mean±std summary per environment."""
	env_names = sorted({env_name for run in runs for env_name in run["per_env"]})
	per_env = {}
	for env_name in env_names:
		niche_counts = [run["per_env"].get(env_name, EMPTY_BUCKET)["niche_count"] for run in runs]
		gladiator_counts = [run["per_env"].get(env_name, EMPTY_BUCKET)["gladiator_count"] for run in runs]
		avg_scores = [run["avg_score_by_env"][env_name] for run in runs if env_name in run["avg_score_by_env"]]
		late_scores = [run["late_avg_score_by_env"][env_name] for run in runs if env_name in run["late_avg_score_by_env"]]
		per_env[env_name] = {
			"niche_count_mean": round(float(np.mean(niche_counts)), 2),
			"niche_count_std": round(float(np.std(niche_counts)), 2),
			"gladiator_count_mean": round(float(np.mean(gladiator_counts)), 2),
			"gladiator_count_std": round(float(np.std(gladiator_counts)), 2),
			"avg_score_mean": round(float(np.mean(avg_scores)), 1) if avg_scores else None,
			"avg_score_std": round(float(np.std(avg_scores)), 1) if avg_scores else None,
			"late_avg_score_mean": round(float(np.mean(late_scores)), 1) if late_scores else None,
			"present_in_seeds": len(avg_scores),
		}
	return {
		"pop_size": pop_size,
		"episodes_per_phase": episodes_per_phase,
		"seeds": [run["seed"] for run in runs],
		"total_wall_time_seconds": round(sum(run["wall_time_seconds"] for run in runs), 2),
		"per_env": per_env,
		# Kept so a later comparison can run a real paired test on matching seeds instead of
		# eyeballing aggregate means — a single seed's outcome is noisy enough (see module
		# docstring) that the raw per-seed values matter for any statistical claim.
		"raw_runs": runs,
	}


def run_benchmark(pop_size: int, episodes_per_phase: int, max_threads: int, seeds: list, recognizer_overrides: dict = None) -> dict:
	"""Run every seed and aggregate into a mean±std summary — the unit of comparison this
	tool reports, since a single seed's outcome is not trustworthy on its own (see module
	docstring). Prints one progress line per seed as it finishes, since a large --seeds list
	can take hours end to end."""
	recognizer_overrides = recognizer_overrides or {}
	runs = []
	for index, seed in enumerate(seeds, start=1):
		run = _run_single_seed(pop_size, episodes_per_phase, max_threads, seed, recognizer_overrides)
		summary = ", ".join(f"{env_name}={bucket['gladiator_count']}g/{bucket['niche_count']}n" for env_name, bucket in run["per_env"].items())
		print(f"[seed {index}/{len(seeds)}] seed={seed} done in {run['wall_time_seconds']}s | {summary}", flush=True)
		runs.append(run)
	return _aggregate_runs(pop_size, episodes_per_phase, runs)


def _print_report(result: dict) -> None:
	"""Print a human-readable mean±std summary table."""
	num_seeds = len(result["seeds"])
	print(
		f"pop_size={result['pop_size']} episodes_per_phase={result['episodes_per_phase']} "
		f"seeds={result['seeds']} total_wall_time={result['total_wall_time_seconds']}s"
	)
	header = f"{'Env':<20} | {'Niches':>12} | {'Gladiators':>12} | {'Avg Score':>14} | {'Late-Half Avg':>14} | {'Seen in':>7}"
	print(header)
	for env_name, bucket in result["per_env"].items():
		niches = f"{bucket['niche_count_mean']:.1f}±{bucket['niche_count_std']:.1f}"
		gladiators = f"{bucket['gladiator_count_mean']:.1f}±{bucket['gladiator_count_std']:.1f}"
		avg_score = f"{bucket['avg_score_mean']:.1f}" if bucket["avg_score_mean"] is not None else "n/a"
		late_avg = f"{bucket['late_avg_score_mean']:.1f}" if bucket["late_avg_score_mean"] is not None else "n/a"
		seen = f"{bucket['present_in_seeds']}/{num_seeds}"
		print(f"{env_name:<20} | {niches:>12} | {gladiators:>12} | {avg_score:>14} | {late_avg:>14} | {seen:>7}")


def _paired_deltas(baseline_runs: list, current_runs: list, env_name: str, metric: str) -> np.ndarray:
	"""Per-seed (current - baseline) deltas for one env/metric, matched by seed value —
	only seeds present in both run lists are used."""
	baseline_by_seed = {run["seed"]: run["per_env"].get(env_name, EMPTY_BUCKET)[metric] for run in baseline_runs}
	current_by_seed = {run["seed"]: run["per_env"].get(env_name, EMPTY_BUCKET)[metric] for run in current_runs}
	common_seeds = sorted(set(baseline_by_seed) & set(current_by_seed))
	return np.array([current_by_seed[seed] - baseline_by_seed[seed] for seed in common_seeds], dtype=float)


def _describe_paired_significance(deltas: np.ndarray) -> str:
	"""Summarize a paired-delta sample as 'mean±SEM (n=.., significant/not @ ~95%)'.

	Uses a normal approximation (|mean| > 1.96·SEM) rather than an exact Student's t test —
	no scipy dependency, and close enough once n gets into the range this tool asks for
	(the gap between the t and z critical values is under 5% by n≈30)."""
	num_seeds = len(deltas)
	if num_seeds < 2:
		return "not enough paired seeds"
	mean_delta = float(np.mean(deltas))
	sem = float(np.std(deltas, ddof=1)) / np.sqrt(num_seeds)
	if sem == 0:
		significance = "significant" if mean_delta != 0 else "no change"
	else:
		significance = "significant" if abs(mean_delta) > 1.96 * sem else "not significant"
	return f"{mean_delta:+.2f}±{sem:.2f} (n={num_seeds}, {significance} @ ~95%)"


def _print_comparison(baseline: dict, current: dict) -> None:
	"""Print a before/after diff of mean niche/gladiator counts per true environment, plus a
	paired significance check on matching seeds when both results retain raw per-seed data."""
	empty = {"niche_count_mean": 0.0, "gladiator_count_mean": 0.0}
	print(f"\n{'Env':<20} | {'Niches (before->after)':>24} | {'Gladiators (before->after)':>28}")
	for env_name in sorted(set(baseline["per_env"]) | set(current["per_env"])):
		before = baseline["per_env"].get(env_name, empty)
		after = current["per_env"].get(env_name, empty)
		niches = f"{before['niche_count_mean']:.1f} -> {after['niche_count_mean']:.1f}"
		gladiators = f"{before['gladiator_count_mean']:.1f} -> {after['gladiator_count_mean']:.1f}"
		print(f"{env_name:<20} | {niches:>24} | {gladiators:>28}")
	print(f"total_wall_time: {baseline['total_wall_time_seconds']}s -> {current['total_wall_time_seconds']}s")

	if "raw_runs" not in baseline or "raw_runs" not in current:
		print("\n(no paired significance check — one of the two results predates raw per-seed storage)")
		return

	print("\nPaired gladiator-count delta (current - baseline) per environment:")
	for env_name in sorted(set(baseline["per_env"]) | set(current["per_env"])):
		deltas = _paired_deltas(baseline["raw_runs"], current["raw_runs"], env_name, "gladiator_count")
		print(f"  {env_name:<20} | {_describe_paired_significance(deltas)}")


def _parse_seeds(raw: str) -> list:
	"""Parse a --seeds argument: comma-separated ints and/or 'a-b' ranges, e.g. '1-5,10,12-15'."""
	seeds = []
	for token in raw.split(","):
		if "-" in token:
			start, end = token.split("-")
			seeds.extend(range(int(start), int(end) + 1))
		else:
			seeds.append(int(token))
	return seeds


def main() -> None:
	"""CLI entry point."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--pop-size", type=int, default=20)
	parser.add_argument("--episodes-per-phase", type=int, default=200)
	parser.add_argument("--max-threads", type=int, default=1)
	parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS, help="Comma-separated seed list, e.g. 1,2,3 or 1-32.")
	parser.add_argument("--recognizer-threshold", type=float, default=0.90)
	parser.add_argument("--recognizer-ema-alpha", type=float, default=0.05)
	parser.add_argument("--save", type=str, default=None, help="Path to save this run's summary as JSON.")
	parser.add_argument("--compare", type=str, default=None, help="Path to a previously saved summary to diff against.")
	args = parser.parse_args()

	recognizer_overrides = {
		"recognizer_threshold": args.recognizer_threshold,
		"recognizer_ema_alpha": args.recognizer_ema_alpha,
	}
	result = run_benchmark(args.pop_size, args.episodes_per_phase, args.max_threads, args.seeds, recognizer_overrides)
	_print_report(result)

	if args.compare:
		with open(args.compare, encoding="utf-8") as baseline_file:
			baseline = json.load(baseline_file)
		_print_comparison(baseline, result)

	if args.save:
		os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
		with open(args.save, "w", encoding="utf-8") as save_file:
			json.dump(result, save_file, indent=2)
		print(f"\nSaved summary to {args.save}")


if __name__ == "__main__":
	main()
