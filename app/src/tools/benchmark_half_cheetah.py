#!/usr/bin/env python
"""Measure each HalfCheetah challenge agent's throughput on this machine and estimate the full study's time and cost.

Builds every agent exactly as half_cheetah.py does, plays it through the shared runner's own episode loop
(one untimed warm-up episode for graph tracing and SAC's replay warm-up, then at least --train-steps timed
env-steps; ENA's count includes its candidate evaluations, as the budget does), times frozen evaluation
episodes, and records the process memory. From those rates and the study's protocol constants it
estimates one replicate's duration (agents run one after another inside a replicate) and the whole
study's wall-clock and machine cost (replicates run --workers at a time).

Run it ON THE TARGET MACHINE, with the thread setting the study will use, and with nothing else running:
the rates are only as good as the conditions they were measured under.

Usage:
	python src/tools/benchmark_half_cheetah.py --threads-per-worker 2 --workers 30 --price-per-hour 2.9 --save bench.json
	python src/tools/benchmark_half_cheetah.py --load bench.json --budget-per-env 2000000 --replicates 20 --workers 20
"""

# General imports
import argparse
import json
import math
import os
import sys
import time

# Libs imports
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
import half_cheetah
from cartpole_ena_ablation import _play_episode, configure_worker_threads, evaluate_agent

EPISODE_STEPS = 1000  # HalfCheetah-v5's time limit


def _consumed_steps(agent, played: int) -> int:
	"""Env-steps an agent has consumed: ENA's own counter (candidate evaluations included), else the played steps."""
	return agent.total_env_steps if hasattr(agent, "total_env_steps") else played


def measure_agent(name: str, train_steps: int, eval_episodes: int) -> dict:
	"""Training and inference rates (seconds per env-step) and process memory for one challenge agent."""
	agent = half_cheetah.build_agent(name, max_agent_threads=1)
	agent.set_learning(True)
	if hasattr(agent, "set_total_steps"):
		agent.set_total_steps(len(half_cheetah.TRAINING_EPISODES) * half_cheetah.FIXED_BUDGET_PER_ENV)
	env = half_cheetah.make_training_worlds()[-1]

	_score, played = _play_episode(agent, env)  # warm-up, untimed
	start_steps = _consumed_steps(agent, played)
	start = time.perf_counter()
	while _consumed_steps(agent, played) - start_steps < train_steps:
		_score, steps = _play_episode(agent, env)
		played += steps
	train_seconds = time.perf_counter() - start
	measured = _consumed_steps(agent, played) - start_steps

	start = time.perf_counter()
	_mean, eval_steps = evaluate_agent(agent, half_cheetah.make_training_worlds()[-1], eval_episodes)
	eval_seconds = time.perf_counter() - start

	return {
		"agent": name,
		"train_steps_measured": int(measured),
		"train_seconds_per_step": train_seconds / measured,
		"eval_seconds_per_step": eval_seconds / eval_steps,
		"rss_mb": psutil.Process().memory_info().rss / 1e6,
	}


# pylint: disable=too-many-arguments
def replicate_seconds(
	measurement: dict,
	*,
	budget_per_env: int,
	n_envs: int,
	test_episodes: list,
	episode_steps: int = EPISODE_STEPS,
) -> float:
	"""One agent's seconds in one replicate: the training budget and the frozen test."""
	training = n_envs * budget_per_env * measurement["train_seconds_per_step"]
	test = sum(test_episodes) * episode_steps * measurement["eval_seconds_per_step"]
	return training + test


def estimate_study(agent_seconds: dict, *, replicates: int, workers: int, threads_per_worker: int, price_per_hour: float) -> dict:
	"""Wall-clock and cost of the study: agents sequential within a replicate, replicates `workers` at a time."""
	replicate_hours = sum(agent_seconds.values()) / 3600
	waves = math.ceil(replicates / workers)
	wall_hours = waves * replicate_hours
	return {
		"replicate_hours": replicate_hours,
		"waves": waves,
		"wall_hours": wall_hours,
		"cost": wall_hours * price_per_hour,
		"vcpus_needed": workers * threads_per_worker,
		"core_hours": replicates * replicate_hours * threads_per_worker,
		"bottleneck_agent": max(agent_seconds, key=agent_seconds.get),
	}


def parse_args(argv: list = None) -> argparse.Namespace:
	"""CLI: what to measure (or load), and the study/machine to estimate for."""
	agents = [row["name"] for row in half_cheetah.CHALLENGE_AGENTS]
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--agents", nargs="+", default=agents, choices=agents)
	parser.add_argument("--train-steps", type=int, default=20_000, help="Timed training env-steps per agent (after one warm-up episode).")
	parser.add_argument("--eval-episodes", type=int, default=3, help="Timed frozen evaluation episodes per agent.")
	parser.add_argument("--threads-per-worker", type=int, default=1, help="TensorFlow threads, as the study will run (--threads-per-worker).")
	parser.add_argument("--workers", type=int, default=None, help="Parallel replicates (default: cores / threads-per-worker).")
	parser.add_argument("--replicates", type=int, default=half_cheetah.NUM_EXPERIMENTS)
	parser.add_argument("--budget-per-env", type=int, default=half_cheetah.FIXED_BUDGET_PER_ENV)
	parser.add_argument("--price-per-hour", type=float, default=0.0, help="Machine price (your provider's, USD/h).")
	parser.add_argument("--save", default=None, help="Write the measurements to this JSON file.")
	parser.add_argument("--load", default=None, help="Re-estimate from saved measurements instead of measuring.")
	args = parser.parse_args(argv)
	if args.workers is None:
		args.workers = max(1, (os.cpu_count() or 1) // args.threads_per_worker)
	return args


def main(argv: list = None) -> None:
	"""Measure (or load) and print the per-agent rates and the study estimate."""
	args = parse_args(argv)
	configure_worker_threads(args.threads_per_worker)
	if args.load:
		with open(args.load, encoding="utf-8") as saved:
			measurements = json.load(saved)
	else:
		measurements = []
		for name in args.agents:
			print(f"Measuring {name}...", flush=True)
			measurements.append(measure_agent(name, args.train_steps, args.eval_episodes))
		if args.save:
			with open(args.save, "w", encoding="utf-8") as saved:
				json.dump(measurements, saved, indent=2)

	protocol = {
		"budget_per_env": args.budget_per_env,
		"n_envs": len(half_cheetah.TRAINING_EPISODES),
		"test_episodes": half_cheetah.TESTING_EPISODES,
	}
	agent_seconds = {row["agent"]: replicate_seconds(row, **protocol) for row in measurements}
	print(f"\n{'Agent':14s} {'train steps/s':>14s} {'eval steps/s':>13s} {'hours/replicate':>16s}")
	for row in measurements:
		print(
			f"{row['agent']:14s} {1 / row['train_seconds_per_step']:14,.0f} {1 / row['eval_seconds_per_step']:13,.0f} "
			f"{agent_seconds[row['agent']] / 3600:16.2f}"
		)
	estimate = estimate_study(
		agent_seconds,
		replicates=args.replicates,
		workers=args.workers,
		threads_per_worker=args.threads_per_worker,
		price_per_hour=args.price_per_hour,
	)
	print(f"\nProtocol: {protocol['budget_per_env']:,} steps x {protocol['n_envs']} envs, " f"test {protocol['test_episodes']}")
	print(f"Memory per worker (all agents built): ~{max(row['rss_mb'] for row in measurements):,.0f} MB")
	print(f"One replicate: {estimate['replicate_hours']:.1f} h (bottleneck: {estimate['bottleneck_agent']})")
	print(
		f"{args.replicates} replicates, {args.workers} workers x {args.threads_per_worker} threads ({estimate['vcpus_needed']} vCPUs): "
		f"{estimate['waves']} wave(s), {estimate['wall_hours']:.1f} h wall-clock, {estimate['core_hours']:,.0f} core-hours"
	)
	if args.price_per_hour:
		print(f"Machine cost at ${args.price_per_hour:.2f}/h: ${estimate['cost']:,.0f}")


if __name__ == "__main__":
	main()
