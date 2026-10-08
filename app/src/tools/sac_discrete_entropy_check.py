#!/usr/bin/env python
"""Why the CartPole comparison's SAC-Discrete uses a target entropy of 0.5 log|A|: measure the entropy temperature per ratio.

SAC-Discrete (Christodoulou, 2019) tunes its temperature alpha toward a target policy entropy of
ratio * log|A|; the paper uses ratio 0.98. With CartPole's two actions that is nearly the maximum
entropy, which the policy cannot hold while learning, so alpha keeps growing. This trains the
challenge's own SAC-Discrete (same network, same hyperparameters, only the ratio changed) on the
first training environment for a fixed number of steps, per ratio and seed, and logs alpha and the
critics' value on states from the agent's own replay buffer. The paper's sentence on the choice is
generated from the summary (paper_results.py).

	python src/tools/sac_discrete_entropy_check.py                      # 0.98, 0.89, 0.5 x 3 seeds, 150k steps
	python src/tools/sac_discrete_entropy_check.py --steps 2000 --seeds 1 --output-dir /tmp/sacd   # smoke test
"""

# General imports
import argparse
import concurrent.futures
import os
import sys

# Libs imports
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from agents import SACDiscreteAgent
from cartpole_challenge import SAC_DISCRETE_TARGET_ENTROPY, mlp_brain
from cartpole_ena_ablation import env_seed_for, make_env, make_training_worlds, play, seed_experiment

PAPER_RATIO = 0.98  # Christodoulou (2019)
CLEANRL_RATIO = 0.89  # CleanRL's choice
RATIOS = (PAPER_RATIO, CLEANRL_RATIO, SAC_DISCRETE_TARGET_ENTROPY)
STEPS = 150_000
SEEDS = 3
LOG_EVERY = 2_000
PROBE_STATES = 256
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sac_discrete_entropy_outputs")


def probe_value(agent: SACDiscreteAgent) -> float:
	"""Mean over the first PROBE_STATES buffered states of max_a min(Q1, Q2): how large the critics' values have grown."""
	states = agent.buffer.states[: min(agent.buffer.size, PROBE_STATES)]
	if states.shape[0] == 0:
		return float("nan")
	values = np.minimum(agent.critic_1(states).numpy(), agent.critic_2(states).numpy())
	return float(values.max(axis=1).mean())


def run_one(ratio: float, seed: int, steps: int = STEPS, log_every: int = LOG_EVERY) -> list:
	"""Train the challenge's SAC-Discrete with one target-entropy ratio; return its (env_steps, alpha, value) trace."""
	seed_experiment(seed)
	agent = SACDiscreteAgent(brain=mlp_brain, make_env=make_env, target_entropy_ratio=ratio)
	agent.set_name(f"SAC-Discrete-{ratio}")
	trace = []

	def record(progress: int, _total: int) -> None:
		if not trace or progress - trace[-1]["env_steps"] >= log_every:
			trace.append(
				{"ratio": ratio, "seed": seed, "env_steps": progress, "alpha": float(np.exp(agent.log_alpha.numpy())), "value": probe_value(agent)}
			)

	play(agent, make_training_worlds()[:1], [None], steps_per_env=[steps], on_episode=record, env_seed=env_seed_for(seed, "training"))
	return trace


def summarize(trace: pd.DataFrame) -> pd.DataFrame:
	"""Per ratio, across seeds: median alpha at the first log and its median peak, median peak critic value, seeds."""
	per_seed = (
		trace.sort_values("env_steps")
		.groupby(["ratio", "seed"])
		.agg(
			alpha_start=("alpha", "first"),
			alpha_max=("alpha", "max"),
			alpha_end=("alpha", "last"),
			value_max=("value", "max"),
			steps=("env_steps", "max"),
		)
	)
	summary = per_seed.groupby("ratio").agg(
		alpha_start=("alpha_start", "median"),
		alpha_max=("alpha_max", "median"),
		alpha_end=("alpha_end", "median"),
		value_max=("value_max", "median"),
		steps=("steps", "min"),
		seeds=("alpha_max", "size"),
	)
	return summary.reset_index().sort_values("ratio", ascending=False)


def main() -> None:
	"""Run every ratio x seed in parallel and write trace.csv and summary.csv."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
	parser.add_argument("--seeds", type=int, default=SEEDS, help="Seeds per ratio (replicate ids 0..seeds-1).")
	parser.add_argument("--steps", type=int, default=STEPS)
	parser.add_argument("--workers", type=int, default=3)
	parser.add_argument("--output-dir", default=OUTPUT_DIR)
	args = parser.parse_args()
	jobs = [(ratio, seed) for ratio in args.ratios for seed in range(args.seeds)]
	with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
		traces = list(executor.map(run_one, *zip(*jobs), [args.steps] * len(jobs)))
	trace = pd.DataFrame([row for rows in traces for row in rows])
	os.makedirs(args.output_dir, exist_ok=True)
	trace.to_csv(os.path.join(args.output_dir, "trace.csv"), index=False)
	summary = summarize(trace)
	summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
	print(summary.to_string(index=False))


if __name__ == "__main__":
	main()
