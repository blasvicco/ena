#!/usr/bin/env python
"""HalfCheetah design probes (docs/PLAN_half_cheetah.md, R3): one agent, a chosen sequence of environments, the study's own code.

Trains one agent (built exactly as half_cheetah.py builds it) on a sequence of HalfCheetah environments at a
step budget per environment, with the study's frozen evaluation checks, then tests it frozen on all five
environments. Writes, into --output-dir:
  - live_training.csv / live_evaluation.csv: per episode, and every evaluation check;
  - trace.csv: every --trace-every steps, the agent's internal health (SAC: temperature alpha and critic
    values; PPO: learning rate and policy std) and the process memory;
  - summary.json: per training environment, the return at the end of its phase (last check) and the best
    check; per test environment, the mean and std of the frozen test; and, for every training environment
    left behind, how far its test return fell below the end-of-phase return (forgetting, P2).

	python src/tools/half_cheetah_probe.py --agent PPO --worlds Earth-Standard --output-dir half_cheetah_probe_outputs/p1_ppo
	python src/tools/half_cheetah_probe.py --agent PPO --output-dir half_cheetah_probe_outputs/p2_ppo   # Mars -> Jupiter -> Earth
"""

# General imports
import argparse
import json
import os
import sys

# Libs imports
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
import half_cheetah
from cartpole_ena_ablation import configure_worker_threads, env_seed_for, play, seed_experiment
from tools.detection_replay import Recorder
from tools.early_stopping import StoppingRule

TRAINING_ORDER = ["Mars-Slippery", "Jupiter-Rough", "Earth-Standard"]


def worlds_by_name() -> dict:
	"""Every HalfCheetah environment of the study, by name (a fresh instance each call)."""
	return {world.unwrapped.name: world for world in half_cheetah.make_training_worlds() + half_cheetah.make_unseen_worlds()}


def agent_health(agent) -> dict:
	"""The internal quantities that diverged or froze in earlier studies, for the agent's type."""
	health = {}
	if hasattr(agent, "log_alpha") and hasattr(agent, "critic_1") and getattr(agent.buffer, "size", 0) >= 256:
		indices = np.random.default_rng(0).integers(0, agent.buffer.size, size=256)
		health["alpha"] = float(np.exp(agent.log_alpha.numpy()))
		health["q_mean"] = float(np.mean(agent.critic_1([agent.buffer.states[indices], agent.buffer.actions[indices]]).numpy()))
	elif hasattr(agent, "log_alpha"):
		health["alpha"] = float(np.exp(agent.log_alpha.numpy()))
	if hasattr(agent, "_scheduled_learning_rate"):
		health["learning_rate"] = float(agent._scheduled_learning_rate())  # pylint: disable=protected-access
	if getattr(agent, "log_std", None) is not None:
		health["policy_std"] = float(np.mean(np.exp(agent.log_std.numpy())))
	return health


def forgetting(end_of_phase: dict, test_means: dict) -> dict:
	"""For each training environment left behind: its test return and the drop (%) below its end-of-phase return."""
	result = {}
	for name, phase_return in end_of_phase.items():
		if name not in test_means or name == list(end_of_phase)[-1]:
			continue
		drop = 100.0 * (phase_return - test_means[name]) / abs(phase_return) if phase_return else float("nan")
		result[name] = {"end_of_phase": phase_return, "test": test_means[name], "drop_percent": drop}
	return result


def summarize(evaluations: pd.DataFrame, testing: pd.DataFrame, order: list) -> dict:
	"""End-of-phase and best check per training environment, frozen-test mean/std per environment, and forgetting."""
	end_of_phase = {name: float(evaluations[evaluations["env_id"] == name]["eval_mean"].iloc[-1]) for name in order}
	best = {name: float(evaluations[evaluations["env_id"] == name]["eval_mean"].max()) for name in order}
	test = testing.groupby("env_id", sort=False)["score"].agg(["mean", "std"])
	test_means = {name: float(row["mean"]) for name, row in test.iterrows()}
	return {
		"end_of_phase": end_of_phase,
		"best_check": best,
		"test": {name: {"mean": float(row["mean"]), "std": float(row["std"])} for name, row in test.iterrows()},
		"forgetting": forgetting(end_of_phase, test_means),
	}


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def probe(
	agent_name: str,
	order: list,
	steps_per_env: int,
	test_episodes: int,
	output_dir: str,
	replicate: int = 0,
	trace_every: int = 10_000,
	threads: int = None,
	ena_overrides: dict = None,
	record_detection: bool = False,
) -> dict:
	"""Train agent_name on the environments in order, test it frozen on all five, and write the probe's outputs."""
	os.makedirs(output_dir, exist_ok=True)
	seed_experiment(replicate)
	agent = half_cheetah.build_agent(agent_name, max_agent_threads=threads or os.cpu_count(), **(ena_overrides or {}))
	worlds = [worlds_by_name()[name] for name in order]
	eval_worlds = [worlds_by_name()[name] for name in order]
	rule = StoppingRule(
		check_every=half_cheetah.CHECK_EVERY,
		eval_episodes=half_cheetah.EVAL_EPISODES,
		solved_score=half_cheetah.SOLVED_SCORE,
		plateau_checks=half_cheetah.PLATEAU_CHECKS,
		plateau_min_gain=half_cheetah.PLATEAU_MIN_GAIN,
		max_steps=steps_per_env,
		stop_when_solved=False,
		stop_on_plateau=False,
		check_after_solve=True,
	)
	if hasattr(agent, "set_total_steps"):
		agent.set_total_steps(steps_per_env * len(order))
	recorder = Recorder(agent._skills["env_detection"]) if record_detection else None  # pylint: disable=protected-access
	trace, process = [], psutil.Process()

	def record(progress: int, _total: int) -> None:
		if not trace or progress - trace[-1]["steps"] >= trace_every:
			trace.append({"steps": progress, "rss_mb": process.memory_info().rss / 2**20, **agent_health(agent)})

	training = play(
		agent,
		worlds,
		[None] * len(worlds),
		live_log_path=os.path.join(output_dir, "live_training.csv"),
		on_episode=record,
		stopping=rule,
		eval_environments=eval_worlds,
		env_seed=env_seed_for(replicate, "training"),
	)
	evaluations = pd.DataFrame(training.attrs["evaluations"])
	evaluations.to_csv(os.path.join(output_dir, "live_evaluation.csv"), index=False)
	pd.DataFrame(trace).to_csv(os.path.join(output_dir, "trace.csv"), index=False)
	if recorder is not None:
		recorder.save(os.path.join(output_dir, "detection.npz"))

	agent.set_learning(False, output_dir=output_dir)
	if hasattr(agent, "load_gladiators"):
		agent.load_gladiators()
	seed_experiment(replicate)
	test_worlds = list(worlds_by_name().values())
	testing = play(
		agent,
		test_worlds,
		[test_episodes] * len(test_worlds),
		live_log_path=os.path.join(output_dir, "live_testing.csv"),
		env_seed=env_seed_for(replicate, "testing"),
	)
	summary = {
		"agent": agent_name,
		"order": order,
		"steps_per_env": steps_per_env,
		"ena_overrides": ena_overrides or {},
		**summarize(evaluations, testing, order),
	}
	with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as output:
		json.dump(summary, output, indent=2)
	return summary


def parse_overrides(pairs: list) -> dict:
	"""KEY=VALUE strings to a dict, values as int, float or text."""
	overrides = {}
	for pair in pairs:
		key, value = pair.split("=", 1)
		for cast in (int, float):
			try:
				value = cast(value)
				break
			except ValueError:
				continue
		overrides[key] = value
	return overrides


def main() -> None:
	"""Run one probe."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--agent", required=True, choices=["ENA-Combined", "PPO", "SAC"])
	parser.add_argument("--worlds", nargs="+", default=TRAINING_ORDER, help="Training environments, in order.")
	parser.add_argument("--steps-per-env", type=int, default=half_cheetah.FIXED_BUDGET_PER_ENV)
	parser.add_argument("--test-episodes", type=int, default=10)
	parser.add_argument("--trace-every", type=int, default=10_000)
	parser.add_argument("--threads", type=int, default=None, help="TensorFlow and ENA evaluation threads (default: every core).")
	parser.add_argument(
		"--ena", nargs="*", default=[], metavar="KEY=VALUE", help="ENAgent argument overrides for calibration, e.g. pop_size=50 max_eval_steps=100."
	)
	parser.add_argument("--record-detection", action="store_true", help="ENA: save the detector's training calls for detection_replay.py.")
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()
	configure_worker_threads(args.threads)
	summary = probe(
		args.agent,
		args.worlds,
		args.steps_per_env,
		args.test_episodes,
		args.output_dir,
		trace_every=args.trace_every,
		threads=args.threads,
		ena_overrides=parse_overrides(args.ena),
		record_detection=args.record_detection,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
