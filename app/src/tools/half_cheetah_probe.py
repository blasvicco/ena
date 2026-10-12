#!/usr/bin/env python
"""HalfCheetah design probes (docs/PLAN_half_cheetah.md, R3): one agent, a chosen sequence of environments, the study's own code.

Trains one agent (built exactly as half_cheetah.py builds it) on a sequence of HalfCheetah environments at a
step budget per environment, as the study trains (no evaluation checks), then tests it frozen on all five
environments. Writes, into --output-dir:
  - live_training.csv: per episode;
  - trace.csv: every --trace-every steps, the agent's internal health (SAC: temperature alpha and critic
    values; PPO: learning rate and policy std) and the process memory;
  - body.csv (with --body-episodes): after the frozen test, extra frozen episodes per training environment recording
    the body at every step (forward position and speed, torso height and pitch, floor contacts), to tell running from
    a physics exploit (flipping, sliding on the back);
  - summary.json: per training environment, the return at the end of its phase (the study's end_of_phase: mean
    training return over its last 100k env-steps); per test environment, the mean and std of the frozen test; and, for every training environment
    left behind, how far its test return fell below the end-of-phase return (forgetting, P2).

	python src/tools/half_cheetah_probe.py --agent PPO --worlds Earth-Standard --output-dir half_cheetah_probe_outputs/p1_ppo
	python src/tools/half_cheetah_probe.py --agent PPO --output-dir half_cheetah_probe_outputs/p2_ppo   # Earth -> Heavy -> Mars
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
from tools.agent_health import agent_health
from tools.cheetah_results import end_of_phase

TRAINING_ORDER = ["Earth-Standard", "Heavy-Rough", "Mars-Slippery"]


def worlds_by_name() -> dict:
	"""Every HalfCheetah environment of the study, by name (a fresh instance each call)."""
	return {world.unwrapped.name: world for world in half_cheetah.make_training_worlds() + half_cheetah.make_unseen_worlds()}


def forgetting(phase_ends: dict, test_means: dict) -> dict:
	"""For each training environment left behind: its test return and the drop (%) below its end-of-phase return."""
	result = {}
	for name, phase_return in phase_ends.items():
		if name not in test_means or name == list(phase_ends)[-1]:
			continue
		drop = 100.0 * (phase_return - test_means[name]) / abs(phase_return) if phase_return else float("nan")
		result[name] = {"end_of_phase": phase_return, "test": test_means[name], "drop_percent": drop}
	return result


def _body_row(name: str, episode: int, data, floor: int, body: set) -> dict:
	"""One step of the body: forward position and speed, torso height and pitch (HalfCheetah's root joints), whether
	nothing touches the floor and whether the torso or head does."""
	touching = {
		contact.geom2 if contact.geom1 == floor else contact.geom1 for contact in data.contact[: data.ncon] if floor in (contact.geom1, contact.geom2)
	}
	return {
		"env_id": name,
		"episode": episode,
		"x": float(data.qpos[0]),
		"z": float(data.qpos[1]),
		"pitch": float(data.qpos[2]),
		"vx": float(data.qvel[0]),
		"airborne": not touching,
		"body_on_floor": bool(touching & body),
	}


def _body_episode(agent, env, episode: int, seed: int) -> list:
	"""One frozen episode through the agent's act(), one _body_row per step."""
	model = env.unwrapped.model
	floor, body = model.geom("floor").id, {model.geom("torso").id, model.geom("head").id}
	state, _ = env.reset(seed=seed)
	rows, done = [], False
	while not done:
		state, _reward, terminated, truncated, _action = agent.act(env, np.asarray(state, dtype=np.float32))
		done = terminated or truncated
		rows.append(_body_row(env.unwrapped.name, episode, env.unwrapped.data, floor, body))
	return rows


def body_rollouts(agent, world_names: list, episodes: int, seed: int) -> pd.DataFrame:
	"""Frozen episodes per environment through the agent's act() (learning off; for ENA, its current individual)."""
	rows = []
	for world_offset, name in enumerate(world_names):
		env = worlds_by_name()[name]
		for episode in range(episodes):
			rows += _body_episode(agent, env, episode, seed + 100 * world_offset + episode)
	return pd.DataFrame(rows)


def summarize_body(steps: pd.DataFrame) -> dict:
	"""Per environment: forward speed (mean, max), mean distance per episode, torso height (mean, min), and the shares of
	steps flipped (|pitch| > pi/2), airborne, and with the torso or head on the floor."""
	summary = {}
	for name, group in steps.groupby("env_id", sort=False):
		distance = group.groupby("episode")["x"].agg(lambda x: x.iloc[-1] - x.iloc[0])
		summary[name] = {
			"speed_mean": float(group["vx"].mean()),
			"speed_max": float(group["vx"].max()),
			"distance_per_episode": float(distance.mean()),
			"height_mean": float(group["z"].mean()),
			"height_min": float(group["z"].min()),
			"share_flipped": float((group["pitch"].abs() > np.pi / 2).mean()),
			"share_airborne": float(group["airborne"].mean()),
			"share_body_on_floor": float(group["body_on_floor"].mean()),
		}
	return summary


def summarize(training: pd.DataFrame, testing: pd.DataFrame, order: list) -> dict:
	"""End of phase per training environment (end_of_phase), frozen-test mean/std per environment, and forgetting."""
	end = {name: end_of_phase(training, name) for name in order}
	test = testing.groupby("env_id", sort=False)["score"].agg(["mean", "std"])
	test_means = {name: float(row["mean"]) for name, row in test.iterrows()}
	return {
		"end_of_phase": end,
		"test": {name: {"mean": float(row["mean"]), "std": float(row["std"])} for name, row in test.iterrows()},
		"forgetting": forgetting(end, test_means),
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
	test_worlds: list = None,
	body_episodes: int = 0,
) -> dict:
	"""Train agent_name on the environments in order, test it frozen on all five, and write the probe's outputs."""
	os.makedirs(output_dir, exist_ok=True)
	seed_experiment(replicate)
	agent = half_cheetah.build_agent(agent_name, max_agent_threads=threads or os.cpu_count(), **(ena_overrides or {}))
	worlds = [worlds_by_name()[name] for name in order]
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
		steps_per_env=[steps_per_env] * len(worlds),
		env_seed=env_seed_for(replicate, "training"),
	)
	pd.DataFrame(trace).to_csv(os.path.join(output_dir, "trace.csv"), index=False)
	if recorder is not None:
		recorder.save(os.path.join(output_dir, "detection.npz"))

	write_olympics(agent, output_dir)
	enter_test(agent, output_dir)
	seed_experiment(replicate)
	test_worlds = [worlds_by_name()[name] for name in test_worlds] if test_worlds else list(worlds_by_name().values())
	testing = play(
		agent,
		test_worlds,
		[test_episodes] * len(test_worlds),
		live_log_path=os.path.join(output_dir, "live_testing.csv"),
		env_seed=env_seed_for(replicate, "testing"),
	)
	body = {}
	if body_episodes:
		steps = body_rollouts(agent, order, body_episodes, env_seed_for(replicate, "testing"))
		steps.to_csv(os.path.join(output_dir, "body.csv"), index=False)
		body = summarize_body(steps)
	summary = {
		"agent": agent_name,
		"order": order,
		"steps_per_env": steps_per_env,
		"ena_overrides": ena_overrides or {},
		**summarize(training, testing, order),
		**({"body": body} if body else {}),
	}
	with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as output:
		json.dump(summary, output, indent=2)
	return summary


def enter_test(agent, output_dir: str) -> None:
	"""Freeze the agent as the studies do (cartpole_ena_ablation.run_replicate): re-introduce ENA's gladiators
	first, then enter test mode, so test-time selection is built on the re-introduced population."""
	if hasattr(agent, "load_gladiators"):
		agent.load_gladiators()
	agent.set_learning(False, output_dir=output_dir)


def write_olympics(agent, output_dir: str) -> None:
	"""olympics.csv: one row per competitor per Olympiad, with the mean and std of its Olympic returns."""
	rows = getattr(agent, "olympics_history", [])
	if not rows:
		return
	os.makedirs(output_dir, exist_ok=True)
	table = pd.DataFrame(rows)
	table["mean"] = table["returns"].apply(np.mean)
	table["std"] = table["returns"].apply(np.std)
	table.to_csv(os.path.join(output_dir, "olympics.csv"), index=False)


def _layer_sizes(value: str) -> tuple:
	"""'32x16' -> (32, 16); ValueError otherwise."""
	sizes = tuple(int(size) for size in value.split("x"))
	if len(sizes) < 2:
		raise ValueError(value)
	return sizes


def parse_overrides(pairs: list) -> dict:
	"""KEY=VALUE strings to a dict, values as int, float, layer sizes (32x16 -> (32, 16)) or text; dotted keys nest
	(recognizer_probe.features.wavelet=db4)."""
	overrides = {}
	for pair in pairs:
		key, value = pair.split("=", 1)
		for cast in (int, float, _layer_sizes):
			try:
				value = cast(value)
				break
			except ValueError:
				continue
		*parents, leaf = key.split(".")
		target = overrides
		for parent in parents:
			target = target.setdefault(parent, {})
		target[leaf] = value
	return overrides


def main() -> None:
	"""Run one probe."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--agent", required=True, choices=["ENA-Combined", "PPO", "SAC"])
	parser.add_argument("--worlds", nargs="+", default=TRAINING_ORDER, help="Training environments, in order.")
	parser.add_argument("--steps-per-env", type=int, default=half_cheetah.FIXED_BUDGET_PER_ENV)
	parser.add_argument("--test-episodes", type=int, default=10)
	parser.add_argument("--replicate", type=int, default=0, help="Seed index (the study's seeding of replicate i).")
	parser.add_argument("--trace-every", type=int, default=10_000)
	parser.add_argument("--threads", type=int, default=None, help="TensorFlow and ENA evaluation threads (default: every core).")
	parser.add_argument(
		"--ena", nargs="*", default=[], metavar="KEY=VALUE", help="ENAgent argument overrides for calibration, e.g. pop_size=50 max_eval_steps=100."
	)
	parser.add_argument("--test-worlds", nargs="+", default=None, help="Frozen-test environments, in order (default: all five).")
	parser.add_argument("--record-detection", action="store_true", help="ENA: save the detector's training calls for detection_replay.py.")
	parser.add_argument("--body-episodes", type=int, default=0, help="Extra frozen episodes per training environment recording the body.")
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()
	configure_worker_threads(args.threads)
	summary = probe(
		args.agent,
		args.worlds,
		args.steps_per_env,
		args.test_episodes,
		args.output_dir,
		replicate=args.replicate,
		trace_every=args.trace_every,
		threads=args.threads,
		ena_overrides=parse_overrides(args.ena),
		record_detection=args.record_detection,
		test_worlds=args.test_worlds,
		body_episodes=args.body_episodes,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
