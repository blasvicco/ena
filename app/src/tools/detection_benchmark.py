#!/usr/bin/env python
"""Benchmark an environment-detection driver on scripted inputs, without training an agent."""
# The detector sees exactly what ENA hands it: the raw state, the action taken and the next state, one transition
# at a time, with an evaluation at every episode end. This script plays scripted episodes in a sequence of
# environments (e.g. Earth -> Heavy -> Mars), changing the acting policy every `policy_block` episodes to mimic
# a learning population (gaits, random, near-still; the same policy family in every environment, so only the
# physics differ), records the calls in detection_replay's format, replays them through a fresh detector, and
# scores the result:
#
#   - every switch confirmed within `window` episodes (delays);
#   - no confirmed change inside an environment (inside);
#   - one niche per environment: each environment's majority niche covers most of its episodes (purity), and
#     different environments have different majority niches (distinct).
#
# A match_niche call follows every episode: stricter than ENA, which binds niches only at Hall-of-Fame inductions.
#
#         python src/tools/detection_benchmark.py --domain half_cheetah --strategy dynamics_model

# General imports
from typing import Callable
import argparse
from collections import Counter
import math
import os
import sys

# Libs imports
import gymnasium as gym
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from agents.ena.skills.env_detection.main import EnvironmentRecognizer
from tools.detection_replay import EVALUATE, MATCH, OBSERVE, CHIRP, START, parse_overrides, replay

PURITY_TARGET = 0.9  # share of an environment's episodes its majority niche must cover
SEED_RANGE = 2**31 - 1


def continuous_policy(rng: np.random.Generator, action_dim: int) -> Callable[[int, np.ndarray, np.random.Generator], np.ndarray]:
	"""A policy from the continuous family: near-still, uniform random, or a sinusoidal gait with random parameters."""
	kind = rng.choice(["still", "random", "gait", "gait"])
	if kind == "still":
		return lambda _step, _state, noise: np.clip(noise.normal(0.0, 0.05, size=action_dim), -1.0, 1.0)
	if kind == "random":
		return lambda _step, _state, noise: noise.uniform(-1.0, 1.0, size=action_dim)
	frequency, amplitude = rng.uniform(0.03, 0.2), rng.uniform(0.3, 1.0)
	phases = rng.uniform(0.0, 2.0 * math.pi, size=action_dim)
	return lambda step, _state, noise: np.clip(
		amplitude * np.sin(2.0 * math.pi * frequency * step + phases) + noise.normal(0.0, 0.1, size=action_dim), -1.0, 1.0
	)


def discrete_policy(rng: np.random.Generator, state_dim: int) -> Callable[[int, np.ndarray, np.random.Generator], int]:
	"""A policy from the discrete (CartPole) family: a perturbed linear balancing controller with some random actions, or random."""
	if rng.uniform() < 0.25:
		return lambda _step, _state, noise: int(noise.integers(2))
	gains = np.zeros(state_dim)
	gains[2:] = 1.0  # push toward the pole's fall (angle and angular velocity)
	weights, epsilon = gains + rng.normal(0.0, 0.3, size=state_dim), rng.uniform(0.0, 0.3)
	return lambda _step, state, noise: int(noise.integers(2)) if noise.uniform() < epsilon else int(np.dot(weights, state) > 0.0)


def fixed_policy(state_dim: int, action_dim: int, is_continuous: bool) -> Callable[[int, np.ndarray, np.random.Generator], int | np.ndarray]:
	"""One behavior for every episode of every world: a fixed sinusoidal gait (continuous) or a fixed balancing controller."""
	if is_continuous:
		phases = np.linspace(0.0, math.pi, action_dim)
		return lambda step, _state, noise: np.clip(
			0.8 * np.sin(2.0 * math.pi * 0.1 * step + phases) + noise.normal(0.0, 0.1, size=action_dim), -1.0, 1.0
		)
	gains = np.zeros(state_dim)
	gains[2:] = 1.0
	return lambda _step, state, noise: int(noise.integers(2)) if noise.uniform() < 0.1 else int(np.dot(gains, state) > 0.0)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def generate(
	worlds: list,
	episodes_per_world: int,
	policy_block: int,
	seed: int,
	max_steps: int,
	is_continuous: bool,
	chirp_actions: np.ndarray | None = None,
	policy_family: str = "drifting",
) -> tuple:
	"""Play the scripted episodes; return (recording in detection_replay's format, environment name per episode)."""
	# With chirp_actions, each episode ends with that fixed chirp played as ENAgent plays it (a fresh reset, the
	# environment's random stream restored afterwards) and recorded as a CHIRP event before the EVALUATE.
	# policy_family 'drifting' draws a new policy every policy_block episodes; 'fixed' plays one behavior throughout.
	if policy_family not in ("drifting", "fixed"):
		raise ValueError(f"Unknown policy family: {policy_family!r}")
	rng = np.random.default_rng(seed)
	kinds, learn, states, actions, next_states, env_ids, chirps = [], [], [], [], [], [], []
	for world in worlds:
		state_dim = world.observation_space.shape[0]
		policy = None
		for episode in range(episodes_per_world):
			if policy_family == "fixed":
				policy = fixed_policy(state_dim, world.action_space.shape[0] if is_continuous else 2, is_continuous)
			elif episode % policy_block == 0:
				policy = continuous_policy(rng, world.action_space.shape[0]) if is_continuous else discrete_policy(rng, state_dim)
			noise = np.random.default_rng(rng.integers(SEED_RANGE))
			state, _ = world.reset(seed=int(rng.integers(SEED_RANGE)))
			kinds.append(START)
			learn.append(False)
			for step in range(max_steps):
				action = policy(step, state, noise)
				next_state, _reward, terminated, truncated, _ = world.step(action)
				kinds.append(OBSERVE)
				learn.append(False)
				states.append(np.asarray(state, dtype=np.float32))
				actions.append(np.asarray(action, dtype=np.float32) if is_continuous else action)
				next_states.append(np.asarray(next_state, dtype=np.float32))
				state = next_state
				if terminated or truncated:
					break
			if chirp_actions is not None:
				chirps.append(play_chirp(world, chirp_actions))
				kinds.append(CHIRP)
				learn.append(False)
			kinds += [EVALUATE, MATCH]
			learn += [True, False]
			env_ids.append(world.unwrapped.name)
	recording = {
		"kinds": np.asarray(kinds, dtype=np.int8),
		"learn": np.asarray(learn, dtype=bool),
		"outcomes": np.zeros(len(kinds), dtype=np.float32),
		"states": np.asarray(states),
		"actions": np.asarray(actions),
		"next_states": np.asarray(next_states),
		"chirps": np.asarray(chirps),
	}
	return recording, env_ids


def play_chirp(world: gym.Env, chirp_actions: np.ndarray | None) -> np.ndarray:
	"""Return the chirp's observations from a fresh reset, as ENAgent plays it, restoring the world's random stream."""
	# The chirp's observations from a fresh reset (start first, padded if it ends early), as ENAgent.__run_chirp
	# plays it; the world's random stream is restored afterwards.
	random_state = world.unwrapped.np_random.bit_generator.state
	observation, _ = world.reset()
	observations = [observation]
	for action in chirp_actions:
		observation, _reward, terminated, truncated, _ = world.step(action)
		observations.append(observation)
		if terminated or truncated:
			break
	world.unwrapped.np_random.bit_generator.state = random_state
	observations += [observations[-1]] * (len(chirp_actions) + 1 - len(observations))
	return np.asarray(observations, dtype=np.float32)


def score(replayed: pd.DataFrame, env_ids: list, window: int) -> dict:
	"""Switch delays, switches caught within `window` episodes, false changes, and niche purity per environment."""
	shifts = replayed["shift"].to_numpy()
	names = list(dict.fromkeys(env_ids))
	starts = [env_ids.index(name) for name in names]
	delays = []
	for start in starts[1:]:
		later = np.flatnonzero(shifts[start:])
		delays.append(int(later[0]) if later.size else -1)
	caught = sum(1 for delay in delays if 0 <= delay < window)
	purity, majority = {}, {}
	for name in names:
		niches = [niche for niche, env in zip(replayed["niche"], env_ids) if env == name and not pd.isna(niche)]
		if not niches:
			purity[name], majority[name] = 0.0, None
			continue
		niche, count = Counter(int(niche) for niche in niches).most_common(1)[0]
		purity[name], majority[name] = count / len(niches), niche
	return {
		"switches": len(starts) - 1,
		"delays": delays,
		"caught": caught,
		"changes": int(shifts.sum()),
		"inside": int(shifts.sum()) - caught,
		"niches": int(pd.Series(replayed["niche"]).dropna().nunique()),
		"purity": purity,
		"majority_niches": majority,
		"distinct": None not in majority.values() and len(set(majority.values())) == len(majority),
	}


def meets_target(result: dict) -> bool:
	"""Every switch within the window, no false change, and one niche per environment."""
	return result["caught"] == result["switches"] and result["inside"] == 0 and result["distinct"] and min(result["purity"].values()) >= PURITY_TARGET


# pylint: disable=import-outside-toplevel
def domain_case(domain: str) -> dict:
	"""The benchmark's worlds, schedule and the study's detector settings for a domain."""
	if domain == "cartpole":
		import cartpole_ena_ablation

		return {
			"worlds": cartpole_ena_ablation.make_training_worlds(),
			"episodes_per_world": 60,
			"policy_block": 5,
			"max_steps": 500,
			"is_continuous": False,
			"action_dim": 2,
			"window": 3,
			"settings": {},  # ENAgent's defaults are CartPole's
		}
	import half_cheetah

	return {
		"worlds": half_cheetah.make_training_worlds(),
		"episodes_per_world": 40,
		"policy_block": 4,
		"max_steps": 1_000,
		"is_continuous": True,
		"action_dim": 6,
		"window": 5,
		"settings": dict(half_cheetah.ENA_DETECTION),
	}


def run(domain: str, strategy: str = None, overrides: dict = None, seed: int = 0, policy_family: str = "drifting") -> dict:
	"""Generate the domain's scripted inputs, replay them through a fresh detector, and score them."""
	# The driver is `strategy`, else the study's (the settings' recognizer_strategy), else dynamics_model.
	case = domain_case(domain)
	settings = {**case["settings"], **(overrides or {})}
	strategy = strategy or settings.pop("recognizer_strategy", "dynamics_model")
	settings.pop("recognizer_strategy", None)
	chirp_actions = None
	if strategy == "chirp":
		state_dim = case["worlds"][0].observation_space.shape[0]
		chirp_actions = EnvironmentRecognizer(
			"chirp", state_dim, action_dim=case["action_dim"], is_continuous=case["is_continuous"], chirp=settings.get("recognizer_chirp")
		).chirp_actions
	recording, env_ids = generate(
		case["worlds"],
		case["episodes_per_world"],
		case["policy_block"],
		seed,
		case["max_steps"],
		case["is_continuous"],
		chirp_actions=chirp_actions,
		policy_family=policy_family,
	)
	replayed = replay(recording, recording["states"].shape[1], case["action_dim"], settings, strategy=strategy, is_continuous=case["is_continuous"])
	return score(replayed, env_ids, case["window"])


def main() -> None:
	"""Benchmark one driver and settings on one domain."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--domain", choices=["cartpole", "half_cheetah"], required=True)
	parser.add_argument("--strategy", default=None, help="Detector driver (default: the study's).")
	parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ENAgent recognizer_* overrides.")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	result = run(args.domain, args.strategy, parse_overrides(args.set), args.seed)
	print(result)
	print(f"Meets the target: {meets_target(result)}")


if __name__ == "__main__":
	main()
