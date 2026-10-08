#!/usr/bin/env python
"""Calibrate ENA's environment detection (dynamics_model) for a domain, from that domain's own transitions.

The detector's settings were tuned on CartPole. This measures, on the domain's training environments, with the
detector's own classes (LinearGaussianDynamics, PMCuSum): each environment's model is fitted on all calibration
policies but one (a training ENA pools many specialists' transitions), and the held-out policy's transitions test
it, so a new policy must not read as new physics. Per tolerance and verification window: false alarms, the
verification gap on the own environment and on the others against the novelty margin, and the detection delay.
Policies: uniform random actions and two sinusoidal gaits.

	python src/tools/detection_calibration.py --output-dir half_cheetah_probe_outputs/detection_calibration
"""

# General imports
import argparse
import itertools
import math
import os
import sys

# Libs imports
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
import half_cheetah
from agents.ena.tools.linear_dynamics import LinearGaussianDynamics
from agents.ena.tools.pm_cusum import PMCuSum

POLICIES = ("random", "gait_slow", "gait_fast")
TOLERANCES = (3.0, 5.0, 8.0)
VERIFY_WINDOWS = (50, 200, 1_000, 3_000)


def policy_action(policy: str, step: int, rng: np.random.Generator, action_dim: int) -> np.ndarray:
	"""An action of one of the calibration policies: uniform random, or a sinusoidal gait with per-joint phases."""
	if policy == "random":
		return rng.uniform(-1.0, 1.0, size=action_dim)
	frequency = {"gait_slow": 0.05, "gait_fast": 0.15}[policy]
	phases = np.linspace(0.0, math.pi, action_dim)
	return np.clip(np.sin(2 * math.pi * frequency * step + phases) + rng.normal(0.0, 0.1, size=action_dim), -1.0, 1.0)


def collect(world, policy: str, steps: int, seed: int) -> tuple:
	"""(features [state, action, 1], targets next_state - state) of one policy in one environment."""
	rng = np.random.default_rng(seed)
	state, _ = world.reset(seed=seed)
	features, targets, episode_step = [], [], 0
	for _ in range(steps):
		action = policy_action(policy, episode_step, rng, world.action_space.shape[0])
		next_state, _reward, terminated, truncated, _ = world.step(action)
		features.append(np.concatenate([state, action, [1.0]]))
		targets.append(next_state - state)
		state, episode_step = next_state, episode_step + 1
		if terminated or truncated:
			state, _ = world.reset()
			episode_step = 0
	return np.asarray(features), np.asarray(targets)


def alarms(model, features, targets, tolerance: float, threshold: float) -> list:
	"""Steps at which PM-CuSum raises an alarm on the model's residuals (restarting after each)."""
	cusum = PMCuSum(targets.shape[1], threshold=threshold)
	residuals = model.standardized_residuals(features, targets) / tolerance
	raised = []
	for index, residual in enumerate(residuals):
		if cusum.update(residual):
			raised.append(index)
			cusum.reset()
	return raised


def gap(model, features, targets, variance_scale: float, var_floor: float) -> float:
	"""Nats per step by which a model fitted on these transitions beats `model` on them (the identification test)."""
	fresh = LinearGaussianDynamics.fit(features, targets, var_floor=var_floor)
	return max(
		float(np.mean(fresh.log_likelihood(features, targets, variance_scale)) - np.mean(model.log_likelihood(features, targets, variance_scale))),
		0.0,
	)


def pooled_model(data: dict, environment: str, held_out: str, var_floor: float = 1e-6):
	"""The environment's model fitted on every policy but the held-out one, as a training ENA pools many specialists."""
	pairs = [data[(environment, policy)] for policy in POLICIES if policy != held_out]
	features, targets = np.vstack([pair[0] for pair in pairs]), np.vstack([pair[1] for pair in pairs])
	model = LinearGaussianDynamics(features.shape[1], targets.shape[1], var_floor=var_floor)
	model.add(features, targets)
	return model


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calibrate(
	data: dict, tolerances=TOLERANCES, windows=VERIFY_WINDOWS, held_out: str = POLICIES[-1], false_alarm_steps: float = 1e8, var_floor: float = 1e-6
) -> pd.DataFrame:
	"""One row per (tolerance, verification window, environment's model), on the held-out policy's transitions:

	- false_alarms_per_100k: PM-CuSum alarms on the model's own environment (each one costs a verification);
	- own_gap: median verification gap (nats/step) on windows of the model's own environment; at most the novelty
	  margin means an alarm there is absorbed instead of confirmed;
	- other_gap_min: the smallest median gap over the other environments; above the margin means a real change is
	  confirmed and a new niche created;
	- delay_mean: mean steps until the alarm on the other environments.

	data: {(environment, policy): (features, targets)}.
	"""
	environments = list(dict.fromkeys(name for name, _policy in data))
	threshold = math.log(false_alarm_steps)
	rows = []
	for name in environments:
		model = pooled_model(data, name, held_out, var_floor)
		own_features, own_targets = data[(name, held_out)]
		for tolerance in tolerances:
			own_alarms = len(alarms(model, own_features, own_targets, tolerance, threshold))
			delays = []
			for other in environments:
				if other != name:
					raised = alarms(model, data[(other, held_out)][0][:5_000], data[(other, held_out)][1][:5_000], tolerance, threshold)
					delays.append(raised[0] if raised else math.inf)
			for window in windows:
				starts = range(0, len(own_features) - window + 1, max(window, 1_000))

				def median_gap(features, targets, starts=starts, window=window, tolerance=tolerance, model=model):
					return float(
						np.median([gap(model, features[s : s + window], targets[s : s + window], tolerance**2, var_floor) for s in starts])
					)

				rows.append(
					{
						"model": name,
						"tolerance": tolerance,
						"verify_window": window,
						"false_alarms_per_100k": 1e5 * own_alarms / len(own_features),
						"own_gap": median_gap(own_features, own_targets),
						"other_gap_min": min(median_gap(*data[(other, held_out)]) for other in environments if other != name),
						"delay_mean": float(np.mean([delay for delay in delays if math.isfinite(delay)]))
						if any(map(math.isfinite, delays))
						else math.inf,
					}
				)
	return pd.DataFrame(rows)


def main() -> None:
	"""Collect HalfCheetah transitions under every calibration policy and print the calibration table."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--steps", type=int, default=50_000, help="Transitions per (environment, policy).")
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()
	worlds = {world.unwrapped.name: world for world in half_cheetah.make_training_worlds()}
	data = {
		(name, policy): collect(worlds[name], policy, args.steps, seed) for seed, (name, policy) in enumerate(itertools.product(worlds, POLICIES))
	}
	table = calibrate(data)
	os.makedirs(args.output_dir, exist_ok=True)
	table.to_csv(os.path.join(args.output_dir, "calibration.csv"), index=False)
	print(table.to_string(index=False))


if __name__ == "__main__":
	main()
