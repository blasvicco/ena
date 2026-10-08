"""Tests for detection_calibration.py — the domain calibration of ENA's environment detection."""

# Libs imports
import numpy as np

# App imports
from helpers import step
from tools.detection_calibration import POLICIES, calibrate, policy_action


def _linear_system(matrix: np.ndarray, steps: int, seed: int) -> tuple:
	"""Transitions of next - state = matrix @ [state, action, 1] + small noise, as (features, targets)."""
	rng = np.random.default_rng(seed)
	features = np.column_stack([rng.normal(size=(steps, 2)), rng.uniform(-1, 1, size=(steps, 1)), np.ones(steps)])
	return features, features @ matrix.T + rng.normal(0.0, 0.01, size=(steps, 2))


def test_policy_actions_stay_in_the_action_range() -> None:
	"""Every calibration policy produces actions in [-1, 1]."""
	with step("Act/Assert: 100 steps of each policy."):
		rng = np.random.default_rng(0)
		for policy in POLICIES:
			assert all(np.all(np.abs(policy_action(policy, t, rng, 6)) <= 1.0) for t in range(100))


def test_calibration_separates_different_physics_and_accepts_the_same_physics() -> None:
	"""Two linear systems: own windows explained (gap ~0), the other system's windows not (large gap), no false alarms."""
	with step("Arrange: two distinct linear dynamics, every policy a different sample of the same system."):
		systems = {"A": np.array([[0.1, 0.0, 0.5, 0.0], [0.0, -0.1, 0.0, 0.2]]), "B": np.array([[0.1, 0.0, -0.5, 1.0], [0.0, 0.3, 0.4, -0.2]])}
		data = {
			(name, policy): _linear_system(matrix, 3_000, seed)
			for seed, (name, matrix, policy) in enumerate((n, m, p) for n, m in systems.items() for p in POLICIES)
		}

	with step("Act: calibrate at one tolerance and one window."):
		table = calibrate(data, tolerances=(3.0,), windows=(1_000,)).set_index("model")

	with step("Assert: own gap ~0 and no false alarms; the other system is far beyond a 2-nat margin and detected."):
		for name in systems:
			assert table.loc[name, "own_gap"] < 0.5
			assert table.loc[name, "false_alarms_per_100k"] == 0
			assert table.loc[name, "other_gap_min"] > 2.0
			assert np.isfinite(table.loc[name, "delay_mean"])
