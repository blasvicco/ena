"""Tests for detection_ablation.py — the chirp driver's excitation x feature ablation and the baselines, on the benchmark."""

# Libs imports
import pandas as pd

# App imports
from helpers import step
from tools import detection_ablation
from tools.detection_ablation import pass_table, run_baselines, run_grid


def test_the_grid_scores_every_excitation_feature_and_seed(monkeypatch) -> None:
	"""One row per (excitation, features, seed) with the benchmark's score and verdict."""
	with step("Arrange: a 2 x 2 grid on CartPole, 2 seeds (fast)."):
		monkeypatch.setattr(detection_ablation, "EXCITATIONS", {"chirp": ("chirp", {}), "impulse": ("impulse", {})})
		monkeypatch.setattr(detection_ablation, "FEATURES", {"raw": {"kind": "raw"}, "dwt_haar": {"kind": "wavelet"}})

	with step("Act: run it on CartPole with 16-step chirps."):
		rows = run_grid("cartpole", seeds=[0, 1], workers=1, chirp_steps=16)

	with step("Assert: 8 rows with the score's fields."):
		assert len(rows) == 8
		assert set(rows.columns) >= {"excitation", "features", "seed", "passes", "changes", "inside", "niches", "min_purity", "max_delay"}


def test_the_pass_table_counts_seeds_passing_per_excitation_and_features() -> None:
	"""The paper's table: seeds passing (of all) per excitation (row) and feature type (column)."""
	with step("Arrange: rows for 2 excitations x 1 feature type x 2 seeds."):
		rows = pd.DataFrame({"excitation": ["a", "a", "b", "b"], "features": ["f"] * 4, "seed": [0, 1, 0, 1], "passes": [True, True, False, True]})

	with step("Act/Assert: 2 of 2 for a, 1 of 2 for b."):
		table = pass_table(rows)
		assert table.loc["a", "f"] == 2 and table.loc["b", "f"] == 1


def test_the_baselines_compare_detectors_and_behaviors(monkeypatch) -> None:
	"""Rows for each baseline (driver, policy family) and seed."""
	with step("Arrange: two baselines on CartPole."):
		monkeypatch.setattr(
			detection_ablation,
			"BASELINES",
			{"dyn drifting": ("dynamics_model", {}, "drifting"), "dyn fixed": ("dynamics_model", {}, "fixed")},
		)

	with step("Act: one seed."):
		rows = run_baselines("cartpole", seeds=[0], workers=1)

	with step("Assert: one row per baseline."):
		assert list(rows["baseline"]) == ["dyn drifting", "dyn fixed"]


def test_the_three_seed_sets_are_disjoint_and_ordered() -> None:
	"""Screening, ablation grid and confirmation never share a seed (no result is reported on seeds it was chosen on)."""
	with step("Act/Assert: disjoint consecutive ranges."):
		sets = [set(detection_ablation.SELECTION_SEEDS), set(detection_ablation.SEEDS), set(detection_ablation.CONFIRMATION_SEEDS)]
		assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
		assert max(sets[0]) < min(sets[1]) and max(sets[1]) < min(sets[2])


def test_the_study_labels_are_the_halfcheetah_detectors_configuration() -> None:
	"""The ablation's labels for the HalfCheetah study's chirp settings: the Mexican-hat train and Haar packets."""
	with step("Act: labels of the driver's defaults."):
		from agents.ena.skills.env_detection.driver.chirp import DEFAULTS  # pylint: disable=import-outside-toplevel

		labels = detection_ablation.study_labels(DEFAULTS)

	with step("Assert: the chosen cell."):
		assert labels == ("Wavelet train (Mexican hat)", "Packets Haar")


def test_the_cartpole_check_runs_cartpoles_detector_on_its_own_benchmark() -> None:
	"""The benchmark's sanity row: CartPole's transition-model detector on CartPole, one row per seed."""
	with step("Act: two seeds."):
		rows = detection_ablation.run_cartpole_check(seeds=[35, 36], workers=1)

	with step("Assert: two scored rows."):
		assert list(rows["seed"]) == [35, 36] and "passes" in rows.columns
