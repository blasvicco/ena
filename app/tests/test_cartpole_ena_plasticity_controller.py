"""Tests for cartpole_ena_plasticity_controller.py — ENA-Combined with each alternative Plasticity controller."""

# App imports
from helpers import step
import cartpole_ena_plasticity_controller
from cartpole_ena_plasticity_controller import CONTROLLER_VARIANTS
from cartpole_ena_ablation import ABLATION_VARIANTS, parse_args


def test_each_row_differs_from_combined_only_in_the_plasticity_driver() -> None:
	"""Quadratic and Mamdani fuzzy rows: ENA-Combined's exact configuration with plasticity_algorithm swapped."""
	with step("Arrange: ENA-Combined's configuration, without name and Plasticity driver."):
		combined = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
		shared = {key: value for key, value in combined.items() if key not in ("name", "plasticity_algorithm")}

	with step("Assert: the two alternatives, each equal to ENA-Combined apart from the driver."):
		assert {row["name"]: row["plasticity_algorithm"] for row in CONTROLLER_VARIANTS} == {
			"ENA-Combined-QuadraticPlasticity": "quadratic",
			"ENA-Combined-FuzzyPlasticity": "fuzzylogic",
		}
		for row in CONTROLLER_VARIANTS:
			assert {key: value for key, value in row.items() if key not in ("name", "plasticity_algorithm")} == shared


def test_the_linear_ramp_reference_is_not_rerun() -> None:
	"""linear_ramp is the ablation's own ENA-Combined (same replicate, same run), so no row repeats it."""
	with step("Assert: no linear_ramp row."):
		assert all(row["plasticity_algorithm"] != "linear_ramp" for row in CONTROLLER_VARIANTS)


def test_defaults_are_30_replicates_without_random_into_their_own_directory() -> None:
	"""Same replicate count as the ablation it's compared against; separate output directory."""
	with step("Act: parse the study's defaults."):
		args = parse_args(
			[],
			variant_table=CONTROLLER_VARIANTS,
			defaults={
				"output_dir": cartpole_ena_plasticity_controller.OUTPUT_DIR,
				"num_experiments": cartpole_ena_plasticity_controller.NUM_EXPERIMENTS,
				"include_random": False,
			},
		)

	with step("Assert: 30 replicates, no Random, both rows, the study's own output directory."):
		assert args.variants == ["ENA-Combined-QuadraticPlasticity", "ENA-Combined-FuzzyPlasticity"]
		assert args.num_experiments == 30
		assert args.include_random is False
		assert args.output_dir.endswith("cartpole_ena_plasticity_controller_outputs")
