"""Tests for cartpole_ena_fuzzylogic.py — ENA-Combined with the Mamdani fuzzy Plasticity driver."""

# App imports
from helpers import step
import cartpole_ena_fuzzylogic
from cartpole_ena_fuzzylogic import FUZZY_VARIANTS
from cartpole_ena_performance import PERFORMANCE_VARIANTS, parse_args


def test_the_only_difference_from_combined_is_the_fuzzy_plasticity_driver() -> None:
	"""One row: ENA-Combined's exact configuration with plasticity_algorithm swapped to fuzzylogic."""
	with step("Assert: one row; equal to ENA-Combined except name and plasticity driver."):
		combined = next(row for row in PERFORMANCE_VARIANTS if row["name"] == "ENA-Combined")
		assert len(FUZZY_VARIANTS) == 1
		fuzzy = FUZZY_VARIANTS[0]
		assert fuzzy["plasticity_algorithm"] == "fuzzylogic"
		assert {key: value for key, value in fuzzy.items() if key not in ("name", "plasticity_algorithm")} == {
			key: value for key, value in combined.items() if key not in ("name", "plasticity_algorithm")
		}


def test_defaults_are_50_replicates_without_random_into_their_own_directory() -> None:
	"""Same replicate count as the performance study it's compared against; separate output directory."""
	with step("Act: parse the study's defaults."):
		args = parse_args(
			[],
			variant_table=FUZZY_VARIANTS,
			defaults={
				"output_dir": cartpole_ena_fuzzylogic.OUTPUT_DIR,
				"num_experiments": cartpole_ena_fuzzylogic.NUM_EXPERIMENTS,
				"include_random": False,
			},
		)

	with step("Assert: 50 replicates, no Random, the real-fuzzy output directory."):
		assert args.variants == ["ENA-Combined-FuzzyPlasticity"]
		assert args.num_experiments == 50
		assert args.include_random is False
		assert args.output_dir.endswith("cartpole_ena_performance_real_fuzzylogic_outputs")
