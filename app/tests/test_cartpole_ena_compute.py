"""Tests for cartpole_ena_compute.py's plasticity-cost matrix."""

# App imports
from helpers import step
import cartpole_ena_compute
from cartpole_ena_compute import COMPUTE_VARIANTS
from cartpole_ena_performance import PERFORMANCE_VARIANTS, parse_args


def test_rows_vary_only_the_plasticity_driver_over_zero_linear_ramp_full() -> None:
	"""The full agent's Hall of Fame and env detection are held fixed; only plasticity changes."""
	with step("Assert: three rows, the three plasticity drivers, everything else identical to ENA-Combined."):
		combined = next(row for row in PERFORMANCE_VARIANTS if row["name"] == "ENA-Combined")
		assert sorted(row["plasticity_algorithm"] for row in COMPUTE_VARIANTS) == ["full", "linear_ramp", "zero"]
		for row in COMPUTE_VARIANTS:
			assert row["hall_of_fame_driver"] == combined["hall_of_fame_driver"]
			assert row["recognizer_strategy"] == combined["recognizer_strategy"]


def test_linear_ramp_row_matches_the_performance_studys_combined_row() -> None:
	"""Its cost distribution is taken from the performance study's 50 ENA-Combined replicates, so the configs must match."""
	with step("Assert: identical skill configuration."):
		linear_ramp = next(row for row in COMPUTE_VARIANTS if row["plasticity_algorithm"] == "linear_ramp")
		combined = next(row for row in PERFORMANCE_VARIANTS if row["name"] == "ENA-Combined")
		assert {key: value for key, value in linear_ramp.items() if key != "name"} == {key: value for key, value in combined.items() if key != "name"}


def test_defaults_are_one_replicate_without_random_into_its_own_directory() -> None:
	"""The costs of zero and full are deterministic, so one replicate; separate from the performance study's outputs."""
	with step("Act: parse the compute study's defaults."):
		args = parse_args(
			[],
			variant_table=COMPUTE_VARIANTS,
			defaults={
				"output_dir": cartpole_ena_compute.OUTPUT_DIR,
				"num_experiments": cartpole_ena_compute.NUM_EXPERIMENTS,
				"include_random": False,
			},
		)

	with step("Assert: all three rows, one replicate, no Random, compute output dir."):
		assert args.variants == [row["name"] for row in COMPUTE_VARIANTS]
		assert args.num_experiments == 1
		assert args.include_random is False
		assert args.output_dir.endswith("cartpole_ena_compute_outputs")
