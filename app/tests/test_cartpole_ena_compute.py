"""Tests for cartpole_ena_compute.py — full search effort, against the ablation's runs of the other two levels."""

# General imports
import os

# App imports
from helpers import step
import cartpole_ena_compute
from cartpole_ena_compute import COMPUTE_VARIANTS, REFERENCES
from cartpole_ena_ablation import ABLATION_VARIANTS, parse_args


def _without(row, keys):
	return {key: value for key, value in row.items() if key not in keys}


def test_the_only_row_is_ena_combined_at_full_search_effort() -> None:
	"""One row: ENA-Combined with Plasticity fixed at full, everything else identical."""
	with step("Assert: a single row, full Plasticity, otherwise ENA-Combined."):
		combined = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
		assert len(COMPUTE_VARIANTS) == 1
		assert COMPUTE_VARIANTS[0]["plasticity_algorithm"] == "full"
		assert _without(COMPUTE_VARIANTS[0], ("name", "plasticity_algorithm")) == _without(combined, ("name", "plasticity_algorithm"))


def test_references_are_the_ablation_rows_of_the_same_configuration_at_zero_and_ramp_effort() -> None:
	"""Zero effort is ENA-HoF+EnvDetection and the ramp is ENA-Combined: same config as full apart from Plasticity."""
	with step("Assert: each reference exists in the ablation with that Plasticity driver and full's other settings."):
		rows = {row["name"]: row for row in ABLATION_VARIANTS}
		for plasticity, name in REFERENCES.items():
			assert rows[name]["plasticity_algorithm"] == plasticity
			assert _without(rows[name], ("name", "plasticity_algorithm")) == _without(COMPUTE_VARIANTS[0], ("name", "plasticity_algorithm"))


def test_defaults_are_one_replicate_without_random_into_its_own_directory() -> None:
	"""One replicate (the author's decision), no Random, its own output directory."""
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

	with step("Assert: the full row, one replicate, no Random, compute output dir."):
		assert args.variants == ["ENA-Combined-PlasticityFull"]
		assert args.num_experiments == 1
		assert args.include_random is False
		assert args.output_dir.endswith(os.path.join("outputs", "cartpole_ena_compute_outputs"))
