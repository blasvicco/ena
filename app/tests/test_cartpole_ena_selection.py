"""Tests for cartpole_ena_selection.py — ENA-Combined with each simpler test-time selection driver."""

# App imports
from helpers import step
import cartpole_ena_selection
from cartpole_ena_selection import SELECTION_VARIANTS
from cartpole_ena_ablation import ABLATION_VARIANTS, parse_args


def test_each_row_differs_from_combined_only_in_the_selection_driver() -> None:
	"""NicheMatch and BestTrust rows: ENA-Combined's exact configuration plus a selection_driver."""
	with step("Arrange: ENA-Combined's configuration, without its name."):
		combined = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
		shared = {key: value for key, value in combined.items() if key != "name"}

	with step("Assert: the two drivers, each row equal to ENA-Combined apart from name and driver."):
		assert {row["name"]: row["selection_driver"] for row in SELECTION_VARIANTS} == {
			"ENA-Combined-NicheMatch": "niche_match",
			"ENA-Combined-BestTrust": "best_trust",
		}
		for row in SELECTION_VARIANTS:
			assert {key: value for key, value in row.items() if key not in ("name", "selection_driver")} == shared


def test_the_bpr_reference_is_not_rerun() -> None:
	"""BPR is the ablation's own ENA-Combined (same replicate, same run), so no row repeats it."""
	with step("Assert: no bpr row."):
		assert all(row["selection_driver"] != "bpr" for row in SELECTION_VARIANTS)


def test_defaults_are_30_replicates_without_random_into_their_own_directory() -> None:
	"""Same replicate count as the ablation it's compared against; separate output directory."""
	with step("Act: parse the study's defaults."):
		args = parse_args(
			[],
			variant_table=SELECTION_VARIANTS,
			defaults={
				"output_dir": cartpole_ena_selection.OUTPUT_DIR,
				"num_experiments": cartpole_ena_selection.NUM_EXPERIMENTS,
				"include_random": False,
			},
		)

	with step("Assert: 30 replicates, no Random, both rows, the study's own output directory."):
		assert args.variants == ["ENA-Combined-NicheMatch", "ENA-Combined-BestTrust"]
		assert args.num_experiments == 30
		assert args.include_random is False
		assert args.output_dir.endswith("cartpole_ena_selection_outputs")
