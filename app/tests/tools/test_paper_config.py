"""Tests for paper_config.py — the paper's configuration numbers come from the code that ran."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
import cartpole_challenge
import cartpole_ena_ablation
from tools.paper_config import _number, cartpole_rows, cheetah_rows, config_entries, environment_table, chirp_entries


@pytest.mark.parametrize("value, expected", [(2_500_000, "2,500,000"), (475.0, "475"), (0.05, "0.05"), (2.5, "2.5"), (3.0, "3")])
def test_number_prints_integers_with_separators_and_decimals_short(value, expected) -> None:
	"""Settings read the way the paper writes them."""
	with step("Act/Assert: formatted value."):
		assert _number(value) == expected


def test_config_entries_read_the_settings_the_studies_use() -> None:
	"""Spot checks against the study code itself and the documented detector/selection settings."""
	with step("Act: collect every entry."):
		entries = config_entries()

	with step("Assert: protocol constants match the scripts; ENA's settings are those of the built agent."):
		assert entries["cfg/challenge/budget"] == f"{cartpole_challenge.FIXED_BUDGET_PER_ENV:,}"
		assert entries["cfg/cartpole/train"] == f"{cartpole_ena_ablation.TRAINING_EPISODES_ENA[0]:,}"
		assert entries["cfg/cartpole/maxsteps"] == "500"
		assert entries["cfg/ena/detect/gammaexp"] == "8"
		assert entries["cfg/ena/bpr/mix"] == "0.95"
		assert (entries["cfg/ena/fuzzy/trendrecent"], entries["cfg/ena/fuzzy/trendbefore"], entries["cfg/ena/fuzzy/trendmin"]) == ("5", "45", "20")
		assert all(value for value in entries.values())


def test_cartpole_rows_list_training_worlds_as_seen_and_test_only_worlds_as_unseen() -> None:
	"""Physics come from the environments themselves; gravity is printed as a magnitude."""
	with step("Arrange: one training and one test-only world."):

		def world(name, gravity, masspole, length):
			return SimpleNamespace(unwrapped=SimpleNamespace(name=name, gravity=gravity, masspole=masspole, length=length))

		rows = cartpole_rows([world("Mars-Light", 3.7, 0.05, 0.1)], [world("Brown-dwarf", -50.0, 0.1, 0.5)])

	with step("Assert: the environments' names, |g|, seen flags."):
		assert rows == [["Mars-Light", "3.7", "0.05", "0.1", "yes"], ["Brown-dwarf", "50", "0.1", "0.5", "no"]]


def test_cheetah_rows_come_from_the_study_worlds() -> None:
	"""Three training worlds then the two never-trained ones, wind's head component included."""
	with step("Act: build the rows."):
		rows = cheetah_rows()

	with step("Assert: five rows; High-Wind blows at -20 m/s and is unseen."):
		assert len(rows) == 5
		assert rows[-1][0] == "High-Wind" and rows[-1][3] == "-20" and rows[-1][4] == "no"


def test_environment_table_has_a_header_one_row_per_environment_and_the_label() -> None:
	"""A LaTeX table the setup section inputs."""
	with step("Act: render two rows."):
		table = environment_table([["Mars", "3.7"], ["Earth", "9.8"]], ["Environment", "$g$"], "Envs", "tab:envs")

	with step("Assert: header, both rows, label."):
		assert "\\textbf{Environment} & \\textbf{$g$} \\\\" in table
		assert "Mars & 3.7 \\\\" in table and "Earth & 9.8 \\\\" in table
		assert "\\label{tab:envs}" in table


def test_chirp_entries_read_the_halfcheetah_detector_and_benchmark() -> None:
	"""The chirp detector's settings as the HalfCheetah study builds it, and the benchmark's protocol."""
	with step("Act: the chirp entries."):
		entries = chirp_entries()

	with step("Assert: the chosen configuration and the benchmark constants."):
		assert entries["cfg/cheetah/chirp/steps"] == "64"
		assert entries["cfg/cheetah/chirp/excitation"] == "wavelet train" and entries["cfg/cheetah/chirp/excitationwavelet"] == "Mexican hat"
		assert entries["cfg/cheetah/chirp/transform"] == "wavelet packet" and entries["cfg/cheetah/chirp/featurewavelet"] == "Haar"
		assert (entries["cfg/cheetah/chirp/level"], entries["cfg/cheetah/chirp/bands"]) == ("3", "8")
		assert (entries["cfg/cheetah/chirp/window"], entries["cfg/cheetah/chirp/minchirps"], entries["cfg/cheetah/chirp/margin"]) == ("4", "8", "3")
		assert entries["cfg/cheetah/chirp/share"] == "6.4"
		assert entries["cfg/detbench/episodes"] == "40" and entries["cfg/detbench/block"] == "4" and entries["cfg/detbench/window"] == "5"
		assert entries["cfg/detbench/purity"] == "90"
		assert (entries["cfg/detbench/seeds"], entries["cfg/detbench/selectionseeds"], entries["cfg/detbench/confirmseeds"]) == ("30", "5", "30")
		assert (entries["cfg/detbench/seedfirst"], entries["cfg/detbench/seedlast"]) == ("5", "34")
		assert (entries["cfg/detbench/confirmfirst"], entries["cfg/detbench/confirmlast"]) == ("35", "64")
