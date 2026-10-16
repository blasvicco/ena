"""Tests for paper_config.py — the paper's configuration numbers come from the code that ran."""

# General imports
from types import SimpleNamespace

# Libs imports
import pytest

# App imports
from helpers import step
import cartpole_challenge
import cartpole_ena_ablation
from agents import SACDiscreteAgent
from tools.paper_config import _number, agent_settings, cartpole_rows, cheetah_rows, config_entries, environment_table, chirp_entries
from tools.paper_stats import MINUS


@pytest.mark.parametrize(
	"value, expected",
	[
		(2_500_000, "2,500,000"),
		(475.0, "475"),
		(0.05, "0.05"),
		(2.5, "2.5"),
		(3.0, "3"),
		(-20.0, f"{MINUS}20"),
		(3e-4, "$3\\times10^{-4}$"),
		(2.0633e-5, "$2.0633\\times10^{-5}$"),
	],
)
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
		assert (entries["cfg/cheetah/maxsteps"], entries["cfg/cheetah/maxeval"]) == ("1,000", "200")
		assert entries["cfg/ena/detect/gammaexp"] == "8"
		assert entries["cfg/ena/bpr/mix"] == "0.95"
		assert (entries["cfg/ena/fuzzy/trendrecent"], entries["cfg/ena/fuzzy/trendbefore"], entries["cfg/ena/fuzzy/trendmin"]) == ("5", "45", "20")
		assert entries["cfg/ena/brain"] == "4 $\\times$ 2 ReLU" and entries["cfg/cheetahena/brain"] == "32 $\\times$ 16 tanh"
		assert entries["cfg/challenge/mlpnet"] == "64 $\\times$ 64 ReLU" and entries["cfg/challenge/pponet"] == "64 $\\times$ 64 tanh"
		assert entries["cfg/ppo/entropycoef"] == "0.01" and entries["cfg/ppo/anneallr"] == "no"
		assert entries["cfg/sacdiscrete/targetentropyratio"] == f"{cartpole_challenge.SAC_DISCRETE_TARGET_ENTROPY:g}"
		assert entries["cfg/cheetahena/olympicsinterval"] == "200,000"
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
	"""Three training worlds then the two never-trained ones, wind's head component and air density included."""
	with step("Act: build the rows."):
		rows = cheetah_rows()

	with step("Assert: five rows; High-Wind blows at -20 m/s through air and is unseen; Ice-Floor has no air."):
		assert len(rows) == 5
		assert rows[-1][0] == "High-Wind" and rows[-1][3] == f"{MINUS}20" and rows[-1][4] == "1.2" and rows[-1][5] == "no"
		assert rows[3][0] == "Ice-Floor" and rows[3][3:5] == ["0", "0"]


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


def test_agent_settings_read_the_class_defaults_and_the_study_overrides() -> None:
	"""Every keyword setting of the class, the study's override winning; keys without underscores, flags as yes/no."""
	with step("Act: SAC-Discrete's settings with one override."):
		entries = agent_settings("sacd", SACDiscreteAgent, {"target_entropy_ratio": 0.5})

	with step("Assert: the override, a class default, and no key with an underscore."):
		assert entries["cfg/sacd/targetentropyratio"] == "0.5"
		assert entries["cfg/sacd/batchsize"] == "64" and entries["cfg/sacd/learningstarts"] == "1,000"
		assert not any("_" in key for key in entries)
