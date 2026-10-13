"""Tests for half_cheetah_probe.py — the HalfCheetah design probes (forgetting, learning, health)."""

# General imports
import json

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
import half_cheetah
from tools import half_cheetah_probe
from tools.half_cheetah_probe import forgetting, parse_overrides, probe, summarize, summarize_body


def test_forgetting_reports_the_drop_below_the_end_of_phase_return_for_environments_left_behind() -> None:
	"""The last training environment is not 'left behind'; earlier ones are compared with their own phase's end."""
	with step("Act: Mars ended at 4000 and tests at 1000; Earth (last) is skipped."):
		result = forgetting({"Mars": 4000.0, "Earth": 5000.0}, {"Mars": 1000.0, "Earth": 5100.0})

	with step("Assert: a 75% drop on Mars, nothing for Earth."):
		assert result == {"Mars": {"end_of_phase": 4000.0, "test": 1000.0, "drop_percent": 75.0}}


def test_summarize_takes_the_training_return_at_the_end_of_each_phase_and_the_test_mean_per_environment() -> None:
	"""End of phase = the mean training return over the phase's last 100k env-steps, as the study's analysis takes it."""
	with step("Arrange: two training episodes per environment (the first outside the last 100k), a test of two environments."):
		training = pd.DataFrame(
			{"env_id": ["Mars", "Mars", "Earth", "Earth"], "env_steps": [100_000, 500_000, 600_000, 1_000_000], "score": [100.0, 300.0, 500.0, 400.0]}
		)
		testing = pd.DataFrame({"env_id": ["Mars", "Mars", "Earth", "Earth"], "score": [50.0, 70.0, 380.0, 420.0]})

	with step("Act: summarize."):
		summary = summarize(training, testing, ["Mars", "Earth"])

	with step("Assert: end of phase 300 / 400, test means 60 / 400, Mars drop 80%."):
		assert summary["end_of_phase"] == {"Mars": 300.0, "Earth": 400.0}
		assert summary["test"]["Mars"]["mean"] == 60.0
		assert summary["forgetting"]["Mars"]["drop_percent"] == pytest.approx(80.0)


def test_probe_runs_end_to_end_and_writes_every_output(tmp_path) -> None:
	"""A tiny real run: training to the step budget, frozen test on all five environments, trace and summary."""
	with step("Act: PPO, one environment, 2,000 steps, one test episode per environment."):
		summary = probe("PPO", ["Earth-Standard"], 2_000, 1, str(tmp_path), trace_every=500)

	with step("Assert: outputs exist; the test covers the five environments; the trace has PPO's learning rate."):
		for name in ("live_training.csv", "live_testing.csv", "trace.csv", "summary.json"):
			assert (tmp_path / name).exists(), name
		assert len(summary["test"]) == 5
		assert "learning_rate" in pd.read_csv(tmp_path / "trace.csv").columns
		assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["agent"] == "PPO"


def test_parse_overrides_casts_numbers_and_keeps_text() -> None:
	"""Calibration overrides from the command line."""
	with step("Act/Assert: int, float and text values."):
		assert parse_overrides(["pop_size=50", "mutation_noise=0.2", "selection_driver=bpr"]) == {
			"pop_size": 50,
			"mutation_noise": 0.2,
			"selection_driver": "bpr",
		}


def test_ena_overrides_reach_the_agent() -> None:
	"""build_agent passes calibration overrides to ENA and keeps every other study setting."""
	with step("Act: ENA with a smaller population and shallower evaluation."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1, pop_size=50, max_eval_steps=100)

	with step("Assert: overridden values, the rest from ENA_SETTINGS."):
		assert (agent.pop_size, agent.max_eval_steps) == (50, 100)
		assert agent._skills["env_detection"].chirp_actions is not None  # pylint: disable=protected-access


def test_ena_network_size_is_a_calibration_option() -> None:
	"""brain_hidden sets ENA's hidden units per layer (the study's is 32x16)."""
	with step("Act: ENA with 32-unit hidden layers."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1, brain_hidden=32, pop_size=4)

	with step("Assert: 32 units in each hidden layer."):
		model = agent.population[0]["model"]
		assert [layer.units for layer in model.layers] == [32, 32, 6]


def test_command_line_replicate_reaches_the_probe(monkeypatch) -> None:
	"""--replicate picks the seed (finalists of the search tuning run on several seeds); the default is replicate 0."""
	with step("Arrange: record probe's arguments instead of running it."):
		calls = []
		monkeypatch.setattr(half_cheetah_probe, "configure_worker_threads", lambda _threads: None)
		monkeypatch.setattr(half_cheetah_probe, "probe", lambda *args, **kwargs: calls.append(kwargs) or {})

	with step("Act: the command line with and without --replicate."):
		for argv in (["--agent", "PPO", "--output-dir", "out", "--replicate", "2"], ["--agent", "PPO", "--output-dir", "out"]):
			monkeypatch.setattr("sys.argv", ["half_cheetah_probe.py", *argv])
			half_cheetah_probe.main()

	with step("Assert: replicate 2, then 0."):
		assert [call["replicate"] for call in calls] == [2, 0]


def test_entering_the_test_phase_follows_the_studys_order() -> None:
	"""As the studies do: re-introduce the gladiators first, then enter test mode, so test-time selection is built on them."""
	with step("Arrange: an agent stub recording its calls."):
		calls = []

		class _Agent:
			"""Records the order of the test-entry calls."""

			def load_gladiators(self):
				"""Record."""
				calls.append("load_gladiators")

			def set_learning(self, mode, output_dir=None):
				"""Record."""
				calls.append(("set_learning", mode, output_dir))

	with step("Act: enter the test phase."):
		half_cheetah_probe.enter_test(_Agent(), "out")

	with step("Assert: gladiators first, then test mode."):
		assert calls == ["load_gladiators", ("set_learning", False, "out")]


def test_olympics_rows_are_written_when_the_agent_held_any(tmp_path) -> None:
	"""Each competitor's Olympic returns go to olympics.csv (the noise measurement); nothing is written without Olympiads."""
	with step("Arrange: one agent with an Olympics row, one without."):
		row = {"total_env_steps": 10, "niche_id": 0, "individual_id": "a", "returns": [1.0, 3.0], "steps": [5, 5], "evicted": False}
		held, none = type("Held", (), {"olympics_history": [row]})(), type("NoGames", (), {"olympics_history": []})()

	with step("Act: write both."):
		half_cheetah_probe.write_olympics(held, str(tmp_path / "held"))
		half_cheetah_probe.write_olympics(none, str(tmp_path / "none"))

	with step("Assert: a CSV with the mean and std of the returns for the first; nothing for the second."):
		table = pd.read_csv(tmp_path / "held" / "olympics.csv")
		assert table.loc[0, "mean"] == 2.0 and table.loc[0, "individual_id"] == "a"
		assert not (tmp_path / "none" / "olympics.csv").exists()


def test_the_frozen_test_can_be_limited_to_chosen_environments(tmp_path) -> None:
	"""--test-worlds: an Earth-only tuning probe tests on Earth alone, so test-time selection is not steered by other
	environments' episodes (one niche with detection off); the default stays all five."""
	with step("Act: a tiny PPO probe tested on Earth only."):
		summary = probe("PPO", ["Earth-Standard"], 2_000, 1, str(tmp_path), test_worlds=["Earth-Standard"])

	with step("Assert: the test covers Earth alone."):
		assert list(summary["test"]) == ["Earth-Standard"]


def test_parse_overrides_nests_dotted_keys() -> None:
	"""Dotted keys build the nested settings a driver takes (recognizer_probe.features.wavelet=db4)."""
	with step("Act/Assert: two levels of nesting next to a flat key."):
		assert parse_overrides(["recognizer_probe.excitation=impulse", "recognizer_probe.features.wavelet=db4", "pop_size=50"]) == {
			"recognizer_probe": {"excitation": "impulse", "features": {"wavelet": "db4"}},
			"pop_size": 50,
		}


def test_parse_overrides_reads_layer_sizes() -> None:
	"""brain_hidden=32x16 on the command line: one size per hidden layer."""
	with step("Act/Assert: a tapered brain next to a plain int."):
		assert parse_overrides(["brain_hidden=32x16", "pop_size=50"]) == {"brain_hidden": (32, 16), "pop_size": 50}


def test_summarize_body_reports_speed_height_flips_air_time_and_body_contacts() -> None:
	"""Per environment: mean and max forward speed, mean distance per episode, mean and min torso height, and the shares
	of steps flipped (|pitch| > pi/2), airborne (nothing on the floor) and with the torso or head on the floor."""
	with step("Arrange: 2 episodes of 2 steps: one upright and running, one flipped on its back."):
		steps = pd.DataFrame(
			{
				"env_id": ["Mars"] * 4,
				"episode": [0, 0, 1, 1],
				"x": [1.0, 3.0, 0.5, 1.5],
				"z": [0.0, 0.2, -0.3, -0.3],
				"pitch": [0.1, -0.2, 3.0, -2.0],
				"vx": [4.0, 6.0, 1.0, 1.0],
				"airborne": [False, True, False, False],
				"body_on_floor": [False, False, True, True],
			}
		)

	with step("Act: summarize."):
		body = summarize_body(steps)["Mars"]

	with step("Assert: the numbers."):
		assert (body["speed_mean"], body["speed_max"]) == (3.0, 6.0)
		assert body["distance_per_episode"] == pytest.approx(1.5)  # (3 - 1 + 1.5 - 0.5) / 2
		assert (body["height_mean"], body["height_min"]) == (pytest.approx(-0.1), -0.3)
		assert (body["share_flipped"], body["share_airborne"], body["share_body_on_floor"]) == (0.5, 0.25, 0.5)


def test_the_probe_records_the_body_during_extra_frozen_episodes(tmp_path) -> None:
	"""--body-episodes: after the frozen test, that many extra frozen episodes per training environment record the body,
	summarized in summary.json under 'body' and stepwise in body.csv."""
	with step("Act: a tiny PPO probe on Earth with one body episode."):
		summary = probe("PPO", ["Earth-Standard"], 2_000, 1, str(tmp_path), test_worlds=["Earth-Standard"], body_episodes=1)

	with step("Assert: a body summary for Earth and the stepwise file (1,000 steps)."):
		assert set(summary["body"]) == {"Earth-Standard"}
		assert {"speed_mean", "share_flipped", "share_body_on_floor"} <= set(summary["body"]["Earth-Standard"])
		assert len(pd.read_csv(tmp_path / "body.csv")) == 1_000
