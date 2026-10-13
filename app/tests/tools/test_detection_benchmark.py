"""Tests for detection_benchmark.py — scripted detector inputs per environment, replayed and scored."""

# Libs imports
import numpy as np
import pandas as pd
import pytest

# App imports
from helpers import step
import cartpole_ena_ablation
from tools.detection_benchmark import generate, meets_target, run, score
from tools.detection_replay import EVALUATE, MATCH, OBSERVE, CHIRP, START


def test_generate_scripts_episodes_per_world_in_the_recordings_format() -> None:
	"""One START, the episode's OBSERVEs, a training EVALUATE and a MATCH per episode; transitions stored; env per episode."""
	with step("Act: 2 CartPole worlds x 3 episodes, policy changing every 2 episodes, at most 30 steps."):
		worlds = cartpole_ena_ablation.make_training_worlds()[:2]
		recording, env_ids = generate(worlds, episodes_per_world=3, policy_block=2, seed=0, max_steps=30, is_continuous=False)

	with step("Assert: 6 episodes in order; event counts consistent with the transitions."):
		kinds = recording["kinds"]
		assert env_ids == ["Mars-Light"] * 3 + ["Jupiter-Heavy"] * 3
		assert (kinds == START).sum() == 6 and (kinds == EVALUATE).sum() == 6 and (kinds == MATCH).sum() == 6
		assert (kinds == OBSERVE).sum() == len(recording["states"]) == len(recording["next_states"]) == len(recording["actions"])
		assert recording["learn"][kinds == EVALUATE].all()


def test_generate_is_reproducible_from_its_seed() -> None:
	"""Same seed, same inputs: benchmark results are comparable across drivers."""
	with step("Act: generate twice with seed 3."):
		worlds = cartpole_ena_ablation.make_training_worlds()[:1]
		first, _ = generate(worlds, 2, 1, seed=3, max_steps=20, is_continuous=False)
		second, _ = generate(cartpole_ena_ablation.make_training_worlds()[:1], 2, 1, seed=3, max_steps=20, is_continuous=False)

	with step("Assert: identical transitions."):
		assert np.array_equal(first["states"], second["states"]) and np.array_equal(first["actions"], second["actions"])


def _replayed(shifts, niches):
	return pd.DataFrame({"shift": shifts, "niche": niches})


def test_score_reports_switch_delays_false_changes_and_niche_purity() -> None:
	"""Three worlds of four episodes: a switch caught at once, one caught late, one false change; niches per world."""
	with step("Arrange: changes at episodes 2 (false), 4 (Jupiter, delay 0) and 10 (Earth, delay 2); niches 0 / 1 / 2."):
		env_ids = ["Mars"] * 4 + ["Jupiter"] * 4 + ["Earth"] * 4
		shifts = [False, True, False, False, True, False, False, False, False, False, True, False]
		niches = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2]

	with step("Act: score with a 3-episode window."):
		result = score(_replayed(shifts, niches), env_ids, window=3)

	with step("Assert: delays, the false change, purity and distinct majorities."):
		assert result["delays"] == [0, 2]
		assert result["caught"] == 2 and result["inside"] == 1
		assert result["purity"] == {"Mars": 1.0, "Jupiter": 1.0, "Earth": 0.5}
		assert result["majority_niches"] == {"Mars": 0, "Jupiter": 1, "Earth": 0}
		assert not result["distinct"]


def test_meets_target_requires_every_switch_no_false_change_and_one_niche_per_world() -> None:
	"""The pass rule: all switches within the window, zero false changes, each world's majority niche its own."""
	with step("Arrange: a perfect score and three flawed ones."):
		perfect = {"caught": 2, "switches": 2, "inside": 0, "distinct": True, "purity": {"a": 1.0, "b": 0.95}}

	with step("Act/Assert: only the perfect one passes."):
		assert meets_target(perfect)
		assert not meets_target({**perfect, "caught": 1})
		assert not meets_target({**perfect, "inside": 1})
		assert not meets_target({**perfect, "distinct": False})
		assert not meets_target({**perfect, "purity": {"a": 1.0, "b": 0.6}})


def test_the_cartpole_detector_meets_the_benchmark() -> None:
	"""Validates the benchmark itself: CartPole's detection (reported working in the studies) passes on scripted inputs."""
	with step("Act: CartPole Mars -> Jupiter -> Earth, the study's detector settings."):
		result = run("cartpole")

	with step("Assert: both switches caught, no false change, one niche per environment."):
		assert meets_target(result), result


@pytest.mark.parametrize("seed", range(5))
def test_the_half_cheetah_detector_meets_the_benchmark(seed) -> None:
	"""Guards HalfCheetah's detection (the chirp driver): on each of 5 benchmark seeds (other policies and resets),
	Mars -> Heavy -> Earth under drifting policies is detected as CartPole's is."""
	with step(f"Act: HalfCheetah Mars -> Heavy -> Earth, the study's detector settings, benchmark seed {seed}."):
		result = run("half_cheetah", seed=seed)

	with step("Assert: both switches caught, no false change, one niche per environment."):
		assert meets_target(result), result


def test_generate_plays_a_chirp_after_every_episode_when_given_one() -> None:
	"""With chirp actions, each episode ends with a CHIRP event holding its observations (start first), before the EVALUATE."""
	with step("Act: 2 CartPole episodes with a 6-step chirp."):
		worlds = cartpole_ena_ablation.make_training_worlds()[:1]
		recording, _ = generate(worlds, 2, 1, seed=0, max_steps=20, is_continuous=False, chirp_actions=np.array([1, 0, 1, 0, 1, 0]))

	with step("Assert: 2 chirps of 7 observations each; each CHIRP right before an EVALUATE."):
		kinds = recording["kinds"]
		assert recording["chirps"].shape == (2, 7, 4)
		chirp_positions = np.flatnonzero(kinds == CHIRP)
		assert len(chirp_positions) == 2 and all(kinds[position + 1] == EVALUATE for position in chirp_positions)


def test_the_chirp_driver_runs_on_the_benchmark() -> None:
	"""run() builds the chirp from the settings and replays the chirps through the chirp driver."""
	with step("Act: CartPole through the chirp driver with a 16-step chirp and raw features."):
		result = run("cartpole", "chirp", {"recognizer_chirp": {"steps": 16, "features": {"kind": "raw"}}})

	with step("Assert: a scored result over the three worlds."):
		assert result["switches"] == 2 and len(result["purity"]) == 3


def test_a_fixed_policy_family_plays_the_same_gait_in_every_world() -> None:
	"""policy_family='fixed': one gait for every episode of every world (behavior never changes)."""
	with step("Act: 2 HalfCheetah worlds x 2 episodes of 20 steps with the fixed family."):
		import half_cheetah  # pylint: disable=import-outside-toplevel

		fixed, _ = generate(half_cheetah.make_training_worlds()[:2], 2, 1, seed=0, max_steps=20, is_continuous=True, policy_family="fixed")

	with step("Assert: the first and last episodes (different worlds) play the same gait, up to the action noise."):
		episodes = fixed["actions"].reshape(4, 20, 6)
		assert np.corrcoef(episodes[0].ravel(), episodes[3].ravel())[0, 1] > 0.9


def test_a_benchmark_run_is_reproducible_from_its_seed() -> None:
	"""The paper's numbers can be regenerated: the same domain, driver, settings and seed give the same score."""
	with step("Act: the same CartPole run twice."):
		first, second = run("cartpole", seed=7), run("cartpole", seed=7)

	with step("Assert: identical scores."):
		assert first == second
