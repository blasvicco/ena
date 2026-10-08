"""Tests for cartpole_challenge.py — ENA vs. DQN/PPO/SAC-Discrete under equal env-step budgets."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
import cartpole_challenge
import cartpole_ena_ablation
from agents import DQNAgent, ENAgent, PPOAgent, RandomAgent, SACDiscreteAgent
from cartpole_ena_ablation import ABLATION_VARIANTS
from tools.early_stopping import StoppingRule


@pytest.mark.parametrize(
	"name, agent_type",
	[("ENA-Combined", ENAgent), ("DQN", DQNAgent), ("PPO", PPOAgent), ("SAC-Discrete", SACDiscreteAgent), ("Random", RandomAgent)],
)
def test_build_agent_returns_the_named_agent(name, agent_type) -> None:
	"""Every challenge name builds the right agent, named as asked."""
	with step("Act/Assert: build and check type and name."):
		agent = cartpole_challenge.build_agent(name, max_agent_threads=1)
		assert isinstance(agent, agent_type)
		assert agent.get_name() == name


def test_the_challenge_ena_is_the_performance_studys_combined_agent() -> None:
	"""The ENA under test is exactly ENA-Combined from the ablation (same skill drivers)."""
	with step("Act: build it and read its skill drivers' modules."):
		agent = cartpole_challenge.build_agent("ENA-Combined", max_agent_threads=1)
		skills = agent._skills  # pylint: disable=protected-access
		combined = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")

	with step("Assert: same plasticity, Hall of Fame and env-detection drivers, bpr selection."):
		assert type(skills["plasticity"]._instance).__module__.endswith(combined["plasticity_algorithm"])  # pylint: disable=protected-access
		assert type(skills["hall_of_fame"]._instance).__module__.endswith(combined["hall_of_fame_driver"])  # pylint: disable=protected-access
		assert type(skills["env_detection"]._instance).__module__.endswith(combined["recognizer_strategy"])  # pylint: disable=protected-access
		assert type(skills["selection"]._instance).__module__.endswith("bpr")  # pylint: disable=protected-access


def test_build_agent_rejects_an_unknown_name() -> None:
	"""A typo'd agent name is an error, not a silently missing row."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError):
			cartpole_challenge.build_agent("DQN-Typo", max_agent_threads=1)


def test_first_solve_protocol_trains_every_agent_under_the_same_stopping_rule(monkeypatch, tmp_path) -> None:
	"""Fairness: one StoppingRule for all agents, evaluations on separate copies of the training environments."""
	with step("Arrange: capture agent_training's arguments; skip testing."):
		captured = {}
		monkeypatch.setattr(cartpole_ena_ablation, "agent_training", lambda **kwargs: captured.update(kwargs))
		monkeypatch.setattr(cartpole_ena_ablation, "agent_testing", lambda **_kwargs: None)

	with step("Act: one replicate with two baselines and Random."):
		cartpole_challenge.run_experiment(
			0, str(tmp_path), None, [{"name": "DQN"}, {"name": "SAC-Discrete"}], include_random=True, protocol="first-solve"
		)

	with step("Assert: the approved rule, three agents, distinct evaluation env objects for the same physics."):
		rule = captured["stopping"]
		assert isinstance(rule, StoppingRule)
		assert (rule.check_every, rule.eval_episodes, rule.solved_score, rule.plateau_checks, rule.plateau_min_gain, rule.max_steps) == (
			5_000,
			5,
			475.0,
			40,
			10.0,
			3_000_000,
		)
		assert [agent.get_name() for agent in captured["agents"]] == ["DQN", "SAC-Discrete", "Random"]
		names = [world.unwrapped.name for world in captured["worlds"]]
		assert [world.unwrapped.name for world in captured["eval_worlds"]] == names
		assert all(evaluation is not training for evaluation, training in zip(captured["eval_worlds"], captured["worlds"]))
		# Equal step budget: the training chart is drawn against env steps, so every agent spans the same range.
		assert captured["training_x_axis"] == "env_steps"


def test_defaults_run_ena_and_the_three_baselines_without_random_for_50_replicates() -> None:
	"""No flags: ENA, DQN, PPO, SAC-Discrete, 50 replicates, challenge output dir. No Random floor: the ablation reports it."""
	with step("Act: parse the challenge's defaults."):
		args = cartpole_challenge.parse_challenge_args([])

	with step("Assert: the full challenge, without Random."):
		assert args.variants == ["ENA-Combined", "DQN", "PPO", "SAC-Discrete"]
		assert args.include_random is False
		assert args.num_experiments == 50
		assert args.output_dir.endswith("cartpole_challenge_outputs")


def test_fixed_budget_protocol_trains_everyone_to_the_same_budget_without_stopping(monkeypatch, tmp_path) -> None:
	"""Option 3 (default): first solve recorded, no stop on solve or plateau, the same per-environment budget for all."""
	with step("Arrange: capture agent_training's arguments; skip testing."):
		captured = {}
		monkeypatch.setattr(cartpole_ena_ablation, "agent_training", lambda **kwargs: captured.update(kwargs))
		monkeypatch.setattr(cartpole_ena_ablation, "agent_testing", lambda **_kwargs: None)

	with step("Act: one fixed-budget replicate."):
		cartpole_challenge.run_experiment(0, str(tmp_path), None, [{"name": "PPO"}], include_random=False, protocol="fixed-budget")

	with step("Assert: non-stopping rule, the fixed per-environment budget, same checks."):
		rule = captured["stopping"]
		assert (rule.stop_when_solved, rule.stop_on_plateau) == (False, False)
		assert rule.max_steps == cartpole_challenge.FIXED_BUDGET_PER_ENV
		assert (rule.check_every, rule.eval_episodes, rule.solved_score) == (5_000, 5, 475.0)


def test_protocol_defaults_to_fixed_budget_and_rejects_unknown_values() -> None:
	"""The command line chooses the protocol; fixed-budget is the default."""
	with step("Act/Assert: default, explicit first-solve, and a typo."):
		assert cartpole_challenge.parse_challenge_args([]).protocol == "fixed-budget"
		assert cartpole_challenge.parse_challenge_args(["--protocol", "first-solve"]).protocol == "first-solve"
		with pytest.raises(SystemExit):
			cartpole_challenge.parse_challenge_args(["--protocol", "fastest"])


def test_challenge_ppo_uses_a_constant_learning_rate() -> None:
	"""In a non-stationary stream PPO must keep learning at full rate in every environment (author's choice)."""
	with step("Act/Assert: the challenge's PPO doesn't anneal."):
		assert cartpole_challenge.build_agent("PPO", max_agent_threads=1).anneal_lr is False


def test_challenge_sac_discrete_uses_the_stable_target_entropy() -> None:
	"""The paper's 0.98 x log|A| diverges with 2 actions; the challenge uses 0.5 x log|A| (author's choice)."""
	with step("Act/Assert: target entropy is half the maximum log(2)."):
		agent = cartpole_challenge.build_agent("SAC-Discrete", max_agent_threads=1)
		assert agent.target_entropy == pytest.approx(0.5 * np.log(2))
