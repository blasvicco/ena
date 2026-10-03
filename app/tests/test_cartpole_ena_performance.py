"""Tests for cartpole_ena_performance.py's skill matrix and study plumbing — catches a typo'd row before a multi-hour run."""

# Libs imports
import numpy as np
import pandas as pd
import pytest

# App imports
from helpers import step
import cartpole_ena_performance
from tools.early_stopping import StoppingRule
from cartpole_ena_performance import PERFORMANCE_VARIANTS

SKILLS = ("plasticity_algorithm", "hall_of_fame_driver", "recognizer_strategy")
OFF_VALUES = {"plasticity_algorithm": "zero", "hall_of_fame_driver": "none", "recognizer_strategy": "none"}
ON_VALUES = {"plasticity_algorithm": "linear_ramp", "hall_of_fame_driver": "default", "recognizer_strategy": "dynamics_model"}


def _skills_on(variant):
	"""Which of the three studied skills a row turns on."""
	return {skill for skill in SKILLS if variant[skill] == ON_VALUES[skill]}


def test_rows_are_exactly_the_six_meaningful_skill_combinations() -> None:
	"""Every on/off combination of the three skills except env detection without the Hall of Fame (which equals its twin)."""
	with step("Act: collect each row's set of enabled skills."):
		combinations = [frozenset(_skills_on(variant)) for variant in PERFORMANCE_VARIANTS]

	with step("Assert: six unique names, six unique combinations, none with env detection but no Hall of Fame."):
		assert len({variant["name"] for variant in PERFORMANCE_VARIANTS}) == 6
		assert len(set(combinations)) == 6
		assert not [combo for combo in combinations if "recognizer_strategy" in combo and "hall_of_fame_driver" not in combo]


def test_every_skill_is_either_on_or_at_its_off_value() -> None:
	"""No row uses a third value (e.g. plasticity "full") — that belongs to the compute study."""
	with step("Assert: each skill column holds only its on or off value."):
		for variant in PERFORMANCE_VARIANTS:
			for skill in SKILLS:
				assert variant[skill] in (OFF_VALUES[skill], ON_VALUES[skill]), (variant["name"], skill)


@pytest.mark.parametrize(
	"name, expected",
	[
		("ENA-NoSkill", set()),
		("ENA-Plasticity", {"plasticity_algorithm"}),
		("ENA-HoF", {"hall_of_fame_driver"}),
		("ENA-HoF+EnvDetection", {"hall_of_fame_driver", "recognizer_strategy"}),
		("ENA-Plasticity+HoF", {"plasticity_algorithm", "hall_of_fame_driver"}),
		("ENA-Combined", set(SKILLS)),
	],
)
def test_each_row_name_matches_its_skills(name, expected) -> None:
	"""A row's name must describe exactly the skills it enables."""
	with step("Assert: the named row enables exactly the expected skills."):
		variant = next(row for row in PERFORMANCE_VARIANTS if row["name"] == name)
		assert _skills_on(variant) == expected


def test_module_exposes_training_and_testing_episode_budgets() -> None:
	"""Sanity check the shared episode-budget constants used to build every agent."""
	with step("Assert: the training/testing episode-count constants are present and non-empty."):
		assert cartpole_ena_performance.TRAINING_EPISODES_ENA
		assert cartpole_ena_performance.TESTING_EPISODES


def test_seed_experiment_is_distinct_per_replicate_and_repeatable() -> None:
	"""Each replicate gets its own seed, and the same replicate always draws the same numbers."""
	with step("Act: seed replicate 3 twice and replicate 4 once, drawing a NumPy number each time."):
		seed_3 = cartpole_ena_performance.seed_experiment(3)
		draw_3 = np.random.rand()
		cartpole_ena_performance.seed_experiment(3)
		draw_3_again = np.random.rand()
		seed_4 = cartpole_ena_performance.seed_experiment(4)
		draw_4 = np.random.rand()

	with step("Assert: distinct seeds; repeatable within a replicate, different across replicates."):
		assert seed_3 != seed_4
		assert draw_3 == draw_3_again
		assert draw_3 != draw_4


def test_parse_args_defaults_to_the_full_performance_study() -> None:
	"""No flags: all rows, 50 replicates, the Random floor, into OUTPUT_DIR."""
	with step("Act: parse an empty command line."):
		args = cartpole_ena_performance.parse_args([])

	with step("Assert: module defaults."):
		assert args.variants == [row["name"] for row in PERFORMANCE_VARIANTS]
		assert args.num_experiments == cartpole_ena_performance.NUM_EXPERIMENTS == 50
		assert args.include_random is True
		assert args.output_dir == cartpole_ena_performance.OUTPUT_DIR


def test_parse_args_overrides_and_rejects_unknown_rows() -> None:
	"""Flags select rows, replicate count and study directory; a typo'd row name is a usage error."""
	with step("Act: parse an override command line."):
		args = cartpole_ena_performance.parse_args(
			["--variants", "ENA-HoF", "--num-experiments", "3", "--output-dir", "/tmp/hof", "--no-include-random"]
		)

	with step("Assert: overrides applied; an unknown row exits with a usage error."):
		assert (args.variants, args.num_experiments, args.output_dir, args.include_random) == (["ENA-HoF"], 3, "/tmp/hof", False)
		with pytest.raises(SystemExit):
			cartpole_ena_performance.parse_args(["--variants", "ENA-Typo"])


def test_select_variants_keeps_table_order() -> None:
	"""Rows come back in table order, whatever order they were named in."""
	with step("Act/Assert: two names given out of order."):
		rows = cartpole_ena_performance.select_variants(PERFORMANCE_VARIANTS, ["ENA-Combined", "ENA-NoSkill"])
		assert [row["name"] for row in rows] == ["ENA-NoSkill", "ENA-Combined"]


def test_phase_env_steps_splits_cumulative_steps_per_contiguous_environment() -> None:
	"""Per-phase consumption from the cumulative env_steps column."""
	with step("Arrange: 2 Mars episodes then 3 Jupiter episodes."):
		history = pd.DataFrame({"env_id": ["Mars"] * 2 + ["Jupiter"] * 3, "env_steps": [100, 250, 300, 420, 600]})

	with step("Act/Assert: 250 steps on Mars, 350 on Jupiter."):
		assert cartpole_ena_performance.phase_env_steps(history) == [250, 350]


class _BudgetAgent:
	"""Agent stub: records the step budget it was trained under (via the stubbed play())."""

	def __init__(self, name, episodes):
		self._name = name
		self._episodes = episodes
		self.total_steps = None
		self.learn = True

	def get_name(self):
		"""Name."""
		return self._name

	def get_episodes(self):
		"""Episodes per env."""
		return self._episodes

	def set_learning(self, mode, output_dir=None):  # pylint: disable=unused-argument
		"""Record mode."""
		self.learn = mode

	def set_total_steps(self, total):
		"""Record the total budget (PPO's LR-schedule hook)."""
		self.total_steps = total


def test_agent_training_matches_later_agents_to_the_reference_agents_steps_per_env(monkeypatch, tmp_path) -> None:
	"""S16: agents after match_steps_of train under its per-environment env-step consumption."""
	with step("Arrange: a stubbed play() that records budgets; the reference uses 300 then 500 steps."):
		budgets = {}

		def _fake_play(agent, _worlds, _episodes, steps_per_env=None, **_kwargs):
			budgets[agent.get_name()] = steps_per_env
			return pd.DataFrame({"env_id": ["Mars", "Mars", "Jupiter"], "env_steps": [100, 300, 800], "score": [1.0, 1.0, 1.0]})

		monkeypatch.setattr(cartpole_ena_performance, "play", _fake_play)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_performance, helper, lambda *args, **kwargs: None)
		agents = [_BudgetAgent("Before", [5, 5]), _BudgetAgent("Reference", [5, 5]), _BudgetAgent("Follower", [5, 5])]

	with step("Act: train with Reference as the step reference."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_performance.agent_training(agents, 0, str(tmp_path), output_file, None, [None, None], match_steps_of="Reference")

	with step("Assert: agents up to the reference train by episodes; the follower gets [300, 500] (800 in total)."):
		assert budgets == {"Before": None, "Reference": None, "Follower": [300, 500]}
		assert agents[2].total_steps == 800
		assert "800 env-steps" in (tmp_path / "outputs.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("row_trust, expected", [(None, "linear_ramp"), ("zero", "zero")])
def test_build_ena_agent_takes_trust_from_the_row_or_the_study_default(row_trust, expected) -> None:
	"""trust_algorithm is an optional per-row key; rows without it use the study-wide linear_ramp."""
	with step("Arrange: the NoSkill row, with or without a trust override."):
		row = dict(next(variant for variant in PERFORMANCE_VARIANTS if variant["name"] == "ENA-NoSkill"))
		if row_trust is not None:
			row["trust_algorithm"] = row_trust

	with step("Act: build the agent."):
		agent = cartpole_ena_performance.build_ena_agent(row, max_agent_threads=1)

	with step("Assert: its Trust driver module is the expected one."):
		assert type(agent._skills["trust"]._instance).__module__.endswith(expected)  # pylint: disable=protected-access


def test_agent_training_tells_agents_the_true_total_budget_under_a_stopping_rule(monkeypatch, tmp_path) -> None:
	"""Regression: under a stopping rule nothing told PPO its total budget, so its LR schedule used a
	1.5M-step default and hit zero after the first environment. The total is max_steps x environments."""
	with step("Arrange: a stubbed play() and a rule capping each of 3 environments at 2.5M steps."):
		monkeypatch.setattr(
			cartpole_ena_performance, "play", lambda *_args, **_kwargs: pd.DataFrame({"env_id": ["Mars"], "env_steps": [1], "score": [1.0]})
		)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_performance, helper, lambda *args, **kwargs: None)
		agent = _BudgetAgent("PPO-like", [5, 5, 5])
		rule = StoppingRule(max_steps=2_500_000, stop_when_solved=False, stop_on_plateau=False)

	with step("Act: train under the rule on 3 environments."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_performance.agent_training(
				[agent], 0, str(tmp_path), output_file, None, [None, None, None], stopping=rule, eval_worlds=[None] * 3
			)

	with step("Assert: told 7.5M steps."):
		assert agent.total_steps == 7_500_000


def test_parse_args_sets_how_many_replicates_run_in_parallel() -> None:
	"""--workers caps the parallel replicates (default: the module's MAX_WORKERS); it must be positive."""
	with step("Act/Assert: default, override, and a non-positive value."):
		assert cartpole_ena_performance.parse_args([]).workers == cartpole_ena_performance.MAX_WORKERS
		assert cartpole_ena_performance.parse_args(["--workers", "5"]).workers == 5
		with pytest.raises(SystemExit):
			cartpole_ena_performance.parse_args(["--workers", "0"])
