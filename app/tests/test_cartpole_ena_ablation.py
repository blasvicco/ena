"""Tests for cartpole_ena_ablation.py's skill matrix and study plumbing — catches a typo'd row before a multi-hour run."""

# General imports
import concurrent.futures
import os

# Libs imports
import numpy as np
import pandas as pd
import pytest

# App imports
from helpers import step
import cartpole_ena_ablation
from tools.early_stopping import StoppingRule
from cartpole_ena_ablation import ABLATION_VARIANTS

SKILLS = ("plasticity_algorithm", "hall_of_fame_driver", "recognizer_strategy")
OFF_VALUES = {"plasticity_algorithm": "zero", "hall_of_fame_driver": "none", "recognizer_strategy": "none"}
ON_VALUES = {"plasticity_algorithm": "linear_ramp", "hall_of_fame_driver": "default", "recognizer_strategy": "dynamics_model"}


def _skills_on(variant):
	"""Which of the three studied skills a row turns on."""
	return {skill for skill in SKILLS if variant[skill] == ON_VALUES[skill]}


def test_rows_are_exactly_the_six_meaningful_skill_combinations() -> None:
	"""Every on/off combination of the three skills except env detection without the Hall of Fame (which equals its twin)."""
	with step("Act: collect each row's set of enabled skills."):
		combinations = [frozenset(_skills_on(variant)) for variant in ABLATION_VARIANTS]

	with step("Assert: six unique names, six unique combinations, none with env detection but no Hall of Fame."):
		assert len({variant["name"] for variant in ABLATION_VARIANTS}) == 6
		assert len(set(combinations)) == 6
		assert not [combo for combo in combinations if "recognizer_strategy" in combo and "hall_of_fame_driver" not in combo]


def test_every_skill_is_either_on_or_at_its_off_value() -> None:
	"""No row uses a third value (e.g. plasticity "full") — that belongs to the compute study."""
	with step("Assert: each skill column holds only its on or off value."):
		for variant in ABLATION_VARIANTS:
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
		variant = next(row for row in ABLATION_VARIANTS if row["name"] == name)
		assert _skills_on(variant) == expected


def test_module_exposes_training_and_testing_episode_budgets() -> None:
	"""Sanity check the shared episode-budget constants used to build every agent."""
	with step("Assert: the training/testing episode-count constants are present and non-empty."):
		assert cartpole_ena_ablation.TRAINING_EPISODES_ENA
		assert cartpole_ena_ablation.TESTING_EPISODES


def test_seed_experiment_is_distinct_per_replicate_and_repeatable() -> None:
	"""Each replicate gets its own seed, and the same replicate always draws the same numbers."""
	with step("Act: seed replicate 3 twice and replicate 4 once, drawing a NumPy number each time."):
		seed_3 = cartpole_ena_ablation.seed_experiment(3)
		draw_3 = np.random.rand()
		cartpole_ena_ablation.seed_experiment(3)
		draw_3_again = np.random.rand()
		seed_4 = cartpole_ena_ablation.seed_experiment(4)
		draw_4 = np.random.rand()

	with step("Assert: distinct seeds; repeatable within a replicate, different across replicates."):
		assert seed_3 != seed_4
		assert draw_3 == draw_3_again
		assert draw_3 != draw_4


def test_parse_args_defaults_to_the_full_performance_study() -> None:
	"""No flags: all rows, 30 replicates, the Random floor, into OUTPUT_DIR."""
	with step("Act: parse an empty command line."):
		args = cartpole_ena_ablation.parse_args([])

	with step("Assert: module defaults."):
		assert args.variants == [row["name"] for row in ABLATION_VARIANTS]
		assert args.num_experiments == cartpole_ena_ablation.NUM_EXPERIMENTS == 30
		assert args.include_random is True
		assert args.output_dir == cartpole_ena_ablation.OUTPUT_DIR


def test_parse_args_overrides_and_rejects_unknown_rows() -> None:
	"""Flags select rows, replicate count and study directory; a typo'd row name is a usage error."""
	with step("Act: parse an override command line."):
		args = cartpole_ena_ablation.parse_args(
			["--variants", "ENA-HoF", "--num-experiments", "3", "--output-dir", "/tmp/hof", "--no-include-random"]
		)

	with step("Assert: overrides applied; an unknown row exits with a usage error."):
		assert (args.variants, args.num_experiments, args.output_dir, args.include_random) == (["ENA-HoF"], 3, "/tmp/hof", False)
		with pytest.raises(SystemExit):
			cartpole_ena_ablation.parse_args(["--variants", "ENA-Typo"])


def test_select_variants_keeps_table_order() -> None:
	"""Rows come back in table order, whatever order they were named in."""
	with step("Act/Assert: two names given out of order."):
		rows = cartpole_ena_ablation.select_variants(ABLATION_VARIANTS, ["ENA-Combined", "ENA-NoSkill"])
		assert [row["name"] for row in rows] == ["ENA-NoSkill", "ENA-Combined"]


def test_phase_env_steps_splits_cumulative_steps_per_contiguous_environment() -> None:
	"""Per-phase consumption from the cumulative env_steps column."""
	with step("Arrange: 2 Mars episodes then 3 Jupiter episodes."):
		history = pd.DataFrame({"env_id": ["Mars"] * 2 + ["Jupiter"] * 3, "env_steps": [100, 250, 300, 420, 600]})

	with step("Act/Assert: 250 steps on Mars, 350 on Jupiter."):
		assert cartpole_ena_ablation.phase_env_steps(history) == [250, 350]


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

		monkeypatch.setattr(cartpole_ena_ablation, "play", _fake_play)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)
		agents = [_BudgetAgent("Before", [5, 5]), _BudgetAgent("Reference", [5, 5]), _BudgetAgent("Follower", [5, 5])]

	with step("Act: train with Reference as the step reference."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(agents, 0, str(tmp_path), output_file, None, [None, None], match_steps_of="Reference")

	with step("Assert: agents up to the reference train by episodes; the follower gets [300, 500] (800 in total)."):
		assert budgets == {"Before": None, "Reference": None, "Follower": [300, 500]}
		assert agents[2].total_steps == 800
		assert "800 env-steps" in (tmp_path / "outputs.txt").read_text(encoding="utf-8")


def test_agent_training_gives_every_agent_a_fixed_step_budget_per_environment(monkeypatch, tmp_path) -> None:
	"""steps_per_env trains every agent to the same env-step budget per environment, with no checks."""
	# steps_per_env trains every agent to the same env-step budget per environment, with no checks (HalfCheetah);
	# the total reaches PPO-like agents' training-length hook.
	with step("Arrange: a stubbed play() that records budgets."):
		budgets = {}

		def _fake_play(agent, _worlds, _episodes, steps_per_env=None, **kwargs):
			budgets[agent.get_name()] = (steps_per_env, kwargs.get("stopping"))
			return pd.DataFrame({"env_id": ["Mars", "Jupiter"], "env_steps": [300, 800], "score": [1.0, 1.0]})

		monkeypatch.setattr(cartpole_ena_ablation, "play", _fake_play)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)
		agents = [_BudgetAgent("A", [5, 5]), _BudgetAgent("B", [5, 5])]

	with step("Act: train with 400 steps per environment."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(agents, 0, str(tmp_path), output_file, None, [None, None], steps_per_env=[400, 400])

	with step("Assert: both train under [400, 400], no stopping rule, and know the 800-step total."):
		assert budgets == {"A": ([400, 400], None), "B": ([400, 400], None)}
		assert agents[0].total_steps == agents[1].total_steps == 800


@pytest.mark.parametrize("health_every, written", [(None, False), (10_000, True)])
def test_agent_training_logs_agent_health_only_when_asked(monkeypatch, tmp_path, health_every, written) -> None:
	"""health_every writes each agent's internal health to health_{agent}.csv; it is off by default."""
	# health_every (HalfCheetah): each agent's internal health (SAC's temperature and Q-values, PPO's learning rate and
	# policy std) is written to health_{agent}.csv at most every health_every env-steps; off by default (CartPole).
	with step("Arrange: a stubbed play() that reports progress at 10k and 20k steps, and a stubbed health reading."):

		def _fake_play(_agent, _worlds, _episodes, on_episode=None, **_kwargs):
			for progress in (5_000, 10_000, 20_000):
				on_episode(progress, 20_000)
			return pd.DataFrame({"env_id": ["Mars"], "env_steps": [20_000], "score": [1.0]})

		monkeypatch.setattr(cartpole_ena_ablation, "play", _fake_play)
		monkeypatch.setattr(cartpole_ena_ablation, "agent_health", lambda _agent: {"alpha": 0.5})
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)

	with step("Act: train one agent."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training([_BudgetAgent("A", [5])], 0, str(tmp_path), output_file, None, [None], health_every=health_every)

	with step("Assert: the file only when asked, with one row per 10k steps."):
		path = tmp_path / "health_A.csv"
		assert path.exists() is written
		if written:
			assert pd.read_csv(path).to_dict("list") == {"steps": [5_000, 20_000], "alpha": [0.5, 0.5]}


@pytest.mark.parametrize("row_trust, expected", [(None, "linear_ramp"), ("zero", "zero")])
def test_build_ena_agent_takes_trust_from_the_row_or_the_study_default(row_trust, expected) -> None:
	"""trust_algorithm is an optional per-row key; rows without it use the study-wide linear_ramp."""
	with step("Arrange: the NoSkill row, with or without a trust override."):
		row = dict(next(variant for variant in ABLATION_VARIANTS if variant["name"] == "ENA-NoSkill"))
		if row_trust is not None:
			row["trust_algorithm"] = row_trust

	with step("Act: build the agent."):
		agent = cartpole_ena_ablation.build_ena_agent(row, max_agent_threads=1)

	with step("Assert: its Trust driver module is the expected one."):
		assert type(agent._skills["trust"]._instance).__module__.endswith(expected)  # pylint: disable=protected-access


@pytest.mark.parametrize("row_driver, expected", [(None, "bpr"), ("niche_match", "niche_match")])
def test_build_ena_agent_takes_selection_from_the_row_or_the_study_default(row_driver, expected) -> None:
	"""selection_driver is an optional per-row key; rows without it use the study-wide bpr."""
	with step("Arrange: the Combined row, with or without a selection override."):
		row = dict(next(variant for variant in ABLATION_VARIANTS if variant["name"] == "ENA-Combined"))
		if row_driver is not None:
			row["selection_driver"] = row_driver

	with step("Act: build the agent."):
		agent = cartpole_ena_ablation.build_ena_agent(row, max_agent_threads=1)

	with step("Assert: its selection driver module is the expected one."):
		assert type(agent._skills["selection"]._instance).__module__.endswith(expected)  # pylint: disable=protected-access


def test_agent_training_tells_agents_the_true_total_budget_under_a_stopping_rule(monkeypatch, tmp_path) -> None:
	"""Under a stopping rule, agents are told their true total budget: max_steps times the environments."""
	# Regression: under a stopping rule nothing told PPO its total budget, so its LR schedule used a
	# 1.5M-step default and hit zero after the first environment. The total is max_steps x environments.
	with step("Arrange: a stubbed play() and a rule capping each of 3 environments at 2.5M steps."):
		monkeypatch.setattr(
			cartpole_ena_ablation, "play", lambda *_args, **_kwargs: pd.DataFrame({"env_id": ["Mars"], "env_steps": [1], "score": [1.0]})
		)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)
		agent = _BudgetAgent("PPO-like", [5, 5, 5])
		rule = StoppingRule(max_steps=2_500_000, stop_when_solved=False, stop_on_plateau=False)

	with step("Act: train under the rule on 3 environments."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(
				[agent], 0, str(tmp_path), output_file, None, [None, None, None], stopping=rule, eval_worlds=[None] * 3
			)

	with step("Assert: told 7.5M steps."):
		assert agent.total_steps == 7_500_000


def test_parse_args_sets_how_many_replicates_run_in_parallel() -> None:
	"""--workers caps the parallel replicates (default: the module's MAX_WORKERS); it must be positive."""
	with step("Act/Assert: default, override, and a non-positive value."):
		assert cartpole_ena_ablation.parse_args([]).workers == cartpole_ena_ablation.MAX_WORKERS
		assert cartpole_ena_ablation.parse_args(["--workers", "5"]).workers == 5
		with pytest.raises(SystemExit):
			cartpole_ena_ablation.parse_args(["--workers", "0"])


def _stub_reporting(monkeypatch) -> dict:
	"""Stub every metrics/plot helper of the shared runner; return {helper: [kwargs of each call]}."""
	calls = {}
	for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
		calls[helper] = []
		monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, _name=helper, **kwargs: calls[_name].append(kwargs))
	return calls


def test_agent_testing_defaults_stay_cartpoles(monkeypatch, tmp_path) -> None:
	"""Without overrides, agent_testing still tests on CartPole's five worlds, bar and chart range."""
	# Other domains reuse agent_testing; with no overrides it must still test on CartPole's 5 worlds
	# (Neptune, Brown-dwarf added) with the 475 bar and CartPole's chart range.
	with step("Arrange: stubbed play() recording the worlds, and stubbed reporting."):
		played = {}
		monkeypatch.setattr(
			cartpole_ena_ablation,
			"play",
			lambda agent, worlds, _episodes, **_kwargs: played.setdefault("worlds", worlds) is not None
			and pd.DataFrame({"env_id": ["Mars"], "env_steps": [1], "score": [1.0]}),
		)
		calls = _stub_reporting(monkeypatch)
		agent = _BudgetAgent("Agent", [1] * 5)

	with step("Act: test with the default arguments."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_testing([agent], 0, str(tmp_path), output_file, None, ["Mars", "Jupiter", "Earth"])

	with step("Assert: CartPole's worlds, threshold and y-range."):
		names = [world if isinstance(world, str) else world.unwrapped.name for world in played["worlds"]]
		assert names == ["Mars", "Neptune", "Jupiter", "Brown-dwarf", "Earth"]
		assert calls["calculate_paper_metrics"][0]["success_threshold"] == 475.0
		assert calls["plot_academic_comparison"][0]["success_threshold"] == 475.0
		assert calls["plot_academic_comparison"][0]["ylim"] == (-10, 550)


def test_agent_testing_and_training_take_a_domains_worlds_threshold_and_range(monkeypatch, tmp_path) -> None:
	"""A domain runner passes its own test worlds, solved bar and chart range; nothing CartPole leaks in."""
	with step("Arrange: stubbed play(), reporting, and a make_env that must not be called."):
		played = []
		monkeypatch.setattr(
			cartpole_ena_ablation,
			"play",
			lambda agent, worlds, _episodes, **_kwargs: played.append(worlds) or pd.DataFrame({"env_id": ["A"], "env_steps": [1], "score": [1.0]}),
		)
		monkeypatch.setattr(cartpole_ena_ablation, "make_env", lambda *args, **kwargs: pytest.fail("built a CartPole world"))
		calls = _stub_reporting(monkeypatch)
		agent = _BudgetAgent("Agent", [1] * 5)

	with step("Act: train then test with a domain's settings."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(
				[agent], 0, str(tmp_path), output_file, None, ["A", "B", "C"], success_threshold=4800.0, ylim=(-1000, 12000)
			)
			cartpole_ena_ablation.agent_testing(
				[agent],
				0,
				str(tmp_path),
				output_file,
				None,
				["A", "B", "C"],
				test_worlds=["A", "X", "B", "Y", "C"],
				success_threshold=4800.0,
				ylim=(-1000, 12000),
			)

	with step("Assert: the domain's worlds, threshold and range reached play() and every report."):
		assert played == [["A", "B", "C"], ["A", "X", "B", "Y", "C"]]
		assert [call["success_threshold"] for call in calls["calculate_paper_metrics"]] == [4800.0, 4800.0]
		assert [call["success_threshold"] for call in calls["plot_academic_comparison"]] == [4800.0, 4800.0]
		assert [call["ylim"] for call in calls["plot_academic_comparison"]] == [(-1000, 12000), (-1000, 12000)]


@pytest.mark.parametrize(
	"content, complete",
	[
		(None, False),
		("", False),
		("Experiment 3 Log\n==============================\n", False),
		("...\n  --- Experiment 3 Test Results ---\n...", True),
	],
)
def test_a_replicate_is_complete_only_once_its_test_results_are_written(tmp_path, content, complete) -> None:
	"""A replicate counts as complete only once its test results are written, so interrupted ones re-run."""
	# Resume safety: outputs.txt gets its header when a replicate starts, so an interrupted replicate (a
	# preempted cloud machine) must be re-run, not skipped. Finished replicates (CartPole's included) keep being skipped.
	with step("Arrange: a replicate directory with the given outputs.txt (or none)."):
		if content is not None:
			(tmp_path / "outputs.txt").write_text(content, encoding="utf-8")

	with step("Act/Assert: complete only with the test results."):
		assert cartpole_ena_ablation.replicate_complete(str(tmp_path)) is complete


@pytest.mark.parametrize("threads, expected", [(None, []), (2, [("intra", 2), ("inter", 1)])])
def test_worker_threads_are_capped_only_when_asked(monkeypatch, threads, expected) -> None:
	"""On a many-core machine every worker's TensorFlow would otherwise use every core; locally (None) nothing changes."""
	with step("Arrange: record TensorFlow's thread setters."):
		calls = []
		monkeypatch.setattr(cartpole_ena_ablation.tf.config.threading, "set_intra_op_parallelism_threads", lambda n: calls.append(("intra", n)))
		monkeypatch.setattr(cartpole_ena_ablation.tf.config.threading, "set_inter_op_parallelism_threads", lambda n: calls.append(("inter", n)))

	with step("Act: configure."):
		cartpole_ena_ablation.configure_worker_threads(threads)

	with step("Assert: set only when given."):
		assert calls == expected


def test_parse_args_threads_per_worker_defaults_to_uncapped() -> None:
	"""--threads-per-worker is opt-in, so CartPole's launches are unchanged."""
	with step("Act/Assert: default and override."):
		assert cartpole_ena_ablation.parse_args([]).threads_per_worker is None
		assert cartpole_ena_ablation.parse_args(["--threads-per-worker", "2"]).threads_per_worker == 2
		with pytest.raises(SystemExit):
			cartpole_ena_ablation.parse_args(["--threads-per-worker", "0"])


def test_every_agent_of_a_replicate_starts_from_the_same_random_state(monkeypatch, tmp_path) -> None:
	"""Every agent of a replicate starts from the same random state, so variants differ only in their skills."""
	# Common random numbers: the RNGs are re-seeded before each agent is built, so two variants of a
	# replicate start from the same initial population — they differ only in their skills.
	with step("Arrange: capture the built agents; skip training and testing."):
		captured = {}
		monkeypatch.setattr(cartpole_ena_ablation, "agent_training", lambda **kwargs: captured.update(agents=kwargs["agents"]))
		monkeypatch.setattr(cartpole_ena_ablation, "agent_testing", lambda **_kwargs: None)
		combined = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
		twin = dict(combined, name="ENA-Combined-Twin")

	with step("Act: one replicate with ENA-Combined twice."):
		cartpole_ena_ablation.run_experiment(0, str(tmp_path), None, [combined, twin], include_random=False)

	with step("Assert: identical initial weights."):
		first, second = captured["agents"]
		for weights_a, weights_b in zip(first.population[0]["model"].get_weights(), second.population[0]["model"].get_weights()):
			np.testing.assert_array_equal(weights_a, weights_b)


def test_training_and_testing_reseed_before_each_agent(monkeypatch, tmp_path) -> None:
	"""Each agent's training and test phases start from the replicate's seed, whatever the agents before it consumed."""
	with step("Arrange: record seeding and play calls in order."):
		calls = []
		monkeypatch.setattr(cartpole_ena_ablation, "seed_experiment", lambda experiment_id: calls.append(("seed", experiment_id)))

		def _fake_play(agent, _worlds, _episodes, **_kwargs):
			calls.append(("play", agent.get_name()))
			return pd.DataFrame({"env_id": ["Mars"], "env_steps": [1], "score": [1.0]})

		monkeypatch.setattr(cartpole_ena_ablation, "play", _fake_play)
		for helper in ("calculate_paper_metrics", "calculate_compute_cost", "calculate_adaptation", "plot_academic_comparison"):
			monkeypatch.setattr(cartpole_ena_ablation, helper, lambda *args, **kwargs: None)
		agents = [_BudgetAgent("A", [1] * 5), _BudgetAgent("B", [1] * 5)]

	with step("Act: train and test replicate 4."):
		with open(tmp_path / "outputs.txt", "w", encoding="utf-8") as output_file:
			cartpole_ena_ablation.agent_training(agents, 4, str(tmp_path), output_file, None, ["W"] * 3)
			cartpole_ena_ablation.agent_testing(agents, 4, str(tmp_path), output_file, None, ["W"] * 3, test_worlds=["W"] * 5)

	with step("Assert: a reseed with the replicate's id right before every play."):
		assert calls == [("seed", 4), ("play", "A"), ("seed", 4), ("play", "B")] * 2


def test_a_variants_run_is_identical_with_or_without_other_agents_before_it(monkeypatch, tmp_path) -> None:
	"""A variant's run is byte-identical alone, after other agents, or under another name."""
	# End to end: replicate i of ENA-Combined logs byte-identical episodes alone, after other agents, and under
	# another name, so another study can reuse the ablation's run of the same configuration, paired by replicate,
	# instead of re-running it (cartpole_ena_plasticity_controller.py and cartpole_ena_compute.py rely on this).
	with step("Arrange: a tiny episode budget; ENA-Combined alone, and after Random and ENA-NoSkill."):
		monkeypatch.setattr(cartpole_ena_ablation, "TRAINING_EPISODES_ENA", [2, 2, 2])
		monkeypatch.setattr(cartpole_ena_ablation, "TESTING_EPISODES", [1, 1, 1, 1, 1])
		rows = {row["name"]: row for row in ABLATION_VARIANTS}
		alone, after_others = tmp_path / "alone", tmp_path / "after_others"

		renamed = tmp_path / "renamed"

	with step("Act: run replicate 2 alone, after other agents, and under another name (as another study would label it)."):
		cartpole_ena_ablation.run_experiment(2, str(alone), None, [rows["ENA-Combined"]], include_random=False)
		cartpole_ena_ablation.run_experiment(2, str(after_others), None, [rows["ENA-NoSkill"], rows["ENA-Combined"]], include_random=True)
		cartpole_ena_ablation.run_experiment(2, str(renamed), None, [{**rows["ENA-Combined"], "name": "Same-Config"}], include_random=False)

	with step("Assert: ENA-Combined's training and test logs are byte-identical in all three runs."):
		for phase in ("training", "testing"):
			log = f"live_{phase}_ENA-Combined.csv"
			assert (alone / log).read_bytes() == (after_others / log).read_bytes(), log
			assert (alone / log).read_bytes() == (renamed / f"live_{phase}_Same-Config.csv").read_bytes(), log


@pytest.mark.parametrize(
	"num_experiments, pending, workers, expected",
	[
		(1, 1, 6, "LAUNCHING 1 OF 1 ENA COMPUTE REPLICATES, 1 at a time"),
		(30, 24, 6, "LAUNCHING 24 OF 30 ENA COMPUTE REPLICATES (6 already complete), 6 at a time"),
	],
)
def test_launch_summary_states_what_actually_runs(num_experiments, pending, workers, expected) -> None:
	"""Never more parallel replicates than there are left to run, and resumed replicates are counted out."""
	with step("Act/Assert: the banner line."):
		assert cartpole_ena_ablation.launch_summary(num_experiments, pending, workers, "ENA COMPUTE") == expected


def test_study_ids_are_every_replicate_or_only_the_chosen_one() -> None:
	"""--replicate K runs only exp_K (one machine per replicate in the cloud); by default every replicate runs."""
	with step("Act/Assert: all 20, only exp_3, and an out-of-range choice."):
		assert cartpole_ena_ablation.study_ids(20) == list(range(20))
		assert cartpole_ena_ablation.study_ids(20, replicate=3) == [2]
		with pytest.raises(ValueError):
			cartpole_ena_ablation.study_ids(20, replicate=21)


def test_parse_args_takes_one_replicate_within_the_study() -> None:
	"""--replicate defaults to every replicate; it must name one of the study's replicates."""
	with step("Act/Assert: default, a choice, and one beyond --num-experiments."):
		assert cartpole_ena_ablation.parse_args([]).replicate is None
		assert cartpole_ena_ablation.parse_args(["--replicate", "3"]).replicate == 3
		with pytest.raises(SystemExit):
			cartpole_ena_ablation.parse_args(["--num-experiments", "2", "--replicate", "3"])


def _mark_replicate(experiment_id, output_dir, _queue, _variants, _include_random) -> None:
	"""Experiment stub: leave a marker in the replicate's directory."""
	with open(os.path.join(output_dir, "outputs.txt"), "w", encoding="utf-8") as outputs:
		outputs.write(f"replicate {experiment_id} --- Experiment {experiment_id + 1} Test Results ---\n")


def test_run_study_runs_only_the_chosen_replicate(monkeypatch, tmp_path) -> None:
	"""With replicate=3, the study runs replicate id 2 into exp_3 and nothing else (same seeding as a full run)."""
	with step("Arrange: worker threads instead of processes, so the stub experiment runs in-process."):
		monkeypatch.setattr(cartpole_ena_ablation.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)

	with step("Act: a 20-replicate study, replicate 3 only."):
		cartpole_ena_ablation.run_study(20, str(tmp_path), [], False, "TEST", experiment_fn=_mark_replicate, workers=1, replicate=3)

	with step("Assert: only exp_3 ran, as replicate id 2."):
		assert sorted(path.name for path in tmp_path.iterdir()) == ["exp_3"]
		assert (tmp_path / "exp_3" / "outputs.txt").read_text(encoding="utf-8").startswith("replicate 2 ")
