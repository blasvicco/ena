"""Tests for paper_results.py: every number the paper quotes is generated from the study outputs."""

# Libs imports
import math

import pandas as pd
import pytest

# App imports
from helpers import step
from tools.paper_results import (
	cohens_d,
	clear_absent_outputs,
	comparison_entries,
	family_entries,
	completed_replicates,
	compute_ratio_entries,
	candidate_share_entries,
	episode_count_entries,
	compute_rank_entries,
	controller_entries,
	controller_verdict,
	detectable_d,
	ALPHA,
	holm,
	paired_p,
	replicates_for_power,
	detection_entries,
	detection_table,
	format_p,
	parse_training_times,
	reliability_figure,
	slug,
	results_table,
	summary_entries,
	welch_ci,
	with_reference_rows,
	welch_p,
	write_macros,
)


def _results(rows: list, phase: str = "Test") -> dict:
	"""An aggregate_metrics-style {phase: [per-replicate DataFrames]} from (exp, agent, metrics) rows."""
	frames = [pd.DataFrame([{"Agent": agent, "Exp": exp, **metrics}]) for exp, agent, metrics in rows]
	return {phase: frames}


@pytest.mark.parametrize(
	"metric, expected",
	[
		("Reliability (%)", "rel"),
		("Stability Gap (↓)", "gap"),
		("Efficiency (Score/s)", "eff"),
		("Mars-Light Avg", "Mars-Light"),
		("Total Candidate Env-Steps", "candsteps"),
		("Jupiter-Heavy:steps", "Jupiter-Heavy:steps"),
	],
)
def test_slug_gives_short_stable_metric_keys(metric, expected) -> None:
	"""Macro keys must be short and free of characters LaTeX can't put in a control-sequence name."""
	with step("Act/Assert: the slug."):
		assert slug(metric) == expected


@pytest.mark.parametrize("p_value, expected", [(0.0004, "<0.001"), (0.0123, "0.012"), (0.5, "0.500"), (float("nan"), "n/a")])
def test_format_p(p_value, expected) -> None:
	"""p-values below 0.001 are bounded, others shown to 3 decimals."""
	with step("Act/Assert: formatted."):
		assert format_p(p_value) == expected


def test_welch_p_separates_clear_differences_and_handles_degenerate_samples() -> None:
	"""Welch's t-test: tiny p for disjoint samples, large for identical ones; 1 for two equal constants; NaN if n < 2."""
	with step("Act/Assert: the four cases."):
		assert welch_p([90, 91, 92, 93], [40, 41, 42, 43]) < 0.001
		assert welch_p([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
		assert welch_p([5, 5, 5], [5, 5, 5]) == 1.0
		assert math.isnan(welch_p([1], [2, 3]))


def test_parse_training_times_reads_both_log_formats(tmp_path) -> None:
	"""Newer logs add env-steps after the seconds; older ones don't."""
	with step("Arrange: an outputs.txt with both formats."):
		path = tmp_path / "outputs.txt"
		path.write_text(
			"  --- Training Performance vs Time ---\n  ENA-Combined: 3696.76 seconds, 7,519,387 env-steps\n  Random: 217.83 seconds\n\nTraining:\n",
			encoding="utf-8",
		)

	with step("Act/Assert: seconds and env-steps per agent."):
		assert parse_training_times(str(path)) == {"ENA-Combined": (3696.76, 7519387), "Random": (217.83, None)}


def test_summary_entries_give_mean_std_and_replicate_count() -> None:
	"""One mean/std pair per study/agent/phase/metric, plus the study's replicate count."""
	with step("Arrange: two replicates of two agents."):
		results = _results(
			[
				("exp_1", "ENA", {"Reliability (%)": 90.0, "Mars-Light Avg": 480.0}),
				("exp_1", "PPO", {"Reliability (%)": 70.0, "Mars-Light Avg": 490.0}),
				("exp_2", "ENA", {"Reliability (%)": 94.0, "Mars-Light Avg": 470.0}),
				("exp_2", "PPO", {"Reliability (%)": 80.0, "Mars-Light Avg": 500.0}),
			]
		)

	with step("Act: summarize."):
		entries = summary_entries("challenge", results)

	with step("Assert: formatted means, sample std, and n."):
		assert entries["challenge/n"] == "2"
		assert entries["challenge/ENA/test/rel/mean"] == "92.0"
		assert entries["challenge/ENA/test/rel/std"] == "2.8"
		assert entries["challenge/PPO/test/Mars-Light/mean"] == "495.0"


def test_comparison_entries_give_welch_p_and_difference() -> None:
	"""For each pair and metric: the p-value and the difference of means (b - a)."""
	with step("Arrange: clearly different reliabilities."):
		rows = [(f"exp_{index}", "ENA", {"Reliability (%)": 90.0 + index % 3}) for index in range(10)]
		rows += [(f"exp_{index}", "PPO", {"Reliability (%)": 70.0 + index % 3}) for index in range(10)]
		results = _results(rows)

	with step("Act: compare PPO -> ENA on reliability."):
		entries = comparison_entries("challenge", results, "PPO", results, "ENA", "Test", ["Reliability (%)"])

	with step("Assert: keyed by both agents, the metric, p and diff."):
		assert entries["challenge/PPO-vs-ENA/test/rel/p"] == "$p<0.001$"
		assert entries["challenge/PPO-vs-ENA/test/rel/diff"] == "+20.0"


def test_write_macros_writes_one_resdef_per_entry_sorted(tmp_path) -> None:
	"""A generated, do-not-edit file of \\resdef{key}{value} lines, sorted for stable diffs."""
	with step("Act: write two entries."):
		path = tmp_path / "results.tex"
		write_macros({"b/key": "2", "a/key": "<0.001"}, str(path))

	with step("Assert: header and sorted lines; '<' made safe for text mode."):
		text = path.read_text(encoding="utf-8")
		assert text.startswith("% GENERATED")
		assert text.index("\\resdef{a/key}{\\textless{}0.001}") < text.index("\\resdef{b/key}{2}")


def test_results_table_has_one_row_per_agent_with_mean_and_std() -> None:
	"""Appendix table: per-environment mean return and reliability mean +- std, agents in the given order."""
	with step("Arrange: one agent, two replicates."):
		results = _results(
			[
				("exp_1", "ENA-Combined", {"Mars-Light Avg": 480.0, "Reliability (%)": 90.0}),
				("exp_2", "ENA-Combined", {"Mars-Light Avg": 470.0, "Reliability (%)": 94.0}),
			]
		)

	with step("Act: build the table."):
		columns = [("Mars-Light Avg", "Mars-Light", False), ("Reliability (%)", "Reliability (\\%)", True)]
		table = results_table(results, "Test", ["ENA-Combined"], columns, caption="Test", label="tab:test")

	with step("Assert: header, the row, the label, and a width cap so it never overflows the column."):
		assert "Mars-Light" in table
		assert "ENA-Combined & 475.0 & 92.0 $\\pm$ 2.8 \\\\" in table
		assert "\\label{tab:test}" in table
		assert "\\begin{adjustbox}{max width=\\columnwidth}" in table


def test_reliability_figure_writes_a_vector_figure_with_one_box_per_agent(tmp_path, monkeypatch) -> None:
	"""Per-replicate test reliability, one box (plus the replicates as dots) per agent, in the given order."""
	with step("Arrange: three replicates of two agents; keep the figure instead of closing it."):
		rows = [(f"exp_{index}", agent, {"Reliability (%)": value + index}) for index in range(3) for agent, value in (("A", 30.0), ("B", 90.0))]
		figures = []
		monkeypatch.setattr("tools.paper_results.plt.close", figures.append)

	with step("Act: draw it."):
		path = reliability_figure(_results(rows), ["A", "B"], str(tmp_path / "rel.pdf"))

	with step("Assert: a non-empty PDF; the x labels are the agents in order."):
		assert (tmp_path / "rel.pdf").stat().st_size > 0 and path.endswith("rel.pdf")
		assert [label.get_text() for label in figures[0].axes[0].get_xticklabels()] == ["A", "B"]


def _detection_summary():
	"""A detection_stats.summarize()-style frame: one agent, a training and a test phase."""
	rows = [
		("ENA-Combined", "train", "Earth-Standard", "unrecognized", 0.0, 0.0, 6),
		("ENA-Combined", "train", "Earth-Standard", "changes", 1.5, 0.5, 6),
		("ENA-Combined", "train", "Earth-Standard", "first_change", 0.0, 0.0, 6),
		("ENA-Combined", "train", "Earth-Standard", "niches", 1.5, 0.5, 6),
		("ENA-Combined", "test", "Brown-dwarf", "unrecognized", 95.7, 8.7, 6),
		("ENA-Combined", "test", "Brown-dwarf", "changes", 15.8, 21.4, 6),
	]
	return pd.DataFrame(rows, columns=["agent", "phase", "env_id", "metric", "mean", "std", "n"])


def test_detection_entries_give_mean_and_std_per_agent_phase_env_metric() -> None:
	"""Keys study/agent/detect/phase/env/metric/{mean,std}, values to one decimal."""
	with step("Act: build the entries."):
		entries = detection_entries("ablation", _detection_summary())

	with step("Assert: Brown-dwarf's unrecognized share and Earth's niches, mean and std."):
		assert entries["ablation/ENA-Combined/detect/test/Brown-dwarf/unrec/mean"] == "95.7"
		assert entries["ablation/ENA-Combined/detect/test/Brown-dwarf/unrec/std"] == "8.7"
		assert entries["ablation/ENA-Combined/detect/train/Earth-Standard/niches/mean"] == "1.5"
		assert entries["ablation/ENA-Combined/detect/train/Earth-Standard/firstchange/mean"] == "0.0"


def test_detection_table_has_one_row_per_agent_phase_and_environment() -> None:
	"""Rows: training phases then test phases; metrics a phase doesn't have (niches at test, no change) print as --."""
	with step("Act: render the table."):
		table = detection_table(_detection_summary(), ["ENA-Combined"], "Recognition", "tab:detection")

	with step("Assert: both rows, with mean ± std, -- where a value is missing, and the label."):
		assert "ENA-Combined & Training & Earth & 0.0 $\\pm$ 0.0 & 1.5 $\\pm$ 0.5 & 0.0 $\\pm$ 0.0 & 1.5 $\\pm$ 0.5 \\\\" in table
		assert "ENA-Combined & Test & Brown & 95.7 $\\pm$ 8.7 & 15.8 $\\pm$ 21.4 & -- & -- \\\\" in table
		assert table.index("Training & Earth") < table.index("Test & Brown")
		assert "\\label{tab:detection}" in table


def test_welch_ci_brackets_the_difference_of_means() -> None:
	"""95% interval of mean(b) - mean(a): centred on the difference, excluding 0 for clearly separated samples."""
	with step("Act: two samples 20 apart."):
		low, high = welch_ci([70.0, 71.0, 72.0, 70.0, 71.0], [90.0, 91.0, 92.0, 90.0, 91.0])

	with step("Assert: the interval contains +20 and not 0."):
		assert low < 20.0 < high
		assert low > 0


def test_cohens_d_is_the_difference_over_the_average_standard_deviation() -> None:
	"""d = (mean b - mean a) / sqrt((var a + var b) / 2)."""
	with step("Act/Assert: samples with SD 1 whose means differ by 2."):
		assert cohens_d([0.0, 1.0, 2.0], [2.0, 3.0, 4.0]) == pytest.approx(2.0)


def test_power_helpers_match_the_power_analysis_that_fixed_the_replicate_count() -> None:
	"""At n = 30, alpha 0.05/7 and power 0.8 the smallest detectable d is 0.94 (docs/PLAN_ena_review.md); and back."""
	with step("Act/Assert: detectable effect at 30, and replicates needed for it."):
		assert detectable_d(30, 0.05 / 7) == pytest.approx(0.94, abs=0.01)
		assert replicates_for_power(0.94, 0.05 / 7) in (30, 31)
		assert replicates_for_power(0.0, 0.05) is None


def test_holm_adjusts_ordered_p_values_and_keeps_them_monotone() -> None:
	"""Holm: k-th smallest p times (m - k + 1), running maximum, capped at 1; original order kept."""
	with step("Act/Assert: three p-values."):
		assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
		assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_comparison_entries_also_give_the_interval_effect_size_and_replicates_needed() -> None:
	"""Each comparison carries what a reader needs to judge it: CI, d, and n for 80% power at the family's alpha."""
	with step("Arrange: clearly different reliabilities."):
		rows = [(f"exp_{index}", "ENA", {"Reliability (%)": 90.0 + index % 3}) for index in range(10)]
		rows += [(f"exp_{index}", "PPO", {"Reliability (%)": 70.0 + index % 3}) for index in range(10)]
		results = _results(rows)

	with step("Act: compare PPO -> ENA."):
		entries = comparison_entries("challenge", results, "PPO", results, "ENA", "Test", ["Reliability (%)"])

	with step("Assert: interval around +20, a large d, and few replicates needed."):
		key = "challenge/PPO-vs-ENA/test/rel"
		assert float(entries[f"{key}/ci/low"]) < 20.0 < float(entries[f"{key}/ci/high"])
		assert float(entries[f"{key}/d"]) > 5
		assert int(entries[f"{key}/nreq"]) <= 5


def test_ablation_family_entries_give_holm_p_for_the_planned_comparisons_and_the_detectable_effect() -> None:
	"""The planned comparisons are one family: Holm-adjusted p per pair, plus the smallest d detectable at this n."""
	with step("Arrange: 30 replicates; Combined far above NoSkill, HoF equal to NoSkill."):
		rows = []
		for index in range(30):
			rows.append((f"exp_{index}", "ENA-NoSkill", {"Reliability (%)": 40.0 + index % 5}))
			rows.append((f"exp_{index}", "ENA-HoF", {"Reliability (%)": 40.0 + (index + 2) % 5}))
		results = _results(rows)

	with step("Act: the family entries for two pairs."):
		entries = family_entries("ablation", results, [("ENA-NoSkill", "ENA-HoF"), ("ENA-NoSkill", "ENA-Missing")])

	with step("Assert: a Holm p for the pair with data, none for the missing one; the detectable d at n = 30 for a family of 2."):
		assert entries["ablation/ENA-NoSkill-vs-ENA-HoF/test/rel/pholm"] == "$p=1.000$"
		assert not [key for key in entries if "ENA-Missing" in key]
		assert entries["ablation/mde"] == f"{detectable_d(30, 0.05 / 2):.2f}"
		assert entries["ablation/family"] == "2"


def test_paired_p_matches_replicates_by_experiment_and_drops_unmatched_ones() -> None:
	"""Replicate i of each agent is the same seeded run up to the agent's difference: compare within replicate."""
	with step("Arrange: b = a + 1 in every shared replicate (noisy a), plus one unmatched replicate."):
		pooled_a = pd.DataFrame({"Agent": "A", "Exp": ["exp_1", "exp_2", "exp_3", "exp_4"], "Reliability (%)": [10.0, 50.0, 30.0, 90.0]})
		pooled_b = pd.DataFrame({"Agent": "B", "Exp": ["exp_1", "exp_2", "exp_3", "exp_9"], "Reliability (%)": [11.2, 50.9, 31.1, 0.0]})

	with step("Act/Assert: a constant within-replicate gain is significant although a's spread is huge; n = 3."):
		p_value, pairs = paired_p(pooled_a, "A", pooled_b, "B", "Reliability (%)")
		assert pairs == 3
		assert p_value < 0.01


@pytest.mark.parametrize(
	"rel_diff, rel_p, steps_diff, steps_p, qualifies",
	[
		(+6.0, 0.01, 0.0, 1.0, True),  # (a) significantly higher reliability
		(+6.0, 0.20, -1e6, 0.001, True),  # (b) not significantly lower, significantly cheaper
		(-6.0, 0.01, -1e6, 0.001, False),  # cheaper but significantly worse
		(+6.0, 0.20, +1e6, 0.001, False),  # neither
	],
)
def test_controller_verdict_applies_the_pre_registered_rule(rel_diff, rel_p, steps_diff, steps_p, qualifies) -> None:
	"""Adopt an alternative if reliability is significantly higher, or not significantly lower with significantly fewer candidate steps."""
	with step("Act: one alternative's Holm-adjusted outcomes."):
		verdict = controller_verdict({"Alt": {"rel": (rel_diff, rel_p), "candsteps": (steps_diff, steps_p), "rel_mean": 80.0}})

	with step("Assert: adopted only when the rule holds; otherwise the linear ramp stays."):
		assert verdict == ("Alt" if qualifies else "linear ramp")


def test_controller_verdict_prefers_the_more_reliable_of_two_qualifying_alternatives() -> None:
	"""Both qualify: the one with the higher mean test reliability is adopted."""
	with step("Act/Assert: B is more reliable."):
		outcomes = {
			"A": {"rel": (+5.0, 0.01), "candsteps": (0.0, 1.0), "rel_mean": 85.0},
			"B": {"rel": (+7.0, 0.01), "candsteps": (0.0, 1.0), "rel_mean": 87.0},
		}
		assert controller_verdict(outcomes) == "B"


def test_controller_entries_give_holm_and_paired_p_per_alternative_and_the_verdict() -> None:
	"""Holm over the alternatives per metric, a paired p as robustness, and the generated verdict."""
	with step("Arrange: 10 paired replicates; Alt is 10 points more reliable and cheaper."):
		reference = _results([(f"exp_{index}", "Ref", {"Reliability (%)": 70.0 + index % 3}) for index in range(10)])
		reference["Compute Cost"] = [
			pd.DataFrame([{"Agent": "Ref", "Exp": f"exp_{index}", "Total Candidate Env-Steps": 1000.0 + index}]) for index in range(10)
		]
		alternative = _results([(f"exp_{index}", "Alt", {"Reliability (%)": 80.0 + index % 3}) for index in range(10)])
		alternative["Compute Cost"] = [
			pd.DataFrame([{"Agent": "Alt", "Exp": f"exp_{index}", "Total Candidate Env-Steps": 900.0 + index}]) for index in range(10)
		]

	with step("Act: build the entries for the single alternative."):
		entries = controller_entries(reference, "Ref", alternative, ["Alt"])

	with step("Assert: Holm and paired p for reliability, the family size, the verdict."):
		assert entries["controller/Ref-vs-Alt/test/rel/pholm"] == "$p<0.001$"
		assert entries["controller/Ref-vs-Alt/test/rel/ppaired"] == "$p<0.001$"
		assert entries["controller/family"] == "1"
		assert entries["controller/verdict"] == "Alt"


def test_clear_absent_outputs_replaces_a_missing_studys_tables_and_removes_its_figures(tmp_path) -> None:
	"""A study that has not run must not leave numbers from an older run in the paper: placeholder tables, no figures."""
	with step("Arrange: stale challenge and ablation outputs from an earlier run."):
		generated, images = tmp_path / "generated", tmp_path / "images"
		generated.mkdir()
		images.mkdir()
		for name in ("table_challenge_test.tex", "table_challenge_adapt.tex", "table_ablation_test.tex"):
			(generated / name).write_text("stale numbers", encoding="utf-8")
		for name in ("challenge_reliability.pdf", "challenge_test_curves.pdf", "ablation_reliability.pdf"):
			(images / name).write_bytes(b"stale")

	with step("Act: only the ablation is present."):
		clear_absent_outputs({"ablation"}, str(generated), str(images))

	with step("Assert: challenge tables are placeholders, challenge figures gone; the ablation's files untouched."):
		assert "\\todo" in (generated / "table_challenge_test.tex").read_text(encoding="utf-8")
		assert "\\label{tab:challenge_test}" in (generated / "table_challenge_test.tex").read_text(encoding="utf-8")
		assert "stale" not in (generated / "table_challenge_adapt.tex").read_text(encoding="utf-8")
		assert not (images / "challenge_reliability.pdf").exists()
		assert not (images / "challenge_test_curves.pdf").exists()
		assert (generated / "table_ablation_test.tex").read_text(encoding="utf-8") == "stale numbers"
		assert (images / "ablation_reliability.pdf").exists()


def test_with_reference_rows_adds_another_studys_rows_for_the_same_replicates_only() -> None:
	"""A study reusing another's runs of the same configuration takes those agents' rows for its own replicates."""
	with step("Arrange: the study ran exp_1; the reference study has exp_1 and exp_2 of two agents."):
		study = {"Test": [pd.DataFrame([{"Agent": "Full", "Exp": "exp_1", "Reliability (%)": 90.0}])]}
		reference = {
			"Test": [
				pd.DataFrame([{"Agent": agent, "Exp": exp, "Reliability (%)": 80.0} for agent in ("Ramp", "Other")]) for exp in ("exp_1", "exp_2")
			]
		}
		for frame, exp in zip(reference["Test"], ("exp_1", "exp_2")):
			frame["Exp"] = exp

	with step("Act: merge Ramp's rows."):
		merged = with_reference_rows(study, reference, ["Ramp"])

	with step("Assert: Full and Ramp from exp_1 only; Other left out."):
		pooled = pd.concat(merged["Test"])
		assert sorted(zip(pooled["Agent"], pooled["Exp"])) == [("Full", "exp_1"), ("Ramp", "exp_1")]


def test_compute_ratio_entries_divide_candidate_steps_of_the_same_replicate() -> None:
	"""How many times more candidate steps full search spends than the ramp, and the ramp than zero."""
	with step("Arrange: one replicate's compute-cost rows."):
		results = {
			"Compute Cost": [
				pd.DataFrame(
					[
						{"Agent": "Full", "Exp": "exp_1", "Total Candidate Env-Steps": "48,000,000"},
						{"Agent": "Ramp", "Exp": "exp_1", "Total Candidate Env-Steps": "4,000,000"},
						{"Agent": "Zero", "Exp": "exp_1", "Total Candidate Env-Steps": "30,000"},
					]
				)
			]
		}

	with step("Act/Assert: 12x and 133x."):
		entries = compute_ratio_entries(results, full="Full", ramp="Ramp", zero="Zero")
		assert entries["compute/ratio/full-ramp"] == "12"
		assert entries["compute/ratio/ramp-zero"] == "133"


def test_compute_rank_entries_place_the_single_full_run_within_the_rams_replicates() -> None:
	"""Descriptive only: how many of the ramp's replicates reached at least full search's reliability."""
	with step("Arrange: full search at 90; ramp replicates at 70, 85, 90, 95."):
		full = {"Test": [pd.DataFrame([{"Agent": "Full", "Exp": "exp_1", "Reliability (%)": 90.0}])]}
		ramp = {
			"Test": [
				pd.DataFrame([{"Agent": "Ramp", "Exp": f"exp_{index}", "Reliability (%)": value}])
				for index, value in enumerate((70.0, 85.0, 90.0, 95.0))
			]
		}

	with step("Act/Assert: 2 of 4 ramp replicates reached 90 or more."):
		entries = compute_rank_entries(full, "Full", ramp, "Ramp")
		assert entries["compute/rank/atleast"] == "2"
		assert entries["compute/rank/of"] == "4"


def test_completed_replicates_counts_only_replicates_that_wrote_their_test_results(tmp_path) -> None:
	"""A study whose run has started but finished nothing must count as not run (the generator then writes placeholders)."""
	with step("Arrange: one finished replicate, one still training, one without outputs yet."):
		(tmp_path / "exp_1").mkdir()
		(tmp_path / "exp_1" / "outputs.txt").write_text("...\n  --- Experiment 1 Test Results ---\n", encoding="utf-8")
		(tmp_path / "exp_2").mkdir()
		(tmp_path / "exp_2" / "outputs.txt").write_text("Experiment 2 Log\n", encoding="utf-8")
		(tmp_path / "exp_3").mkdir()

	with step("Act/Assert: one."):
		assert completed_replicates(str(tmp_path)) == 1


def test_candidate_share_entries_give_the_share_of_training_steps_spent_on_candidates_per_replicate() -> None:
	"""Per replicate candidate steps / training steps, then mean and std across replicates, in percent."""
	with step("Arrange: two replicates spending 80% and 90% of their training steps on candidates."):
		results = {
			"Compute Cost": [
				pd.DataFrame([{"Agent": "ENA", "Exp": exp, "Total Candidate Env-Steps": candidates}])
				for exp, candidates in (("exp_1", "6,000,000"), ("exp_2", "6,750,000"))
			],
			"Adaptation": [pd.DataFrame([{"Agent": "ENA", "Exp": exp, "Train-Env-Steps": "7,500,000"}]) for exp in ("exp_1", "exp_2")],
		}

	with step("Act/Assert: mean 85.0, std 7.1."):
		entries = candidate_share_entries("challenge", results, "ENA")
		assert entries["challenge/ENA/candshare/mean"] == "85.0"
		assert entries["challenge/ENA/candshare/std"] == "7.1"


def test_episode_count_entries_average_episodes_per_environment_over_complete_replicates(tmp_path) -> None:
	"""Mean episodes per training environment per agent, from the live training logs of finished replicates."""
	with step("Arrange: two finished replicates of one agent (2 environments of 3 and 5 episodes, then 5 and 7), one unfinished."):
		for name, counts, done in (("exp_1", (3, 5), True), ("exp_2", (5, 7), True), ("exp_3", (100, 100), False)):
			exp_dir = tmp_path / name
			exp_dir.mkdir()
			envs = ["A"] * counts[0] + ["B"] * counts[1]
			pd.DataFrame({"env_id": envs, "score": 1.0}).to_csv(exp_dir / "live_training_ENA.csv", index=False)
			(exp_dir / "outputs.txt").write_text("--- Test Results ---" if done else "log", encoding="utf-8")

	with step("Act/Assert: (3 + 5 + 5 + 7) / 4 = 5 episodes per environment."):
		assert episode_count_entries("challenge", str(tmp_path), ["ENA"]) == {"challenge/ENA/episodes/perenv": "5"}


def _challenge(reliabilities: dict, replicates: int) -> dict:
	"""A challenge-like Test phase: per agent, a constant-plus-noise reliability over the replicates."""
	rows = []
	for agent, mean in reliabilities.items():
		rows += [(f"exp_{index}", agent, {"Reliability (%)": mean + (index % 5) - 2}) for index in range(replicates)]
	return _results(rows)


def test_family_entries_correct_the_comparisons_against_each_baseline_as_one_holm_family() -> None:
	"""The comparison: ENA against each of three baselines on 50 replicates, Holm over the three."""
	with step("Act: ENA far above all three baselines."):
		entries = family_entries("challenge", _challenge({"A": 50.0, "B": 60.0, "C": 70.0, "ENA": 90.0}, 50), [(name, "ENA") for name in "ABC"])

	with step("Assert: a Holm p per baseline, the family size, and the detectable effect at n = 50."):
		assert all(entries[f"challenge/{name}-vs-ENA/test/rel/pholm"] == "$p<0.001$" for name in "ABC")
		assert entries["challenge/family"] == "3"
		assert entries["challenge/mde"] == f"{detectable_d(50, ALPHA / 3):.2f}"
