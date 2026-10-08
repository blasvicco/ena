"""Tests for the pure, chart-independent data transforms in tools/metrics.py."""

# Libs imports
import matplotlib.pyplot as plt
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.metrics import (
	_build_transition_matrix,
	_classify_niche_events,
	_niche_env_labels,
	_qualitative_colors,
	calculate_compute_cost,
	estimate_eval_cost,
	plot_academic_comparison,
	plot_specialist_transitions,
)


@pytest.mark.parametrize("distinct_color_count", [3, 4, 5, 8, 14])
def test_qualitative_colors_returns_n_distinct_colors(distinct_color_count: int) -> None:
	"""C3: color assignment must scale to any n without wraparound/reuse."""
	with step("Act: build a color palette of the requested size."):
		colors = _qualitative_colors(distinct_color_count)

	with step("Assert: every color is unique and the palette has the requested size."):
		assert len(colors) == distinct_color_count
		assert len(set(colors)) == distinct_color_count


@pytest.mark.parametrize(
	("plasticity", "expected_num_to_eval", "expected_eval_step_cap"),
	[
		pytest.param(0.0, 2, 10.0, id="zero-plasticity-floors-at-the-minimum"),
		pytest.param(1.0, 20, 100.0, id="full-plasticity-reaches-the-maximum"),
	],
)
def test_estimate_eval_cost_scales_with_plasticity(plasticity: float, expected_num_to_eval: int, expected_eval_step_cap: float) -> None:
	"""Mirrors ena.py train()'s formula exactly — same floors/ceilings at 0.0/1.0 plasticity."""
	with step("Act: estimate eval cost at the given plasticity."):
		num_to_eval, eval_step_cap = estimate_eval_cost(plasticity, pop_size=20, min_eval_steps=10, max_eval_steps=100)

	with step("Assert: it matches ena.py train()'s own num_to_eval/eval_step_cap at this plasticity."):
		assert num_to_eval == expected_num_to_eval
		assert eval_step_cap == pytest.approx(expected_eval_step_cap)


def test_calculate_compute_cost_summarizes_logged_columns() -> None:
	"""Aggregates the already-logged num_to_eval/eval_step_cap/candidate_env_steps columns."""
	with step("Arrange: a training history with logged compute-cost columns."):
		history = pd.DataFrame(
			{
				"num_to_eval": [2, 10, 20],
				"eval_step_cap": [10.0, 50.0, 100.0],
				"candidate_env_steps": [20.0, 500.0, 2000.0],
			}
		)

	with step("Act: summarize compute cost."):
		cost_df = calculate_compute_cost([history], ["ENA-Plasticity"])

	with step("Assert: the summary reports the mean depth/breadth and total env-steps."):
		row = cost_df.iloc[0]
		assert row["Agent"] == "ENA-Plasticity"
		assert row["Avg Eval Depth"] == "53.3"
		assert row["Avg Candidates/Gen"] == "10.7"
		assert row["Total Candidate Env-Steps"] == "2,520"


def test_calculate_compute_cost_skips_agents_without_the_column() -> None:
	"""An agent with no compute-cost columns (e.g. RandomAgent) must be excluded, not error."""
	with step("Arrange: a history with no compute-cost columns at all."):
		history = pd.DataFrame({"episode": [1, 2], "score": [1.0, 1.0]})

	with step("Act: summarize compute cost."):
		cost_df = calculate_compute_cost([history], ["Random"])

	with step("Assert: the agent is skipped entirely — no row, no error."):
		assert cost_df.empty


@pytest.mark.parametrize(
	("series", "expected_events"),
	[
		pytest.param(
			# niche 0 seen first; 1 is new; back to 0 is a switch to a known id;
			# 2 is new; back to 0 is again a switch to a known id.
			pd.Series([0, 0, 1, 1, 0, 2, 0]),
			[(2, 1, "new"), (4, 0, "switch"), (5, 2, "new"), (6, 0, "switch")],
			id="distinguishes-new-from-switch",
		),
		pytest.param(
			pd.Series([0, -1, float("nan"), 0, 1]),
			[(4, 1, "new")],
			id="ignores-missing-and-unresolved",
		),
	],
)
def test_classify_niche_events(series: pd.Series, expected_events: list) -> None:
	"""C1b: 'new' = never-seen id; 'switch' = a different, already-known id;
	-1/NaN mean 'no niche resolved this episode' and must not count as events."""
	with step("Act: classify the niche-id series into new/switch events."):
		events = _classify_niche_events(series)

	with step("Assert: the emitted events match the expected (index, niche_id, kind) tuples."):
		assert events == expected_events


def test_classify_niche_events_seeds_already_known_niches_as_switches() -> None:
	"""A niche established before this series began (e.g. from training, for a test-phase
	chart) must classify as a 'switch' the first time it appears here, not a spurious 'new
	niche' — this function has no memory of anything outside the series it's given unless
	told what's already known."""
	with step("Arrange: a series whose first value is a pre-known niche, seeded as such."):
		series = pd.Series([5, 5, 7, 5])

	with step("Act: classify with known_niche_ids seeded to include niche 5."):
		events = _classify_niche_events(series, known_niche_ids={5})

	with step("Assert: niche 5 never fires a 'new' event; niche 7 (genuinely unseen) still does."):
		assert events == [(2, 7, "new"), (3, 5, "switch")]


def test_build_transition_matrix_counts_consecutive_changes() -> None:
	"""C4: every consecutive value change is an edge, regardless of new-vs-switch."""
	with step("Arrange: build a niche-id series with several consecutive changes."):
		series = pd.Series([0, 0, 1, 0, 2, 2, 0, 3, 0])

	with step("Act: build the transition matrix."):
		counts = _build_transition_matrix(series)

	with step("Assert: every consecutive (from, to) change is counted as one edge."):
		assert counts == {
			(0, 1): 1,
			(1, 0): 1,
			(0, 2): 1,
			(2, 0): 1,
			(0, 3): 1,
			(3, 0): 1,
		}


def test_build_transition_matrix_surfaces_cross_environment_transitions() -> None:
	"""A transition between niches belonging to two DIFFERENT true environments must
	show up as its own (from, to) cell — the exact pattern the transition matrix is
	meant to reveal (worse than mere within-env fragmentation)."""
	with step("Arrange: build a DataFrame where niches 0/1 share an env and niche 2 is another."):
		# niches 0 and 1 both belong to "Mars-Light" (fragmentation); niche 2 belongs to
		# "Jupiter-Heavy" — the 1 -> 2 switch is a genuine cross-environment transition.
		dataframe = pd.DataFrame(
			{
				"detected_env_id": [0, 0, 1, 1, 2, 2],
				"gladiator_env": [
					"Mars-Light",
					"Mars-Light",
					"Mars-Light",
					"Mars-Light",
					"Jupiter-Heavy",
					"Jupiter-Heavy",
				],
			}
		)

	with step("Act: build the transition matrix and resolve each niche's true environment label."):
		counts = _build_transition_matrix(dataframe["detected_env_id"])
		labels = _niche_env_labels(dataframe)

	with step("Assert: the 1 -> 2 cell crosses a true-environment boundary."):
		assert counts == {(0, 1): 1, (1, 2): 1}
		assert labels == {0: "Mars-Light", 1: "Mars-Light", 2: "Jupiter-Heavy"}
		assert labels[1] != labels[2]  # the (1, 2) cell crosses a true-environment boundary


def test_plot_academic_comparison_niche_markers_match_their_legend_swatch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""C0: every 'New Niche' marker and every 'Niche Switch' marker must share one fixed
	color per kind, matching the legend swatch — regardless of how many distinct niche ids
	are involved (previously colored per-niche-id while the legend only showed one swatch)."""
	with step(
		"Arrange: a history whose detected_env_id produces 'new' events for two "
		"different niche ids and 'switch' events for another, and prevent the figure "
		"from being closed so its artists stay inspectable after the call."
	):
		monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
		dataframe = pd.DataFrame(
			{
				"episode": range(7),
				"score": [100.0, 110.0, 90.0, 95.0, 105.0, 98.0, 102.0],
				"env_id": ["Env-A"] * 7,
				"detected_env_id": [0, 0, 1, 1, 0, 2, 0],
			}
		)

	with step("Act: render the comparison chart."):
		plot_academic_comparison([dataframe], ["Agent"], save_path=str(tmp_path / "chart.png"))

	with step("Assert: every 'new'-kind marker and every 'switch'-kind marker uses one fixed color each, matching what the legend actually shows."):
		axis = plt.gcf().axes[0]
		new_lines = [line for line in axis.lines if line.get_marker() == "o" and line.get_linestyle() == "None"]
		switch_lines = [line for line in axis.lines if line.get_marker() == "x"]

		assert len(new_lines) == 2  # niches 1 and 2 both provisioned as "new"
		assert len(switch_lines) == 2  # niche 0 reclassified as "switch" twice
		assert len({line.get_color() for line in new_lines}) == 1
		assert len({line.get_color() for line in switch_lines}) == 1

		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert "New Niche" in legend_labels
		assert "Niche Switch" in legend_labels
		plt.close(axis.figure)


def test_plot_academic_comparison_seeds_known_niches_to_avoid_spurious_new_niche_markers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""A test-phase chart must not show 'New Niche' for a niche that was actually
	established during training — known_niche_ids_by_history seeds each history's
	already-known niches, so their first appearance here classifies as a 'switch'."""
	with step("Arrange: the same history as above, but with every niche it contains pre-declared as already known (as if all came from training)."):
		monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
		dataframe = pd.DataFrame(
			{
				"episode": range(7),
				"score": [100.0, 110.0, 90.0, 95.0, 105.0, 98.0, 102.0],
				"env_id": ["Env-A"] * 7,
				"detected_env_id": [0, 0, 1, 1, 0, 2, 0],
			}
		)

	with step("Act: render the chart with known_niche_ids_by_history seeding all three niches."):
		plot_academic_comparison(
			[dataframe],
			["Agent"],
			save_path=str(tmp_path / "chart.png"),
			known_niche_ids_by_history=[{0, 1, 2}],
		)

	with step("Assert: no 'new'-kind marker was drawn, and the legend has no 'New Niche' entry."):
		axis = plt.gcf().axes[0]
		new_lines = [line for line in axis.lines if line.get_marker() == "o" and line.get_linestyle() == "None"]
		assert not new_lines

		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert "New Niche" not in legend_labels
		assert "Niche Switch" in legend_labels
		plt.close(axis.figure)


def test_plot_specialist_transitions_explains_the_niche_ring_in_the_legend() -> None:
	"""The outer ring around a gladiator marker encodes its recognized niche ID (distinct
	from the fill color, which groups by ground-truth env name) — since ring colors are
	per-niche and change every run, a single generic legend entry explains what the ring
	means rather than one entry per color."""
	with step("Arrange: a history with two gladiators sharing an env name but different niches."):
		dataframe = pd.DataFrame(
			{
				"env_id": ["Mars-Light"] * 4,
				"episode": range(4),
				"specialist_id": [1, 2, 1, 2],
				"is_gladiator": [True, True, True, True],
				"gladiator_env": ["Mars-Light"] * 4,
				"internal_env_id": [0, 1, 0, 1],
			}
		)

	with step("Act: render the specialist transitions chart."):
		plot_specialist_transitions(dataframe, pop_size=5)

	with step("Assert: the legend includes an entry explaining the niche-ID ring."):
		axis = plt.gcf().axes[0]
		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert any("Recognized niche id" in label for label in legend_labels)
		plt.close(axis.figure)


def test_plot_specialist_transitions_omits_ring_legend_entry_without_niche_data() -> None:
	"""No internal_env_id data (or no gladiators) means no rings are drawn — the legend
	must not claim a ring meaning that isn't actually on the chart."""
	with step("Arrange: a history with no gladiators at all."):
		dataframe = pd.DataFrame(
			{
				"env_id": ["Mars-Light"] * 3,
				"episode": range(3),
				"specialist_id": [1, 2, 3],
				"is_gladiator": [False, False, False],
				"gladiator_env": [None, None, None],
			}
		)

	with step("Act: render the specialist transitions chart."):
		plot_specialist_transitions(dataframe, pop_size=5, save_path=None)

	with step("Assert: the legend has no niche-ring explanation."):
		axis = plt.gcf().axes[0]
		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert not any("Recognized niche id" in label for label in legend_labels)
		plt.close(axis.figure)


def _detection_history():
	"""Two 4-episode phases: A recognized as niche 0; B (unseen physics) mostly unrecognized (-1), with 2 confirmed shifts."""
	return pd.DataFrame(
		{
			"episode": range(8),
			"score": [500.0] * 8,
			"env_id": ["Env-A"] * 4 + ["Env-B"] * 4,
			"detected_env_id": [0, 0, 0, 0, -1, 2, -1, -1],
			"env_shift_detected": [False, False, False, False, True, False, True, False],
		}
	)


def test_plot_academic_comparison_marks_confirmed_shifts_and_starts_at_episode_zero(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""A recognizer's confirmed shifts are labelled as such, and the y axis crosses the x axis at episode 0."""
	with step("Arrange: the two-phase detection history; keep the figure open for inspection."):
		monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

	with step("Act: render the comparison chart."):
		plot_academic_comparison([_detection_history()], ["ENA-Combined"], save_path=str(tmp_path / "chart.png"))

	with step("Assert: 'Shift confirmed' (not 'Shift Detected'), two triangles, and the x axis starting at 0."):
		axis = plt.gcf().axes[0]
		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert "Shift confirmed" in legend_labels
		assert "Shift Detected" not in legend_labels
		shifts = [line for line in axis.lines if line.get_marker() == "^"]
		assert list(shifts[0].get_xdata()) == [4, 6]
		assert axis.get_xlim()[0] == 0
		plt.close(axis.figure)


@pytest.mark.parametrize("constant_niche", [0, -1], ids=["none-driver", "random-agent"])
def test_plot_academic_comparison_adds_no_detection_overlay_for_agents_without_env_detection(
	constant_niche, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""No recognizer, no shift markers: the none driver always logs niche 0 and the Random agent always -1, and neither fires."""
	with step("Arrange: a history from an agent without env detection."):
		monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
		history = _detection_history().assign(detected_env_id=constant_niche, env_shift_detected=False)

	with step("Act: render the comparison chart."):
		plot_academic_comparison([history], ["ENA-HoF"], save_path=str(tmp_path / "chart.png"))

	with step("Assert: no shift legend entry."):
		axis = plt.gcf().axes[0]
		legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
		assert "Shift confirmed" not in legend_labels
		plt.close(axis.figure)


def _budget_history(episodes_per_env, steps_per_episode):
	"""Two environments, each played for episodes_per_env episodes of steps_per_episode env-steps (cumulative env_steps)."""
	rows = []
	for env_id in ("Env-A", "Env-B"):
		rows += [{"env_id": env_id, "score": 100.0} for _ in range(episodes_per_env)]
	frame = pd.DataFrame(rows)
	frame["episode"] = range(1, len(frame) + 1)
	frame["env_steps"] = [steps_per_episode * (index + 1) for index in range(len(frame))]
	return frame


def test_plot_academic_comparison_can_put_agents_with_different_episode_lengths_on_one_step_axis(tmp_path, monkeypatch) -> None:
	"""Under an equal step budget, an agent that spends most steps on candidate evaluation plays far fewer episodes;
	on an environment-step axis both agents span the same range and their switches line up."""
	with step("Arrange: a slow agent (4 episodes/env of 1000 steps) and a fast one (40 of 100); keep the figure open."):
		monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
		slow, fast = _budget_history(4, 1000), _budget_history(40, 100)

	with step("Act: plot on the env-step axis."):
		plot_academic_comparison([slow, fast], ["Slow", "Fast"], save_path=str(tmp_path / "chart.png"), x_axis="env_steps")

	with step("Assert: both curves end at 8000 steps, both switches at 4000, the x label names steps."):
		figure = plt.gcf()
		for axis in figure.axes[:2]:
			curve = next(line for line in axis.lines if line.get_label() == "Mean Performance")
			assert curve.get_xdata()[-1] == 8000
			switch_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
			assert [line.get_xdata()[0] for line in switch_lines] == [4000]
			assert axis.get_xlim() == (0.0, 8000.0)
		assert figure.axes[1].get_xlabel() == "Environment Steps"
		plt.close(figure)


def test_plot_academic_comparison_rejects_an_unknown_x_axis(tmp_path) -> None:
	"""Only episodes (default) and env_steps exist."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError):
			plot_academic_comparison([_budget_history(2, 10)], ["A"], save_path=str(tmp_path / "chart.png"), x_axis="seconds")
