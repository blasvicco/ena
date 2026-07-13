"""Tests for the pure, chart-independent data transforms in tools/metrics.py."""

# Libs imports
import pandas as pd
import pytest

# App imports
from helpers import step
from tools.metrics import (
	_build_transition_matrix,
	_classify_niche_events,
	_niche_env_labels,
	_qualitative_colors,
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
