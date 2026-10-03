"""Tests for the Mamdani fuzzy inference tool."""

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.tools.fuzzy_inference import FuzzyVariable, LeftShoulder, MamdaniController, RightShoulder, Triangle

LEVEL = FuzzyVariable({"LOW": LeftShoulder(0.2, 0.5), "MEDIUM": Triangle(0.2, 0.5, 0.8), "HIGH": RightShoulder(0.5, 0.8)})
OUTPUT = FuzzyVariable(
	{"LOW": Triangle(-0.35, 0.1, 0.55), "MEDIUM": Triangle(0.1, 0.55, 1.0), "HIGH": Triangle(0.55, 1.0, 1.45)}, universe=(-0.35, 1.45)
)


@pytest.mark.parametrize(
	"function, points, expected",
	[
		(Triangle(0.2, 0.5, 0.8), [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9], [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]),
		(LeftShoulder(0.2, 0.5), [0.0, 0.2, 0.35, 0.5, 0.9], [1.0, 1.0, 0.5, 0.0, 0.0]),
		(RightShoulder(0.5, 0.8), [0.0, 0.5, 0.65, 0.8, 1.0], [0.0, 0.0, 0.5, 1.0, 1.0]),
	],
	ids=["triangle", "left-shoulder", "right-shoulder"],
)
def test_membership_degrees_at_known_points(function, points, expected) -> None:
	"""Piecewise-linear membership functions, evaluated on arrays."""
	with step("Act/Assert: degrees match the hand-computed values."):
		assert np.allclose(function(np.array(points)), expected)


def test_fuzzify_returns_each_terms_degree() -> None:
	"""A crisp input maps to one degree per linguistic term."""
	with step("Act/Assert: 0.35 is half LOW, half MEDIUM, not HIGH."):
		assert LEVEL.fuzzify(0.35) == pytest.approx({"LOW": 0.5, "MEDIUM": 0.5, "HIGH": 0.0})


def _controller(rules):
	return MamdaniController({"level": LEVEL}, OUTPUT, rules)


@pytest.mark.parametrize("level, expected", [(0.0, 1.0), (0.5, 0.55), (1.0, 0.1)])
def test_a_single_fully_firing_rule_defuzzifies_to_its_output_peak(level, expected) -> None:
	"""Symmetric output triangles: one rule at full strength gives exactly the term's peak (no centroid shrinkage)."""
	with step("Arrange: LOW -> HIGH, MEDIUM -> MEDIUM, HIGH -> LOW."):
		controller = _controller([({"level": "LOW"}, "HIGH"), ({"level": "MEDIUM"}, "MEDIUM"), ({"level": "HIGH"}, "LOW")])

	with step("Act/Assert: the crisp output is the fired term's peak."):
		assert controller.evaluate(level=level) == pytest.approx(expected, abs=1e-3)


def test_two_rules_firing_equally_defuzzify_between_their_peaks() -> None:
	"""Half LOW, half MEDIUM -> a centroid strictly between the HIGH and MEDIUM output peaks."""
	with step("Arrange/Act: the same three rules at level 0.35."):
		controller = _controller([({"level": "LOW"}, "HIGH"), ({"level": "MEDIUM"}, "MEDIUM"), ({"level": "HIGH"}, "LOW")])
		output = controller.evaluate(level=0.35)

	with step("Assert: by symmetry, the midpoint of the two peaks."):
		assert output == pytest.approx((0.55 + 1.0) / 2, abs=1e-3)


def test_rule_antecedents_combine_with_min() -> None:
	"""A two-input rule fires at the minimum of its antecedents' degrees (fuzzy AND)."""
	with step("Arrange: one rule, IF a is HIGH AND b is HIGH THEN HIGH, plus a weaker LOW rule."):
		controller = MamdaniController({"a": LEVEL, "b": LEVEL}, OUTPUT, [({"a": "HIGH", "b": "HIGH"}, "HIGH"), ({"a": "LOW"}, "LOW")])

	with step("Act: a fully HIGH, b only half HIGH; no LOW."):
		strengths = controller.rule_strengths(a=1.0, b=0.65)

	with step("Assert: the AND rule fires at 0.5, the LOW rule not at all."):
		assert strengths == pytest.approx([0.5, 0.0])


def test_no_rule_firing_is_an_error() -> None:
	"""An uncovered input has no defuzzified value — that's a rule-base gap, not a silent default."""
	with step("Arrange: only a HIGH rule."):
		controller = _controller([({"level": "HIGH"}, "LOW")])

	with step("Act/Assert: a fully LOW input raises."):
		with pytest.raises(ValueError):
			controller.evaluate(level=0.0)
