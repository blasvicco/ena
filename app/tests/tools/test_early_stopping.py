"""Tests for the challenge's per-environment early-stopping rule."""

# Libs imports
import pytest

# App imports
from helpers import step
from tools.early_stopping import StoppingRule


def test_checks_fall_due_every_check_every_env_steps() -> None:
	"""A check is due once each further check_every env-steps have been consumed in the phase."""
	with step("Arrange: checks every 20k steps."):
		rule = StoppingRule(check_every=20_000)
		rule.start_phase()

	with step("Act/Assert: due at 20k, not again until 40k (even if an episode overshoots)."):
		assert not rule.due(19_999)
		assert rule.due(21_500)
		rule.record(21_500, 100.0)
		assert not rule.due(39_999)
		assert rule.due(40_000)


def test_solved_at_the_threshold() -> None:
	"""Mean evaluation score >= solved_score ends the phase as solved."""
	with step("Act/Assert: 474.9 continues, 475 solves."):
		rule = StoppingRule(solved_score=475.0)
		rule.start_phase()
		assert rule.record(20_000, 474.9) == "continue"
		assert rule.record(40_000, 475.0) == "solved"


def test_plateau_when_the_last_checks_do_not_beat_the_earlier_best_by_the_minimum_gain() -> None:
	"""No improvement of at least plateau_min_gain within the last plateau_checks checks -> plateau."""
	with step("Arrange: plateau after 3 checks without a 10-point gain."):
		rule = StoppingRule(plateau_checks=3, plateau_min_gain=10.0)
		rule.start_phase()
		statuses = [rule.record(step_count, score) for step_count, score in [(1, 200.0), (2, 205.0), (3, 209.0)]]

	with step("Act: a fourth check, still under 210."):
		statuses.append(rule.record(4, 208.0))

	with step("Assert: undecided until there are more than plateau_checks checks, then plateau."):
		assert statuses == ["continue", "continue", "continue", "plateau"]


def test_steady_improvement_is_never_a_plateau() -> None:
	"""Gains of at least the minimum keep the phase going."""
	with step("Act/Assert: +15 per check for 20 checks, all below the solved bar."):
		rule = StoppingRule(plateau_checks=3, plateau_min_gain=10.0, solved_score=1000.0)
		rule.start_phase()
		assert {rule.record(index, 100.0 + 15.0 * index) for index in range(20)} == {"continue"}


def test_start_phase_forgets_the_previous_environment() -> None:
	"""Each environment is judged on its own checks only."""
	with step("Arrange: a phase that plateaued."):
		rule = StoppingRule(plateau_checks=1, plateau_min_gain=10.0)
		rule.start_phase()
		rule.record(20_000, 300.0)
		assert rule.record(40_000, 300.0) == "plateau"

	with step("Act/Assert: a new phase starts over — first check due at 20k, no plateau yet."):
		rule.start_phase()
		assert rule.due(20_000)
		assert rule.record(20_000, 100.0) == "continue"


@pytest.mark.parametrize("field, value", [("check_every", 0), ("eval_episodes", 0), ("plateau_checks", 0), ("max_steps", 0)])
def test_rejects_non_positive_settings(field, value) -> None:
	"""A zero schedule or cap would stop immediately or never check — a configuration error."""
	with step("Act/Assert: ValueError."):
		with pytest.raises(ValueError):
			StoppingRule(**{field: value})


def test_without_stop_when_solved_checks_end_after_the_first_solve() -> None:
	"""Fixed-budget mode: the first solve is still reported, then no more checks fall due in that environment."""
	with step("Arrange: a rule that doesn't stop on solving."):
		rule = StoppingRule(check_every=10, stop_when_solved=False)
		rule.start_phase()

	with step("Act/Assert: solved at the second check; nothing due afterwards; a new phase checks again."):
		assert rule.record(10, 300.0) == "continue"
		assert rule.record(20, 480.0) == "solved"
		assert not rule.due(1_000_000)
		rule.start_phase()
		assert rule.due(10)


def test_check_after_solve_keeps_checking_to_the_end_of_the_phase() -> None:
	"""HalfCheetah needs the return at the end of each environment (forgetting), not only the first solve."""
	with step("Arrange: a fixed-budget rule that keeps checking, solved at the first check."):
		rule = StoppingRule(check_every=100, solved_score=10.0, stop_when_solved=False, stop_on_plateau=False, check_after_solve=True)
		rule.start_phase()
		rule.record(100, 50.0)

	with step("Act/Assert: the next check still falls due."):
		assert rule.due(200)
