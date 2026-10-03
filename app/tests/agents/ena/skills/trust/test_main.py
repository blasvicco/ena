"""Tests for the Trust facade's reference-trust and population-reset behavior."""

# General imports
from types import SimpleNamespace

# App imports
from helpers import step
from agents.ena.skills.trust.main import Trust


def test_reference_returns_minus_one_below_five_samples() -> None:
	"""Fewer than 5 reward samples means no reliable reference yet."""
	with step("Arrange: a trust facade and a memory with fewer than 5 reward samples."):
		trust = Trust("zero")
		memory = SimpleNamespace(reward_history=[1.0, 2.0])

	with step("Assert: reference() returns -1.0, ranking a fresh slot above any evaluated mutant."):
		assert trust.reference(memory, env_max_steps=1000.0) == -1.0


def test_reset_population_leaves_gladiator_trust_untouched() -> None:
	"""reset_population() must not dilute an established gladiator's trust on an unrelated
	exploratory failure — only non-gladiators get blended toward the reference trust.
	Gladiators are already protected from replacement in __evolve_population; there's no
	reason a failure that has nothing to do with them should erode their trust."""
	with step("Arrange: a population with one gladiator and one non-gladiator, both at a known trust."):
		trust = Trust("zero")
		memory = SimpleNamespace(reward_history=[])
		population = [
			{"trust": 5.0, "is_gladiator": True},
			{"trust": 5.0, "is_gladiator": False},
		]

	with step("Act: reset trust with a large blend factor."):
		trust.reset_population(population, memory, env_max_steps=1000.0, factor=1.0)

	with step("Assert: the gladiator's trust is unchanged; the non-gladiator's was blended toward the reference."):
		assert population[0]["trust"] == 5.0
		assert population[1]["trust"] != 5.0
