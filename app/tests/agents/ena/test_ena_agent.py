"""Tests for ENAgent's cross-phase/cross-mode statistics hygiene."""

# General imports
from typing import Any, Callable

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents.ena.ena import ENAgent


def _run_one_episode(agent: ENAgent, env: Any) -> np.ndarray:
	"""Drive act() through a full episode, returning the final state."""
	state, _ = env.reset()
	state = np.array(state, dtype=np.float32)
	done = False
	while not done:
		next_state, _reward, terminated, truncated, _action = agent.act(env, state)
		done = terminated or truncated
		state = next_state
	return state


def test_set_learning_false_resets_recognizer_statistics(agent: ENAgent) -> None:
	"""S1: entering test mode must not leak training-time recognizer statistics."""
	with step("Arrange: feed training-time statistics into the recognizer's normalizers."):
		batch = np.ones((5, agent.state_dim), dtype=np.float32) * 3.0
		agent.memory.short_term_normalizer.update(batch)
		agent.memory.transition_normalizer.update(batch)
		assert agent.memory.short_term_normalizer.count > 1e-4
		assert agent.memory.transition_normalizer.count > 1e-4

	with step("Act: switch the agent into test mode."):
		agent.set_learning(False)

	with step("Assert: both normalizers are reset to their identity statistics."):
		assert agent.memory.short_term_normalizer.count == 1e-4
		assert agent.memory.transition_normalizer.count == 1e-4
		assert np.allclose(agent.memory.short_term_normalizer.mean, 0.0)
		assert np.allclose(agent.memory.transition_normalizer.mean, 0.0)
		assert np.allclose(agent.memory.short_term_normalizer.var, 1.0)
		assert np.allclose(agent.memory.transition_normalizer.var, 1.0)


@pytest.mark.parametrize("is_training", [True, False], ids=["training-time", "test-time"])
def test_env_shift_reward_history_clearing(agent: ENAgent, monkeypatch: pytest.MonkeyPatch, is_training: bool) -> None:
	"""T3: a detected shift during training must clear the reward history; the training-only
	clear must not fire for test-mode shifts (a separate concern)."""
	with step("Arrange: seed the reward history and script a detected environment shift."):
		if is_training:
			assert agent.learn is True
		else:
			agent.set_learning(False)
		agent.memory.reward_history.extend([1.0, 2.0, 3.0])
		# pylint: disable=protected-access
		monkeypatch.setattr(agent._recognizer, "detect_change", lambda memory: True)

	with step("Act: run one episode, triggering the scripted shift."):
		env = agent.make_env()
		_run_one_episode(agent, env)
		env.close()

	with step("Assert: training-time shifts clear the history; test-time shifts append instead."):
		history = list(agent.memory.reward_history)
		if is_training:
			assert not history
		else:
			# Test mode always appends the just-finished episode's score itself (existing
			# behavior) — the fix must not additionally wipe the history via clear_training_history.
			assert history[:3] == [1.0, 2.0, 3.0]
			assert len(history) == 4


@pytest.mark.parametrize("detect_change_result", [True, False])
def test_last_env_shift_detected_reflects_detect_change(agent: ENAgent, monkeypatch: pytest.MonkeyPatch, detect_change_result: bool) -> None:
	"""C1: expose detect_change()'s real result, independent of niche resolution."""
	with step("Arrange: script the recognizer's detect_change() to return a fixed result."):
		# pylint: disable=protected-access
		monkeypatch.setattr(agent._recognizer, "detect_change", lambda memory: detect_change_result)

	with step("Act: run one episode."):
		env = agent.make_env()
		_run_one_episode(agent, env)
		env.close()

	with step("Assert: last_env_shift_detected mirrors the scripted result."):
		assert agent.last_env_shift_detected is detect_change_result


def _spy_on_fast_call(individual: dict) -> dict:
	"""Wrap an individual's fast_call to capture the first tensor it's fed."""
	captured = {}
	original = individual["fast_call"]

	def spy(state_tensor, training=False):
		captured.setdefault("input", state_tensor.numpy().copy())
		return original(state_tensor, training=training)

	individual["fast_call"] = spy
	return captured


@pytest.mark.parametrize("is_gladiator", [True, False], ids=["gladiator", "non-gladiator"])
def test_evaluate_individual_performance_always_uses_live_normalizer(agent: ENAgent, is_gladiator: bool) -> None:
	"""Restored (Adding-Memory) behavior: every individual — gladiators included — is
	normalized with the live, shared obs_normalizer. No special-casing: gladiators get
	their weights continuously re-projected to match it (see __project_individual_weights),
	so using the live normalizer for them is always correct."""
	with step("Arrange: mark the individual, set distinctive live-normalizer stats, spy on its input."):
		individual = agent.population[0]
		individual["is_gladiator"] = is_gladiator
		agent.memory.obs_normalizer.mean = np.full(agent.state_dim, 2.0)
		agent.memory.obs_normalizer.var = np.full(agent.state_dim, 4.0)
		captured = _spy_on_fast_call(individual)

	with step("Act: evaluate the individual's performance in a seeded env."):
		env = agent.make_env()
		seed = 123
		# pylint: disable=protected-access
		agent._ENAgent__evaluate_individual_performance(individual, env, seed=seed)
		env.close()

	with step("Assert: the first observation fed to the model was normalized with the live normalizer."):
		reference_env = agent.make_env()
		raw_state, _ = reference_env.reset(seed=seed)
		reference_env.close()

		expected = agent.memory.obs_normalizer.normalize(np.array(raw_state))
		assert np.allclose(captured["input"][0], expected, atol=1e-5)


def test_project_individual_weights_no_longer_skips_gladiators(agent: ENAgent) -> None:
	"""Restored (Adding-Memory) behavior: gladiators must be re-projected to the live
	normalizer just like everyone else — freezing them was the "Failed shit!" regression."""
	with step("Arrange: a gladiator with stale last_mu/last_var, and a live normalizer with enough samples."):
		individual = agent.population[0]
		individual["is_gladiator"] = True
		individual["last_mu"] = np.zeros(agent.state_dim)
		individual["last_var"] = np.ones(agent.state_dim)
		agent.memory.obs_normalizer.mean = np.full(agent.state_dim, 5.0)
		agent.memory.obs_normalizer.var = np.full(agent.state_dim, 9.0)
		agent.memory.obs_normalizer.count = agent.max_eval_steps * 3  # past the warmup gate

	with step("Act: project the gladiator's weights."):
		# pylint: disable=protected-access
		agent._ENAgent__project_individual_weights(individual)

	with step("Assert: its last_mu/last_var were updated to match the live normalizer, not left frozen."):
		assert np.allclose(individual["last_mu"], agent.memory.obs_normalizer.mean)
		assert np.allclose(individual["last_var"], agent.memory.obs_normalizer.var)


class _NeverEndsEnv:
	"""Env stub that never terminates — isolates step-cap behavior from CartPole dynamics."""

	def __init__(self, state_dim: int) -> None:
		self._state_dim = state_dim
		self.steps_taken = 0

	def reset(self, seed=None):
		"""Reset the step counter and return a zeroed observation."""
		self.steps_taken = 0
		return np.zeros(self._state_dim, dtype=np.float32), {}

	def step(self, _action):
		"""Take a step that never ends the episode."""
		self.steps_taken += 1
		return np.zeros(self._state_dim, dtype=np.float32), 1.0, False, False, {}


@pytest.mark.parametrize(
	("max_steps_override", "expected_steps"),
	[(10, 10), (None, "max_eval_steps")],
	ids=["explicit-cap", "defaults-to-max-eval-steps"],
)
def test_evaluate_individual_performance_respects_max_steps(
	agent: ENAgent, max_steps_override, expected_steps
) -> None:
	"""Candidate evaluation depth must honor an explicit cap, and fall back to
	max_eval_steps when none is given — the mechanism plasticity-scaled depth relies on."""
	with step("Arrange: an individual and an env that never terminates on its own."):
		individual = agent.population[0]
		env = _NeverEndsEnv(agent.state_dim)
		expected = agent.max_eval_steps if expected_steps == "max_eval_steps" else expected_steps

	with step("Act: evaluate the individual with (or without) an explicit step cap."):
		# pylint: disable=protected-access
		score = agent._ENAgent__evaluate_individual_performance(
			individual, env, seed=0, max_steps=max_steps_override
		)

	with step("Assert: the loop stopped at the expected cap, rescaled to env_max_steps."):
		assert env.steps_taken == expected
		assert score == pytest.approx(agent.env_max_steps)


def test_train_handles_zero_plasticity_without_degenerate_eval_cap(
	build_agent: Callable[..., ENAgent],
) -> None:
	"""The min_eval_steps floor must keep eval_step_cap from degenerating to 0 when
	plasticity is exactly 0.0 (the "zero" plasticity_algorithm) — train() must not hang
	or crash evaluating candidates for a zero-length episode."""
	with step("Arrange: an agent configured with the zero plasticity algorithm."):
		agent = build_agent(plasticity_algorithm="zero", pop_size=5, max_threads=1)
		env = agent.make_env()
		env.unwrapped.name = "Test-Env"  # train() reads this to tag the hall-of-fame origin
		state, _ = env.reset()
		state = np.array(state, dtype=np.float32)

	with step("Act: drive one full episode through act() and train()."):
		done = False
		while not done:
			next_state, reward, terminated, truncated, action = agent.act(env, state)
			done = terminated or truncated
			agent.train(
				env=env,
				step_data={
					"action": action,
					"done": done,
					"next_state": next_state,
					"reward": reward,
					"state": state,
				},
			)
			state = next_state
		env.close()

	with step("Assert: training completed and recorded zero plasticity for the episode."):
		assert agent.memory.plasticity_history[-1] == 0.0


def test_candidate_evaluation_applies_eval_step_cap_uniformly(
	build_agent: Callable[..., ENAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Restored (Adding-Memory) behavior: gladiators get exactly the same plasticity-scaled
	eval_step_cap as every other candidate — no special-casing by is_gladiator."""
	with step("Arrange: an agent at maximal plasticity, with one candidate marked a gladiator."):
		agent = build_agent(
			pop_size=6, max_threads=1, plasticity_algorithm="full", min_eval_steps=7, max_eval_steps=50
		)
		gladiator_idx = 1
		agent.population[gladiator_idx]["is_gladiator"] = True

		captured_caps: dict = {}
		original = agent._ENAgent__evaluate_individual_performance  # pylint: disable=protected-access

		def spy(individual, env, seed=None, max_steps=None):
			captured_caps[individual["individual_id"]] = max_steps
			return original(individual, env, seed=seed, max_steps=max_steps)

		monkeypatch.setattr(agent, "_ENAgent__evaluate_individual_performance", spy)

		env = agent.make_env()
		env.unwrapped.name = "Test-Env"
		state, _ = env.reset()
		state = np.array(state, dtype=np.float32)

	with step("Act: drive one full episode through act() and train()."):
		done = False
		while not done:
			next_state, reward, terminated, truncated, action = agent.act(env, state)
			done = terminated or truncated
			agent.train(
				env=env,
				step_data={
					"action": action,
					"done": done,
					"next_state": next_state,
					"reward": reward,
					"state": state,
				},
			)
			state = next_state
		env.close()

	with step("Assert: every evaluated candidate, gladiator included, got the same full cap."):
		assert captured_caps
		assert all(value == agent.max_eval_steps for value in captured_caps.values())


def test_evolve_population_never_replaces_gladiators(build_agent: Callable[..., ENAgent]) -> None:
	"""T4: gladiators must survive evolution even when every individual looks equally bad."""
	with step("Arrange: build a population where every individual looks equally bad, mark gladiators."):
		agent = build_agent(pop_size=10)
		for individual in agent.population:
			individual["trust"] = -10.0
			individual["age"] = 0

		gladiator_slots = (2, 5)
		gladiator_ids = []
		for slot in gladiator_slots:
			agent.population[slot]["is_gladiator"] = True
			gladiator_ids.append(agent.population[slot]["individual_id"])

	with step("Act: evolve the population."):
		# pylint: disable=protected-access
		agent._ENAgent__evolve_population()

	with step("Assert: every gladiator slot still holds its original gladiator."):
		for slot, gladiator_id in zip(gladiator_slots, gladiator_ids):
			assert agent.population[slot]["individual_id"] == gladiator_id
			assert agent.population[slot]["is_gladiator"] is True
