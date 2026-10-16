"""Tests for ENAgent's use of the Olympics skill: holding Olympiads, evicting, and test-time admission."""

# Libs imports
import numpy as np

# App imports
from helpers import step
from agents.ena.tools.reward_normalizer import normalize_reward

# pylint: disable=protected-access


def _olympic_agent(build_agent, **overrides):
	"""A small ENA with the Hall of Fame on, detection off (one niche, 0) and Olympiads every 10 steps."""
	settings = {"recognizer_strategy": "none", "hall_of_fame_driver": "olympic", "olympics_interval": 10, "olympics_episodes": 2}
	return build_agent(**{**settings, **overrides})


def _named_env(agent):
	"""A fresh CartPole with the name ENA labels gladiators by (the study's worlds carry one)."""
	env = agent.make_env()
	env.unwrapped.name = "CartPole"
	return env


def _archive(agent, *ids, confirmed=False):
	"""Archive one gladiator per id in niche 0, each with a live copy in the population."""
	hall_of_fame = agent._skills["hall_of_fame"]._instance
	for index, individual_id in enumerate(ids):
		individual = agent.population[index]
		individual.update({"individual_id": individual_id, "is_gladiator": True, "internal_env_id": 0, "olympic_confirmed": confirmed})
		hall_of_fame._records[individual_id] = {
			"individual_id": individual_id,
			"model": individual["model"],
			"trust": 1.0,
			"internal_env_id": 0,
			"is_gladiator": True,
			"env_name": "CartPole",
			"last_mu": np.zeros(agent.state_dim),
			"last_var": np.ones(agent.state_dim),
		}
	return hall_of_fame


def test_olympics_are_off_by_default_and_the_olympic_archive_takes_its_settings_from_the_constructor(build_agent) -> None:
	"""The default archive holds no Olympiad (CartPole as before); hall_of_fame_driver='olympic' gets the interval and episodes."""
	with step("Act: the default agent and one with Olympiads."):
		default, olympic = build_agent(), build_agent(hall_of_fame_driver="olympic", olympics_interval=5_000, olympics_episodes=4)

	with step("Assert: disabled by default; the settings reach the periodic driver."):
		assert not default._skills["hall_of_fame"].holds_olympics
		assert olympic._skills["hall_of_fame"]._instance.interval == 5_000
		assert olympic._skills["hall_of_fame"]._instance.episodes == 4


def test_an_olympiad_rescores_survivors_evicts_the_lucky_and_counts_its_steps(build_agent, monkeypatch) -> None:
	"""Each competitor's trust becomes its Olympic mean; those below the population's reference are evicted, never the champion."""
	with step("Arrange: three gladiators with scripted Olympic returns, and a reward history whose reference is 100."):
		agent = _olympic_agent(build_agent)
		hall_of_fame = _archive(agent, "steady", "fair", "lucky")
		agent.memory.reward_history.extend([100.0] * 10)
		agent.memory.max_seen, agent.memory.min_seen = 400.0, 0.0
		returns = {"steady": [300.0, 320.0], "fair": [150.0, 170.0], "lucky": [10.0, 30.0]}
		monkeypatch.setattr(agent, "_ENAgent__run_gladiator", lambda record, _env, _seed: (returns[record["individual_id"]].pop(0), 25))
		steps_before = agent.candidate_env_steps

	with step("Act: hold an Olympiad."):
		agent._ENAgent__hold_olympics(_named_env(agent))

	with step("Assert: steady and fair survive with their Olympic trust and are confirmed; lucky is evicted; 6 x 25 steps counted."):
		assert sorted(record["individual_id"] for record in hall_of_fame.records_in_niche(0)) == ["fair", "steady"]
		steady = hall_of_fame._records["steady"]
		assert steady["trust"] == normalize_reward(310.0, agent.memory, agent.return_scale)
		assert steady["olympic_confirmed"] and hall_of_fame._records["fair"]["olympic_confirmed"]
		assert agent.population[0]["olympic_confirmed"] and agent.population[0]["trust"] == steady["trust"]
		assert agent.population[2]["is_gladiator"] is False
		assert agent.candidate_env_steps - steps_before == 150
		assert {row["individual_id"]: row["evicted"] for row in agent.olympics_history} == {"steady": False, "fair": False, "lucky": True}


def test_a_gladiator_competes_on_real_full_episodes(build_agent) -> None:
	"""Unscripted: every competitor plays the Olympiad's full episodes in a clone of the current environment."""
	with step("Arrange: two archived gladiators."):
		agent = _olympic_agent(build_agent)
		_archive(agent, "a", "b")
		steps_before = agent.candidate_env_steps

	with step("Act: hold an Olympiad on CartPole."):
		agent._ENAgent__hold_olympics(_named_env(agent))

	with step("Assert: two rows, each mean over two episodes; their steps are counted."):
		assert len(agent.olympics_history) == 2
		assert all(len(row["returns"]) == 2 for row in agent.olympics_history)
		assert agent.candidate_env_steps - steps_before == sum(sum(row["steps"]) for row in agent.olympics_history)


def test_training_holds_an_olympiad_once_it_is_due(build_agent, monkeypatch) -> None:
	"""train() checks the Olympics after each episode, and holds one once the interval has passed."""
	with step("Arrange: record Olympiads instead of holding them."):
		agent = _olympic_agent(build_agent, olympics_interval=1)
		agent.memory.reward_history.extend([10.0] * 5)  # the eviction bar exists
		held = []
		monkeypatch.setattr(agent, "_ENAgent__hold_olympics", lambda env: held.append(agent.total_env_steps))
		env = _named_env(agent)

	with step("Act: play and train one CartPole episode."):
		state, _ = env.reset(seed=0)
		done = False
		while not done:
			state, _reward, terminated, truncated, _ = agent.act(env, state)
			done = terminated or truncated
		agent.train(env=env, step_data={"done": True})

	with step("Assert: one Olympiad, held after the episode's steps."):
		assert len(held) == 1 and held[0] >= 1


def test_an_olympiad_waits_until_the_eviction_bar_exists(build_agent, monkeypatch) -> None:
	"""An Olympiad is postponed until the reference trust that sets its eviction bar exists."""
	# With fewer training returns than the reference trust needs, the bar is a placeholder (-1): the Olympiad is
	# postponed rather than confirming every competitor against it.
	with step("Arrange: due, Hall of Fame non-empty, but only 3 training returns; record Olympiads."):
		agent = _olympic_agent(build_agent, olympics_interval=1)
		_archive(agent, "g")
		held = []
		monkeypatch.setattr(agent, "_ENAgent__hold_olympics", lambda env: held.append(agent.total_env_steps))
		env = _named_env(agent)

	with step("Act: play and train one episode (the 4th return)."):
		agent.memory.reward_history.extend([10.0] * 3)
		state, _ = env.reset(seed=0)
		done = False
		while not done:
			state, _reward, terminated, truncated, _ = agent.act(env, state)
			done = terminated or truncated
		agent.train(env=env, step_data={"done": True})

	with step("Assert: postponed, and still due."):
		assert not held
		assert agent._skills["hall_of_fame"].olympics_due(agent.total_env_steps)


def test_test_time_uses_only_gladiators_that_competed(build_agent) -> None:
	"""Provisional gladiators are neither re-introduced nor played at test when the Olympics are on."""
	with step("Arrange: a confirmed and a provisional gladiator."):
		agent = _olympic_agent(build_agent, max_gladiator_ratio=1.0)
		_archive(agent, "confirmed", confirmed=True)
		_archive(agent, "provisional")
		agent.population[0].update({"individual_id": "confirmed", "is_gladiator": True, "olympic_confirmed": True})

	with step("Act: re-introduce gladiators and enter test mode."):
		agent.load_gladiators()
		agent.set_learning(False)

	with step("Assert: only the confirmed gladiator is a gladiator in the test population."):
		gladiators = [individual["individual_id"] for individual in agent.population if individual.get("is_gladiator")]
		assert gladiators == ["confirmed"]


def test_without_the_olympics_provisional_gladiators_stay_usable(build_agent) -> None:
	"""The none driver keeps ENA's test time as before: every gladiator counts."""
	with step("Arrange: a never-confirmed gladiator, Olympics off."):
		agent = build_agent(recognizer_strategy="none")
		_archive(agent, "g")

	with step("Act: enter test mode."):
		agent.set_learning(False)

	with step("Assert: still a gladiator."):
		assert agent.population[0]["is_gladiator"] is True


def test_the_olympic_archive_protects_confirmed_gladiators_and_the_default_does_not(build_agent) -> None:
	"""Only an Olympiad removes a confirmed gladiator: ENA turns the archive's protection on with the Olympics."""
	with step("Act: an agent with and one without the Olympics."):
		olympic, default = _olympic_agent(build_agent), build_agent()

	with step("Assert: protection follows the Olympics."):
		assert olympic._skills["hall_of_fame"]._instance.protect_confirmed is True
		assert default._skills["hall_of_fame"]._instance.protect_confirmed is False


def test_qualifier_seats_reach_the_archive_and_an_olympiad_trims_the_niche_to_its_capacity(build_agent, monkeypatch) -> None:
	"""olympics_qualifier_seats is an agent argument; an Olympiad keeps only the best max_per_niche competitors."""
	with step("Arrange: capacity 2 plus 1 qualifier seat, three gladiators all above the bar."):
		agent = _olympic_agent(build_agent, max_per_niche=2, olympics_qualifier_seats=1)
		hall_of_fame = _archive(agent, "gold", "silver", "bronze")
		agent.memory.reward_history.extend([10.0] * 10)
		agent.memory.max_seen, agent.memory.min_seen = 400.0, 0.0
		returns = {"gold": 300.0, "silver": 200.0, "bronze": 100.0}
		monkeypatch.setattr(agent, "_ENAgent__run_gladiator", lambda record, _env, _seed: (returns[record["individual_id"]], 10))

	with step("Act: hold an Olympiad."):
		agent._ENAgent__hold_olympics(_named_env(agent))

	with step("Assert: the seats reached the archive; bronze relegated though above the bar."):
		assert hall_of_fame.qualifier_seats == 1
		assert sorted(record["individual_id"] for record in hall_of_fame.records_in_niche(0)) == ["gold", "silver"]


def test_test_time_ranks_confirmed_gladiators_by_their_olympic_trust_not_their_drifted_live_trust(build_agent) -> None:
	"""Test mode ranks confirmed gladiators by their Olympic trust, not their drifted live trust."""
	# A live copy's trust keeps moving with single training episodes; entering test mode restores each confirmed
	# gladiator's Olympic trust from its record, so test-time selection starts from the Olympic ranking.
	with step("Arrange: two confirmed gladiators; the weaker one's live copy drifted above the champion."):
		agent = _olympic_agent(build_agent)
		hall_of_fame = _archive(agent, "champion", "runner-up", confirmed=True)
		hall_of_fame._records["champion"].update({"trust": 1.0, "olympic_confirmed": True})
		hall_of_fame._records["runner-up"].update({"trust": 0.7, "olympic_confirmed": True})
		agent.population[0]["trust"], agent.population[1]["trust"] = 0.4, 1.3

	with step("Act: enter test mode."):
		agent.set_learning(False)

	with step("Assert: the live copies carry their Olympic trust again."):
		assert (agent.population[0]["trust"], agent.population[1]["trust"]) == (1.0, 0.7)


def test_an_olympiad_stores_each_survivors_olympic_mean_return_and_reinjection_carries_it(build_agent, monkeypatch) -> None:
	"""BPR's prior needs the raw Olympic mean: on the record and the live copy, and on re-injected copies at test."""
	with step("Arrange: one gladiator scoring 120 and 80 at its Olympiad."):
		agent = _olympic_agent(build_agent, max_gladiator_ratio=1.0)
		hall_of_fame = _archive(agent, "g")
		agent.memory.reward_history.extend([10.0] * 10)
		returns = [120.0, 80.0]
		monkeypatch.setattr(agent, "_ENAgent__run_gladiator", lambda _record, _env, _seed: (returns.pop(0), 10))

	with step("Act: hold the Olympiad, then re-inject for test."):
		agent._ENAgent__hold_olympics(_named_env(agent))
		agent.load_gladiators()

	with step("Assert: 100 on the record and on every copy in the population."):
		assert hall_of_fame._records["g"]["olympic_return"] == 100.0
		assert all(individual["olympic_return"] == 100.0 for individual in agent.population if individual.get("individual_id") == "g")
