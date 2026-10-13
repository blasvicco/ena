"""Tests for ENAgent playing a probing detector's excitation at every episode end."""

# Libs imports
import numpy as np

# App imports
from helpers import step

CHIRP = {"steps": 8, "features": {"kind": "raw"}, "min_chirps": 3}


def _play_episode(agent, env, seed):
	"""One real episode through act(), as the runner plays it; returns the steps played."""
	state, _ = env.reset(seed=seed)
	done, steps = False, 0
	while not done:
		state, _reward, terminated, truncated, _ = agent.act(env, state)
		done, steps = terminated or truncated, steps + 1
	return steps


def test_the_chirp_settings_reach_the_detector(build_agent) -> None:
	"""recognizer_strategy='chirp' with recognizer_chirp settings."""
	with step("Act/Assert: an 8-step chirp of discrete actions for CartPole."):
		agent = build_agent(recognizer_strategy="chirp", recognizer_chirp=CHIRP)
		assert agent._skills["env_detection"].chirp_actions.shape == (8,)  # pylint: disable=protected-access


def test_every_episode_end_plays_the_chirp_in_the_real_environment_and_counts_its_steps(build_agent, monkeypatch) -> None:
	"""The detector gets the chirp's observations (start first); its steps are candidate steps (the budget)."""
	with step("Arrange: capture what the detector receives."):
		agent = build_agent(recognizer_strategy="chirp", recognizer_chirp=CHIRP)
		detector = agent._skills["env_detection"]  # pylint: disable=protected-access
		received = []
		original = detector.observe_chirp
		monkeypatch.setattr(detector, "observe_chirp", lambda response: received.append(np.asarray(response)) or original(response))
		env = agent.make_env()
		before = agent.candidate_env_steps

	with step("Act: one episode."):
		_play_episode(agent, env, seed=0)

	with step("Assert: one chirp of 1 + 8 observations; its steps (up to 8, fewer if the pole falls) counted."):
		assert len(received) == 1 and received[0].shape == (9, 4)
		assert 1 <= agent.candidate_env_steps - before <= 8


def test_the_chirp_leaves_the_environments_random_stream_untouched(build_agent) -> None:
	"""Common random numbers: after a chirpd episode, the next reset gives the same start as without the chirp."""
	with step("Arrange: a probing and a passive agent, each on its own identically seeded CartPole."):
		probing, passive = build_agent(recognizer_strategy="chirp", recognizer_chirp=CHIRP), build_agent()
		env_probing, env_passive = probing.make_env(), passive.make_env()
		for env in (env_probing, env_passive):
			env.reset(seed=11)
			env.action_space.seed(11)

	with step("Act: play one episode on each, then reset both without a seed."):
		_play_episode(probing, env_probing, seed=None)
		_play_episode(passive, env_passive, seed=None)
		next_probing, _ = env_probing.reset()
		next_passive, _ = env_passive.reset()

	with step("Assert: the same next start state."):
		assert np.array_equal(next_probing, next_passive)


def test_passive_drivers_play_no_chirp(build_agent) -> None:
	"""dynamics_model (the default): episode ends cost no extra steps."""
	with step("Act: one episode with the default detector."):
		agent = build_agent()
		env = agent.make_env()
		before = agent.candidate_env_steps
		_play_episode(agent, env, seed=0)

	with step("Assert: no candidate steps (no training call happened)."):
		assert agent.candidate_env_steps == before
