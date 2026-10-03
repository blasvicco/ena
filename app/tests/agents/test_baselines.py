"""Tests for the baseline-agent contract details the CartPole challenge relies on."""

# Libs imports
import numpy as np
import pytest
import tensorflow as tf
from gymnasium import spaces
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents import PPOAgent, SACAgent
from agents.tools.baseline import ABaseline


def _brain(output_dim, input_dim):
	"""Small MLP factory."""
	return Sequential([Input(shape=(input_dim,)), Dense(8, activation="tanh"), Dense(output_dim)])


class _DiscreteEnv:
	"""Fixed observation, two actions, one-step episodes."""

	def __init__(self):
		self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
		self.action_space = spaces.Discrete(2)
		self.unwrapped = self

	def reset(self, seed=None):  # pylint: disable=unused-argument
		"""A fixed observation."""
		return np.array([0.5, -0.5], dtype=np.float32), {}

	def step(self, action):
		"""Terminate immediately."""
		return np.array([0.5, -0.5], dtype=np.float32), float(action), True, False, {}

	def close(self):
		"""No-op."""


class _MinimalBaseline(ABaseline):
	"""The smallest concrete ABaseline."""

	def act(self, env, state):
		"""Unused."""

	def train(self, **kwargs):
		"""Unused."""


def test_abaseline_set_learning_accepts_the_runner_output_dir_keyword() -> None:
	"""The study runner calls set_learning(False, output_dir=...) on every agent; the base must accept it."""
	with step("Act/Assert: no TypeError, mode applied."):
		agent = _MinimalBaseline(brain=None, make_env=None)
		agent.set_learning(False, output_dir="/tmp")
		assert agent.learn is False


def _ppo():
	return PPOAgent(brain=_brain, critic_brain=_brain, make_env=_DiscreteEnv)


def test_ppo_acts_greedily_when_not_learning() -> None:
	"""Frozen evaluation: the most probable action every time, not a sample."""
	with step("Arrange: a frozen PPO agent and its policy's preferred action."):
		agent, env = _ppo(), _DiscreteEnv()
		agent.set_learning(False)
		state, _ = env.reset()
		logits = agent.actor(state[np.newaxis]).numpy()[0]

	with step("Act: 50 evaluation actions."):
		actions = {agent.act(env, state)[4] for _ in range(50)}

	with step("Assert: always the argmax."):
		assert actions == {int(np.argmax(logits))}


@pytest.mark.parametrize("total_steps, expected", [(2048 * 10, 10), (100, 1)])
def test_ppo_set_total_steps_sizes_the_learning_rate_schedule(total_steps, expected) -> None:
	"""Under an env-step budget the LR annealing horizon is total_steps / n_steps rollouts (at least 1)."""
	with step("Act/Assert: the schedule's update count."):
		agent = _ppo()
		agent.set_total_steps(total_steps)
		assert agent._max_updates == expected  # pylint: disable=protected-access


def _forbid_tf_random(monkeypatch):
	"""Make any eager tf.random sampling call fail: under tf.random.set_seed each eager call leaks ~36 KB
	(measured: 30k seeded tf.random.categorical calls grew memory 588 MB -> 1,681 MB; unseeded and NumPy flat)."""

	def _forbidden(*_args, **_kwargs):
		raise AssertionError("eager tf.random sampling leaks memory under a global seed")

	monkeypatch.setattr(tf.random, "categorical", _forbidden)
	monkeypatch.setattr(tf.random, "normal", _forbidden)


def test_ppo_discrete_sampling_avoids_eager_tf_random_and_follows_the_policy(monkeypatch) -> None:
	"""Training-time actions are drawn from softmax(logits) without tf.random, and the cached log-prob matches."""
	with step("Arrange: a learning PPO agent, tf.random sampling forbidden."):
		np.random.seed(0)
		agent, env = _ppo(), _DiscreteEnv()
		state, _ = env.reset()
		normalized = agent._obs_normalizer.normalize(state)  # pylint: disable=protected-access
		probabilities = tf.nn.softmax(agent.actor(normalized[np.newaxis])).numpy()[0]
		_forbid_tf_random(monkeypatch)

	with step("Act: 4000 training-mode actions, checking each cached log-prob."):
		counts = np.zeros(2)
		for _ in range(4000):
			action = agent.act(env, state)[4]
			counts[action] += 1
			assert agent._last_log_prob == pytest.approx(np.log(probabilities[action]), abs=1e-5)  # pylint: disable=protected-access

	with step("Assert: empirical frequencies match the policy's probabilities."):
		assert counts / counts.sum() == pytest.approx(probabilities, abs=0.03)


class _BoxEnv(_DiscreteEnv):
	"""Same as _DiscreteEnv with a 1-D continuous action."""

	def __init__(self):
		super().__init__()
		self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

	def step(self, action):
		"""Terminate immediately."""
		return np.array([0.5, -0.5], dtype=np.float32), 0.0, True, False, {}


def test_ppo_continuous_sampling_avoids_eager_tf_random(monkeypatch) -> None:
	"""Gaussian exploration noise is drawn with NumPy, around the policy mean with the policy's std."""
	with step("Arrange: a continuous PPO agent, tf.random sampling forbidden."):
		np.random.seed(0)
		agent = PPOAgent(brain=_brain, critic_brain=_brain, make_env=_BoxEnv)
		env = _BoxEnv()
		state, _ = env.reset()
		_forbid_tf_random(monkeypatch)

	with step("Act: 2000 unclipped training-mode samples."):
		samples = []
		for _ in range(2000):
			agent.act(env, state)
			samples.append(agent._last_unclipped_action[0])  # pylint: disable=protected-access

	with step("Assert: their spread matches the policy's std."):
		std = float(np.exp(np.clip(agent.log_std.numpy()[0], -20.0, 2.0)))
		assert np.std(samples) == pytest.approx(std, rel=0.1)


def test_sac_continuous_action_noise_avoids_eager_tf_random(monkeypatch) -> None:
	"""SAC's per-step exploration noise must not use eager tf.random either (same leak under seeding)."""
	with step("Arrange: a SAC agent, tf.random sampling forbidden."):
		agent = SACAgent(brain=_brain, make_env=_BoxEnv)
		env = _BoxEnv()
		state, _ = env.reset()
		_forbid_tf_random(monkeypatch)

	with step("Act/Assert: acting works and the action is inside the tanh range."):
		action = agent.act(env, state)[4]
		assert np.all(np.abs(action) <= 1.0)


def test_ppo_with_a_constant_learning_rate_never_anneals() -> None:
	"""anneal_lr=False: every update uses the initial learning rate, however long training runs."""
	with step("Arrange: a constant-rate PPO with a tiny (and therefore irrelevant) schedule horizon."):
		agent = PPOAgent(brain=_brain, critic_brain=_brain, make_env=_DiscreteEnv, learning_rate=3e-4, anneal_lr=False)
		agent.set_total_steps(1)

	with step("Act: ask for the rate of 1,000 consecutive updates."):
		rates = []
		for _ in range(1_000):
			rates.append(agent._scheduled_learning_rate())  # pylint: disable=protected-access
			agent._total_updates += 1  # pylint: disable=protected-access

	with step("Assert: always the initial rate."):
		assert set(rates) == {3e-4}


def test_ppo_annealing_raises_instead_of_silently_freezing_when_its_schedule_runs_out() -> None:
	"""Regression: a schedule sized for 1.5M steps reached zero after the first environment and PPO trained
	frozen for 5M more steps. An exhausted schedule must now fail loudly."""
	with step("Arrange: an annealing PPO whose schedule covers 3 updates."):
		agent = PPOAgent(brain=_brain, critic_brain=_brain, make_env=_DiscreteEnv, learning_rate=3e-4)
		agent.set_total_steps(3 * agent.n_steps)

	with step("Act: three scheduled updates, decreasing."):
		rates = []
		for _ in range(3):
			rates.append(agent._scheduled_learning_rate())  # pylint: disable=protected-access
			agent._total_updates += 1  # pylint: disable=protected-access

	with step("Assert: positive and decreasing, then the fourth raises."):
		assert rates[0] > rates[1] > rates[2] > 0
		with pytest.raises(RuntimeError):
			agent._scheduled_learning_rate()  # pylint: disable=protected-access
