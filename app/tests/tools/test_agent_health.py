"""Tests for agent_health.py: the internal quantities logged during training, read without side effects."""

# Libs imports
import numpy as np

# App imports
from helpers import step
import half_cheetah
from tools.agent_health import agent_health


def test_sac_health_reads_its_temperature_and_q_values_without_touching_the_global_random_state() -> None:
	"""Reading SAC's health never moves the replicate's random stream, so logging cannot change a run."""
	# SAC's temperature and mean Q over a fixed sample of its buffer; reading it never moves the replicate's NumPy
	# random stream, so logging cannot change a run.
	with step("Arrange: a SAC agent with 300 random transitions in its buffer."):
		agent = half_cheetah.build_agent("SAC", max_agent_threads=1)
		rng = np.random.default_rng(1)
		for _ in range(300):
			agent.buffer.store(rng.normal(size=17), rng.uniform(-1, 1, size=6), 1.0, rng.normal(size=17), 0.0)
		before = np.random.get_state()[1].copy()

	with step("Act: read the health."):
		health = agent_health(agent)

	with step("Assert: temperature 1 at the start, a finite mean Q, and the global stream untouched."):
		assert health["alpha"] == 1.0 and np.isfinite(health["q_mean"])
		assert np.array_equal(np.random.get_state()[1], before)


def test_ppo_health_reads_its_learning_rate_and_policy_std() -> None:
	"""PPO's learning rate (constant in HalfCheetah) and its policy's standard deviation."""
	with step("Act: a fresh HalfCheetah PPO."):
		health = agent_health(half_cheetah.build_agent("PPO", max_agent_threads=1))

	with step("Assert: the zoo's learning rate and exp(-2)."):
		assert set(health) == {"learning_rate", "policy_std"}
		assert np.isclose(health["policy_std"], np.exp(-2.0))
