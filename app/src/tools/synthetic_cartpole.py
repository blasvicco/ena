"""Fast, dependency-light re-implementation of gymnasium's CartPole-v1 physics.

Used by benchmark_ena.py to exercise ENAgent's real control flow (selection, trust,
HallOfFame, EnvironmentRecognizer) without gym.make()'s registry lookup and wrapper-stack
construction overhead — the dominant per-episode cost given ENA builds a fresh env per
candidate evaluation (thread_eval_task) and per episode-start probe (NeutralProbeRecognizer).
The governing equations, constants, and termination/reward semantics are identical to
gymnasium's CartPoleEnv, so scores and physics-fingerprint statistics stay comparable.
"""

# General imports
import math

# Libs imports
import numpy as np
from gymnasium.spaces import Box, Discrete


# pylint: disable=too-many-instance-attributes
class SyntheticCartPoleEnv:
	"""CartPole-v1 physics/termination/reward, parameterized by gravity/masspole/length."""

	MASSCART = 1.0
	FORCE_MAG = 10.0
	TAU = 0.02
	X_THRESHOLD = 2.4
	THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360
	MAX_EPISODE_STEPS = 500

	def __init__(self, gravity: float, masspole: float, length: float, name: str) -> None:
		self.gravity = gravity
		self.masspole = masspole
		self.length = length
		self.name = name
		self.total_mass = masspole + self.MASSCART
		self.polemass_length = masspole * length

		self.action_space = Discrete(2)
		high = np.array([self.X_THRESHOLD * 2, np.inf, self.THETA_THRESHOLD_RADIANS * 2, np.inf], dtype=np.float32)
		self.observation_space = Box(-high, high, dtype=np.float32)

		self._state = np.zeros(4, dtype=np.float32)
		self._steps = 0
		self._rng = np.random.default_rng()

	@property
	def unwrapped(self) -> "SyntheticCartPoleEnv":
		"""Match gymnasium's env.unwrapped convention (ena.py/recognizer.py read physics off it)."""
		return self

	def reset(self, seed: int = None) -> tuple:
		"""Reset to a small random state, matching CartPole-v1's reset distribution."""
		if seed is not None:
			self._rng = np.random.default_rng(seed)
		self._state = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
		self._steps = 0
		return np.copy(self._state), {}

	def step(self, action: int) -> tuple:
		"""Advance one physics step using CartPole-v1's governing equations."""
		cart_x, x_dot, theta, theta_dot = self._state
		force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
		costheta = math.cos(theta)
		sintheta = math.sin(theta)

		temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
		theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
		x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

		cart_x = cart_x + self.TAU * x_dot
		x_dot = x_dot + self.TAU * x_acc
		theta = theta + self.TAU * theta_dot
		theta_dot = theta_dot + self.TAU * theta_acc

		self._state = np.array([cart_x, x_dot, theta, theta_dot], dtype=np.float32)
		self._steps += 1

		terminated = bool(
			cart_x < -self.X_THRESHOLD or cart_x > self.X_THRESHOLD or theta < -self.THETA_THRESHOLD_RADIANS or theta > self.THETA_THRESHOLD_RADIANS
		)
		truncated = self._steps >= self.MAX_EPISODE_STEPS
		return np.copy(self._state), 1.0, terminated, truncated, {}

	def close(self) -> None:
		"""No resources to release — kept for interface parity with gymnasium envs."""


def make_synthetic_env(
	gravity: float = 9.8,
	masspole: float = 0.1,
	name: str = "Earth",
	length: float = 0.5,
) -> SyntheticCartPoleEnv:
	"""Build a SyntheticCartPoleEnv — drop-in for cartpole_challenge.py's make_env, minus gym.make()'s
	registry/wrapper overhead."""
	return SyntheticCartPoleEnv(gravity=gravity, masspole=masspole, length=length, name=name)
