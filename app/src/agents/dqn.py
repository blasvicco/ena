"""DQN Agent — pure TensorFlow/Keras implementation."""

# Algorithm: Deep Q-Network with Experience Replay and Target Network
# Reference: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
# https://doi.org/10.1038/nature14236

# General imports
import collections
import random

# Libs imports
import numpy as np
import tensorflow as tf
from keras import Model, Input, optimizers
from keras.layers import Dense

# App imports
from .baseline import ABaseline


def _build_q_network(state_dim, action_dim, hidden=(64, 64)):
	"""Build a Q-network mapping states to per-action Q-values."""
	inputs = Input(shape=(state_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="relu")(_x_)
	outputs = Dense(action_dim)(_x_)
	return Model(inputs, outputs)


# pylint: disable=too-many-instance-attributes
class DQNAgent(ABaseline):
	"""Deep Q-Network with experience replay and a target network."""

	# Implements the algorithm described in Mnih et al. (2015), with:
	# - epsilon-greedy exploration with linear annealing
	# - fixed target network updated every `target_update_freq` steps
	# - uniform random experience replay

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		make_env,
		batch_size=64,
		buffer_size=50_000,
		epsilon_start=1.0,
		epsilon_end=0.01,
		epsilon_decay_steps=10_000,
		gamma=0.99,
		learning_rate=3e-4,
		learning_starts=1_000,
		target_update_freq=1_000,
		train_freq=4,
	):
		"""Initialize DQNAgent."""
		super().__init__()
		self.learn = True
		self.best_individual_idx = "DQN"

		# Hyperparameters
		self.batch_size = batch_size
		self.gamma = gamma
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay_steps = epsilon_decay_steps
		self.learning_starts = learning_starts
		self.target_update_freq = target_update_freq
		self.train_freq = train_freq

		# Infer dims from a temporary env
		tmp_env = make_env()
		state_dim = int(tmp_env.observation_space.shape[0])
		self.action_dim = int(tmp_env.action_space.n)
		tmp_env.close()

		# Networks
		self.q_net = _build_q_network(state_dim, self.action_dim)
		self.target_net = _build_q_network(state_dim, self.action_dim)
		self._sync_target()

		self.optimizer = optimizers.Adam(learning_rate=learning_rate)

		# Replay buffer
		self.replay_buffer = collections.deque(maxlen=buffer_size)
		self.current_step = 0

	# ------------------------------------------------------------------
	# ABaseline interface
	# ------------------------------------------------------------------

	def set_learning(self, mode):
		"""Switch between train and eval mode."""
		self.learn = mode

	def act(self, env, state):
		"""Epsilon-greedy action selection."""
		if self.learn:
			epsilon = max(
				self.epsilon_end,
				self.epsilon_start
				- (self.epsilon_start - self.epsilon_end)
				* self.current_step
				/ self.epsilon_decay_steps,
			)
			if random.random() < epsilon:
				action = env.action_space.sample()
			else:
				action = self._greedy_action(state)
		else:
			action = self._greedy_action(state)

		next_state, reward, terminated, truncated, _ = env.step(action)
		return next_state, reward, terminated, truncated, action

	def train(self, _, step_data):
		"""Store transition and run a gradient step on schedule."""
		self.replay_buffer.append(
			(
				step_data["state"],
				step_data["action"],
				step_data["reward"],
				step_data["next_state"],
				float(step_data["done"]),
			)
		)

		self.current_step += 1

		if (
			self.current_step >= self.learning_starts
			and self.current_step % self.train_freq == 0
		):
			self._gradient_step()

		if self.current_step % self.target_update_freq == 0:
			self._sync_target()

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _greedy_action(self, state):
		"""Return the action with the highest Q-value."""
		obs = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
		q_values = self.q_net(obs, training=False)
		return int(tf.argmax(q_values, axis=1).numpy()[0])

	def _sync_target(self):
		"""Hard-copy Q-network weights into the target network."""
		self.target_net.set_weights(self.q_net.get_weights())

	@tf.function
	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def _train_step(self, states, actions, rewards, next_states, dones):
		"""Single minibatch Bellman update."""
		# Compute TD targets using the frozen target network
		next_q = self.target_net(next_states, training=False)
		max_next_q = tf.reduce_max(next_q, axis=1)
		targets = rewards + self.gamma * max_next_q * (1.0 - dones)

		with tf.GradientTape() as tape:
			q_values = self.q_net(states, training=True)
			# Select Q-values for the taken actions
			action_mask = tf.one_hot(actions, self.action_dim)
			predicted = tf.reduce_sum(q_values * action_mask, axis=1)
			loss = tf.reduce_mean(tf.square(targets - predicted))

		grads = tape.gradient(loss, self.q_net.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

	def _gradient_step(self):
		"""Sample a minibatch and run one Bellman update."""
		if len(self.replay_buffer) < self.batch_size:
			return

		batch = random.sample(self.replay_buffer, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		self._train_step(
			tf.convert_to_tensor(states, dtype=tf.float32),
			tf.convert_to_tensor(actions, dtype=tf.int32),
			tf.convert_to_tensor(rewards, dtype=tf.float32),
			tf.convert_to_tensor(next_states, dtype=tf.float32),
			tf.convert_to_tensor(dones, dtype=tf.float32),
		)
