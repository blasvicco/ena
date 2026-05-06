"""SAC Agent — Soft Actor-Critic implementation."""

# Algorithm: Soft Actor-Critic (SAC v2)
# Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018)
# https://arxiv.org/abs/1812.05905

# Libs imports
import numpy as np
import tensorflow as tf
from keras import Input, Model, layers, optimizers

# App imports
from .baseline import ABaseline

# pylint: disable=too-many-instance-attributes
class ReplayBuffer:
	"""Simple circular buffer for experience replay."""

	def __init__(self, capacity, state_dim, action_dim):
		self.capacity = capacity
		self.pointer = 0
		self.size = 0

		self.states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
		self.rewards = np.zeros((capacity, 1), dtype=np.float32)
		self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.dones = np.zeros((capacity, 1), dtype=np.float32)

	# pylint: disable=too-many-arguments, too-many-positional-arguments
	def store(self, state, action, reward, next_state, done):
		"""Store a transition in the buffer."""
		self.states[self.pointer] = state
		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.next_states[self.pointer] = next_state
		self.dones[self.pointer] = done

		self.pointer = (self.pointer + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size):
		"""Randomly sample a batch of transitions."""
		indices = np.random.randint(0, self.size, size=batch_size)
		return (
			self.states[indices],
			self.actions[indices],
			self.rewards[indices],
			self.next_states[indices],
			self.dones[indices],
		)

# pylint: disable=too-many-instance-attributes
class SACAgent(ABaseline):
	"""Soft Actor-Critic Agent with Twin-Q and automatic entropy tuning."""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		brain,
		make_env,
		batch_size=256,
		buffer_capacity=100000,
		gamma=0.99,
		learning_rate=3e-4,
		tau=0.005,
		update_frequency=20,
		gradient_steps=None,
	):
		super().__init__(brain, make_env)
		self.learn = True

		# Hyperparameters
		self.batch_size = batch_size
		self.gamma = gamma
		self.tau = tau
		self.update_frequency = update_frequency
		self.gradient_steps = (
			gradient_steps if gradient_steps is not None else update_frequency
		)
		self.env_steps = 0

		# Infer dims
		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])
		self.action_dim = int(tmp_env.action_space.shape[0])
		tmp_env.close()

		# Networks
		# SAC v2 needs specifically structured actor/critic.
		# We'll use the 'brain' as a factory for the shared parts.
		self.actor = self.brain(self.action_dim * 2, self.state_dim)
		self.critic_1 = self._build_critic()
		self.critic_2 = self._build_critic()

		# Target Networks
		self.target_critic_1 = self._build_critic()
		self.target_critic_2 = self._build_critic()
		self.target_critic_1.set_weights(self.critic_1.get_weights())
		self.target_critic_2.set_weights(self.critic_2.get_weights())

		# Optimizers
		self.actor_optimizer = optimizers.Adam(learning_rate=learning_rate)
		self.critic_optimizer = optimizers.Adam(learning_rate=learning_rate)

		# Entropy Tuning (Standard SAC v2)
		self.target_entropy = -float(self.action_dim)
		self.log_alpha = tf.Variable(0.0, trainable=True, dtype=tf.float32)
		self.alpha_optimizer = optimizers.Adam(learning_rate=learning_rate)

		# Buffer
		self.buffer = ReplayBuffer(buffer_capacity, self.state_dim, self.action_dim)

	def _build_critic(self):
		"""Builds a Q-network that takes (state, action) as input."""
		state_input = Input(shape=(self.state_dim,))
		action_input = Input(shape=(self.action_dim,))
		concatenated = layers.Concatenate()([state_input, action_input])

		# We reuse the same hidden architecture as the brain
		# The brain factory expects (action_output_dim, input_dim)
		core_model = self.brain(1, self.state_dim + self.action_dim)
		output_layer = core_model(concatenated)

		return Model(inputs=[state_input, action_input], outputs=output_layer)

	# pylint: disable=too-many-arguments
	def act(self, env, state):
		"""Sample action from the policy using Gaussian reprocessing."""
		obs_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
		mean, log_standard_deviation = self._get_policy_outputs(obs_tensor)
		standard_deviation = tf.exp(log_standard_deviation)

		# Reparameterization trick for sampling
		gaussian_noise = tf.random.normal(tf.shape(mean))
		action_raw = mean + gaussian_noise * standard_deviation
		action_squashed = tf.tanh(action_raw)

		# Scale/Clip action to env space happens within gym if needed,
		# but HalfCheetah expects [-1, 1] which tanh provides.
		action = action_squashed.numpy()[0]

		next_obs, reward, terminated, truncated, _ = env.step(action)
		return next_obs, reward, terminated, truncated, action

	@tf.function
	def _get_policy_outputs(self, state_tensor):
		"""Splits actor output into mean and log_standard_deviation."""
		outputs = self.actor(state_tensor, training=False)
		mean, log_standard_deviation = tf.split(outputs, 2, axis=-1)
		log_standard_deviation = tf.clip_by_value(log_standard_deviation, -20.0, 2.0)
		return mean, log_standard_deviation

	def train(self, **kwargs):
		"""Store transition and update networks if buffer has enough data."""
		step_data = kwargs["step_data"]
		self.buffer.store(
			step_data["state"],
			step_data["action"],
			step_data["reward"],
			step_data["next_state"],
			float(step_data["done"]),
		)

		self.env_steps += 1

		if (
			self.buffer.size >= self.batch_size
			and self.env_steps % self.update_frequency == 0
		):
			for _ in range(self.gradient_steps):
				(
					batch_states,
					batch_actions,
					batch_rewards,
					batch_next_states,
					batch_dones,
				) = self.buffer.sample(self.batch_size)

				self._update_networks(
					tf.convert_to_tensor(batch_states),
					tf.convert_to_tensor(batch_actions),
					tf.convert_to_tensor(batch_rewards),
					tf.convert_to_tensor(batch_next_states),
					tf.convert_to_tensor(batch_dones),
				)

	@tf.function
	# pylint: disable=too-many-locals
	def _update_networks(self, states, actions, rewards, next_states, dones):
		"""Main SAC optimization step."""
		alpha = tf.exp(self.log_alpha)

		# 1. Update Critics
		with tf.GradientTape(persistent=True) as critic_tape:
			# Target action selection
			actor_outputs = self.actor(next_states, training=True)
			(
				next_mean,
				next_log_standard_deviation,
			) = tf.split(actor_outputs, 2, axis=-1)
			next_log_standard_deviation = tf.clip_by_value(
				next_log_standard_deviation, -20.0, 2.0
			)
			next_standard_deviation = tf.exp(next_log_standard_deviation)

			next_noise = tf.random.normal(tf.shape(next_mean))
			next_actions_raw = next_mean + next_noise * next_standard_deviation
			next_actions = tf.tanh(next_actions_raw)

			# Next log_prob
			next_log_probabilities = self._calculate_log_probability(
				next_actions_raw, next_actions, next_mean, next_log_standard_deviation
			)

			# Target Q values
			target_q1 = self.target_critic_1([next_states, next_actions])
			target_q2 = self.target_critic_2([next_states, next_actions])
			target_q_minimum = (
				tf.minimum(target_q1, target_q2) - alpha * next_log_probabilities
			)

			target_q_final = rewards + (1.0 - dones) * self.gamma * target_q_minimum

			# Current Q values
			current_q1 = self.critic_1([states, actions])
			current_q2 = self.critic_2([states, actions])

			critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q_final))
			critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q_final))
			critic_loss_total = critic1_loss + critic2_loss

		# Apply Critic Gradients jointly (preventing optimizer variable mismatch)
		critic_variables = (
			self.critic_1.trainable_variables + self.critic_2.trainable_variables
		)
		critic_gradients = critic_tape.gradient(critic_loss_total, critic_variables)
		self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

		# 2. Update Actor
		with tf.GradientTape() as actor_tape:
			actor_outputs = self.actor(states, training=True)
			(
				current_mean,
				current_log_standard_deviation,
			) = tf.split(actor_outputs, 2, axis=-1)
			current_log_standard_deviation = tf.clip_by_value(
				current_log_standard_deviation, -20.0, 2.0
			)
			current_standard_deviation = tf.exp(current_log_standard_deviation)

			current_noise = tf.random.normal(tf.shape(current_mean))
			current_actions_raw = (
				current_mean + current_noise * current_standard_deviation
			)
			current_actions = tf.tanh(current_actions_raw)

			current_log_probabilities = self._calculate_log_probability(
				current_actions_raw,
				current_actions,
				current_mean,
				current_log_standard_deviation,
			)

			q1_value = self.critic_1([states, current_actions])
			q2_value = self.critic_2([states, current_actions])
			q_minimum = tf.minimum(q1_value, q2_value)

			actor_loss = tf.reduce_mean(alpha * current_log_probabilities - q_minimum)

		# Apply Actor Gradients
		actor_gradients = actor_tape.gradient(
			actor_loss, self.actor.trainable_variables
		)
		self.actor_optimizer.apply_gradients(
			zip(actor_gradients, self.actor.trainable_variables)
		)

		# 3. Update Alpha (Entropy Coefficient)
		with tf.GradientTape() as alpha_tape:
			alpha_loss = -tf.reduce_mean(
				self.log_alpha * (current_log_probabilities + self.target_entropy)
			)

		alpha_gradients = alpha_tape.gradient(alpha_loss, [self.log_alpha])
		self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))

		# 4. Soft Update Target Networks
		self._soft_update(self.target_critic_1, self.critic_1)
		self._soft_update(self.target_critic_2, self.critic_2)

	def _calculate_log_probability(
		self, raw_action, squashed_action, mean, log_standard_deviation
	):
		"""Gaussian log-probability with tanh squashing correction."""
		standard_deviation = tf.exp(log_standard_deviation)
		log_probability = -0.5 * (
			tf.square((raw_action - mean) / (standard_deviation + 1e-8))
			+ 2.0 * log_standard_deviation
			+ tf.math.log(2.0 * np.pi)
		)
		log_probability = tf.reduce_sum(log_probability, axis=-1, keepdims=True)
		# Tanh correction: log(1 - tanh(x)^2)
		log_probability -= tf.reduce_sum(
			tf.math.log(1.0 - tf.square(squashed_action) + 1e-6),
			axis=-1,
			keepdims=True,
		)
		return log_probability

	def _soft_update(self, target_network, source_network):
		"""Polyak averaging for target networks."""
		for target_weight, source_weight in zip(
			target_network.weights, source_network.weights
		):
			target_weight.assign(
				target_weight * (1.0 - self.tau) + source_weight * self.tau
			)
