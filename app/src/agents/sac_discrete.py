"""SAC-Discrete Agent — Soft Actor-Critic for discrete action spaces."""

# Algorithm: Soft Actor-Critic for Discrete Action Settings (SAC-Discrete)
# Reference: Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019)
# https://arxiv.org/abs/1910.07207

# Libs imports
import numpy as np
import tensorflow as tf
from keras import optimizers

# App imports
from .sac import ReplayBuffer
from .tools.baseline import ABaseline, transition


# pylint: disable=too-many-instance-attributes
class SACDiscreteAgent(ABaseline):
	"""SAC with a categorical policy, twin Q-networks over all actions, and automatic entropy tuning."""

	# Differences from continuous SAC (Christodoulou, 2019):
	# - the actor outputs a categorical distribution and each critic outputs Q(s, .) for every
	#   action, so the soft state value V(s) = sum_a pi(a|s) [min Q(s, a) - alpha log pi(a|s)]
	#   is computed exactly instead of from a sampled action (no reparameterization);
	# - the actor loss is sum_a pi(a|s) [alpha log pi(a|s) - min Q(s, a)];
	# - the target entropy is target_entropy_ratio * log|A| (the paper uses 0.98 of the maximum).
	# Targets are soft-updated with tau, as in SAC v2. Updates run every train_freq env-steps
	# once learning_starts transitions are stored. Evaluation (learn=False) acts greedily.

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		brain,
		make_env,
		batch_size=64,
		buffer_capacity=100_000,
		gamma=0.99,
		learning_rate=3e-4,
		learning_starts=1_000,
		target_entropy_ratio=0.98,
		tau=0.005,
		train_freq=4,
	):
		super().__init__(brain, make_env)
		self.learn = True

		self.batch_size = batch_size
		self.gamma = gamma
		self.learning_starts = learning_starts
		self.tau = tau
		self.train_freq = train_freq
		self.env_steps = 0

		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])
		self.action_dim = int(tmp_env.action_space.n)
		tmp_env.close()

		# The brain factory builds every network: brain(output_dim, input_dim).
		self.actor = self.brain(self.action_dim, self.state_dim)
		self.critic_1 = self.brain(self.action_dim, self.state_dim)
		self.critic_2 = self.brain(self.action_dim, self.state_dim)
		self.target_critic_1 = self.brain(self.action_dim, self.state_dim)
		self.target_critic_2 = self.brain(self.action_dim, self.state_dim)
		self.target_critic_1.set_weights(self.critic_1.get_weights())
		self.target_critic_2.set_weights(self.critic_2.get_weights())

		self.actor_optimizer = optimizers.Adam(learning_rate=learning_rate)
		self.critic_optimizer = optimizers.Adam(learning_rate=learning_rate)
		self.alpha_optimizer = optimizers.Adam(learning_rate=learning_rate)
		self.target_entropy = target_entropy_ratio * float(np.log(self.action_dim))
		self.log_alpha = tf.Variable(0.0, trainable=True, dtype=tf.float32)

		self.buffer = ReplayBuffer(buffer_capacity, self.state_dim, 1)

	def set_learning(self, mode, output_dir=None):
		"""Switch between training (sample from the policy, update) and evaluation (greedy, frozen)."""
		self.learn = mode

	def act(self, env, state):
		"""Sample an action from the policy while learning; take the most probable one otherwise."""
		probabilities = self._policy(tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)).numpy()[0]
		if self.learn:
			action = int(np.random.choice(self.action_dim, p=probabilities / probabilities.sum()))
		else:
			action = int(np.argmax(probabilities))
		next_state, reward, terminated, truncated, _ = env.step(action)
		return next_state, reward, terminated, truncated, action

	def train(self, **kwargs):
		"""Store the transition and, on schedule, run one SAC-Discrete update on a sampled batch."""
		step_data = kwargs["step_data"]
		# Bootstrap through time-limit truncation: only a true termination zeroes the target.
		self.buffer.store(*transition(step_data))
		self.env_steps += 1

		if self.buffer.size >= max(self.batch_size, self.learning_starts) and self.env_steps % self.train_freq == 0:
			states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
			self._update_networks(
				tf.convert_to_tensor(states),
				tf.convert_to_tensor(actions[:, 0].astype(np.int32)),
				tf.convert_to_tensor(rewards),
				tf.convert_to_tensor(next_states),
				tf.convert_to_tensor(dones),
			)

	@tf.function
	def _policy(self, states):
		"""Categorical action probabilities."""
		return tf.nn.softmax(self.actor(states, training=False), axis=-1)

	@tf.function
	# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
	def _update_networks(self, states, actions, rewards, next_states, dones):
		"""One SAC-Discrete step: critics, actor, entropy temperature, then Polyak target update."""
		alpha = tf.exp(self.log_alpha)

		# 1. Critics: y = r + gamma (1 - d) V(s'), V exact over the categorical policy.
		next_log_probs = tf.nn.log_softmax(self.actor(next_states, training=True), axis=-1)
		next_probs = tf.exp(next_log_probs)
		next_q = tf.minimum(self.target_critic_1(next_states), self.target_critic_2(next_states))
		next_value = tf.reduce_sum(next_probs * (next_q - alpha * next_log_probs), axis=-1, keepdims=True)
		target = tf.stop_gradient(rewards + (1.0 - dones) * self.gamma * next_value)

		action_mask = tf.one_hot(actions, self.action_dim)
		with tf.GradientTape() as critic_tape:
			q_one = tf.reduce_sum(self.critic_1(states, training=True) * action_mask, axis=-1, keepdims=True)
			q_two = tf.reduce_sum(self.critic_2(states, training=True) * action_mask, axis=-1, keepdims=True)
			critic_loss = tf.reduce_mean(tf.square(q_one - target)) + tf.reduce_mean(tf.square(q_two - target))
		critic_variables = self.critic_1.trainable_variables + self.critic_2.trainable_variables
		self.critic_optimizer.apply_gradients(zip(critic_tape.gradient(critic_loss, critic_variables), critic_variables))

		# 2. Actor: minimize sum_a pi(a|s) [alpha log pi(a|s) - min Q(s, a)].
		q_min = tf.stop_gradient(tf.minimum(self.critic_1(states), self.critic_2(states)))
		with tf.GradientTape() as actor_tape:
			log_probs = tf.nn.log_softmax(self.actor(states, training=True), axis=-1)
			probs = tf.exp(log_probs)
			actor_loss = tf.reduce_mean(tf.reduce_sum(probs * (alpha * log_probs - q_min), axis=-1))
		self.actor_optimizer.apply_gradients(zip(actor_tape.gradient(actor_loss, self.actor.trainable_variables), self.actor.trainable_variables))

		# 3. Temperature: drive the policy entropy toward target_entropy.
		entropy = tf.stop_gradient(-tf.reduce_sum(probs * log_probs, axis=-1))
		with tf.GradientTape() as alpha_tape:
			alpha_loss = tf.reduce_mean(self.log_alpha * (entropy - self.target_entropy))
		self.alpha_optimizer.apply_gradients(zip(alpha_tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha]))

		# 4. Polyak averaging of the target critics.
		for target_network, source_network in ((self.target_critic_1, self.critic_1), (self.target_critic_2, self.critic_2)):
			for target_weight, source_weight in zip(target_network.weights, source_network.weights):
				target_weight.assign(target_weight * (1.0 - self.tau) + source_weight * self.tau)
