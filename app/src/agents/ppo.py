"""PPO Agent — pure TensorFlow/Keras implementation."""

# Algorithm: Proximal Policy Optimization (Clip variant)
# Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
# https://arxiv.org/abs/1707.06347

# Libs imports
import numpy as np
import tensorflow as tf
from keras import Model, Input, optimizers
from keras.layers import Dense

# App imports
from .baseline import ABaseline


def _build_network(input_dim, output_dim, hidden=(64, 64), output_activation=None):
	"""Shared factory for actor and critic networks."""
	inputs = Input(shape=(input_dim,))
	_x_ = inputs
	for units in hidden:
		_x_ = Dense(units, activation="tanh")(_x_)
	outputs = Dense(output_dim, activation=output_activation)(_x_)
	return Model(inputs, outputs)


# pylint: disable=too-many-instance-attributes
class PPOAgent(ABaseline):
	"""Proximal Policy Optimization with clipped surrogate objective."""

	# Implements the actor-critic variant with Generalized Advantage Estimation (GAE)
	# as described in Schulman et al. (2017).

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		make_env,
		batch_size=64,
		clip_ratio=0.2,
		entropy_coef=0.01,
		gae_lambda=0.95,
		gamma=0.99,
		learning_rate=3e-4,
		n_epochs=10,
		n_steps=2048,
		value_coef=0.5,
	):
		"""Initialize PPOAgent."""
		super().__init__()
		self.learn = True
		self.best_individual_idx = "PPO"

		# Hyperparameters
		self.n_steps = n_steps
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_ratio = clip_ratio
		self.entropy_coef = entropy_coef
		self.value_coef = value_coef

		# Infer dims from a temporary env
		tmp_env = make_env()
		state_dim = int(tmp_env.observation_space.shape[0])
		action_dim = int(tmp_env.action_space.n)
		tmp_env.close()

		# Networks
		self.actor = _build_network(state_dim, action_dim)
		self.critic = _build_network(state_dim, 1)
		self.optimizer = optimizers.Adam(learning_rate=learning_rate)

		# Last action values
		self._last_log_prob = None
		self._last_value = None

		# Rollout buffer (filled up to n_steps, then flushed)
		self._buf_states = []
		self._buf_actions = []
		self._buf_rewards = []
		self._buf_dones = []
		self._buf_log_probs = []
		self._buf_values = []

	# ------------------------------------------------------------------
	# ABaseline interface
	# ------------------------------------------------------------------

	def set_learning(self, mode):
		"""Switch between train and eval mode."""
		self.learn = mode

	def act(self, env, state):
		"""Sample action from the current policy."""
		obs = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
		logits = self.actor(obs, training=False)
		value = self.critic(obs, training=False)

		dist = tf.random.categorical(logits, 1)
		action = int(dist.numpy()[0, 0])
		log_prob = self._log_prob(logits, action)

		# Cache for train()
		self._last_log_prob = float(log_prob.numpy())
		self._last_value = float(value.numpy()[0, 0])

		next_state, reward, terminated, truncated, _ = env.step(action)
		return next_state, reward, terminated, truncated, action

	def train(self, env, step_data):
		"""Buffer the transition; run PPO update when buffer is full."""
		self._buf_states.append(step_data["state"])
		self._buf_actions.append(step_data["action"])
		self._buf_rewards.append(step_data["reward"])
		self._buf_dones.append(float(step_data["done"]))
		self._buf_log_probs.append(self._last_log_prob)
		self._buf_values.append(self._last_value)

		if len(self._buf_states) >= self.n_steps:
			# Bootstrap value for the last state
			last_obs = tf.convert_to_tensor(
				step_data["next_state"][np.newaxis], dtype=tf.float32
			)
			last_value = float(self.critic(last_obs, training=False).numpy()[0, 0])
			self._update(last_value)
			self._flush_buffer()

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _log_prob(self, logits, action):
		"""Log probability of a discrete action given logits."""
		log_probs = tf.nn.log_softmax(logits)
		return log_probs[0, action]

	def _compute_gae(self, last_value):
		"""Generalized Advantage Estimation (Schulman et al. 2016)."""
		rewards = np.array(self._buf_rewards, dtype=np.float32)
		dones = np.array(self._buf_dones, dtype=np.float32)
		values = np.array(self._buf_values + [last_value], dtype=np.float32)

		advantages = np.zeros_like(rewards)
		gae = 0.0
		for _t_ in reversed(range(len(rewards))):
			delta = (
				rewards[_t_]
				+ self.gamma * values[_t_ + 1] * (1 - dones[_t_])
				- values[_t_]
			)
			gae = delta + self.gamma * self.gae_lambda * (1 - dones[_t_]) * gae
			advantages[_t_] = gae

		returns = advantages + values[:-1]
		return advantages, returns

	def _update(self, last_value):
		"""Run PPO gradient updates over the collected rollout."""
		advantages, returns = self._compute_gae(last_value)

		# Normalize advantages
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		states = tf.convert_to_tensor(self._buf_states, dtype=tf.float32)
		actions = tf.convert_to_tensor(self._buf_actions, dtype=tf.int32)
		old_log_probs = tf.convert_to_tensor(self._buf_log_probs, dtype=tf.float32)
		advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
		returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

		states_len = len(self._buf_states)
		indices = np.arange(states_len)

		for _ in range(self.n_epochs):
			np.random.shuffle(indices)
			for start in range(0, states_len, self.batch_size):
				batch_idx = indices[start : start + self.batch_size]
				self._gradient_step(
					tf.gather(states, batch_idx),
					tf.gather(actions, batch_idx),
					tf.gather(old_log_probs, batch_idx),
					tf.gather(advantages_tf, batch_idx),
					tf.gather(returns_tf, batch_idx),
				)

	@tf.function
	# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
	def _gradient_step(self, states, actions, old_log_probs, advantages, returns):
		"""Single minibatch gradient step."""
		with tf.GradientTape() as tape:
			logits = self.actor(states, training=True)
			values = tf.squeeze(self.critic(states, training=True), axis=1)

			# Policy loss (clipped surrogate objective)
			log_probs = tf.reduce_sum(
				tf.one_hot(actions, logits.shape[-1]) * tf.nn.log_softmax(logits),
				axis=1,
			)
			ratio = tf.exp(log_probs - old_log_probs)
			clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
			policy_loss = -tf.reduce_mean(
				tf.minimum(ratio * advantages, clipped * advantages)
			)

			# Value loss
			value_loss = tf.reduce_mean(tf.square(returns - values))

			# Entropy bonus (encourages exploration)
			probs = tf.nn.softmax(logits)
			entropy = -tf.reduce_mean(
				tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
			)

			loss = (
				policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
			)

		params = self.actor.trainable_variables + self.critic.trainable_variables
		grads = tape.gradient(loss, params)
		self.optimizer.apply_gradients(zip(grads, params))

	def _flush_buffer(self):
		"""Clear rollout buffer."""
		self._buf_states.clear()
		self._buf_actions.clear()
		self._buf_rewards.clear()
		self._buf_dones.clear()
		self._buf_log_probs.clear()
		self._buf_values.clear()
