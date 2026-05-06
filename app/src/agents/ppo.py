"""PPO Agent — pure TensorFlow/Keras implementation."""

# Algorithm: Proximal Policy Optimization (Clip variant)
# Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
# https://arxiv.org/abs/1707.06347

# Libs imports
import numpy as np
import tensorflow as tf
from keras import optimizers

# App imports
from .baseline import ABaseline
from .normalizers import _IdentityNormalizer, _RunningNormalizer

# pylint: disable=too-many-instance-attributes
class PPOAgent(ABaseline):
	"""Proximal Policy Optimization with clipped surrogate objective."""

	# Implements the actor-critic variant with Generalized Advantage Estimation (GAE)
	# as described in Schulman et al. (2017).

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		brain,
		make_env,
		critic_brain=None,
		batch_size=64,
		clip_ratio=0.2,
		entropy_coef=0.0,  # No evidence entropy helps continuous control
		gae_lambda=0.95,
		gamma=0.99,
		learning_rate=3e-4,
		n_epochs=10,
		n_steps=2048,
		value_coef=0.5,
	):
		"""Initialize PPOAgent."""
		super().__init__(brain, make_env)
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
		self.initial_lr = learning_rate

		# Annealing state
		self._total_updates = 0
		self._max_updates = 1  # Will be set in set_episodes

		# Infer dims from a temporary env
		tmp_env = self.make_env()
		self.is_continuous = hasattr(tmp_env.action_space, "low")
		self._mode = "continuous" if self.is_continuous else "discrete"
		state_dim = int(tmp_env.observation_space.shape[0])
		self.action_dim = (
			int(tmp_env.action_space.shape[0])
			if self.is_continuous
			else int(tmp_env.action_space.n)
		)
		actor_output_dim = self.action_dim
		tmp_env.close()

		# Networks
		self.actor = self.brain(actor_output_dim, state_dim)
		self.critic = (critic_brain or self.brain)(1, state_dim)

		# State-independent log_std for Gaussian policy (continuous only)
		if self.is_continuous:
			# Initializing log_std to -2.0 (std ~ 0.14) is standard for MuJoCo
			self.log_std = tf.Variable(
				tf.fill([self.action_dim], -2.0),
				trainable=True,
			)

		self.optimizer = optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

		# Observation & Reward normalization
		# For continuous: running mean/variance normalization (critical for MuJoCo)
		# For discrete: identity pass-through (CartPole doesn't need it)
		if self.is_continuous:
			self._obs_normalizer = _RunningNormalizer(state_dim)
			self._reward_normalizer = _RunningNormalizer(1)
		else:
			self._obs_normalizer = _IdentityNormalizer(state_dim)
			self._reward_normalizer = _IdentityNormalizer(1)

		# Last action values
		self._last_log_prob = None
		self._last_value = None
		self._last_unclipped_action = None

		# Rollout buffer (filled up to n_steps, then flushed)
		self._buf_states = []
		self._buf_actions = []
		self._buf_rewards = []
		self._buf_boundaries = []
		self._buf_trunc_values = []
		self._buf_log_probs = []
		self._buf_values = []

	# ------------------------------------------------------------------
	# ABaseline interface
	# ------------------------------------------------------------------

	def set_episodes(self, episodes):
		"""Set episodes and compute max updates for LR annealing."""
		super().set_episodes(episodes)
		# Assume 1000 steps per episode for HalfCheetah/Mujoco
		total_steps = sum(episodes) * 1000
		self._max_updates = max(1, total_steps // self.n_steps)

	def set_learning(self, mode):
		"""Switch between train and eval mode."""
		self.learn = mode

	def act(self, env, state):
		"""Sample action from the current policy."""
		state = self._obs_normalizer.normalize(state)
		obs = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
		logits = self.actor(obs, training=False)
		value = self.critic(obs, training=False)

		action, log_prob = getattr(self, f"_sample_{self._mode}")(logits)

		# Cache for train()
		self._last_log_prob = float(tf.squeeze(log_prob).numpy())
		self._last_value = float(value.numpy()[0, 0])

		next_obs, reward, terminated, truncated, _ = env.step(action)
		return next_obs, reward, terminated, truncated, action

	def _sample_continuous(self, logits):
		"""Sample action from Gaussian policy."""
		mean = logits
		log_std = tf.clip_by_value(self.log_std, -20.0, 2.0)
		std = tf.exp(log_std)
		noise = std * tf.random.normal(tf.shape(mean))
		action_tensor = mean + noise
		self._last_unclipped_action = action_tensor.numpy()[0]
		log_prob = self._log_prob(logits, action_tensor)
		action = np.clip(self._last_unclipped_action, -1.0, 1.0)
		return action, log_prob

	def _sample_discrete(self, logits):
		"""Sample action from categorical policy."""
		dist = tf.random.categorical(logits, 1)
		action = int(dist.numpy()[0, 0])
		log_prob = self._log_prob(logits, action)
		return action, log_prob

	def train(self, **kwargs):
		"""Buffer the transition; run PPO update when buffer is full."""
		step_data = kwargs["step_data"]

		# Buffer state, action, reward (mode-specific normalization)
		getattr(self, f"_buffer_step_{self._mode}")(step_data)

		is_boundary = float(step_data.get("done", False))
		terminated = float(step_data.get("terminated", is_boundary))

		# Evaluate truncation bootstrap immediately
		if is_boundary and not terminated:
			next_state = self._obs_normalizer.normalize(step_data["next_state"])
			next_obs = tf.convert_to_tensor(next_state[np.newaxis], dtype=tf.float32)
			trunc_value = float(self.critic(next_obs, training=False).numpy()[0, 0])
		else:
			trunc_value = 0.0

		self._buf_boundaries.append(is_boundary)
		self._buf_trunc_values.append(trunc_value)
		self._buf_log_probs.append(self._last_log_prob)
		self._buf_values.append(self._last_value)

		if len(self._buf_states) >= self.n_steps:
			if is_boundary:
				# On true termination, bootstrap is 0.
				# On truncation, use the trunc_value we already computed above.
				last_value = trunc_value
			else:
				next_state = self._obs_normalizer.normalize(step_data["next_state"])
				last_obs = tf.convert_to_tensor(
					next_state[np.newaxis], dtype=tf.float32
				)
				last_value = float(self.critic(last_obs, training=False).numpy()[0, 0])
			self._update(last_value)
			self._flush_buffer()

	def _buffer_step_continuous(self, step_data):
		"""Buffer state/action/reward with continuous-specific normalization."""
		raw_state = step_data["state"]
		if self.learn:
			self._obs_normalizer.update(raw_state[np.newaxis])
		self._buf_states.append(self._obs_normalizer.normalize(raw_state))
		self._buf_actions.append(self._last_unclipped_action)

		# Reward scaling: divide by running std (no mean subtraction)
		reward = step_data["reward"]
		self._reward_normalizer.update(np.array([[reward]]))
		scaled_reward = float(reward / np.sqrt(self._reward_normalizer.var[0] + 1e-8))
		self._buf_rewards.append(scaled_reward)

	def _buffer_step_discrete(self, step_data):
		"""Buffer state/action/reward for discrete environments."""
		self._buf_states.append(step_data["state"])
		self._buf_actions.append(step_data["action"])
		self._buf_rewards.append(step_data["reward"])

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _log_prob(self, logits, action):
		"""Log probability of an action given logits (dispatches by mode)."""
		return getattr(self, f"_log_prob_{self._mode}")(logits, action)

	def _log_prob_continuous(self, logits, action):
		"""Gaussian log-probability."""
		mean = logits
		log_std = tf.clip_by_value(self.log_std, -20.0, 2.0)
		std = tf.exp(log_std)
		variance = tf.square(std)
		log_prob_density = -0.5 * (
			tf.square(action - mean) / (variance + 1e-8)
			+ 2 * log_std
			+ tf.math.log(2 * np.pi)
		)
		return tf.reduce_sum(log_prob_density, axis=-1)

	def _log_prob_discrete(self, logits, action):
		"""Categorical log-probability."""
		log_probs = tf.nn.log_softmax(logits)
		return log_probs[0, action]

	def _compute_gae(self, last_value):
		"""Generalized Advantage Estimation (Schulman et al. 2016)."""
		rewards = np.array(self._buf_rewards, dtype=np.float32)
		boundaries = np.array(self._buf_boundaries, dtype=np.float32)
		trunc_values = np.array(self._buf_trunc_values, dtype=np.float32)
		values = np.array(self._buf_values + [last_value], dtype=np.float32)

		advantages = np.zeros_like(rewards)
		gae = 0.0
		for _t_ in reversed(range(len(rewards))):
			if boundaries[_t_]:
				next_val = trunc_values[_t_]
			else:
				next_val = values[_t_ + 1]

			delta = rewards[_t_] + self.gamma * next_val - values[_t_]
			gae = delta + self.gamma * self.gae_lambda * (1 - boundaries[_t_]) * gae
			advantages[_t_] = gae

		returns = advantages + values[:-1]
		return advantages, returns

	def _update(self, last_value):
		"""Run PPO gradient updates over the collected rollout."""
		# 1. Linear LR Annealing
		frac = 1.0 - (self._total_updates / self._max_updates)
		self.optimizer.learning_rate.assign(self.initial_lr * max(frac, 0.0))
		self._total_updates += 1

		# 2. Compute Advantages
		advantages, returns = self._compute_gae(last_value)

		states = tf.convert_to_tensor(self._buf_states, dtype=tf.float32)
		# Use float32 for continuous envs, int32 for discrete
		actions = tf.convert_to_tensor(
			self._buf_actions,
			dtype=tf.float32 if self.is_continuous else tf.int32,
		)
		old_log_probs = tf.convert_to_tensor(self._buf_log_probs, dtype=tf.float32)
		advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
		returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

		states_len = len(self._buf_states)
		indices = np.arange(states_len)

		for _ in range(self.n_epochs):
			np.random.shuffle(indices)
			for start in range(0, states_len, self.batch_size):
				batch_idx = indices[start : start + self.batch_size]

				# Normalize advantages at the micro-batch level
				mb_adv = tf.gather(advantages_tf, batch_idx)
				mb_adv = (mb_adv - tf.reduce_mean(mb_adv)) / (
					tf.math.reduce_std(mb_adv) + 1e-8
				)

				self._gradient_step(
					tf.gather(states, batch_idx),
					tf.gather(actions, batch_idx),
					tf.gather(old_log_probs, batch_idx),
					mb_adv,
					tf.gather(returns_tf, batch_idx),
				)

	@tf.function
	# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
	def _gradient_step(self, states, actions, old_log_probs, advantages, returns):
		"""Single minibatch gradient step."""
		with tf.GradientTape() as tape:
			logits = self.actor(states, training=True)
			values = tf.squeeze(self.critic(states, training=True), axis=1)

			# Policy loss (clipped surrogate objective) — dispatched by mode
			log_probs, entropy = getattr(self, f"_compute_policy_{self._mode}")(
				logits, actions
			)

			ratio = tf.exp(log_probs - old_log_probs)
			clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
			policy_loss = -tf.reduce_mean(
				tf.minimum(ratio * advantages, clipped * advantages)
			)

			# Value loss
			value_loss = tf.reduce_mean(tf.square(returns - values))

			loss = (
				policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
			)

		# Split params and clip gradients separately to avoid value loss dominating
		actor_params = getattr(self, f"_get_actor_params_{self._mode}")()
		critic_params = self.critic.trainable_variables
		all_params = actor_params + critic_params

		grads = tape.gradient(loss, all_params)

		actor_grads = grads[: len(actor_params)]
		critic_grads = grads[len(actor_params) :]

		actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
		critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)

		self.optimizer.apply_gradients(zip(actor_grads + critic_grads, all_params))

	def _compute_policy_continuous(self, logits, actions):
		"""Compute log-probabilities and entropy for Gaussian policy."""
		log_probs = self._log_prob(logits, actions)
		log_std = tf.clip_by_value(self.log_std, -20.0, 2.0)
		entropy = tf.reduce_mean(
			tf.reduce_sum(
				0.5 + 0.5 * tf.math.log(2 * np.pi) + log_std,
				axis=-1,
			)
		)
		return log_probs, entropy

	def _compute_policy_discrete(self, logits, actions):
		"""Compute log-probabilities and entropy for categorical policy."""
		log_probs = tf.reduce_sum(
			tf.one_hot(actions, logits.shape[-1]) * tf.nn.log_softmax(logits),
			axis=1,
		)
		# Entropy bonus (encourages exploration)
		probs = tf.nn.softmax(logits)
		entropy = -tf.reduce_mean(
			tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
		)
		return log_probs, entropy

	def _get_actor_params_continuous(self):
		"""Actor trainable variables including log_std."""
		return list(self.actor.trainable_variables) + [self.log_std]

	def _get_actor_params_discrete(self):
		"""Actor trainable variables."""
		return list(self.actor.trainable_variables)

	def _flush_buffer(self):
		"""Clear rollout buffer."""
		self._buf_states.clear()
		self._buf_actions.clear()
		self._buf_rewards.clear()
		self._buf_boundaries.clear()
		self._buf_trunc_values.clear()
		self._buf_log_probs.clear()
		self._buf_values.clear()
