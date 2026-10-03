"""Neural selection driver: contextual bandit mapping context to niche ID via REINFORCE."""

# Libs imports
import numpy as np
import tensorflow as tf
from keras import Model, Input, initializers, optimizers
from keras.layers import Dense

# App imports
from agents.ena.skills.selection.driver.abstract import ADriver
from agents.ena.tools.reward_normalizer import normalize_reward


# pylint: disable=too-many-instance-attributes
class Instance(ADriver):
	"""Contextual bandit that maps context → best niche ID via REINFORCE."""

	# pylint: disable=too-many-arguments, too-many-positional-arguments
	def __init__(
		self,
		context_dim,
		buffer_size=64,
		controller_episode_rate=0.1,
		controller_action_epsilon=0.2,
		hidden_sizes=(64, 32),
		learning_rate=1e-3,
		max_niches=32,
		**_kwargs,
	):
		"""Constructor"""
		self._active = False
		self._active_context = None
		self._active_niche = None
		self._current_niche = None
		self._smoothed_probs = None

		self._context_dim = context_dim
		self._episode_rate = controller_episode_rate
		self._action_epsilon = controller_action_epsilon
		self._frozen = False
		self._buffer_size = buffer_size
		self._hidden_sizes = hidden_sizes
		self._learning_rate = learning_rate
		self._current_niches = max_niches

		self._model = self._build_model(context_dim, self._current_niches, hidden_sizes)
		self._optimizer = optimizers.Adam(learning_rate=learning_rate)

	# ---------------------------------------------------------------
	# Public API
	# ---------------------------------------------------------------

	def reset_episode(self):
		"""Called at episode start. Resets selection state."""
		self._active = False
		self._active_context = None
		self._active_niche = None
		self._current_niche = None
		self._smoothed_probs = None

	def select_individual(self, population, recognizer, memory, learn):
		"""Entry point. Returns the population index of the selected individual."""
		if learn:
			return self._select_train(population, recognizer, memory)
		return self._select_test(population, recognizer, memory)

	def on_step(self, memory, env_detection, learn):
		"""Record a mid-episode context sample while this episode's exploration is active."""
		if learn and self._active and (memory.steps_this_episode == 1 or memory.steps_this_episode % 50 == 0):
			self.record(memory, env_detection.compute_full_vector(memory), self._active_niche, 0.0)

	def on_learn_episode_end(self, memory, env_detection, env_max_steps):
		"""Backfill the episode reward into buffered contexts and update once the buffer fills."""
		if not self._active:
			return
		norm_reward = normalize_reward(memory.last_episode_score, memory, env_max_steps)
		num_recorded = (memory.steps_this_episode // 50) + 1
		for index in range(1, num_recorded + 1):
			if len(memory.rewards) >= index:
				memory.rewards[-index] = norm_reward
		self.record(memory, env_detection.compute_full_vector(memory), self._active_niche, norm_reward)
		if len(memory.contexts) >= self._buffer_size:
			self.update(memory)

	def record_episode_result(self, score, memory, recognizer=None):
		"""No-op for neural driver (it uses record/update instead)."""
		return False

	def evaluate(self, population):
		"""No-op for neural driver."""

	def set_mode(self, memory, mode):
		"""Switch between train (unfrozen) and test (frozen) mode."""
		if mode:
			self._frozen = False
		else:
			if memory and memory.contexts:
				self.update(memory)
			self._frozen = True

	def record(self, memory, context_vector, niche_id, reward):
		"""Store a (context, niche, reward) experience for training."""
		if self._frozen:
			return

		if niche_id >= self._current_niches:
			self._expand_to_niches(niche_id + 1)

		memory.contexts.append(context_vector.copy())
		memory.actions.append(niche_id)
		memory.rewards.append(reward)

	# pylint: disable=too-many-locals
	def update(self, memory):
		"""REINFORCE policy-gradient update from buffered experiences."""
		if self._frozen or len(memory.contexts) < 2:
			return

		contexts_np = np.array(memory.contexts, dtype=np.float32)
		actions_np = np.array(memory.actions, dtype=np.int32)
		rewards_np = np.array(memory.rewards, dtype=np.float32)

		perm = np.random.permutation(len(actions_np))
		contexts_np = contexts_np[perm]
		actions_np = actions_np[perm]
		rewards_np = rewards_np[perm]

		contexts = tf.convert_to_tensor(contexts_np)
		actions = actions_np
		rewards = rewards_np

		baseline = rewards.mean()
		advantages = rewards - baseline

		with tf.GradientTape() as tape:
			probs = self._model(contexts, training=True)
			batch_indices = tf.range(len(actions))
			indices = tf.stack([batch_indices, tf.constant(actions)], axis=1)
			chosen_probs = tf.gather_nd(probs, indices)
			log_probs = tf.math.log(chosen_probs + 1e-8)
			loss = -tf.reduce_mean(log_probs * tf.constant(advantages))

		grads = tape.gradient(loss, self._model.trainable_variables)
		self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

		memory.clear_selection_buffers()

	# ---------------------------------------------------------------
	# Private — Selection
	# ---------------------------------------------------------------

	def _select_niche(self, context_vector, known_niche_ids=None, explore=True):
		"""Return (niche_id, confidence) from the NN for the given context."""
		ctx = tf.convert_to_tensor(context_vector.reshape(1, -1), dtype=tf.float32)
		raw_probs = self._model(ctx, training=False).numpy()[0]

		masked_probs = raw_probs.copy()
		if known_niche_ids:
			for idx, _ in enumerate(masked_probs):
				if idx not in known_niche_ids:
					masked_probs[idx] = 0.0
			total = masked_probs.sum()
			if total > 0:
				masked_probs = masked_probs / total

		if explore and not self._frozen and known_niche_ids and np.random.rand() < self._action_epsilon:
			niche_id = int(np.random.choice(sorted(known_niche_ids)))
		else:
			niche_id = int(np.argmax(masked_probs))

		confidence = float(masked_probs[niche_id]) if niche_id < len(masked_probs) else 0.0
		return niche_id, confidence

	def _select_test(self, population, recognizer, memory):
		"""Test-time selection using NN probability momentum with softmax masking."""
		ctx = recognizer.compute_full_vector(memory)
		ctx_tensor = tf.convert_to_tensor(ctx.reshape(1, -1), dtype=tf.float32)
		raw_probs = self._model(ctx_tensor, training=False).numpy()[0]

		known_niche_ids = recognizer.known_niche_ids
		masked_probs = raw_probs.copy()
		if known_niche_ids:
			for idx, _ in enumerate(masked_probs):
				if idx not in known_niche_ids:
					masked_probs[idx] = 0.0
			total = masked_probs.sum()
			if total > 0:
				masked_probs = masked_probs / total

		self._smoothed_probs = masked_probs if self._smoothed_probs is None else 0.5 * self._smoothed_probs + 0.5 * masked_probs

		best_niche = int(np.argmax(self._smoothed_probs))
		self._current_niche = best_niche

		return self._best_gladiator_in_niche_or_default(population, best_niche)

	def _select_train(self, population, recognizer, memory):
		"""Train-time selection: REINFORCE exploration + trust-argmax."""
		ctx = recognizer.compute_full_vector(memory)
		known_niche_ids = recognizer.known_niche_ids
		if not self._frozen and np.random.rand() < self._episode_rate:
			niche_id, _ = self._select_niche(ctx, known_niche_ids=known_niche_ids, explore=True)
			best_idx = self._best_gladiator_in_niche_or_default(population, niche_id)
			actual_niche = population[best_idx].get("internal_env_id", -1)

			self._active = True
			self._active_context = ctx
			self._active_niche = actual_niche if actual_niche >= 0 else niche_id
			return best_idx

		trust_scores = [ind["trust"] for ind in population]
		best_idx = int(np.argmax(trust_scores))

		gladiator_niche = population[best_idx].get("internal_env_id", -1)
		if not self._frozen and gladiator_niche >= 0:
			self._active = True
			self._active_context = ctx
			self._active_niche = gladiator_niche

		return best_idx

	@staticmethod
	def _best_gladiator_in_niche_or_default(population, niche_id):
		"""Returns best gladiator for the niche, or highest trust overall."""
		slots = [index for index, g in enumerate(population) if g.get("internal_env_id") == niche_id and g.get("is_gladiator", False)]
		if slots:
			return max(slots, key=lambda index: population[index]["trust"])

		gladiators = [index for index, g in enumerate(population) if g.get("is_gladiator", False)]
		if gladiators:
			return max(gladiators, key=lambda index: population[index]["trust"])

		non_gladiators = [index for index, g in enumerate(population) if not g.get("is_gladiator", False)]
		return np.random.choice(non_gladiators) if non_gladiators else 0

	# ---------------------------------------------------------------
	# Private — Model
	# ---------------------------------------------------------------

	@staticmethod
	def _build_model(context_dim, num_niches, hidden_sizes):
		"""Build a small MLP with softmax output of given size."""
		init_hidden = initializers.Orthogonal(gain=1.0)
		init_output = initializers.Orthogonal(gain=0.01)

		inputs = Input(shape=(context_dim,))
		_x_ = inputs
		for units in hidden_sizes:
			_x_ = Dense(units, activation="relu", kernel_initializer=init_hidden)(_x_)
		outputs = Dense(
			num_niches,
			activation="softmax",
			kernel_initializer=init_output,
		)(_x_)
		return Model(inputs, outputs, name="selection_controller")

	def _expand_to_niches(self, new_size):
		old_size = self._current_niches
		if new_size <= old_size:
			return

		new_model = self._build_model(self._context_dim, new_size, self._hidden_sizes)

		for old_layer, new_layer in zip(self._model.layers[:-1], new_model.layers[:-1]):
			new_layer.set_weights(old_layer.get_weights())

		old_output_layer = self._model.layers[-1]
		new_output_layer = new_model.layers[-1]

		old_kernel, old_bias = old_output_layer.get_weights()
		new_kernel, new_bias = new_output_layer.get_weights()

		new_kernel[:, :old_size] = old_kernel
		new_bias[:old_size] = old_bias
		new_bias[old_size:] = -10.0

		new_output_layer.set_weights([new_kernel, new_bias])

		self._model = new_model
		self._current_niches = new_size
