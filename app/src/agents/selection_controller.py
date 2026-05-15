"""Trainable Selection Controller (Contextual Bandit) for EN Agent."""
# A small neural network that learns to map a context vector (from the
# EnvironmentRecognizer) to the specialist niche that performs best in
# that context.  Trained with REINFORCE during training episodes; frozen
# at test time for zero-shot niche selection.  The output layer grows
# dynamically as new niches are discovered.

# Libs imports
import numpy as np
import tensorflow as tf
from keras import Model, Input, initializers, optimizers
from keras.layers import Dense


# pylint: disable=too-many-instance-attributes
class SelectionController:
	"""Contextual bandit that maps context → best niche ID."""

	# pylint: disable=too-many-arguments, too-many-positional-arguments
	def __init__(
		self,
		context_dim,
		buffer_size=64,
		controller_episode_rate=0.1,
		controller_action_epsilon=0.2,
		hidden_sizes=(64, 32),
		learning_rate=1e-3,
		max_niches=32,  # now treated as initial capacity
	):
		"""Constructor"""
		self._active = False
		self._active_context = None
		self._active_niche = None

		self._context_dim = context_dim
		self._episode_rate = controller_episode_rate
		self._action_epsilon = controller_action_epsilon
		self._frozen = False
		self._buffer_size = buffer_size
		self._hidden_sizes = hidden_sizes
		self._learning_rate = learning_rate
		self._current_niches = max_niches  # tracks actual output size

		# Build the initial model
		self._model = self._build_model(context_dim, self._current_niches, hidden_sizes)
		self._optimizer = optimizers.Adam(learning_rate=learning_rate)

		self._model = self._build_model(context_dim, self._current_niches, hidden_sizes)
		self._optimizer = optimizers.Adam(learning_rate=learning_rate)

	# ---------------------------------------------------------------
	# Public API
	# ---------------------------------------------------------------

	def select_niche(self, context_vector, explore=True):
		"""Return (niche_id, confidence)."""
		ctx = tf.convert_to_tensor(context_vector.reshape(1, -1), dtype=tf.float32)
		probs = self._model(ctx, training=False).numpy()[0]
		niche_id = (
			int(np.random.randint(0, self._current_niches))
			if explore and not self._frozen and np.random.rand() < self._action_epsilon
			else int(np.argmax(probs))
		)
		confidence = float(probs[niche_id])
		return niche_id, confidence

	def record(self, memory, context_vector, niche_id, reward):
		"""Store a (context, niche, reward) experience for training."""
		# Dynamically expands the output layer if niche_id is beyond current capacity.
		if self._frozen:
			return

		# Expand the model if necessary
		if niche_id >= self._current_niches:
			self._expand_to_niches(niche_id + 1)

		# Now niche_id is valid
		memory.contexts.append(context_vector.copy())
		memory.actions.append(niche_id)
		memory.rewards.append(reward)

		if len(memory.contexts) >= self._buffer_size:
			self.update(memory)

	def start_episode(self, memory, context, explore=True):
		"""Called at the beginning of an episode."""
		# If training and the episode‑rate coin says 'controller', returns (niche_id, True).
		# Otherwise returns (None, False).
		self._active = False
		self._active_context = None
		self._active_niche = None

		if explore and not self._frozen and np.random.rand() < self._episode_rate:
			niche_id, _ = self.select_niche(context, explore=True)
			self._active = True
			self._active_context = context
			self._active_niche = niche_id
		return self._active_niche, self._active

	def complete_episode(self, memory, reward, normalize_fn=None):
		"""Call at end of episode with the episode score."""
		if not self._active:
			return
		norm_reward = normalize_fn(reward) if normalize_fn else reward
		self.record(memory, self._active_context, self._active_niche, norm_reward)
		self._active = False
		self._active_context = None
		self._active_niche = None

	# pylint: disable=too-many-locals
	def update(self, memory):
		"""REINFORCE policy-gradient update from buffered experiences."""
		if self._frozen or len(memory.contexts) < 2:
			return

		# Convert to numpy arrays
		contexts_np = np.array(memory.contexts, dtype=np.float32)
		actions_np = np.array(memory.actions, dtype=np.int32)
		rewards_np = np.array(memory.rewards, dtype=np.float32)

		# Shuffle to break sequential ordering
		perm = np.random.permutation(len(actions_np))
		contexts_np = contexts_np[perm]
		actions_np = actions_np[perm]
		rewards_np = rewards_np[perm]

		# Proceed with REINFORCE (unchanged)
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

	def set_mode(self, memory, mode):
		"""Switch between train (unfrozen) and test (frozen) mode."""
		if mode:
			self._frozen = False
		else:
			if memory and memory.contexts:
				self.update(memory)
			self._frozen = True

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

		# Copy hidden layers wholesale
		for old_layer, new_layer in zip(self._model.layers[:-1], new_model.layers[:-1]):
			new_layer.set_weights(old_layer.get_weights())

		old_output_layer = self._model.layers[-1]
		new_output_layer = new_model.layers[-1]

		old_kernel, old_bias = old_output_layer.get_weights()
		new_kernel, new_bias = new_output_layer.get_weights()

		# Copy the old part
		new_kernel[:, :old_size] = old_kernel
		new_bias[:old_size] = old_bias

		# IMPORTANT: set the bias for new niches to a very negative value so
		# their softmax probability is ≈ 0, preserving old policy.
		new_bias[old_size:] = -10.0  # new columns get ~0 probability

		new_output_layer.set_weights([new_kernel, new_bias])

		# Keep the same optimiser so momentum is not lost
		self._model = new_model
		# self._optimizer stays the same (reuse Adam instance)
		self._current_niches = new_size
