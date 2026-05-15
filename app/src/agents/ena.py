"""EN Agent"""
# General imports
import random
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Libs imports
import numpy as np
from keras import initializers
import tensorflow as tf

# App imports
from .abstract import AAgent
from .hall_of_fame import HallOfFame
from .recognizer import EnvironmentRecognizer
from .selection_controller import SelectionController
from .memory import AgentMemory


# pylint: disable=too-many-instance-attributes
class ENAgent(AAgent):
	"""The ENAgent class"""

	# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,dangerous-default-value
	def __init__(
		self,
		brain,
		make_env,
		controller_episode_rate=0.03,
		controller_max_niches=32,
		env_max_steps=1000.0,
		exploration_rate=0.20,
		max_per_niche=5,
		max_gladiator_ratio=0.5,
		history_size=100,
		max_eval_steps=500,
		max_threads=1,
		mutation_noise=0.1,
		mutation_rate=0.01,
		physics=["gravity", "masspole", "length"],
		plasticity_algorithm="zero",
		pop_size=30,
		recognizer_threshold=0.7,
		trust_algorithm="zero",
	):
		super().__init__(brain, make_env)
		self.exploration_rate = exploration_rate
		self.learn = True
		self.env_max_steps = float(env_max_steps)
		self.max_eval_steps = max_eval_steps
		self.mutation_rate = mutation_rate
		self.mutation_noise = mutation_noise
		self.physics = physics
		self.plasticity_algorithm = plasticity_algorithm
		self.pop_size = pop_size
		self.trust_algorithm = trust_algorithm

		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])

		self.memory = AgentMemory(history_size, self.state_dim, max_eval_steps)

		# UNSUPERVISED ENVIRONMENT RECOGNITION
		self._recognizer = EnvironmentRecognizer(
			self.state_dim,
			similarity_threshold=recognizer_threshold,
		)

		# SELECTION CONTROLLER (Contextual Bandit)
		context_dim = 4 * self.state_dim
		self._selection_controller = SelectionController(
			buffer_size=32,
			context_dim=context_dim,
			controller_episode_rate=controller_episode_rate,
			max_niches=controller_max_niches,
		)

		self.is_continuous = not hasattr(tmp_env.action_space, "n")

		self.action_dim = (
			int(tmp_env.action_space.shape[0])
			if self.is_continuous
			else int(tmp_env.action_space.n)
		)
		tmp_env.close()

		# Hall of Fame
		self.hall_of_fame = HallOfFame(
			max_per_niche, max_gladiator_ratio, brain, self.action_dim, self.state_dim
		)

		# Initialize population
		self.best_individual_idx = 0
		self._episode_started = False
		self.population = [self.__build_individual() for _ in range(self.pop_size)]

		# Standard Parallelization Setup
		self.max_threads = max_threads

	# pylint: disable=too-many-locals
	def act(self, env, state):
		"""Act as the ENAgent"""
		# 1. Selection at episode start
		if not self._episode_started:
			# Compute context once (used by the controller for episode routing)
			context = self._recognizer.compute_physical_vector(self.memory)

			niche_id = None
			drives = False

			if self.learn:
				# Training: let the controller optionally take over
				niche_id, drives = self._selection_controller.start_episode(
					self.memory,
					context,
					explore=True,
				)

			if drives and niche_id is not None:
				self.__select_individual_for_niche(niche_id)
			else:
				self.__select_individual()

			self._episode_started = True
			self.__project_individual_weights(self.population[self.best_individual_idx])

		# 2. Execution
		chosen = self.population[self.best_individual_idx]

		self.memory.obs_normalizer.update(state[np.newaxis, :])
		self._recognizer.update_state(state, self.memory)
		norm_state = self.memory.obs_normalizer.normalize(state)
		state_tensor = tf.convert_to_tensor(norm_state.reshape(1, -1), dtype=tf.float32)
		action_output = chosen["fast_call"](state_tensor, training=False)
		action_numpy = action_output.numpy()[0]

		action = (
			np.clip(action_numpy, -1.0, 1.0)
			if self.is_continuous
			else int(np.argmax(action_numpy))
		)

		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated

		# Track transition dynamics (physics signature)
		self.memory.update_transition(state, next_state)

		# 3. Internal State Management
		self.memory.current_episode_reward += reward

		# 4. Episode over
		if done:
			self.memory.last_episode_score = self.memory.current_episode_reward

			if self.learn:
				# Let the controller process the episode outcome (no-op if it wasn't active)
				self._selection_controller.complete_episode(
					self.memory,
					self.memory.last_episode_score,
					normalize_fn=self.__normalize_reward,
				)
			else:
				# Test mode: update stats only — NO trust updates, NO UCB1
				self.memory.reward_history.append(self.memory.last_episode_score)
				self.memory.max_seen = max(self.memory.max_seen, self.memory.last_episode_score)
				self.memory.min_seen = min(self.memory.min_seen, self.memory.last_episode_score)

			self.memory.current_episode_reward = 0
			self._episode_started = False

		return next_state, reward, terminated, truncated, action

	@property
	def plasticity_history(self):
		return self.memory.plasticity_history

	@property
	def current_env_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._recognizer.current_niche_id

	def load_gladiators(self, restore_trust=True):
		"""Re-introduces Gladiators into the active population with Niche Distribution."""
		if self.hall_of_fame:
			self.hall_of_fame.inject_gladiators(
				self.population,
				self.pop_size,
				self.memory.obs_normalizer,
				restore_trust=restore_trust,
			)

	def set_learning(self, mode, output_dir=None):
		"""Reset trust and age for all population members to 0 (except gladiators)."""
		self.best_individual_idx = 0
		self.memory.max_seen = -np.inf
		self.memory.min_seen = np.inf
		self.memory.reward_history.clear()
		for individual in self.population:
			if not individual.get("is_gladiator", False):
				individual["trust"] = 0.0
				individual["age"] = 0

		self.learn = mode
		self._selection_controller.set_mode(self.memory, self.learn)

		# Save niche details when entering test mode if output_dir is provided
		if not mode and output_dir is not None:
			self.hall_of_fame.save_niche_details(output_dir, self.get_name())

	# pylint: disable=too-many-locals
	def train(self, **kwargs):
		"""Handles the evolutionary search."""
		env, step_data = kwargs["env"], kwargs["step_data"]
		if not step_data["done"]:
			return

		# 1. Update Global Statistics
		self.memory.reward_history.append(self.memory.last_episode_score)
		self.memory.max_seen = max(self.memory.max_seen, self.memory.last_episode_score)
		self.memory.min_seen = min(self.memory.min_seen, self.memory.last_episode_score)

		# 2. Calculate Failure Signal once
		failure_signal = self.__get_trust_decay(self.memory.last_episode_score)

		# 3. Update the current Specialist's Trust
		chosen = self.population[self.best_individual_idx]
		chosen["age"] += 1
		norm_score = self.__normalize_reward(self.memory.last_episode_score)
		chosen["trust"] = (1 - failure_signal) * chosen["trust"] + (
			failure_signal * norm_score
		)
		# 4. Compute plasticity for this episode
		plasticity = self.__get_plasticity(self.memory.last_episode_score)
		self.memory.plasticity_history.append(plasticity)

		# 5. Monitor for environment shifts
		if self._recognizer.detect_change(self.memory):
			self.memory.reset_recognizer_statistics()

		# 6. Update Hall of Fame and origin tags
		chosen["env_name"] = env.unwrapped.name
		self.hall_of_fame.update(
			chosen,
			chosen["trust"],
			self.population,
			self.memory,
			self._recognizer,
			self.memory.obs_normalizer,
		)

		# 7. Only reset trust (challenge the specialist) when failing
		if failure_signal > 0.1:
			self.__reset_trust(factor=failure_signal)

		num_to_eval = max(2, int(self.pop_size * plasticity))
		other_indices = [
			index for index in range(self.pop_size) if index != self.best_individual_idx
		]
		test_indices = random.sample(
			other_indices, min(len(other_indices), num_to_eval)
		)

		gen_seed = np.random.randint(0, 9999)
		env_params = {
			key: getattr(env.unwrapped, key)
			for key in self.physics
			if hasattr(env.unwrapped, key)
		}

		def thread_eval_task(task_idx):
			individual_to_eval = self.population[task_idx]
			local_env = self.make_env(name=env.unwrapped.name, **env_params)
			score = self.__evaluate_individual_performance(
				individual_to_eval, local_env, seed=gen_seed
			)
			local_env.close()
			return task_idx, score

		with ThreadPoolExecutor(max_workers=self.max_threads) as _executor:
			eval_results = list(_executor.map(thread_eval_task, test_indices))

		for task_idx, score in eval_results:
			individual_signal = self.__get_trust_decay(score)
			eval_norm_score = self.__normalize_reward(score)
			old = self.population[task_idx]["trust"]
			self.population[task_idx]["trust"] = (1 - individual_signal) * old + (
				individual_signal * eval_norm_score
			)

		# 8. Breed and Mutate
		self.__evolve_population()

		# 9. Final selection for the next training episode
		self.__select_individual()

	# ---------------------------------------------------------------
	# Private methods
	# ---------------------------------------------------------------

	def __best_gladiator_in_niche(self, niche_id):
		"""Return the index of the best gladiator for the given niche, or None."""
		slots = [
			index
			for index, individual in enumerate(self.population)
			if individual.get("internal_env_id") == niche_id
			and individual.get("is_gladiator", False)
		]
		if not slots:
			return None
		return max(slots, key=lambda i: self.population[i]["trust"])

	def __build_individual(self, model=None, existing_fast_call=None):
		"""Build the NN model or reset an existing one in-place."""
		if model is None:
			model = self.brain(self.action_dim, self.state_dim)
		else:
			for index, layer in enumerate(model.layers):
				if (
					hasattr(layer, "kernel_initializer")
					and layer.kernel_initializer is not None
				):
					gain = 0.1 if index == len(model.layers) - 1 else 1.0
					initializer = initializers.Orthogonal(gain=gain)
					layer.kernel.assign(initializer(shape=layer.kernel.shape))
				if (
					hasattr(layer, "bias_initializer")
					and layer.bias_initializer is not None
				):
					layer.bias.assign(initializers.Zeros()(shape=layer.bias.shape))

		fast_call = (
			existing_fast_call
			if existing_fast_call is not None
			else tf.function(model, reduce_retracing=True)
		)

		return {
			"age": 0,
			"individual_id": str(uuid.uuid4()),
			"model": model,
			"fast_call": fast_call,
			"trust": self.__get_dynamic_ref_trust(),
			"is_gladiator": False,
			"last_mu": np.copy(self.memory.obs_normalizer.mean),
			"last_var": np.copy(self.memory.obs_normalizer.var),
		}

	def __crossover(self, parent_1, parent_2, target):
		"""Performs crossover directly into the target individual's model."""
		# USP: Align both parents to the current normalizer before mixing their weights
		self.__project_individual_weights(parent_1)
		self.__project_individual_weights(parent_2)

		weights_1 = parent_1["model"].get_weights()
		weights_2 = parent_2["model"].get_weights()

		new_weights = []
		for weight_1, weight_2 in zip(weights_1, weights_2):
			mask = np.random.random(weight_1.shape) < 0.5
			new_weights.append(np.where(mask, weight_1, weight_2))

		target["model"].set_weights(new_weights)
		target["individual_id"] = str(uuid.uuid4())
		target["age"] = max(parent_1["age"], parent_2["age"])
		target["trust"] = self.__get_dynamic_ref_trust()
		target["is_gladiator"] = False
		# Because parents were projected to current normalizer, the child inherits it
		target["last_mu"] = np.copy(self.memory.obs_normalizer.mean)
		target["last_var"] = np.copy(self.memory.obs_normalizer.var)

	def __evaluate_individual_performance(self, individual, env, seed=None):
		"""Evaluate an individual for a single episode and return the score."""
		self.__project_individual_weights(individual)

		state, _ = env.reset(seed=seed)
		done, score, steps = False, 0, 0

		while not done and steps < self.max_eval_steps:
			norm_state = self.memory.obs_normalizer.normalize(state)
			state_tensor = tf.convert_to_tensor(
				norm_state.reshape(1, -1), dtype=tf.float32
			)
			action_output = individual["fast_call"](state_tensor, training=False)
			action_numpy = action_output.numpy()[0]

			action = (
				np.clip(action_numpy, -1.0, 1.0)
				if self.is_continuous
				else int(np.argmax(action_numpy))
			)

			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			score += reward
			steps += 1

		# Scale truncated eval score to full-episode equivalent using env_max_steps
		if not done:
			score = score * (self.env_max_steps / self.max_eval_steps)

		return float(score)

	def __evolve_population(self, replace_count=None):
		"""Evolution: reuses model buffers to guarantee zero session bloat."""
		# 1. Rank by trust/age
		indices = list(range(self.pop_size))
		indices.sort(
			key=lambda i: (self.population[i]["trust"], self.population[i]["age"]),
			reverse=True,
		)

		# 2. Strategy Budget
		if replace_count is None:
			replace_count = int(self.pop_size * 0.8)
		replace_count = max(1, min(replace_count, self.pop_size - 2))

		elite_pool_size = max(2, int(self.pop_size * 0.2))
		elite_indices = indices[:elite_pool_size]

		# Kill the WORST individuals — but strictly protect gladiators
		to_replace_indices = [
			i
			for i in indices[-replace_count:]
			if not self.population[i].get("is_gladiator", False)
		]
		random.shuffle(to_replace_indices)
		actual_replace_count = len(to_replace_indices)

		mutant_count = int(actual_replace_count * 0.4)
		crossover_count = int(actual_replace_count * 0.4)

		for _ in range(mutant_count):
			if not to_replace_indices:
				break
			target_idx = to_replace_indices.pop(0)
			parent = self.population[random.choice(elite_indices)]
			self.__mutate(parent, self.population[target_idx])

		# --- Fill Crossovers ---
		for _ in range(crossover_count):
			if not to_replace_indices:
				break
			target_idx = to_replace_indices.pop(0)
			parent_1, parent_2 = random.sample(
				[self.population[i] for i in elite_indices], 2
			)
			self.__crossover(parent_1, parent_2, self.population[target_idx])

		# --- Fill Fresh (Diversity) ---
		while to_replace_indices:
			target_idx = to_replace_indices.pop(0)
			target = self.population[target_idx]
			fresh_data = self.__build_individual(
				model=target["model"], existing_fast_call=target["fast_call"]
			)
			target.update(fresh_data)

	# pylint: disable=unused-private-member
	def __full(self, _):
		"""Full."""
		return 1.0

	# pylint: disable=unused-private-member
	def __fuzzylogic(self, reward):
		"""Fuzzy logic membership: high reward."""
		if len(self.memory.reward_history) < 10:
			return 1.0

		low_limit = np.percentile(self.memory.reward_history, 10)
		high_limit = np.percentile(self.memory.reward_history, 90)

		# Adaptive margin: 5% of peak performance, minimum 2.0 points
		margin = max(2.0, high_limit * 0.05)
		if (high_limit - low_limit) < margin:
			low_limit = max(0.0, high_limit - margin)

		if reward >= high_limit:
			return 0.1
		if reward <= low_limit:
			return 1.0

		_mu = (reward - low_limit) / (high_limit - low_limit + 1e-8)
		return 1.0 - (_mu * 0.9)

	def __get_dynamic_ref_trust(self):
		"""Uses the moving average of performance as reference trust."""
		if len(self.memory.reward_history) < 5:
			# Returning 0.0 creates an artificial ceiling in negative-reward environments.
			# We return -1.0 so that evaluated mutants (even those with negative scores)
			# have a higher trust than a fully "reset" unevaluated slot.
			return -1.0
		return self.__normalize_reward(float(np.mean(self.memory.reward_history)))

	def __get_plasticity(self, reward):
		"""Modular dispatch to chosen plasticity algorithm."""
		algorithm = self.plasticity_algorithm if self.learn else "zero"
		try:
			method = getattr(self, f"_ENAgent__{algorithm}")
			return method(reward)
		except AttributeError:
			return 1.0

	def __get_trust_decay(self, reward):
		"""Modular dispatch for trust decay logic."""
		try:
			method = getattr(self, f"_ENAgent__{self.trust_algorithm}")
			return method(reward)
		except AttributeError:
			return 0.0

	def __mutate(self, parent, target):
		"""Performs mutation into the target individual's model."""
		target["model"].set_weights(parent["model"].get_weights())
		target["age"] = parent["age"]
		target["individual_id"] = str(uuid.uuid4())
		target["trust"] = self.__get_dynamic_ref_trust()
		target["is_gladiator"] = False

		# Adaptive mutation noise — linked to current plasticity
		plasticity = self.memory.plasticity_history[-1] if self.memory.plasticity_history else 1.0
		noise_mag = 0.01 + ((self.mutation_noise - 0.01) * plasticity)

		# USP: Inherit the parent's alignment state. The child's weights are still
		# in the parent's space until explicitly projected during evaluation.
		target["last_mu"] = np.copy(parent["last_mu"])
		target["last_var"] = np.copy(parent["last_var"])

		weights = target["model"].get_weights()
		new_weights = []
		for weight in weights:
			mask = np.random.random(weight.shape) < self.mutation_rate
			noise = np.random.normal(0, noise_mag, weight.shape)
			weight[mask] += noise[mask]
			new_weights.append(weight)
		target["model"].set_weights(new_weights)

	def __normalize_reward(self, reward: float) -> float:
		"""Percentile-based reward normalization. Removed [0, 1] clip to preserve selection pressure."""
		if len(self.memory.reward_history) < 10:
			# Warmup: scale against env_max_steps so trust values are distinct
			return float(np.clip(reward / self.env_max_steps, -2.0, 5.0))
		low = float(np.percentile(self.memory.reward_history, 5))
		high = float(np.percentile(self.memory.reward_history, 95))

		# Linear scaling relative to historical percentiles, but ALLOW values > 1.0
		# to reward individuals that break the current records.
		norm = (reward - low) / (high - low + 1e-8)
		return float(np.clip(norm, -2.0, 5.0))

	def __project_individual_weights(self, individual):
		"""Universal Snapshot Projection: aligns first layer weights to current normalizer."""
		if self.memory.obs_normalizer.count < self.max_eval_steps * 2:
			return

		mean_old = individual["last_mu"]
		variance_old = individual["last_var"]
		mean_new = self.memory.obs_normalizer.mean
		variance_new = self.memory.obs_normalizer.var

		if np.allclose(mean_old, mean_new) and np.allclose(variance_old, variance_new):
			return

		weights = individual["model"].get_weights()
		weight_first_layer = weights[0]
		bias_first_layer = weights[1]

		std_old = np.sqrt(variance_old + 1e-8)
		std_new = np.sqrt(variance_new + 1e-8)

		scale_factors = np.clip(std_new / std_old, 0.1, 10.0)
		weight_new = weight_first_layer * scale_factors[:, np.newaxis]

		shift_factors = (mean_new - mean_old) / std_old
		bias_new = bias_first_layer + np.dot(shift_factors, weight_first_layer)

		weights[0] = weight_new
		weights[1] = bias_new

		individual["model"].set_weights(weights)
		individual["last_mu"] = np.copy(mean_new)
		individual["last_var"] = np.copy(variance_new)

	# pylint: disable=unused-private-member
	def __quadratic(self, reward):
		"""Aggressive plasticity algorithm using global benchmarks."""
		if len(self.memory.reward_history) < 10:
			return 1.0

		denominator = (self.memory.max_seen - self.memory.min_seen) + 1e-8
		normalized_error = (self.memory.max_seen - reward) / denominator
		return max(0.1, min(1.0, normalized_error**2))

	def __reset_trust(self, factor=1.0):
		"""Blend all population trust toward the dynamic reference."""
		for individual in self.population:
			individual["trust"] = (individual["trust"] * (1.0 - factor)) + (
				self.__get_dynamic_ref_trust() * factor
			)

	def __select_individual(self):
		"""Selects the best individual from the population."""
		if self.learn:
			# Train mode: pure trust argmax
			trust_scores = [individual["trust"] for individual in self.population]
			self.best_individual_idx = int(np.argmax(trust_scores))
			return
		# Test mode: use the trained SelectionController
		self.__select_individual_test()

	def __select_individual_for_niche(self, niche_id):
		"""Select the best gladiator from a specific recognised niche."""
		# All gladiators that belong to the requested niche
		gladiator_slots = [
			index
			for index, individual in enumerate(self.population)
			if individual.get("internal_env_id") == niche_id
			and individual.get("is_gladiator", False)
		]
		if gladiator_slots:
			# Pick the one with the highest stored (training) trust
			self.best_individual_idx = max(
				gladiator_slots, key=lambda index: self.population[index]["trust"]
			)
		else:
			# Niche has no gladiator yet → fallback to a random non‑gladiator
			non_gladiator_slots = [
				index
				for index, individual in enumerate(self.population)
				if not individual.get("is_gladiator", False)
			]
			self.best_individual_idx = (
				random.choice(non_gladiator_slots) if non_gladiator_slots else 0
			)

	def __select_individual_test(self):
		"""Test‑time specialist selection with recognizer → controller → closest gladiator / mutant fallback."""
		context = self._recognizer.compute_physical_vector(self.memory)

		# 1) Try recognizer first (robust physics signatures)
		rec_niche_id, rec_sim = self._recognizer.match(self.memory)
		gladiator_idx = self.__best_gladiator_in_niche(rec_niche_id)
		if gladiator_idx is not None:
			self.best_individual_idx = gladiator_idx
			return

		# 2) Only if recognizer fails, use the controller with a higher confidence bar
		niche_id, confidence = self._selection_controller.select_niche(context, explore=False)
		gladiator_idx = self.__best_gladiator_in_niche(niche_id)
		if gladiator_idx is not None and confidence > 0.8:   # much stricter
			self.best_individual_idx = gladiator_idx
			return

		# 3) Fallback: mostly try the closest gladiator (zero-shot transfer), sometimes try a random mutant
		non_gladiators = [i for i, g in enumerate(self.population) if not g.get("is_gladiator")]
		gladiators = [i for i, g in enumerate(self.population) if g.get("is_gladiator")]

		# 75% chance to use the closest gladiator for zero-shot transfer, if any exist
		if gladiators and random.random() < 0.75:
			best_sim = -float("inf")
			best_idx = random.choice(gladiators)  # Default to random gladiator if search fails
			for nid, prototype in self._recognizer._physical_prototypes.items():
				sim = self._recognizer._cosine_similarity(context, prototype)
				if sim > best_sim:
					idx = self.__best_gladiator_in_niche(nid)
					if idx is not None:
						best_sim = sim
						best_idx = idx
			self.best_individual_idx = best_idx
			return

		# 4) Absolute fallback to a random mutant (25% chance or if no gladiators exist)
		self.best_individual_idx = random.choice(non_gladiators) if non_gladiators else 0

	# pylint: disable=unused-private-member
	def __zero(self, _):
		"""Zero."""
		return 0.0
