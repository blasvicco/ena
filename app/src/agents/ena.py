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
from .detector import EnvironmentDetector
from .hall_of_fame import HallOfFame
from .normalizers import _IdentityNormalizer, _RunningNormalizer


# pylint: disable=too-many-instance-attributes
class ENAgent(AAgent):
	"""The ENAgent class"""

	# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,dangerous-default-value
	def __init__(
		self,
		brain,
		make_env,
		exploration_rate=0.20,
		history_size=100,
		gladiator_amounts=5,
		env_max_steps=1000.0,
		max_eval_steps=500,
		max_threads=1,
		mutation_rate=0.01,
		mutation_noise=0.1,
		physics=["gravity", "masspole", "length"],
		plasticity_algorithm="zero",
		pop_size=30,
		trust_algorithm="zero",
		env_window_size=5,
		env_short_window_size=None,
		env_cooldown_window=10,
		env_detection_threshold=-50.0,
	):
		super().__init__(brain, make_env)
		self.exploration_rate = exploration_rate
		self.last_episode_score = 0
		self.learn = True
		self.env_max_steps = float(env_max_steps)
		self.max_eval_steps = max_eval_steps
		self.mutation_rate = mutation_rate
		self.mutation_noise = mutation_noise
		self.physics = physics
		self.plasticity_algorithm = plasticity_algorithm
		self.plasticity_history = []
		self.pop_size = pop_size
		self.trust_algorithm = trust_algorithm

		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])

		# AGNOSTIC ENVIRONMENT DETECTION
		self._env_detector = EnvironmentDetector(
			window_size=env_window_size,
			short_window_size=env_short_window_size,
			cooldown_window=env_cooldown_window,
			performance_derivative_threshold=env_detection_threshold,
		)

		self.is_continuous = not hasattr(tmp_env.action_space, "n")
		if self.is_continuous:
			self.action_dim = int(tmp_env.action_space.shape[0])
			self._obs_normalizer = _RunningNormalizer(
				self.state_dim,
				max_count=self.max_eval_steps * 10, # 10 is the window size.
			)
		else:
			self.action_dim = int(tmp_env.action_space.n)
			self._obs_normalizer = _IdentityNormalizer(self.state_dim)
		tmp_env.close()

		# AGNOSTIC STATISTICS
		self.reward_history = deque(maxlen=history_size)
		self.max_seen = -np.inf
		self.min_seen = np.inf
		self.episodes_seen = 0
		self.steps_seen = 0

		# Hall of Fame
		self.hall_of_fame = HallOfFame(gladiator_amounts, brain, self.action_dim, self.state_dim)

		# Initialize population
		self.best_individual_idx = 0
		self.current_episode_reward = 0
		self._episode_started = False
		self.population = [self.__build_individual() for _ in range(self.pop_size)]

		# Standard Parallelization Setup
		self.max_threads = max_threads

	# pylint: disable=too-many-locals
	def act(self, env, state):
		"""Act as the ENAgent"""
		# 1. Selection Logic — only pick a new specialist at episode start (D4: flag)
		if not self._episode_started:
			is_exploring = False
			if self.learn and np.random.rand() < self.exploration_rate:
				self.best_individual_idx = np.random.randint(self.pop_size)
				is_exploring = True

			# Reactive adaptation — only evaluate actual Specialist performance, NOT random exploration!
			if not is_exploring:
				raw_score = round(float(self.last_episode_score))
				self._env_detector.detect(raw_score)
				# Select best individual (which might be a newly loaded gladiator)
				self.__select_individual()

			self._episode_started = True

			# USP: Project the chosen specialist's weights to the current normalizer
			# BEFORE it starts acting, so it isn't out of sync with the drifting stats.
			self.__project_individual_weights(self.population[self.best_individual_idx])

		# 2. Execution
		chosen = self.population[self.best_individual_idx]

		# Always update normalizer to allow zero-shot adaptation
		self._obs_normalizer.update(state[np.newaxis, :])
		norm_state = self._obs_normalizer.normalize(state)
		state_tensor = tf.convert_to_tensor(norm_state.reshape(1, -1), dtype=tf.float32)
		action_output = chosen["fast_call"](state_tensor, training=False)
		action_numpy = action_output.numpy()[0]

		action = (
			np.clip(action_numpy, -1.0, 1.0)
			if self.is_continuous
			else int(np.argmax(action_numpy))
		)

		next_state, reward, terminated, truncated, _ = env.step(action)
		self.steps_seen += 1
		done = terminated or truncated

		# 3. Internal State Management
		self.current_episode_reward += reward

		# 4. Episode over
		if done:
			self.last_episode_score = self.current_episode_reward

			if not self.learn:
				# In test mode: update stats here (train() is not called)
				self.episodes_seen += 1
				self.reward_history.append(self.last_episode_score)
				self.max_seen = max(self.max_seen, self.last_episode_score)
				self.min_seen = min(self.min_seen, self.last_episode_score)

				decay = self.__get_trust_decay(self.last_episode_score)
				old_trust = self.population[self.best_individual_idx]["trust"]
				norm_score = self.__normalize_reward(self.last_episode_score)
				new_trust = (1 - decay) * old_trust + (decay * norm_score)
				self.population[self.best_individual_idx]["trust"] = new_trust

				# Report this episode to HallOfFame UCB1 accounting
				self.hall_of_fame.record_episode_result(self.best_individual_idx, norm_score)

			self.current_episode_reward = 0
			self._episode_started = False

		return next_state, reward, terminated, truncated, action

	def load_gladiators(self):
		"""Re-introduces Gladiators into the active population with Niche Distribution."""
		if self.hall_of_fame:
			self.hall_of_fame.inject_gladiators(
				self.population,
				self.pop_size,
				self._obs_normalizer,
				restore_trust=self.learn,
			)

	@property
	def current_env_id(self) -> int:
		"""Current detected environment ID."""
		return self._env_detector.current_env_id

	def set_learning(self, mode):
		"""Reset trust and age for all population members to 0."""
		self.learn = mode
		for guy in self.population:
			guy["trust"] = 0.0
			guy["age"] = 0
		self.best_individual_idx = 0

		# When switching to test mode, clear the training reward history so that
		# the fuzzylogic trust/plasticity algorithms reference test-phase scores only.
		# Without this, training scores (~800-1500) poison the percentile thresholds,
		# causing any test score (~200) to trigger decay=1.0 and instantly destroy
		# every gladiator's trust after a single episode.
		if not mode:
			self.reward_history.clear()
			self.max_seen = -np.inf
			self.min_seen = np.inf

	# pylint: disable=too-many-locals
	def train(self, **kwargs):
		"""Handles the evolutionary search."""
		env, step_data = kwargs["env"], kwargs["step_data"]
		if not step_data["done"]:
			return

		# 1. Update Global Statistics
		self.episodes_seen += 1
		self.reward_history.append(self.last_episode_score)
		self.max_seen = max(self.max_seen, self.last_episode_score)
		self.min_seen = min(self.min_seen, self.last_episode_score)

		# 2. Calculate Failure Signal once
		failure_signal = self.__get_trust_decay(self.last_episode_score)

		# 3. Update the current Specialist's Trust
		chosen = self.population[self.best_individual_idx]
		chosen["age"] += 1
		norm_score = self.__normalize_reward(self.last_episode_score)
		old_trust = chosen["trust"]
		chosen["trust"] = (1 - failure_signal) * old_trust + (failure_signal * norm_score)
		# 4. Update Hall of Fame and origin tags
		chosen["env_name"] = env.unwrapped.name
		self.hall_of_fame.update(
			chosen, old_trust,
			self.population, self.plasticity_history,
			self._env_detector, self._obs_normalizer,
		)

		# 5. Only reset trust (challenge the specialist) when failing
		if failure_signal > 0.1:
			self.__reset_trust(factor=failure_signal)

		plasticity = self.__get_plasticity(self.last_episode_score)
		self.plasticity_history.append(plasticity)

		num_to_eval = max(2, int(self.pop_size * plasticity))
		other_indices = [i for i in range(self.pop_size) if i != self.best_individual_idx]
		test_indices = random.sample(other_indices, min(len(other_indices), num_to_eval))

		gen_seed = np.random.randint(0, 9999)

		env_params = {
			key: getattr(env.unwrapped, key)
			for key in self.physics
			if hasattr(env.unwrapped, key)
		}

		def thread_eval_task(task_idx):
			guy_to_eval = self.population[task_idx]
			local_env = self.make_env(name=env.unwrapped.name, **env_params)
			score = self.__evaluate_individual_performance(guy_to_eval, local_env, seed=gen_seed)
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

		# 6. Breed and Mutate
		self.__evolve_population()

		# 7. Final selection for the next training episode
		self.__select_individual()

	# ---------------------------------------------------------------
	# Private methods
	# ---------------------------------------------------------------

	def __build_individual(self, model=None, existing_fast_call=None):
		"""Build the NN model or reset an existing one in-place."""
		if model is None:
			model = self.brain(self.action_dim, self.state_dim)
		else:
			for i, layer in enumerate(model.layers):
				if hasattr(layer, "kernel_initializer") and layer.kernel_initializer is not None:
					gain = 0.1 if i == len(model.layers) - 1 else 1.0
					initializer = initializers.Orthogonal(gain=gain)
					layer.kernel.assign(initializer(shape=layer.kernel.shape))
				if hasattr(layer, "bias_initializer") and layer.bias_initializer is not None:
					layer.bias.assign(initializers.Zeros()(shape=layer.bias.shape))

		fast_call = existing_fast_call if existing_fast_call is not None else tf.function(model, reduce_retracing=True)

		return {
			"age": 0,
			"individual_id": str(uuid.uuid4()),
			"model": model,
			"fast_call": fast_call,
			"trust": self.__get_dynamic_ref_trust(),
			"is_gladiator": False,
			"last_mu": np.copy(self._obs_normalizer.mean),
			"last_var": np.copy(self._obs_normalizer.var),
		}

	def __crossover(self, parent_1, parent_2, target):
		"""Performs crossover directly into the target individual's model."""
		# USP: Align both parents to the current normalizer before mixing their weights
		self.__project_individual_weights(parent_1)
		self.__project_individual_weights(parent_2)

		weights_1 = parent_1["model"].get_weights()
		weights_2 = parent_2["model"].get_weights()

		new_weights = []
		for w1, w2 in zip(weights_1, weights_2):
			mask = np.random.random(w1.shape) < 0.5
			new_weights.append(np.where(mask, w1, w2))

		target["model"].set_weights(new_weights)
		target["individual_id"] = str(uuid.uuid4())
		target["age"] = max(parent_1["age"], parent_2["age"])
		target["trust"] = self.__get_dynamic_ref_trust()
		target["is_gladiator"] = False
		# Because parents were projected to current normalizer, the child inherits it
		target["last_mu"] = np.copy(self._obs_normalizer.mean)
		target["last_var"] = np.copy(self._obs_normalizer.var)

	def __evaluate_individual_performance(self, individual, env, seed=None):
		"""Evaluate an individual for a single episode and return the score."""
		self.__project_individual_weights(individual)

		state, _ = env.reset(seed=seed)
		done, score, steps = False, 0, 0

		while not done and steps < self.max_eval_steps:
			norm_state = self._obs_normalizer.normalize(state)
			state_tensor = tf.convert_to_tensor(norm_state.reshape(1, -1), dtype=tf.float32)
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
			i for i in indices[-replace_count:]
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
				model=target["model"],
				existing_fast_call=target["fast_call"]
			)
			target.update(fresh_data)

	# pylint: disable=unused-private-member
	def __full(self, _):
		"""Full."""
		return 1.0

	# pylint: disable=unused-private-member
	def __fuzzylogic(self, reward):
		"""Fuzzy logic membership: high reward."""
		if len(self.reward_history) < 10:
			return 1.0

		low_limit = np.percentile(self.reward_history, 20)
		high_limit = np.percentile(self.reward_history, 90)
		if reward >= high_limit:
			return 0.1
		if reward <= low_limit:
			return 1.0

		_mu = (reward - low_limit) / (high_limit - low_limit + 1e-8)
		return 1.0 - (_mu * 0.9)

	def __get_dynamic_ref_trust(self):
		"""Uses the moving average of performance as reference trust."""
		if len(self.reward_history) < 5:
			# Returning 0.0 creates an artificial ceiling in negative-reward environments.
			# We return -1.0 so that evaluated mutants (even those with negative scores)
			# have a higher trust than a fully "reset" unevaluated slot.
			return -1.0
		return self.__normalize_reward(float(np.mean(self.reward_history)))

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
		plasticity = self.plasticity_history[-1] if self.plasticity_history else 1.0
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
		if len(self.reward_history) < 10:
			# Warmup: scale against env_max_steps so trust values are distinct
			return float(np.clip(reward / self.env_max_steps, -2.0, 5.0))
		low = float(np.percentile(self.reward_history, 5))
		high = float(np.percentile(self.reward_history, 95))

		# Linear scaling relative to historical percentiles, but ALLOW values > 1.0
		# to reward individuals that break the current records.
		norm = (reward - low) / (high - low + 1e-8)
		return float(np.clip(norm, -2.0, 5.0))

	def __project_individual_weights(self, individual):
		"""Universal Snapshot Projection: aligns first layer weights to current normalizer."""
		if not self.is_continuous:
			return
		if self._obs_normalizer.count < self.max_eval_steps * 2:
			return

		mean_old = individual["last_mu"]
		variance_old = individual["last_var"]
		mean_new = self._obs_normalizer.mean
		variance_new = self._obs_normalizer.var

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
		if len(self.reward_history) < 10:
			return 1.0

		denominator = (self.max_seen - self.min_seen) + 1e-8
		normalized_error = (self.max_seen - reward) / denominator
		return max(0.1, min(1.0, normalized_error**2))

	def __reset_trust(self, factor=1.0):
		"""Blend all population trust toward the dynamic reference."""
		for guy in self.population:
			guy["trust"] = (guy["trust"] * (1.0 - factor)) + (
				self.__get_dynamic_ref_trust() * factor
			)

	def __select_individual(self):
		"""Selects the best individual from the population.
		In test mode: delegates to HallOfFame UCB1 to identify the best niche slot.
		In train mode: uses pure trust argmax.
		"""
		if not self.learn:
			failure_signal = self.__get_trust_decay(self.last_episode_score)
			if failure_signal > 0.6 and np.random.rand() < self.exploration_rate:
				# panic mode
				self.best_individual_idx = random.choice(
					[index for index, individual in enumerate(self.population) if not individual["is_gladiator"]]
				)
				return

			# niche gladiators
			niche_slots = self.hall_of_fame.select_best_gladiator_slot()
			if niche_slots:
				self.best_individual_idx = max(
					niche_slots,
					key=lambda index: self.population[index]["trust"],
				)
				return
		trust_scores = [guy["trust"] for guy in self.population]
		self.best_individual_idx = int(np.argmax(trust_scores))

	# pylint: disable=unused-private-member
	def __zero(self, _):
		"""Zero."""
		return 0.0
