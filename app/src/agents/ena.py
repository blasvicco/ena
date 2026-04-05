"""EN Agent"""
# General imports
import random
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Libs imports
import numpy as np
from numpy.linalg import norm
from keras import initializers
import tensorflow as tf

# App imports
from .abstract import AAgent

# pylint: disable=too-many-instance-attributes
class ENAgent(AAgent):
	"""The ENAgent class"""

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		action_size,
		brain,
		make_env,
		exploration_rate=0.20,
		history_size=100,
		gladiator_amounts=5,
		max_eval_steps=500,
		max_threads=1,
		plasticity_algorithm="zero",
		pop_size=30,
		mutation_rate=0.01,
		trust_algorithm="zero",
	):
		super().__init__(brain, make_env)
		self.action_size = action_size
		self.exploration_rate = exploration_rate
		self.last_episode_score = 0
		self.learn = True
		self.max_eval_steps = max_eval_steps
		self.mutation_rate = mutation_rate
		self.plasticity_algorithm = plasticity_algorithm
		self.plasticity_history = []
		self.pop_size = pop_size
		self.trust_algorithm = trust_algorithm

		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])
		self.action_dim = int(tmp_env.action_space.n)
		tmp_env.close()

		# AGNOSTIC STATISTICS
		self.reward_history = deque(maxlen=history_size)
		self.max_seen = -np.inf
		self.min_seen = np.inf
		self.episodes_seen = 0

		# Hall of Fame
		self.gladiator_amounts = gladiator_amounts
		self.hall_of_fame = {}

		# Initialize population with age and score metadata
		self.best_individual_idx = 0
		self.current_episode_reward = 0
		self.population = [self.__build_individual() for _ in range(pop_size)]

		# Standard Parallelization Setup
		self.max_threads = max_threads

	# pylint: disable=too-many-locals
	def act(self, env, state):
		"""Act as the ENAgent"""
		# 1. Selection Logic
		# We only pick a NEW specialist at the very start of an episode
		# (when current_episode_reward is 0)
		if self.current_episode_reward == 0:
			if self.learn and np.random.rand() < self.exploration_rate:
				self.best_individual_idx = np.random.randint(self.pop_size)
			else:
				trust_scores = [guy["trust"] for guy in self.population]
				self.best_individual_idx = np.argmax(trust_scores)

		# 2. Execution
		chosen = self.population[self.best_individual_idx]

		state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
		q_values = chosen["model"](state_tensor, training=False)
		action = np.argmax(q_values.numpy()[0])

		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated

		# 3. Internal State Management
		self.current_episode_reward += reward

		# 4. EPISODE OVER: Update trust autonomously
		if done:
			self.last_episode_score = self.current_episode_reward

			if not self.learn:
				decay = self.__get_trust_decay(self.last_episode_score)
				old_trust = self.population[self.best_individual_idx]["trust"]
				norm_score = self.last_episode_score / 500.0
				new_trust = (1 - decay) * old_trust + (decay * norm_score)
				self.population[self.best_individual_idx]["trust"] = new_trust

			self.current_episode_reward = 0  # Reset for next episode

		return next_state, reward, terminated, truncated, action

	def load_gladiators(self):
		"""Re-introduces the Gladiators into the active population."""
		if not self.hall_of_fame:
			return

		# Convert dict values to list
		gladiators = list(self.hall_of_fame.values())
		for index, gladiator in enumerate(gladiators):
			if index >= self.pop_size:
				break

			# Overwrite population slot
			target = self.population[index]
			target["age"] = 0
			target["id"] = gladiator["id"]
			target["model"].set_weights(gladiator["model"].get_weights())
			# target["trust"] = self.__get_dynamic_ref_trust()
			target["trust"] = gladiator["trust"]

	def set_learning(self, mode):
		"""Reset trust"""
		self.learn = mode
		for guy in self.population:
			guy["trust"] = self.__get_dynamic_ref_trust()
			guy["age"] = 0
		self.best_individual_idx = 0

	# pylint: disable=too-many-locals
	def train(self, **kwargs):
		"""Handles the evolutionary search."""
		env, step_data = kwargs["env"], kwargs["step_data"]
		# Only evolve at the end of an episode
		if not step_data["done"]:
			return

		# 1. Update Global Statistics
		self.episodes_seen += 1
		self.reward_history.append(self.last_episode_score)
		self.max_seen = max(self.max_seen, self.last_episode_score)
		self.min_seen = min(self.min_seen, self.last_episode_score)

		# 2. Calculate the "Failure Signal" (Decay) ONCE
		# This value is 0.1 if performing well, and 1.0 if failing.
		failure_signal = self.__get_trust_decay(self.last_episode_score)

		# 3. Update the current Specialist's Trust
		chosen = self.population[self.best_individual_idx]
		# NORMALIZATION: divide score by 500.0
		norm_score = self.last_episode_score / 500.0
		old_trust = chosen["trust"]
		chosen["trust"] = (1 - failure_signal) * old_trust + (
			failure_signal * norm_score
		)

		# 4. Trigger Global Reset (Forget old winners)
		# We use the same failure_signal. If it's high (> 0.1), we reset everyone.
		if failure_signal > 0.1:
			self.__update_hall_of_fame(chosen, old_trust)
			self.__reset_trust(factor=failure_signal)

		# 5. Evaluate others based on Plasticity (Standard Parallelization)
		plasticity = self.__get_plasticity(self.last_episode_score)
		self.plasticity_history.append(plasticity)

		num_to_eval = max(2, int(self.pop_size * plasticity))
		other_indices = [
			index for index in range(self.pop_size) if index != self.best_individual_idx
		]
		test_indices = random.sample(
			other_indices, min(len(other_indices), num_to_eval)
		)

		gen_seed = np.random.randint(0, 9999)

		# Sync environment parameters (Gravity, Mass, etc.)
		env_params = {
			"gravity": getattr(env.unwrapped, "gravity", 9.8),
			"name": getattr(env.unwrapped, "name", "Unknown"),
			"mass_pole": getattr(env.unwrapped, "masspole", 0.1),
			"pole_length": getattr(env.unwrapped, "length", 0.5),
		}

		def thread_eval_task(task_idx):
			guy_to_eval = self.population[task_idx]
			# Create a fresh local env for the thread to avoid race conditions
			local_env = self.make_env(
				gravity=env_params["gravity"],
				name=env_params["name"],
				mass_pole=env_params["mass_pole"],
				pole_length=env_params["pole_length"],
			)
			score = self.__evaluate_individual_performance(
				guy_to_eval["model"], local_env, seed=gen_seed
			)
			local_env.close()
			return task_idx, score

		# Execute Glads in parallel
		with ThreadPoolExecutor(max_workers=self.max_threads) as _executor:
			eval_results = list(_executor.map(thread_eval_task, test_indices))

		# Update population with results
		for idx, score in eval_results:
			self.population[idx]["trust"] = score / 500.0
			self.population[idx]["age"] += 1

		# 6. Breed and Mutate
		self.__evolve_population()

		# 7. Final selection for the next training episode
		trusts = [guy["trust"] for guy in self.population]
		self.best_individual_idx = np.argmax(trusts)

	# Private methods
	def __build_individual(self, model=None):
		"""Build the NN model or reset an existing one in-place."""
		if model is None:
			model = self.brain(self.action_dim, self.state_dim)
		else:
			# Resets existing model weights to a fresh state to avoid leakage
			initializer = initializers.Orthogonal(gain=1.0)
			for layer in model.layers:
				if hasattr(layer, "kernel_initializer"):
					layer.kernel.assign(initializer(layer.kernel.shape))
				if hasattr(layer, "bias_initializer"):
					layer.bias.assign(initializers.Zeros()(layer.bias.shape))

		return {
			"age": 0,
			"id": str(uuid.uuid4()),
			"model": model,
			"trust": self.__get_dynamic_ref_trust(),
		}

	def __calculate_genetic_similarity(self, ind_a, ind_b_record):
		"""
		Calculates the Cosine Similarity between a candidate and a HoF record.
		Returns: float between -1.0 (opposite) and 1.0 (identical)
		"""
		# 1. Extract and Flatten Weights
		# Candidate is a live dictionary, HoF record is a stored dictionary
		weights_a = ind_a["model"].get_weights()
		weights_b = ind_b_record["model"].get_weights()

		# Flatten list of arrays into a single 1D vector
		vec_a = np.concatenate([w.flatten() for w in weights_a])
		vec_b = np.concatenate([w.flatten() for w in weights_b])

		# 2. Compute Cosine Similarity: (A . B) / (|A| * |B|)
		# A value of 1.0 means they are genetically identical strategies
		dot_product = np.dot(vec_a, vec_b)
		norm_a = norm(vec_a) + 1e-8
		norm_b = norm(vec_b) + 1e-8

		similarity = dot_product / (norm_a * norm_b)
		return similarity

	def __crossover(self, parent_1, parent_2, target):
		"""Performs crossover directly into the target individual's model."""
		weights_1, weights_2 = (
			parent_1["model"].get_weights(),
			parent_2["model"].get_weights(),
		)
		new_weights = []
		for weight_1, weight_2 in zip(weights_1, weights_2):
			mask = np.random.random(weight_1.shape) < 0.5
			combined = np.where(mask, weight_1, weight_2)
			new_weights.append(combined)

		target["model"].set_weights(new_weights)
		target["age"] = max(parent_1["age"], parent_2["age"])
		target["trust"] = self.__get_dynamic_ref_trust()
		target["id"] = str(uuid.uuid4())

	def __get_dynamic_ref_trust(self):
		"""Replaces ref_trust. Uses the moving average of performance."""
		if len(self.reward_history) < 5:
			return 0.0
		return np.mean(self.reward_history) / 500.0

	def __evaluate_individual_performance(self, individual_model, env, seed=None):
		"""Helper method to run a single episode for a given individual returning its score."""
		state, _ = env.reset(seed=seed)
		done, score, steps = False, 0, 0

		while not done and steps < self.max_eval_steps:
			state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
			action_values = individual_model(state_tensor, training=False)
			state, reward, terminated, truncated, _ = env.step(
				np.argmax(action_values.numpy()[0])
			)
			done = terminated or truncated
			score += reward
			steps += 1

		# If the episode was capped (not a natural termination), extrapolate
		# to the full episode length so selection pressure is preserved.
		# A capped individual was still alive — reward it proportionally.
		if not done:
			score = score * (500.0 / self.max_eval_steps)

		return float(score)

	def __evolve_population(self):
		"""Evolution method that reuses model buffers to guarantee zero session bloat."""
		# 1. Rank indices by trust/performance to find who to keep and who to kill
		indices = list(range(self.pop_size))
		indices.sort(
			key=lambda index: (
				self.population[index]["trust"],
				self.population[index]["age"],
			),
			reverse=True,
		)

		# 2. Define our "Strategy Budget" for the whole population
		elite_count = max(2, int(self.pop_size * 0.2))
		mutant_count = int(self.pop_size * 0.3)
		crossover_count = int(self.pop_size * 0.3)

		elite_indices = indices[:elite_count]
		to_replace_indices = indices[elite_count:]
		random.shuffle(to_replace_indices)

		# 3. Fill the replacement slots based on our budget (REUSING EXISTING MODELS)
		current_ptr = 0

		# --- Fill Mutants ---
		for _ in range(mutant_count):
			if current_ptr >= len(to_replace_indices):
				break
			target_idx = to_replace_indices[current_ptr]
			parent = self.population[random.choice(elite_indices)]
			self.__mutate(parent, self.population[target_idx])
			current_ptr += 1

		# --- Fill Crossovers ---
		for _ in range(crossover_count):
			if current_ptr >= len(to_replace_indices):
				break
			target_idx = to_replace_indices[current_ptr]
			parent_1, parent_2 = random.sample(
				[self.population[index] for index in elite_indices], 2
			)
			self.__crossover(parent_1, parent_2, self.population[target_idx])
			current_ptr += 1

		# --- Fill Fresh (Diversity) ---
		while current_ptr < len(to_replace_indices):
			target_idx = to_replace_indices[current_ptr]
			# Reset the metadata but keep the existing model object
			fresh_data = self.__build_individual(
				model=self.population[target_idx]["model"]
			)
			self.population[target_idx].update(fresh_data)
			current_ptr += 1

	# pylint: disable=unused-private-member
	def __quadratic(self, reward):
		"""Aggressive algorithm."""
		if self.episodes_seen < 5:
			return 1.0
		denominator = (self.max_seen - self.min_seen) + 1e-8
		normalized_error = (self.max_seen - reward) / denominator
		return max(0.1, min(1.0, normalized_error**2))

	# pylint: disable=unused-private-member
	def __full(self, _):
		"""Full."""
		return 1.0

	# pylint: disable=unused-private-member
	def __fuzzylogic(self, reward):
		"""Fuzzy logic membership: high reward."""
		if len(self.reward_history) < 10:
			return 1.0

		# We define "Low" as the 20th percentile and "High" as the 90th percentile
		low_limit = np.percentile(self.reward_history, 20)
		high_limit = np.percentile(self.reward_history, 90)

		if reward >= high_limit:
			return 0.1
		if reward <= low_limit:
			return 1.0

		# Calculate membership (_mu) within the dynamic range
		_mu = (reward - low_limit) / (high_limit - low_limit + 1e-8)
		return 1.0 - (_mu * 0.9)

	def __get_plasticity(self, reward):
		"""Modular dispatch to chosen algorithm using getattr."""
		plasticity_algorithm = self.plasticity_algorithm if self.learn else "zero"
		try:
			# Dynamically call the private method based on string name
			method_name = f"_ENAgent__{plasticity_algorithm}"
			method = getattr(self, method_name)
			return method(reward)
		except AttributeError:
			return 1.0  # Default to full if algorithm name is invalid

	def __get_trust_decay(self, reward):
		"""Modular dispatch for Trust Reset logic"""
		try:
			method_name = f"_ENAgent__{self.trust_algorithm}"
			method = getattr(self, method_name)
			return method(reward)
		except AttributeError:
			return 0.0  # Default to no reset

	def __mutate(self, parent, target):
		"""Performs mutation directly into the target individual's model."""
		target["model"].set_weights(parent["model"].get_weights())
		target["age"] = parent["age"]
		target["id"] = str(uuid.uuid4())
		target["trust"] = self.__get_dynamic_ref_trust()

		weights = target["model"].get_weights()
		new_weights = []
		for weight in weights:
			mask = np.random.random(weight.shape) < self.mutation_rate
			noise = np.random.normal(0, 0.1, weight.shape)
			weight[mask] += noise[mask]
			new_weights.append(weight)
		target["model"].set_weights(new_weights)

	def __reset_trust(self, factor=1.0):
		"""Reset trust"""
		for guy in self.population:
			guy["trust"] = (guy["trust"] * (1.0 - factor)) + (
				self.__get_dynamic_ref_trust() * factor
			)

	def __save_to_hof(self, individual, trust, z_score):
		"""Helper to create a deep copy for the archive"""
		backup = self.__build_individual()
		backup["id"] = individual["id"]  # Keep original ID
		backup["induction_z_score"] = z_score
		backup["model"].set_weights(individual["model"].get_weights())
		backup["trust"] = trust
		self.hall_of_fame[individual["id"]] = backup

	def __update_hall_of_fame(self, winner, old_trust):
		"""Update Hall of Fame"""
		# --- STEP 0: MEMORY DECAY ("Age-of-Discovery") ---
		# Every time we try to update the HoF, the existing legends fade slightly.
		# This ensures that if a strategy is not being "re-validated", it eventually
		# becomes vulnerable to replacement.
		MEMORY_DECAY_RATE = 0.995  # Retain 99.5% of glory per check

		for item in self.hall_of_fame.values():
			# We decay the 'Genius Score', not the Trust.
			# This means the agent remembers how to play (Trust), but forgets
			# how 'special' that player was (Z-Score).
			item["induction_z_score"] *= MEMORY_DECAY_RATE

		# --- STEP 1: CALCULATE CONTEXT & METRICS ---
		current_trusts = [guy["trust"] for guy in self.population]
		avg_trust = np.mean(current_trusts)
		std_trust = np.std(current_trusts) + 1e-8

		# Calculate "Genius" Score (Z-Score)
		current_z_score = (old_trust - avg_trust) / std_trust

		# Calculate Entry Threshold (Lower bar if Plasticity is high/Chaos)
		current_plasticity = (
			self.plasticity_history[-1] if self.plasticity_history else 1.0
		)
		z_score_required = 1.5 * (1.0 - (current_plasticity * 0.5))

		# Early Exit: If they aren't impressive enough relative to CURRENT peers
		if current_z_score < z_score_required:
			return

		ind_id = winner["id"]

		# --- STEP 2: CHECK FOR EXISTING ID ---
		# If this specific agent ID is already in HoF, just update stats
		if ind_id in self.hall_of_fame:
			if current_z_score > self.hall_of_fame[ind_id].get("induction_z_score", 0):
				self.hall_of_fame[ind_id]["trust"] = old_trust
				self.hall_of_fame[ind_id]["induction_z_score"] = current_z_score
			return

		# --- STEP 3: NICHE PROTECTION (Genetic Similarity) ---
		# Check if a "Strategic Twin" already exists in the HoF
		SIMILARITY_THRESHOLD = 0.95  # 95% similar weights = same species
		for hof_id, hof_record in self.hall_of_fame.items():
			similarity = self.__calculate_genetic_similarity(winner, hof_record)

			if similarity > SIMILARITY_THRESHOLD:
				# TWIN FOUND!
				# We compete ONLY against this specific twin, not the weakest link.
				# This protects unique niches from being overwritten by duplicates.

				existing_z = hof_record.get("induction_z_score", -99)
				if current_z_score > existing_z:
					del self.hall_of_fame[hof_id]
					self.__save_to_hof(winner, old_trust, current_z_score)
				else:
					# If the new guy isn't smarter than his twin, we discard him.
					# We do NOT let him challenge a different niche.
					pass
				return  # Exit immediately after handling the twin interaction

		# --- STEP 4: OPEN SLOT INDUCTION ---
		# If no twin was found and we have space, welcome aboard!
		if len(self.hall_of_fame) < self.gladiator_amounts:
			self.__save_to_hof(winner, old_trust, current_z_score)
			return

		# --- STEP 5: COMPETITIVE REPLACEMENT (Full HoF) ---
		# If we are here, the HoF is full AND this agent is a unique new species.
		# Now they have earned the right to challenge the globally weakest link.

		weakest_id = min(
			self.hall_of_fame,
			key=lambda k: self.hall_of_fame[k].get("induction_z_score", 0),
		)
		weakest_z = self.hall_of_fame[weakest_id].get("induction_z_score", 0)

		if current_z_score > weakest_z:
			del self.hall_of_fame[weakest_id]
			self.__save_to_hof(winner, old_trust, current_z_score)

	# pylint: disable=unused-private-member
	def __zero(self, _):
		"""Zero"""
		return 0.0
