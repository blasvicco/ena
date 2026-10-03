"""EN Agent"""
# General imports
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

# Libs imports
import numpy as np
from keras import initializers
import tensorflow as tf

# App imports
from ..tools.abstract import AAgent
from .memory import AgentMemory
from .skills.crossover.main import Crossover
from .skills.env_detection.main import EnvironmentRecognizer
from .skills.hall_of_fame.main import HallOfFame
from .skills.mutation.main import Mutation
from .skills.plasticity.main import Plasticity
from .skills.selection.main import SelectionController
from .skills.trust.main import Trust
from .tools.inference import infer_action
from .tools.reward_normalizer import normalize_reward
from .tools.weight_projection import project_individual_weights


# pylint: disable=too-many-instance-attributes
class ENAgent(AAgent):
	"""The ENAgent class"""

	# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,dangerous-default-value
	def __init__(
		self,
		brain,
		make_env,
		controller_episode_rate: float = 0.03,
		controller_max_niches: int = 32,
		selection_driver: str = "heuristic",
		env_max_steps: float = 1000.0,
		exploration_rate: float = 0.20,
		hall_of_fame_driver: str = "default",
		max_per_niche: int = 5,
		max_gladiator_ratio: float = 0.5,
		history_size: int = 100,
		max_eval_steps: int = 500,
		min_eval_steps: int = 10,
		max_threads: int = 1,
		mutation_noise: float = 0.1,
		mutation_rate: float = 0.01,
		physics: list = ["gravity", "masspole", "length"],
		plasticity_algorithm: str = "zero",
		pop_size: int = 30,
		recognizer_threshold: float = 0.90,
		recognizer_ema_alpha: float = 0.10,
		recognizer_strategy: str = "live",
		recognizer_confirmation_window: int = 3,
		recognizer_residual_tolerance: float = 3.0,
		recognizer_novelty_margin: float = 2.0,
		recognizer_cusum_false_alarm_steps: float = 1e8,
		genetic_threshold: float = 0.80,
		genetic_bonus: float = 0.50,
		trust_algorithm: str = "zero",
	) -> None:
		super().__init__(brain, make_env)
		self.learn = True
		self.env_max_steps = float(env_max_steps)
		self.max_eval_steps = max_eval_steps
		self.min_eval_steps = min_eval_steps
		self.physics = physics
		self.pop_size = pop_size

		tmp_env = self.make_env()
		self.state_dim = int(tmp_env.observation_space.shape[0])
		self.is_continuous = not hasattr(tmp_env.action_space, "n")
		self.action_dim = int(tmp_env.action_space.shape[0]) if self.is_continuous else int(tmp_env.action_space.n)
		tmp_env.close()

		self.memory = AgentMemory(history_size, self.state_dim, max_eval_steps, ema_alpha=recognizer_ema_alpha)

		# Skills (pluggable, one facade per mechanism — see agents/ena/skills/)
		self._skills = {
			"env_detection": EnvironmentRecognizer(
				recognizer_strategy,
				self.state_dim,
				similarity_threshold=recognizer_threshold,
				ema_alpha=recognizer_ema_alpha,
				genetic_threshold=genetic_threshold,
				genetic_bonus=genetic_bonus,
				confirmation_window=recognizer_confirmation_window,
				action_dim=self.action_dim,
				is_continuous=self.is_continuous,
				residual_tolerance=recognizer_residual_tolerance,
				novelty_margin=recognizer_novelty_margin,
				cusum_false_alarm_steps=recognizer_cusum_false_alarm_steps,
			),
		}

		context_dim = len(self._skills["env_detection"].compute_full_vector(self.memory))
		self._skills["selection"] = SelectionController(
			selection_driver,
			buffer_size=32,
			context_dim=context_dim,
			controller_episode_rate=controller_episode_rate,
			env_max_steps=self.env_max_steps,
			exploration_rate=exploration_rate,
			max_niches=controller_max_niches,
		)

		self._skills["hall_of_fame"] = HallOfFame(
			hall_of_fame_driver,
			max_per_niche=max_per_niche,
			max_gladiator_ratio=max_gladiator_ratio,
			brain=brain,
			action_dim=self.action_dim,
			state_dim=self.state_dim,
		)
		self._skills["mutation"] = Mutation("gaussian", mutation_rate=mutation_rate, mutation_noise=mutation_noise)
		self._skills["crossover"] = Crossover("uniform_mask")
		self._skills["trust"] = Trust(trust_algorithm)
		self._skills["plasticity"] = Plasticity(plasticity_algorithm)

		# Initialize population
		self.best_individual_idx = 0
		self._episode_started = False
		self._last_env_shift_detected = False
		self.population = [self.__build_individual() for _ in range(self.pop_size)]

		# Standard Parallelization Setup
		self.max_threads = max_threads

		# Environment-step accounting: every step taken in an environment, played or spent
		# evaluating candidates (lets experiments equalize interaction budgets across agents).
		self.played_env_steps = 0
		self.candidate_env_steps = 0
		self._env_steps_lock = threading.Lock()

	def act(self, env, state: np.ndarray) -> tuple:
		"""Act as the ENAgent: select, execute, memorize, and monitor for an environment shift."""
		if not self._episode_started:
			self.__start_episode()

		if self.memory.steps_this_episode == 0:
			self.__select_individual()
		self.memory.steps_this_episode += 1

		chosen = self.population[self.best_individual_idx]
		self.memory.obs_normalizer.update(state[np.newaxis, :])
		self._skills["env_detection"].update(state, self.memory)
		action = infer_action(chosen, state, self.memory, self.is_continuous)

		next_state, reward, terminated, truncated, _ = env.step(action)
		self.played_env_steps += 1
		self.memory.record_step(state, next_state, reward)
		self._skills["env_detection"].observe(state, action, next_state, self.memory)
		self._skills["selection"].on_step(self.memory, self._skills["env_detection"], self.learn)

		if terminated or truncated:
			self.__end_episode()

		return next_state, reward, terminated, truncated, action

	@property
	def plasticity_history(self) -> list:
		"""Per-episode plasticity values recorded so far."""
		return self.memory.plasticity_history

	def evaluation_episode(self, env) -> tuple:
		"""Play one full frozen episode with the highest-trust individual on env; return (score, steps).

		Side-effect free for training: no trust, memory, selection or env-detection update, and the
		steps are not added to played_env_steps/candidate_env_steps (callers report them separately).
		The highest-trust individual is the one training-time selection exploits.
		"""
		best = max(self.population, key=lambda individual: individual["trust"])
		score, steps, _done = self.__run_individual(best, env, None, None)
		return score, steps

	@property
	def total_env_steps(self) -> int:
		"""Every environment step consumed so far: played episodes plus candidate evaluations."""
		return self.played_env_steps + self.candidate_env_steps

	@property
	def current_env_id(self) -> int:
		"""Current recognized niche ID (stable across recurring environments)."""
		return self._skills["env_detection"].current_niche_id

	@property
	def known_niche_ids(self) -> set:
		"""Niche IDs already established, distinguishing a genuinely new niche from one merely
		appearing for the first time in a slice of history (e.g. a test-phase chart)."""
		return self._skills["env_detection"].known_niche_ids

	@property
	def last_env_shift_detected(self) -> bool:
		"""Whether detect_change() fired at the end of the most recently completed episode."""
		# Independent of niche resolution, so callers can tell a real shift from a -1/no-match.
		return self._last_env_shift_detected

	def load_gladiators(self, restore_trust: bool = True) -> None:
		"""Re-introduces Gladiators into the active population with Niche Distribution."""
		if self._skills["hall_of_fame"]:
			self._skills["hall_of_fame"].inject_gladiators(
				self.population,
				self.pop_size,
				self.memory.obs_normalizer,
				restore_trust=restore_trust,
			)

	def set_learning(self, mode: bool, output_dir: str = None) -> None:
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
		self._skills["selection"].set_mode(self.memory, self.learn)

		# Trigger initial evaluation when entering test mode
		if not mode:
			# Flush leftover training-phase recognizer statistics so match() compares
			# the actual test-phase physics against fresh data, not stale training stats.
			self.memory.reset_recognizer_statistics()
			self._skills["env_detection"].reset()
			self._skills["selection"].evaluate(self.population)

		# Save niche details when entering test mode if output_dir is provided
		if not mode and output_dir is not None:
			self._skills["hall_of_fame"].save_niche_details(output_dir, self.get_name())

	# pylint: disable=too-many-locals
	def train(self, **kwargs) -> None:
		"""Handles the evolutionary search."""
		env, step_data = kwargs["env"], kwargs["step_data"]
		if not step_data["done"]:
			return

		# 1. Update Global Statistics
		self.memory.reward_history.append(self.memory.last_episode_score)
		self.memory.max_seen = max(self.memory.max_seen, self.memory.last_episode_score)
		self.memory.min_seen = min(self.memory.min_seen, self.memory.last_episode_score)

		# 2. Calculate Failure Signal once
		failure_signal = self._skills["trust"].decay(self.memory.last_episode_score, self.memory)

		# 3. Update the current Specialist's Trust
		chosen = self.population[self.best_individual_idx]
		chosen["age"] += 1
		norm_score = normalize_reward(self.memory.last_episode_score, self.memory, self.env_max_steps)
		chosen["trust"] = (1 - failure_signal) * chosen["trust"] + (failure_signal * norm_score)
		# 4. Compute plasticity for this episode
		plasticity = self._skills["plasticity"].compute(self.memory.last_episode_score, self.memory, self.learn)
		self.memory.plasticity_history.append(plasticity)

		# 6. Update Hall of Fame. The current env name is handed over rather than written onto
		# chosen: a gladiator's env label is fixed at induction, and a foreign-niche success
		# is archived as a clone carrying this name instead.
		self._skills["hall_of_fame"].update(
			chosen,
			chosen["trust"],
			self.population,
			self.memory,
			self._skills["env_detection"],
			self.memory.obs_normalizer,
			env_name=env.unwrapped.name,
		)

		# 7. Only reset trust (challenge the specialist) when failing
		if failure_signal > 0.1:
			self._skills["trust"].reset_population(self.population, self.memory, self.env_max_steps, factor=failure_signal)

		# tools/metrics.py's estimate_eval_cost() mirrors this formula for display (live
		# dashboard / compute-cost report) — update both if this one changes.
		num_to_eval = max(2, int(self.pop_size * plasticity))
		other_indices = [index for index in range(self.pop_size) if index != self.best_individual_idx]
		test_indices = random.sample(other_indices, min(len(other_indices), num_to_eval))

		# Depth-scale candidate evaluation by the same plasticity signal that scales breadth
		# (num_to_eval above): confident/converged generations evaluate candidates cheaply
		# (min_eval_steps floor, never 0), uncertain ones get the full max_eval_steps for a
		# reliable signal. Applies uniformly — no special-casing by is_gladiator.
		eval_step_cap = int(self.min_eval_steps + (self.max_eval_steps - self.min_eval_steps) * plasticity)

		gen_seed = np.random.randint(0, 9999)
		env_params = {key: getattr(env.unwrapped, key) for key in self.physics if hasattr(env.unwrapped, key)}

		def thread_eval_task(task_idx: int) -> tuple:
			"""Evaluate one population member in its own environment instance."""
			individual_to_eval = self.population[task_idx]
			local_env = self.make_env(name=env.unwrapped.name, **env_params)
			score = self.__evaluate_individual_performance(individual_to_eval, local_env, seed=gen_seed, max_steps=eval_step_cap)
			local_env.close()
			return task_idx, score

		with ThreadPoolExecutor(max_workers=self.max_threads) as _executor:
			eval_results = list(_executor.map(thread_eval_task, test_indices))

		for task_idx, score in eval_results:
			individual_signal = self._skills["trust"].decay(score, self.memory)
			eval_norm_score = normalize_reward(score, self.memory, self.env_max_steps)
			old = self.population[task_idx]["trust"]
			self.population[task_idx]["trust"] = (1 - individual_signal) * old + (individual_signal * eval_norm_score)

		# 8. Breed and Mutate
		self.__evolve_population()

	# ---------------------------------------------------------------
	# Private methods
	# ---------------------------------------------------------------

	def __build_individual(self, model=None, existing_fast_call=None):
		"""Build the NN model or reset an existing one in-place."""
		if model is None:
			model = self.brain(self.action_dim, self.state_dim)
		else:
			for index, layer in enumerate(model.layers):
				if hasattr(layer, "kernel_initializer") and layer.kernel_initializer is not None:
					gain = 0.1 if index == len(model.layers) - 1 else 1.0
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
			"trust": self._skills["trust"].reference(self.memory, self.env_max_steps),
			"is_gladiator": False,
			"last_mu": np.copy(self.memory.obs_normalizer.mean),
			"last_var": np.copy(self.memory.obs_normalizer.var),
		}

	def __end_episode(self) -> None:
		"""Finalize the episode: score bookkeeping, driver finalization, and shift detection."""
		self.memory.last_episode_score = self.memory.current_episode_reward
		self._skills["selection"].on_episode_end(self.memory, self.learn, self.population, self._skills["env_detection"], self.env_max_steps)

		self._last_env_shift_detected = self._skills["env_detection"].evaluate(self.memory, self.learn)
		if self._last_env_shift_detected and not self.learn:
			self._skills["selection"].evaluate(self.population)

		self.memory.current_episode_reward = 0
		self.memory.steps_this_episode = 0
		self._episode_started = False

	def __evaluate_individual_performance(self, individual, env, seed=None, max_steps=None):
		"""Evaluate an individual for a single episode and return the score."""
		eval_step_cap = max_steps if max_steps is not None else self.max_eval_steps
		score, steps, done = self.__run_individual(individual, env, seed, eval_step_cap)

		with self._env_steps_lock:  # candidates are evaluated from a thread pool
			self.candidate_env_steps += steps

		# Scale truncated eval score to full-episode equivalent using env_max_steps
		if not done:
			score = score * (self.env_max_steps / eval_step_cap)

		return float(score)

	def __run_individual(self, individual, env, seed, step_cap):
		"""Play one episode with an individual (live-normalized inputs); return (score, steps, done)."""
		project_individual_weights(individual, self.memory.obs_normalizer, self.max_eval_steps)
		state, _ = env.reset(seed=seed)
		done, score, steps = False, 0, 0

		while not done and (step_cap is None or steps < step_cap):
			norm_state = self.memory.obs_normalizer.normalize(state)
			state_tensor = tf.convert_to_tensor(norm_state.reshape(1, -1), dtype=tf.float32)
			action_output = individual["fast_call"](state_tensor, training=False)
			action_numpy = action_output.numpy()[0]

			action = np.clip(action_numpy, -1.0, 1.0) if self.is_continuous else int(np.argmax(action_numpy))

			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			score += reward
			steps += 1

		return float(score), steps, done

	def __evolve_population(self, replace_count=None):
		"""Evolution: reuses model buffers to guarantee zero session bloat."""
		# 1. Rank by trust/age
		indices = list(range(self.pop_size))
		indices.sort(
			key=lambda index: (self.population[index]["trust"], self.population[index]["age"]),
			reverse=True,
		)

		# 2. Strategy Budget
		if replace_count is None:
			replace_count = int(self.pop_size * 0.8)
		replace_count = max(1, min(replace_count, self.pop_size - 2))

		elite_pool_size = max(2, int(self.pop_size * 0.2))
		elite_indices = indices[:elite_pool_size]

		# Kill the WORST individuals — but strictly protect gladiators
		to_replace_indices = [index for index in indices[-replace_count:] if not self.population[index].get("is_gladiator", False)]
		random.shuffle(to_replace_indices)
		actual_replace_count = len(to_replace_indices)

		mutant_count = int(actual_replace_count * 0.4)
		crossover_count = int(actual_replace_count * 0.4)
		default_trust = self._skills["trust"].reference(self.memory, self.env_max_steps)
		plasticity = self.memory.plasticity_history[-1] if self.memory.plasticity_history else 1.0

		for _ in range(mutant_count):
			if not to_replace_indices:
				break
			target_idx = to_replace_indices.pop(0)
			parent = self.population[random.choice(elite_indices)]
			self._skills["mutation"].apply(parent, self.population[target_idx], plasticity, default_trust)

		# --- Fill Crossovers ---
		for _ in range(crossover_count):
			if not to_replace_indices:
				break
			target_idx = to_replace_indices.pop(0)
			parent_1, parent_2 = random.sample([self.population[index] for index in elite_indices], 2)
			self._skills["crossover"].apply(
				parent_1, parent_2, self.population[target_idx], self.memory.obs_normalizer, self.max_eval_steps, default_trust
			)

		# --- Fill Fresh (Diversity) ---
		while to_replace_indices:
			target_idx = to_replace_indices.pop(0)
			target = self.population[target_idx]
			fresh_data = self.__build_individual(model=target["model"], existing_fast_call=target["fast_call"])
			target.update(fresh_data)

	def __select_individual(self) -> None:
		"""Pick this episode's active individual and project its weights for evaluation."""
		self.best_individual_idx = self._skills["selection"].select_individual(
			self.population,
			self._skills["env_detection"],
			self.memory,
			learn=self.learn,
		)
		project_individual_weights(self.population[self.best_individual_idx], self.memory.obs_normalizer, self.max_eval_steps)

	def __start_episode(self) -> None:
		"""Reset per-episode selection/step state and signal the episode start to env_detection."""
		self._skills["selection"].reset_episode()
		self.memory.steps_this_episode = 0
		self._episode_started = True
		self._skills["env_detection"].start_episode()
