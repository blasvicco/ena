"""Tests for the HalfCheetah challenge runner (physics variants, agent table, protocol wiring)."""

# General imports
import itertools
import os

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents import ENAgent, PPOAgent, SACAgent
import half_cheetah
import cartpole_ena_ablation

# pylint: disable=duplicate-code

ROLLOUT_STEPS = 300
# Smallest max |state difference| (over the rollout) that counts as "physically different".
MIN_DIVERGENCE = 0.5


def _rollout(env, seed: int = 0, steps: int = ROLLOUT_STEPS) -> np.ndarray:
	"""States visited from a seeded reset under a fixed, seeded random action sequence."""
	actions = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(steps, env.action_space.shape[0]))
	state, _ = env.reset(seed=seed)
	states = [state]
	for action in actions:
		state, _, _, _, _ = env.step(action)
		states.append(state)
	env.close()
	return np.asarray(states)


def _all_worlds() -> list:
	"""Every distinct world of the study: the training worlds plus the never-trained test worlds."""
	worlds = half_cheetah.make_training_worlds() + half_cheetah.make_unseen_worlds()
	return worlds


def test_study_has_five_distinct_named_worlds() -> None:
	"""3 training + 2 never-trained test worlds, each with its own name."""
	with step("Act: build every world."):
		names = [env.unwrapped.name for env in _all_worlds()]

	with step("Assert: five unique names, three of them training worlds."):
		assert len(names) == 5
		assert len(set(names)) == 5
		assert len(half_cheetah.make_training_worlds()) == 3


def test_every_pair_of_worlds_produces_different_physics() -> None:
	"""Every pair of HalfCheetah worlds gives clearly different trajectories for the same seed and actions."""
	# Stage 0 gate: same seed and actions must give clearly different trajectories in every pair of worlds
	# (the old make_env left Ice-Floor and High-Wind bit-identical to Earth).
	with step("Arrange: roll out every world under the same seed and actions."):
		rollouts = {env.unwrapped.name: _rollout(env) for env in _all_worlds()}

	with step("Assert: every pair diverges by more than MIN_DIVERGENCE."):
		too_close = {
			(first, second): float(np.max(np.abs(rollouts[first] - rollouts[second])))
			for first, second in itertools.combinations(rollouts, 2)
			if np.max(np.abs(rollouts[first] - rollouts[second])) <= MIN_DIVERGENCE
		}
		assert not too_close, f"Physically indistinguishable worlds: {too_close}"


@pytest.mark.parametrize("world_index", range(5))
def test_same_world_twice_is_identical(world_index) -> None:
	"""Reproducibility: two fresh copies of one world give bit-identical trajectories."""
	with step("Act: roll out two fresh copies of the same world."):
		first = _rollout(_all_worlds()[world_index])
		second = _rollout(_all_worlds()[world_index])

	with step("Assert: identical."):
		np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("world_index", range(5))
def test_ena_can_clone_every_world_from_its_exposed_physics(world_index) -> None:
	"""ENA can rebuild every world from its exposed physics parameters."""
	# Regression: ENA evaluates candidates on make_env(name=..., **{attr: env.unwrapped.attr for attr in physics});
	# wind was exposed as a float, which make_env can't take back, so ENA crashed on its first episode.
	with step("Arrange: a world and ENA's physics list."):
		physics = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1).physics
		world = _all_worlds()[world_index]

	with step("Act: clone it the way ENA does."):
		params = {key: getattr(world.unwrapped, key) for key in physics if hasattr(world.unwrapped, key)}
		clone = half_cheetah.make_env(name=world.unwrapped.name, **params)

	with step("Assert: every physics attribute is exposed, and the clone behaves identically."):
		assert set(params) == set(physics)
		np.testing.assert_array_equal(_rollout(clone), _rollout(world))


def test_friction_applies_to_every_contact() -> None:
	"""A world's friction is set on every geom, since MuJoCo uses the larger of two contacting frictions."""
	# MuJoCo uses the larger of the two contacting geoms' frictions, so a world's friction must be set on
	# every geom (the cheetah's feet included), not just the floor.
	with step("Act: build a low-friction world."):
		env = half_cheetah.make_env(friction=0.05, name="Low")

	with step("Assert: every geom's sliding friction is the world's, and it is exposed for ENA."):
		np.testing.assert_allclose(env.unwrapped.model.geom_friction[:, 0], 0.05)
		assert env.unwrapped.friction == 0.05


def test_wind_gets_a_medium_to_act_through() -> None:
	"""Wind only acts through fluid forces, which need a non-zero medium density; calm worlds stay vacuum (the stock HalfCheetah)."""
	with step("Act: build a windy and a calm world."):
		windy = half_cheetah.make_env(wind=(-20.0, 0.0, 0.0), name="Windy")
		calm = half_cheetah.make_env(name="Calm")

	with step("Assert: the windy one has air and the wind; the calm one is unchanged."):
		assert windy.unwrapped.model.opt.density > 0
		np.testing.assert_allclose(windy.unwrapped.model.opt.wind, [-20.0, 0.0, 0.0])
		assert calm.unwrapped.model.opt.density == 0
		np.testing.assert_allclose(calm.unwrapped.model.opt.wind, 0.0)


def _skill_driver(agent, skill: str) -> str:
	"""Module name of the driver behind one of ENA's skill facades."""
	return type(agent._skills[skill]._instance).__module__.rsplit(".", 1)[-1]  # pylint: disable=protected-access


@pytest.mark.parametrize("name, agent_type", [("ENA-Combined", ENAgent), ("PPO", PPOAgent), ("SAC", SACAgent)])
def test_build_agent_returns_the_named_agent(name, agent_type) -> None:
	"""Every challenge agent builds, with its name set."""
	with step("Act: build it."):
		agent = half_cheetah.build_agent(name, max_agent_threads=1)

	with step("Assert: type and name."):
		assert isinstance(agent, agent_type)
		assert agent.get_name() == name


def test_build_agent_rejects_an_unknown_name() -> None:
	"""DQN and SAC-Discrete are discrete-action agents, not part of this domain; the study has no Random floor."""
	with step("Act/Assert: unknown names raise."):
		for name in ("DQN", "SAC-Discrete", "Random", "Nope"):
			with pytest.raises(ValueError):
				half_cheetah.build_agent(name, max_agent_threads=1)


def test_the_challenge_ena_is_the_combined_agent_with_this_domains_physics() -> None:
	"""The HalfCheetah ENA is ENA-Combined with this domain's physics and the chirp detector."""
	# CartPole's ENA-Combined skills (bpr, linear_ramp plasticity and trust, HoF on), except environment detection: the
	# chirp driver, since dynamics_model misfires on HalfCheetah (detection benchmark). Episodes are 1000 steps; the
	# context is gravity, friction and wind.
	with step("Act: build ENA."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1)

	with step("Assert: the Combined skill set and the HalfCheetah settings."):
		assert agent.is_continuous
		assert {skill: _skill_driver(agent, skill) for skill in ("env_detection", "selection", "plasticity", "trust", "hall_of_fame")} == {
			"env_detection": "chirp",
			"selection": "bpr",
			"plasticity": "linear_ramp",
			"trust": "linear_ramp",
			"hall_of_fame": "olympic",
		}
		assert (agent.max_episode_steps, agent.return_scale) == (1000, 4800.0)
		assert agent.physics == ["gravity", "friction", "wind"]


def test_the_study_ena_has_the_frozen_search_settings() -> None:
	"""The study's ENA uses the search settings frozen by the tuning probes."""
	# The search tuning's choice (T5): a 32x16 tanh brain, population 100, depth 200, mutation rate 0.25 and noise 0.1,
	# and the olympic Hall of Fame (an Olympiad every 200k steps, 3 episodes, 2 qualifier seats).
	# pylint: disable=protected-access
	with step("Act: build the study's ENA."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1)

	with step("Assert: the frozen settings."):
		assert [layer.units for layer in agent.population[0]["model"].layers] == [32, 16, 6]
		assert (agent.pop_size, agent.max_eval_steps) == (100, 200)
		mutation = agent._skills["mutation"]._instance
		assert (mutation._mutation_rate, mutation._mutation_noise) == (0.25, 0.1)
		hall_of_fame = agent._skills["hall_of_fame"]._instance
		assert (hall_of_fame.interval, hall_of_fame.episodes, hall_of_fame.qualifier_seats) == (200_000, 3, 2)


def test_baselines_use_a_constant_lr_and_the_standard_target_entropy() -> None:
	"""CartPole lessons: PPO's LR must not anneal toward an unknown end; SAC uses -dim(A) = -6."""
	with step("Act: build both baselines."):
		ppo = half_cheetah.build_agent("PPO", max_agent_threads=1)
		sac = half_cheetah.build_agent("SAC", max_agent_threads=1)

	with step("Assert: continuous PPO with a constant LR; SAC's target entropy -6."):
		assert ppo.is_continuous
		assert ppo.anneal_lr is False
		assert sac.target_entropy == -6.0


def test_parse_args_defaults() -> None:
	"""No flags: all agents (no Random floor) into outputs/half_cheetah_outputs. There is one protocol, a fixed step budget, so no --protocol flag."""
	with step("Act: parse defaults and overrides."):
		args = half_cheetah.parse_challenge_args([])
		overridden = half_cheetah.parse_challenge_args(["--workers", "5", "--num-experiments", "10"])

	with step("Assert: defaults and overrides."):
		assert args.variants == ["ENA-Combined", "PPO", "SAC"]
		assert args.include_random is False
		assert args.num_experiments == half_cheetah.NUM_EXPERIMENTS
		assert args.output_dir.endswith(os.path.join("outputs", "half_cheetah_outputs"))
		assert (overridden.workers, overridden.num_experiments) == (5, 10)

	with step("Assert: --protocol is not an option."):
		with pytest.raises(SystemExit):
			half_cheetah.parse_challenge_args(["--protocol", "first-solve"])


def test_run_experiment_has_no_random_floor(tmp_path) -> None:
	"""The study runs ENA, PPO and SAC only; asking for the Random floor is an error, not a silent extra agent."""
	with step("Act/Assert: include_random raises."):
		with pytest.raises(ValueError):
			half_cheetah.run_experiment(0, str(tmp_path), None, [{"name": "PPO"}], include_random=True)


def test_run_experiment_wires_the_domain_into_the_shared_runner(monkeypatch, tmp_path) -> None:
	"""One HalfCheetah replicate runs through the shared runner with this domain's budget, worlds and chart."""
	# One replicate: seeded; every agent trains FIXED_BUDGET_PER_ENV env-steps per training world, with no checks (no
	# stopping rule, no evaluation copies); tested on the 5 HalfCheetah worlds (never CartPole's) with this domain's
	# reference line and range.
	with step("Arrange: capture the shared runner's arguments and the seed; stub agent construction."):
		captured = {}

		def _record_training(**kwargs):
			captured["training"] = kwargs

		def _record_testing(**kwargs):
			captured["testing"] = kwargs

		def _record_seed(experiment_id):
			captured["seed_id"] = experiment_id

		def _build_stub(name, _max_agent_threads):
			return _NamedAgent(name)

		monkeypatch.setattr(cartpole_ena_ablation, "agent_training", _record_training)
		monkeypatch.setattr(cartpole_ena_ablation, "agent_testing", _record_testing)
		monkeypatch.setattr(half_cheetah, "seed_experiment", _record_seed)
		monkeypatch.setattr(half_cheetah, "build_agent", _build_stub)

	with step("Act: replicate 3, fixed budget, PPO and SAC."):
		half_cheetah.run_experiment(3, str(tmp_path), None, [{"name": "PPO"}, {"name": "SAC"}], include_random=False)

	with step("Assert: seeding, training wiring, testing wiring."):
		assert captured["seed_id"] == 3
		training, testing = captured["training"], captured["testing"]
		assert [agent.get_name() for agent in training["agents"]] == ["PPO", "SAC"]
		assert training["steps_per_env"] == [half_cheetah.FIXED_BUDGET_PER_ENV] * 3
		assert training["health_every"] == 10_000
		assert "stopping" not in training and "eval_worlds" not in training
		assert training["training_x_axis"] == "env_steps"
		assert [world.unwrapped.name for world in training["worlds"]] == ["Earth-Standard", "Heavy-Rough", "Mars-Slippery"]
		assert [world.unwrapped.name for world in testing["test_worlds"]] == [
			"Earth-Standard",
			"Ice-Floor",
			"Heavy-Rough",
			"High-Wind",
			"Mars-Slippery",
		]
		for kwargs in (training, testing):
			assert kwargs["success_threshold"] == half_cheetah.SOLVED_SCORE
			assert kwargs["ylim"] == half_cheetah.Y_LIM
		assert all(agent.episodes == half_cheetah.TESTING_EPISODES for agent in testing["agents"])


class _NamedAgent:
	"""Agent stub for run_experiment's wiring."""

	def __init__(self, name):
		self._name = name
		self.episodes = None

	def get_name(self):
		"""Name."""
		return self._name

	def set_episodes(self, episodes):
		"""Record the episode budget."""
		self.episodes = episodes


def test_ppo_is_built_with_the_zoos_tuned_halfcheetah_settings() -> None:
	"""PPO uses RL Baselines3 Zoo's validated HalfCheetah configuration (see half_cheetah.py), not our own choice."""
	with step("Act: build the study's PPO."):
		agent = half_cheetah.build_agent("PPO", max_agent_threads=1)

	with step("Assert: the zoo's values, and 256x256 ReLU networks."):
		assert (agent.n_steps, agent.batch_size, agent.n_epochs, agent.gamma, agent.gae_lambda, agent.clip_ratio) == (512, 64, 20, 0.98, 0.92, 0.1)
		assert (agent.initial_lr, agent.entropy_coef, agent.value_coef, agent.max_grad_norm) == (2.0633e-5, 4.01762e-4, 0.58096, 0.8)
		assert agent.reward_scaling == "return" and agent.anneal_lr is False
		assert np.allclose(agent.log_std.numpy(), -2.0)
		assert [layer.units for layer in agent.actor.layers if hasattr(layer, "units")] == [256, 256, 6]


def test_sac_is_built_with_the_sb3_defaults_and_the_zoos_warm_up() -> None:
	"""SAC: 1M buffer, 10k random warm-up, SB3's learning rate, batch, tau and gamma; same update ratio."""
	with step("Act: build the study's SAC."):
		agent = half_cheetah.build_agent("SAC", max_agent_threads=1)

	with step("Assert: the reference values."):
		assert (agent.buffer.capacity, agent.learning_starts, agent.batch_size, agent.tau, agent.gamma) == (1_000_000, 10_000, 256, 0.005, 0.99)
		assert agent.gradient_steps == agent.update_frequency  # one gradient step per environment step
		assert agent.target_entropy == -6.0
		assert agent.fused_updates and not agent.jit_compile  # an implementation detail: same steps, fewer Python calls


def test_ena_detects_halfcheetah_environments_with_the_chirp_driver() -> None:
	"""The study's ENA plays a chirp: a 64-step Mexican-hat wavelet train, analysed by Haar wavelet packets (one energy per band)."""
	# pylint: disable=protected-access
	with step("Act: build the study's ENA."):
		detector = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1)._skills["env_detection"]._instance

	with step("Assert: the chirp driver with the chosen configuration."):
		assert type(detector).__module__.endswith("driver.chirp")
		assert detector.chirp_actions.shape == (64, 6)
		assert detector._settings["excitation"] == "wavelet_train" and detector._settings["excitation_params"] == {"wavelet": "mexh"}
		assert detector._settings["features"] == {"kind": "wavelet", "transform": "packet", "level": 3}


def test_ena_brain_is_the_es_mujoco_policy_network() -> None:
	"""Salimans et al. (2017), MuJoCo: tanh hidden layers and normc weights; tanh output in the [-1, 1] action range; zero biases."""
	# The output keeps normc std 1.0 (not their 0.01): every individual of the first population acts, rather than standing still.
	with step("Act: a 64x64 ENA brain."):
		layers = half_cheetah.ena_brain(6, 17, hidden=64).layers

	with step("Assert: two tanh hidden layers of 64, a tanh output of the action size, normc std 1.0 throughout."):
		assert [layer.units for layer in layers] == [64, 64, 6]
		assert [layer.activation.__name__ for layer in layers] == ["tanh", "tanh", "tanh"]
		assert [type(layer.kernel_initializer) for layer in layers] == [half_cheetah.Normc] * 3
		assert [layer.kernel_initializer.std for layer in layers] == [1.0, 1.0, 1.0]
		assert all(not np.any(layer.bias.numpy()) for layer in layers)


def test_ena_brain_takes_one_size_per_hidden_layer() -> None:
	"""A tapered brain (32x16): the first hidden layer of 32, the second of 16; an int keeps both layers equal."""
	with step("Act: a 32x16 brain and a 32 brain."):
		tapered, square = half_cheetah.ena_brain(6, 17, hidden=(32, 16)).layers, half_cheetah.ena_brain(6, 17, hidden=32).layers

	with step("Assert: the layer sizes, all tanh."):
		assert [layer.units for layer in tapered] == [32, 16, 6]
		assert [layer.units for layer in square] == [32, 32, 6]
		assert [layer.activation.__name__ for layer in tapered] == ["tanh", "tanh", "tanh"]


@pytest.mark.parametrize("std", [1.0, 0.01])
def test_normc_draws_columns_of_norm_std(std) -> None:
	"""normc (the ES starter's normc_initializer): Gaussian weights, each unit's incoming weight vector rescaled to norm std."""
	with step("Act: two draws of a 17x64 kernel."):
		initializer = half_cheetah.Normc(std)
		first, second = np.asarray(initializer((17, 64))), np.asarray(initializer((17, 64)))

	with step("Assert: every column has norm std, and each draw is new."):
		np.testing.assert_allclose(np.linalg.norm(first, axis=0), std, rtol=1e-5)
		assert not np.allclose(first, second)


def test_normc_is_reproducible_under_the_experiment_seed() -> None:
	"""Seeded runs draw the same initial population: normc follows Keras' global seed, as every other initializer does."""
	# pylint: disable=import-outside-toplevel
	import keras

	with step("Act: draw after seeding, twice."):
		draws = []
		for _ in range(2):
			keras.utils.set_random_seed(7)
			draws.append(np.asarray(half_cheetah.Normc(1.0)((17, 8))))

	with step("Assert: identical draws."):
		np.testing.assert_array_equal(draws[0], draws[1])


def test_ena_brain_starts_moving() -> None:
	"""A fresh brain's actions on normalized observations are clearly non-zero, but not saturated at the action bounds."""
	with step("Act: a fresh 32x32 brain's actions on standard-normal observations."):
		actions = np.asarray(half_cheetah.ena_brain(6, 17, hidden=32)(np.random.default_rng(0).normal(size=(1_000, 17)).astype(np.float32)))

	with step("Assert: a typical action size between 0.2 and 0.8."):
		assert 0.2 < np.sqrt(np.mean(actions**2)) < 0.8


def test_ena_brain_reset_redraws_from_normc() -> None:
	"""ENA resets an individual by calling its layers' initializers: a reset brain keeps normc's column norms."""
	with step("Arrange: a brain, then a redraw of its kernels as ENA's reset does."):
		brain = half_cheetah.ena_brain(6, 17, hidden=8)
		for layer in brain.layers:
			layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape))

	with step("Assert: every layer's columns have norm 1."):
		for layer in brain.layers:
			np.testing.assert_allclose(np.linalg.norm(layer.kernel.numpy(), axis=0), 1.0, rtol=1e-5)


def test_the_worlds_are_the_author_s_definitions_in_order() -> None:
	"""The HalfCheetah worlds are the author's definitions, in training and test order."""
	# Training, in order: Earth-Standard (Earth's gravity and stock friction, a soft 5 m/s head wind), Heavy-Rough (1.5 g,
	# friction 1.0), Mars-Slippery (Mars's gravity, friction 0.2). Never trained: Ice-Floor (friction 0.05) and High-Wind
	# (a 20 m/s head wind).
	with step("Act: every world's physics, in order."):
		physics = [
			(world.unwrapped.name, world.unwrapped.gravity, world.unwrapped.friction, world.unwrapped.wind)
			for world in half_cheetah.make_training_worlds() + half_cheetah.make_unseen_worlds()
		]

	with step("Assert: the definitions."):
		assert physics == [
			("Earth-Standard", -9.8, 0.4, (-5.0, 0.0, 0.0)),
			("Heavy-Rough", -14.7, 1.0, (0.0, 0.0, 0.0)),
			("Mars-Slippery", -3.7, 0.2, (0.0, 0.0, 0.0)),
			("Ice-Floor", -9.8, 0.05, (0.0, 0.0, 0.0)),
			("High-Wind", -9.8, 0.4, (-20.0, 0.0, 0.0)),
		]


def test_the_heavy_world_stays_within_the_published_gravity_range() -> None:
	"""Heavy-Rough's gravity stays within the published HalfCheetah range (1.5 g)."""
	# Heavy-Rough replaces Jupiter (about 2.5 g, beyond any HalfCheetah benchmark): 1.5x Earth's gravity, the top of the published
	# range (CARL; multi-task HalfCheetah), on rough ground (friction 1.0).
	with step("Act: the training worlds' physics."):
		heavy = next(world for world in half_cheetah.make_training_worlds() if world.unwrapped.name == "Heavy-Rough").unwrapped

	with step("Assert: g = 1.5 x 9.8 and friction 1.0."):
		assert heavy.gravity == pytest.approx(-14.7) and heavy.friction == pytest.approx(1.0)


def test_hardware_line_names_the_machine_and_every_thread_count() -> None:
	"""Each replicate's hardware line names the CPU, the core count and every thread count."""
	# Each replicate records where it ran, so the paper's wall-clock times are comparable: the CPU, the core count,
	# TensorFlow's thread pool and ENA's evaluation threads.
	with step("Act: the line for 2 ENA threads."):
		line = half_cheetah.hardware_line(2)

	with step("Assert: the cores, both thread counts, and a CPU description."):
		assert line.startswith("Hardware: ")
		assert f"{os.cpu_count()} cores" in line
		assert "TensorFlow threads " in line and line.endswith("ENA threads 2")
