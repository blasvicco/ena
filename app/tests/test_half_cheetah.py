"""Tests for the HalfCheetah challenge runner (physics variants, agent table, protocol wiring)."""

# General imports
import itertools

# Libs imports
import numpy as np
import pytest

# App imports
from helpers import step
from agents import ENAgent, PPOAgent, RandomAgent, SACAgent
from tools.early_stopping import StoppingRule
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
	"""Stage 0 gate: same seed and actions must give clearly different trajectories in every pair of worlds
	(the old make_env left Ice-Floor and High-Wind bit-identical to Earth)."""
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
	"""Regression: ENA evaluates candidates on make_env(name=..., **{attr: env.unwrapped.attr for attr in physics});
	wind was exposed as a float, which make_env can't take back, so ENA crashed on its first episode."""
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
	"""MuJoCo uses the larger of the two contacting geoms' frictions, so a world's friction must be set on
	every geom (the cheetah's feet included), not just the floor."""
	with step("Act: build a low-friction world."):
		env = half_cheetah.make_env(friction=0.05, name="Low")

	with step("Assert: every geom's sliding friction is the world's, and it is exposed for ENA."):
		np.testing.assert_allclose(env.unwrapped.model.geom_friction[:, 0], 0.05)
		assert env.unwrapped.friction == 0.05


def test_wind_gets_a_medium_to_act_through() -> None:
	"""Wind only acts through fluid forces, which need a non-zero medium density; calm worlds stay vacuum
	(the stock HalfCheetah)."""
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


@pytest.mark.parametrize("name, agent_type", [("ENA-Combined", ENAgent), ("PPO", PPOAgent), ("SAC", SACAgent), ("Random", RandomAgent)])
def test_build_agent_returns_the_named_agent(name, agent_type) -> None:
	"""Every challenge agent builds, with its name set."""
	with step("Act: build it."):
		agent = half_cheetah.build_agent(name, max_agent_threads=1)

	with step("Assert: type and name."):
		assert isinstance(agent, agent_type)
		assert agent.get_name() == name


def test_build_agent_rejects_an_unknown_name() -> None:
	"""DQN and SAC-Discrete are discrete-action agents, not part of this domain."""
	with step("Act/Assert: unknown names raise."):
		for name in ("DQN", "SAC-Discrete", "Nope"):
			with pytest.raises(ValueError):
				half_cheetah.build_agent(name, max_agent_threads=1)


def test_the_challenge_ena_is_the_combined_agent_with_this_domains_physics() -> None:
	"""Same skills as CartPole's ENA-Combined (dynamics_model + bpr, linear_ramp plasticity and trust, HoF on);
	HalfCheetah episodes are 1000 steps and its context is gravity, friction and wind."""
	with step("Act: build ENA."):
		agent = half_cheetah.build_agent("ENA-Combined", max_agent_threads=1)

	with step("Assert: the Combined skill set and the HalfCheetah settings."):
		assert agent.is_continuous
		assert {skill: _skill_driver(agent, skill) for skill in ("env_detection", "selection", "plasticity", "trust", "hall_of_fame")} == {
			"env_detection": "dynamics_model",
			"selection": "bpr",
			"plasticity": "linear_ramp",
			"trust": "linear_ramp",
			"hall_of_fame": "default",
		}
		assert (agent.max_episode_steps, agent.return_scale) == (1000, 4800.0)
		assert agent.physics == ["gravity", "friction", "wind"]


def test_baselines_use_a_constant_lr_and_the_standard_target_entropy() -> None:
	"""CartPole lessons: PPO's LR must not anneal toward an unknown end; SAC uses -dim(A) = -6."""
	with step("Act: build both baselines."):
		ppo = half_cheetah.build_agent("PPO", max_agent_threads=1)
		sac = half_cheetah.build_agent("SAC", max_agent_threads=1)

	with step("Assert: continuous PPO with a constant LR; SAC's target entropy -6."):
		assert ppo.is_continuous
		assert ppo.anneal_lr is False
		assert sac.target_entropy == -6.0


@pytest.mark.parametrize("protocol, stops, cap", [("fixed-budget", False, "FIXED_BUDGET_PER_ENV"), ("first-solve", True, "MAX_STEPS_PER_ENV")])
def test_stopping_rule_per_protocol(protocol, stops, cap) -> None:
	"""Both protocols share the evaluation checks; only fixed-budget never stops before its budget."""
	with step("Act: the rule."):
		rule = half_cheetah.make_stopping_rule(protocol)

	with step("Assert: the domain's checks and bar, and the protocol's stopping and cap."):
		assert isinstance(rule, StoppingRule)
		assert (rule.check_every, rule.eval_episodes, rule.solved_score) == (
			half_cheetah.CHECK_EVERY,
			half_cheetah.EVAL_EPISODES,
			half_cheetah.SOLVED_SCORE,
		)
		assert (rule.stop_when_solved, rule.stop_on_plateau) == (stops, stops)
		assert rule.max_steps == getattr(half_cheetah, cap)


def test_stopping_rule_rejects_an_unknown_protocol() -> None:
	"""A typo'd protocol must not silently fall back to one of the two."""
	with step("Act/Assert: raises."):
		with pytest.raises(ValueError):
			half_cheetah.make_stopping_rule("fastest")


def test_parse_args_defaults_and_protocol_output_dirs() -> None:
	"""No flags: all agents plus Random, fixed-budget into half_cheetah_outputs; first-solve has its own dir."""
	with step("Act: parse defaults and the first-solve protocol with workers."):
		args = half_cheetah.parse_challenge_args([])
		first_solve = half_cheetah.parse_challenge_args(["--protocol", "first-solve", "--workers", "5", "--num-experiments", "10"])

	with step("Assert: defaults and overrides."):
		assert args.variants == ["ENA-Combined", "PPO", "SAC"]
		assert args.include_random is True
		assert args.protocol == "fixed-budget"
		assert args.num_experiments == half_cheetah.NUM_EXPERIMENTS
		assert args.output_dir.endswith("half_cheetah_outputs")
		assert first_solve.output_dir.endswith("half_cheetah_first_solve_outputs")
		assert (first_solve.workers, first_solve.num_experiments) == (5, 10)


def test_run_experiment_wires_the_domain_into_the_shared_runner(monkeypatch, tmp_path) -> None:
	"""One replicate: seeded; every agent under one stopping rule with separate evaluation copies of the
	training worlds; tested on the 5 HalfCheetah worlds (never CartPole's) with this domain's bar and range."""
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

	with step("Act: replicate 3, fixed budget, PPO and SAC plus Random."):
		half_cheetah.run_experiment(3, str(tmp_path), None, [{"name": "PPO"}, {"name": "SAC"}], include_random=True)

	with step("Assert: seeding, training wiring, testing wiring."):
		assert captured["seed_id"] == 3
		training, testing = captured["training"], captured["testing"]
		assert [agent.get_name() for agent in training["agents"]] == ["PPO", "SAC", "Random"]
		assert training["stopping"].max_steps == half_cheetah.FIXED_BUDGET_PER_ENV
		assert training["training_x_axis"] == "env_steps"
		names = [world.unwrapped.name for world in training["worlds"]]
		assert names == ["Mars-Slippery", "Jupiter-Rough", "Earth-Standard"]
		assert [world.unwrapped.name for world in training["eval_worlds"]] == names
		assert all(evaluation is not world for evaluation, world in zip(training["eval_worlds"], training["worlds"]))
		assert [world.unwrapped.name for world in testing["test_worlds"]] == [
			"Mars-Slippery",
			"Ice-Floor",
			"Jupiter-Rough",
			"High-Wind",
			"Earth-Standard",
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


def test_ena_uses_the_halfcheetah_detection_calibration() -> None:
	"""The calibrated detector settings reach ENA's detector (verification window, margin, windows)."""
	# pylint: disable=protected-access
	with step("Act: build the study's ENA."):
		detector = (
			half_cheetah.build_agent("ENA-Combined", max_agent_threads=1)._skills["env_detection"]._instance
		)  # pylint: disable=protected-access

	with step("Assert: the calibrated values."):
		assert (detector._min_verify_samples, detector._novelty_margin, detector._recent.maxlen, detector._min_samples) == (1_000, 3.0, 3_000, 1_000)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_ena_brain_hidden_layers_take_the_requested_activation(activation) -> None:
	"""The search-tuning probes compare ReLU and tanh hidden layers; the output stays tanh (the action range)."""
	with step(f"Act: a 64x64 ENA brain with {activation} hidden layers."):
		layers = half_cheetah.ena_brain(6, 17, hidden=64, activation=activation).layers

	with step("Assert: two hidden layers of 64 with that activation, tanh output of the action size."):
		assert [layer.units for layer in layers] == [64, 64, 6]
		assert [layer.activation.__name__ for layer in layers] == [activation, activation, "tanh"]


def test_ena_brain_defaults_to_relu_hidden_layers() -> None:
	"""The default keeps the brain every earlier HalfCheetah probe used."""
	with step("Act/Assert: hidden layers ReLU by default."):
		assert [layer.activation.__name__ for layer in half_cheetah.ena_brain(6, 17).layers] == ["relu", "relu", "tanh"]
