"""Characterization test: the default Hall of Fame's decisions over a scripted sequence of updates."""
# The golden file was recorded from the driver as it ran every reported experiment (2026-10-03), before the
# removal of unused code that day. Any change to what the archive keeps, evicts, refreshes or clones makes
# this test fail.

# General imports
import json
import os
from types import SimpleNamespace

# Libs imports
import numpy as np
from keras import Input, Sequential
from keras.layers import Dense

# App imports
from helpers import step
from agents.ena.skills.hall_of_fame.driver.default import Instance

# pylint: disable=protected-access,duplicate-code

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hof_characterization.json")


def _brain(action_dim, state_dim):
	"""A minimal one-layer model."""
	return Sequential([Input(shape=(state_dim,)), Dense(action_dim)])


class _Recognizer:  # pylint: disable=too-few-public-methods
	"""Resolves every winner to the niche set for the current step."""

	def __init__(self):
		self.current_niche_id = 0

	def match_niche(self, _memory, _winner, _hall_of_fame, population=None):  # pylint: disable=unused-argument
		"""Resolve to the current niche."""
		return self.current_niche_id


def _individual(name, seed):
	"""A non-gladiator population member with deterministic weights."""
	model = _brain(2, 2)
	rng = np.random.default_rng(seed)
	model.set_weights([rng.normal(size=weight.shape) for weight in model.get_weights()])
	return {"individual_id": name, "model": model, "trust": 0.0, "is_gladiator": False}


def _twin(name, source, seed):
	"""A member whose weights are the source's plus tiny noise (cosine similarity > 0.95)."""
	twin = _individual(name, seed)
	rng = np.random.default_rng(seed)
	twin["model"].set_weights([weight + 1e-3 * rng.normal(size=weight.shape) for weight in source["model"].get_weights()])
	return twin


def _fingerprint(model) -> float:
	"""A deterministic summary of a model's weights."""
	return round(float(sum(np.sum(weight) for weight in model.get_weights())), 6)


def _snapshot(hall_of_fame, population) -> dict:
	"""Every live decision: archive records (no generated ids) and the population's gladiator tags."""
	# pylint: disable-next=protected-access
	records = sorted(
		[record["trust"], record["internal_env_id"], record["env_name"], _fingerprint(record["model"])] for record in hall_of_fame._records.values()
	)
	tags = [
		[member["individual_id"], member.get("is_gladiator", False), member.get("internal_env_id", -1), member.get("env_name")]
		for member in population
	]
	return {"records": records, "population": tags}


def run_scenario() -> list:
	"""Drive update() through every path; return a snapshot after each step."""
	hall_of_fame = Instance(2, 0.5, _brain, action_dim=2, state_dim=2)
	recognizer = _Recognizer()
	obs_normalizer = SimpleNamespace(mean=np.zeros(2), var=np.ones(2))
	population = [_individual(f"m{index}", index) for index in range(6)]
	population.append(_twin("twin-of-m0", population[0], 100))
	population.append(_twin("twin-of-m1", population[1], 101))
	by_id = {member["individual_id"]: member for member in population}

	# (winner, trust, niche, env name, plasticity): the gate, inductions, a full niche, evictions,
	# twins better and worse, same-niche refreshes, cross-niche clones and their refreshes.
	script = [
		("m0", 0.1, 0, "Mars", 0.1),  # fails the gate (low trust, peers equal)
		("m0", 5.0, 0, "Mars", 0.1),  # induction, niche 0
		("m1", 4.0, 0, "Mars", 1.0),  # induction, niche 0 now full
		("m2", 3.0, 0, "Mars", 0.5),  # full niche, weaker than the weakest: rejected
		("m2", 6.0, 0, "Mars", 0.5),  # full niche: evicts the weakest (m1)
		("twin-of-m0", 4.5, 0, "Mars", 0.5),  # twin of m0, worse: rejected
		("twin-of-m0", 9.0, 0, "Mars", 0.5),  # twin of m0, better: replaces m0
		("m3", 7.0, 1, "Jupiter", 0.2),  # induction, niche 1
		("m2", 8.0, 1, "Jupiter", 0.2),  # gladiator of niche 0 wins in niche 1: cloned
		("m2", 8.5, 1, "Jupiter", 0.2),  # second foreign win, higher trust: clone refreshed
		("m2", 6.5, 0, "Earth", 0.2),  # same-niche win, above its stored 6.0: refreshed, label kept
		("m2", 9.5, 0, "Earth", 0.2),  # same-niche improvement again: refreshed
		("twin-of-m1", 2.0, 2, "Neptune", 1.0),  # low trust but high plasticity lowers the gate
		("m4", 0.6, 2, "Neptune", 0.0),  # passes the gate on absolute trust (>= 0.5)
		("m5", 10.0, 1, "Jupiter", 0.0),  # full niche 1: evicts its weakest
	]
	snapshots = []
	for winner_id, trust, niche_id, env_name, plasticity in script:
		recognizer.current_niche_id = niche_id
		winner = by_id[winner_id]
		winner["trust"] = trust
		memory = SimpleNamespace(plasticity_history=[plasticity])
		hall_of_fame.update(winner, trust, population, memory, recognizer, obs_normalizer, env_name=env_name)
		snapshots.append(_snapshot(hall_of_fame, population))
	return snapshots


def test_hall_of_fame_decisions_match_the_recorded_behavior() -> None:
	"""Every step's archive and population state equals the golden recording."""
	with step("Arrange: the golden recording."):
		with open(GOLDEN_PATH, encoding="utf-8") as golden_file:
			golden = json.load(golden_file)

	with step("Act: replay the scenario."):
		snapshots = run_scenario()

	with step("Assert: identical, step by step."):
		assert len(snapshots) == len(golden)
		for index, (snapshot, expected) in enumerate(zip(snapshots, golden)):
			assert json.loads(json.dumps(snapshot)) == expected, f"step {index} differs"
