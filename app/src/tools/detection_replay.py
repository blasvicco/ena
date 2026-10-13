#!/usr/bin/env python
"""Replay a recorded ENA run's environment-detection calls through a fresh detector with other settings.

A live ENA run costs an hour; the detector's behavior depends only on the transitions it observes and on the
order of its calls. record() wraps an agent's detector so a chirp run writes that call sequence (episode starts,
transitions, episode-end evaluations, niche assignments) to an .npz file; replay() feeds it to a new detector
built with any settings and reports, per training episode, the confirmed changes and the niche assigned.

Approximation: the transitions are the recorded run's. A detector with other settings would have changed ENA's
behavior (fewer resets, other policies), so replay compares settings on the same evidence; the chosen settings
are then confirmed by a live run. Replayed with the recorded settings it must reproduce the recorded changes
exactly (check_replay), or the replay is not trusted.

	python src/tools/detection_replay.py --recording half_cheetah_probe_outputs/<run>/detection.npz \\
		--training-log half_cheetah_probe_outputs/<run>/live_training.csv --set recognizer_min_samples=10000
"""

# General imports
import argparse
import os
import sys
from types import SimpleNamespace

# Libs imports
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from agents.ena.skills.env_detection.main import EnvironmentRecognizer

START, OBSERVE, EVALUATE, MATCH, CHIRP = 0, 1, 2, 3, 4
# ENAgent argument -> detector (driver) keyword.
SETTING_NAMES = {
	"recognizer_residual_tolerance": "residual_tolerance",
	"recognizer_novelty_margin": "novelty_margin",
	"recognizer_cusum_false_alarm_steps": "cusum_false_alarm_steps",
	"recognizer_var_floor": "var_floor",
	"recognizer_min_samples": "min_samples",
	"recognizer_min_verify_samples": "min_verify_samples",
	"recognizer_recent_size": "recent_size",
	"recognizer_chirp": "chirp",
}


class Recorder:  # pylint: disable=too-many-instance-attributes
	"""Wraps a detector facade's calls, keeping their order, every transition and every outcome."""

	def __init__(self, facade):
		self.kinds, self.learn, self.outcomes, self.states, self.actions, self.next_states = [], [], [], [], [], []
		self.chirps = []
		self._start_episode, self._observe = facade.start_episode, facade.observe
		self._evaluate, self._match_niche, self._observe_chirp = facade.evaluate, facade.match_niche, facade.observe_chirp
		facade.start_episode, facade.observe, facade.evaluate, facade.match_niche = self.start_episode, self.observe, self.evaluate, self.match_niche
		facade.observe_chirp = self.observe_chirp

	def _event(self, kind: int, learn: bool = False, outcome: float = 0.0) -> None:
		self.kinds.append(kind)
		self.learn.append(learn)
		self.outcomes.append(outcome)

	def start_episode(self) -> None:
		"""Record, then delegate."""
		self._event(START)
		self._start_episode()

	def observe(self, state, action, next_state, memory) -> None:
		"""Record the transition, then delegate."""
		self._event(OBSERVE)
		self.states.append(np.asarray(state, dtype=np.float32))
		self.actions.append(np.asarray(action, dtype=np.float32))
		self.next_states.append(np.asarray(next_state, dtype=np.float32))
		self._observe(state, action, next_state, memory)

	def evaluate(self, memory, learn: bool) -> bool:
		"""Delegate, then record whether a change was confirmed."""
		shifted = self._evaluate(memory, learn)
		self._event(EVALUATE, learn, float(shifted))
		return shifted

	def match_niche(self, memory, current_individual, hall_of_fame, population=None) -> int:
		"""Delegate, then record the niche assigned."""
		niche = self._match_niche(memory, current_individual, hall_of_fame, population=population)
		self._event(MATCH, outcome=float(niche))
		return niche

	def observe_chirp(self, response) -> None:
		"""Record a chirp's observations, then delegate."""
		self._event(CHIRP)
		self.chirps.append(np.asarray(response, dtype=np.float32))
		self._observe_chirp(response)

	def save(self, path: str) -> None:
		"""Write the call sequence, transitions and chirps."""
		np.savez_compressed(
			path,
			kinds=np.asarray(self.kinds, dtype=np.int8),
			learn=np.asarray(self.learn, dtype=bool),
			outcomes=np.asarray(self.outcomes, dtype=np.float32),
			states=np.asarray(self.states),
			actions=np.asarray(self.actions),
			next_states=np.asarray(self.next_states),
			chirps=np.asarray(self.chirps),
		)


def _memory() -> SimpleNamespace:
	"""The only memory the detector's facade touches: the observation normalizer reset and the history flush."""
	return SimpleNamespace(obs_normalizer=SimpleNamespace(reset=lambda: None), clear_training_history=lambda: None)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def replay(recording: dict, state_dim: int, action_dim: int, settings: dict, strategy: str = None, is_continuous: bool = True) -> pd.DataFrame:
	"""One row per recorded training episode: whether a fresh detector confirmed a change, and the niche it assigned.

	The driver is `strategy`, else the settings' recognizer_strategy, else dynamics_model."""
	settings = dict(settings)
	strategy = strategy or settings.pop("recognizer_strategy", "dynamics_model")
	settings.pop("recognizer_strategy", None)
	detector = EnvironmentRecognizer(
		strategy, state_dim, action_dim=action_dim, is_continuous=is_continuous, **{SETTING_NAMES[key]: value for key, value in settings.items()}
	)
	memory, rows, transition, chirp, row = _memory(), [], 0, 0, {"shift": False, "niche": np.nan}
	for kind, learn in zip(recording["kinds"], recording["learn"]):
		if kind == START:
			detector.start_episode()
			row = {"shift": False, "niche": np.nan}
		elif kind == OBSERVE:
			detector.observe(recording["states"][transition], recording["actions"][transition], recording["next_states"][transition], memory)
			transition += 1
		elif kind == CHIRP:
			detector.observe_chirp(recording["chirps"][chirp])
			chirp += 1
		elif kind == EVALUATE:
			row["shift"] = bool(detector.evaluate(memory, bool(learn)))
			if learn:
				rows.append(row)
		elif kind == MATCH and rows:
			rows[-1]["niche"] = detector.match_niche(memory, None, None)
	return pd.DataFrame(rows)


def recorded_shifts(recording: dict) -> np.ndarray:
	"""The recorded run's confirmed changes, per training episode."""
	return recording["outcomes"][(recording["kinds"] == EVALUATE) & recording["learn"]].astype(bool)


def check_replay(recording: dict, replayed: pd.DataFrame) -> bool:
	"""Whether a replay with the recorded settings reproduces the recorded changes exactly."""
	return bool(np.array_equal(recorded_shifts(recording), replayed["shift"].to_numpy()))


def summarize(replayed: pd.DataFrame, env_ids: pd.Series) -> pd.DataFrame:
	"""Per training environment: episodes, confirmed changes, distinct niches assigned; plus the run's niche count."""
	frame = replayed.assign(env_id=env_ids.to_numpy()[: len(replayed)])
	summary = frame.groupby("env_id", sort=False).agg(episodes=("shift", "size"), changes=("shift", "sum"), niches=("niche", "nunique"))
	summary.loc["all"] = [len(frame), int(frame["shift"].sum()), frame["niche"].nunique()]
	return summary


def main() -> None:
	"""Replay a recording through a driver and settings; reports whether it reproduces the recorded changes (it must,
	when given the settings the run was recorded with)."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--recording", required=True)
	parser.add_argument("--training-log", required=True, help="The run's live_training.csv (environment per episode).")
	parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ENAgent recognizer_* settings (dotted keys nest).")
	parser.add_argument("--strategy", default=None, help="Detector driver (default: the settings' recognizer_strategy, else dynamics_model).")
	args = parser.parse_args()
	# pylint: disable=import-outside-toplevel
	from tools.half_cheetah_probe import parse_overrides

	recording = dict(np.load(args.recording))
	env_ids = pd.read_csv(args.training_log)["env_id"]
	state_dim, action_dim = recording["states"].shape[1], recording["actions"].shape[1]
	replayed = replay(recording, state_dim, action_dim, parse_overrides(args.set), strategy=args.strategy)
	print(f"Reproduces the recorded run: {check_replay(recording, replayed)}")
	print(summarize(replayed, env_ids).to_string())


if __name__ == "__main__":
	main()
