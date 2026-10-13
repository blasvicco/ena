"""Data helpers for the live dashboard: discover studies/replicates/agents and downsample histories for display."""

# General imports
import glob
import os
import re

# Libs imports
import numpy as np
import pandas as pd


def discover_studies(root: str) -> list:
	"""*_outputs directories under root that contain at least one exp_* replicate, most recently modified first."""
	studies = [path for path in glob.glob(os.path.join(root, "*_outputs")) if glob.glob(os.path.join(path, "exp_*"))]
	return sorted(studies, key=os.path.getmtime, reverse=True)


def list_replicates(study: str) -> list:
	"""The study's exp_* replicate names, in numeric order."""
	names = [os.path.basename(path) for path in glob.glob(os.path.join(study, "exp_*")) if os.path.isdir(path)]
	return sorted(names, key=lambda name: int(re.sub(r"\D", "", name) or 0))


def list_agents(replicate_dir: str) -> list:
	"""Agents with a live_training CSV in the replicate, sorted."""
	paths = glob.glob(os.path.join(replicate_dir, "live_training_*.csv"))
	return sorted(re.sub(r"^live_training_|\.csv$", "", os.path.basename(path)) for path in paths)


def read_log(path: str):
	"""Read a live CSV, tolerating a torn last line mid-write; None if unreadable."""
	try:
		log = pd.read_csv(path)
	except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, UnicodeDecodeError, OSError):
		return None
	# A concurrent flush can leave the last row half-written (NaN in always-populated columns).
	if not log.empty and log.iloc[-1][[column for column in ("episode", "score") if column in log]].isna().any():
		log = log.iloc[:-1]
	return log


def binned_scores(frame: pd.DataFrame, max_points: int = 1_000) -> pd.DataFrame:
	"""Mean score in at most max_points consecutive buckets, indexed by env_steps (or episode if absent).

	Sending every episode to the browser froze it on long runs (e.g. 365k Random episodes under a
	7.5M-step budget); a bucket mean keeps the curve's shape at a bounded size.
	"""
	axis = "env_steps" if "env_steps" in frame and frame["env_steps"].notna().all() else "episode"
	if len(frame) <= max_points:
		return frame.set_index(axis)[["score"]]
	buckets = np.arange(len(frame)) * max_points // len(frame)
	binned = frame.groupby(buckets).agg({axis: "last", "score": "mean"})
	return binned.set_index(axis)
