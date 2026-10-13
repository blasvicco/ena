#!/usr/bin/env python
"""Live convergence dashboard: auto-refreshing browser view of the live_training_*.csv files a study writes.

Pick a study (*_outputs dir), a replicate (exp_N) and an agent in the sidebar; the page shows
that agent's latest values and score curve, plus every agent of the replicate overlaid on one
chart. Curves are plotted against env-steps when the CSV has them (agents in the same study can
play very different numbers of episodes for the same step budget) and are downsampled to a
bounded number of points, so long runs don't freeze the browser.

Run it as a separate process — it only reads the CSVs, never touches a training run:

	streamlit run src/tools/live_dashboard.py                  (from app/, inside or outside the container)

Port 8501 is mapped to the host in docker-compose.yml.
"""

# General imports
import os
import sys

# Libs imports
import pandas as pd
import streamlit as st

# App imports
# streamlit run only puts this script's own directory on sys.path; the app package root is src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.live_data import binned_scores, discover_studies, list_agents, list_replicates, read_log  # pylint: disable=wrong-import-position

REFRESH_SECONDS = 5
MAX_POINTS = 1_000
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd.
APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="ENA Live Convergence", layout="wide")
st.title("ENA Live Convergence")

studies = discover_studies(APP_DIR)
if not studies:
	st.info(f"No *_outputs study with exp_* replicates under {APP_DIR} yet.")
	st.stop()
study = st.sidebar.selectbox("Study", studies, format_func=os.path.basename)
replicates = list_replicates(study)
replicate = st.sidebar.selectbox("Replicate", replicates, index=len(replicates) - 1 if replicates else 0)
replicate_dir = os.path.join(study, replicate)
agents = list_agents(replicate_dir)
if not agents:
	st.info(f"No live_training_*.csv in {replicate_dir} yet.")
	st.stop()
agent = st.sidebar.selectbox("Agent", agents)
st.sidebar.caption(f"Refreshes every {REFRESH_SECONDS}s; charts show at most {MAX_POINTS:,} points (bucket means).")


@st.cache_data(max_entries=64)
def _load(path: str, _mtime: float) -> tuple:
	"""(latest row, binned score curve, downsampled plasticity) for a CSV, recomputed only when it changes."""
	# _mtime is part of the cache key (underscore-free args are hashed; this one is passed by value).
	log = read_log(path)
	if log is None or log.empty:
		return None, None, None
	plasticity = None
	if "plasticity" in log and log["plasticity"].notna().any():
		plasticity = log[["plasticity"]].copy()
		plasticity.index = log["env_steps"] if "env_steps" in log else log["episode"]
		plasticity = plasticity.iloc[:: max(1, len(plasticity) // MAX_POINTS)]
	return log.iloc[-1], binned_scores(log, MAX_POINTS), plasticity


def _load_csv(path: str) -> tuple:
	"""Cached _load keyed on the file's modification time; (None, None, None) if it's missing."""
	try:
		return _load(path, os.path.getmtime(path))
	except FileNotFoundError:
		return None, None, None


@st.fragment(run_every=REFRESH_SECONDS)
def _live_view() -> None:
	"""Latest metrics and curves for the selected agent, then every agent of the replicate overlaid."""
	latest, curve, plasticity = _load_csv(os.path.join(replicate_dir, f"live_training_{agent}.csv"))
	if latest is None:
		st.info(f"Waiting for data from {agent}...")
		return

	columns = st.columns(5)
	columns[0].metric("Episode", f"{int(latest['episode']):,}")
	columns[1].metric("Last score", f"{latest['score']:.1f}")
	if pd.notna(latest.get("env_steps")):
		columns[2].metric("Env-steps", f"{int(latest['env_steps']):,}")
	if "env_id" in latest:
		columns[3].metric("Current env", str(latest["env_id"]))
	if pd.notna(latest.get("plasticity")):
		columns[4].metric("Plasticity", f"{latest['plasticity']:.2f}")

	st.subheader(f"{agent}: score")
	st.line_chart(curve)

	if plasticity is not None:
		st.subheader(f"{agent}: plasticity (search effort)")
		st.line_chart(plasticity)

	overlay = {}
	for name in agents:
		_latest, other_curve, _plasticity = _load_csv(os.path.join(replicate_dir, f"live_training_{name}.csv"))
		if other_curve is not None:
			overlay[name] = other_curve["score"]
	if len(overlay) > 1:
		st.subheader(f"{os.path.basename(study)} / {replicate}: all agents (score)")
		st.line_chart(pd.DataFrame(overlay).sort_index().interpolate(limit_area="inside"))

	evaluation_path = os.path.join(replicate_dir, f"live_evaluation_{agent}.csv")
	if os.path.exists(evaluation_path):
		st.subheader(f"{agent}: evaluation checks (written once the agent finishes training)")
		st.dataframe(pd.read_csv(evaluation_path), width="stretch")


_live_view()
