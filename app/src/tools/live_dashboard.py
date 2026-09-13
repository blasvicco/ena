#!/usr/bin/env python
"""Live convergence dashboard: auto-refreshing browser view of a live_training_*.csv
written by cartpole_challenge.py/half_cheetah.py's play() while a run is still in progress.

Run this as a separate process from the training run itself — it only reads the CSV,
never touches the training process. Since app/ is bind-mounted into the container, run it
directly on the host (pip install streamlit there) to get a real browser window without
any docker port mapping:

	streamlit run app/src/tools/live_dashboard.py

Or inside the container, exposing Streamlit's default port (8501) in docker-compose.yml.
"""

# General imports
import glob
import time

# Libs imports
import pandas as pd
import streamlit as st

REFRESH_SECONDS = 3
ROLLING_WINDOW = 20


def _discover_default_csv():
	"""Best-effort: the most recently modified live_training_*.csv under any *_outputs dir."""
	candidates = glob.glob("*_outputs/*/live_training_*.csv") + glob.glob(
		"app/*_outputs/*/live_training_*.csv"
	)
	if not candidates:
		return ""
	return max(candidates, key=lambda path: path)


def _read_log(path):
	"""Read the live-training CSV, tolerating a torn last line mid-write."""
	try:
		return pd.read_csv(path)
	except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
		return None


st.set_page_config(page_title="ENA Live Convergence", layout="wide")
st.title("ENA Live Convergence")

csv_path = st.sidebar.text_input("live_training_*.csv path", value=_discover_default_csv())
st.sidebar.caption(f"Auto-refreshes every {REFRESH_SECONDS}s while this tab stays open.")

placeholder = st.empty()

while True:
	with placeholder.container():
		if not csv_path:
			st.info("No live_training_*.csv found yet — enter a path in the sidebar once training starts.")
		else:
			dataframe = _read_log(csv_path)
			if dataframe is None or dataframe.empty:
				st.info(f"Waiting for data in {csv_path}...")
			else:
				latest = dataframe.iloc[-1]
				columns = st.columns(4)
				columns[0].metric("Episode", int(latest["episode"]))
				columns[1].metric("Last score", f"{latest['score']:.1f}")
				if "plasticity" in dataframe.columns and pd.notna(latest.get("plasticity")):
					columns[2].metric("Last plasticity", f"{latest['plasticity']:.2f}")
				if "env_id" in dataframe.columns:
					columns[3].metric("Current env", str(latest["env_id"]))

				st.subheader("Score (raw + rolling mean)")
				chart_data = pd.DataFrame(
					{
						"score": dataframe["score"],
						"score (rolling mean)": dataframe["score"].rolling(
							window=ROLLING_WINDOW, min_periods=1
						).mean(),
					},
					index=dataframe["episode"],
				)
				st.line_chart(chart_data)

				if "plasticity" in dataframe.columns and dataframe["plasticity"].notna().any():
					st.subheader("Plasticity (search effort)")
					st.line_chart(dataframe.set_index("episode")[["plasticity"]])

	time.sleep(REFRESH_SECONDS)
