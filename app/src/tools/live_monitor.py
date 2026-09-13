#!/usr/bin/env python
"""Live convergence monitor: tails a live_training_*.csv written by cartpole_challenge.py/half_cheetah.py's
play() and re-renders a score/plasticity chart while a run is still in progress.

Run this as a separate process from the training run itself — it only reads the CSV, never
touches the training process. Since app/ is bind-mounted into the container, it works either
inside `docker compose exec` or directly on the host (where a real display is available).

Usage:
	python tools/live_monitor.py cartpole_outputs/exp_1/live_training_ENA-01\\ Fuzzy-Fuzzy.csv
	python tools/live_monitor.py <path/to/live_training.csv> --interval 5 --window 50
"""

# General imports
import argparse
import os

# Libs imports
import matplotlib.pyplot as plt
import pandas as pd


def _read_log(csv_path):
	"""Read the live-training CSV, tolerating a torn last line mid-write."""
	try:
		return pd.read_csv(csv_path)
	except (pd.errors.EmptyDataError, pd.errors.ParserError):
		return None


def _render(axis_score, axis_plasticity, dataframe, window):
	"""Redraw the score/plasticity chart in place from the latest data."""
	axis_score.clear()
	axis_plasticity.clear()

	mean_score = dataframe["score"].rolling(window=window, min_periods=1).mean()
	axis_score.plot(dataframe["episode"], dataframe["score"], alpha=0.2, color="tab:blue")
	axis_score.plot(dataframe["episode"], mean_score, color="tab:blue", linewidth=2)
	axis_score.set_ylabel("Score", color="tab:blue")
	axis_score.set_xlabel("Episode")

	if "plasticity" in dataframe.columns and dataframe["plasticity"].notna().any():
		axis_plasticity.plot(
			dataframe["episode"],
			dataframe["plasticity"],
			color="tab:orange",
			alpha=0.6,
			linewidth=1,
		)
		axis_plasticity.set_ylabel("Plasticity", color="tab:orange")
		axis_plasticity.set_ylim(0, 1.1)

	latest_episode = int(dataframe["episode"].iloc[-1])
	axis_score.set_title(f"Live Convergence — {latest_episode} episodes so far")


def main():
	"""Poll a live_training_*.csv and keep an updating chart on screen until interrupted."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("csv_path", help="Path to the live_training_*.csv written by play().")
	parser.add_argument("--interval", type=float, default=3.0, help="Seconds between refreshes.")
	parser.add_argument("--window", type=int, default=20, help="Rolling-mean window for score.")
	args = parser.parse_args()

	plt.ion()
	figure, axis_score = plt.subplots(figsize=(12, 6))
	axis_plasticity = axis_score.twinx()

	print(f"Watching {args.csv_path} (Ctrl+C to stop)...")
	try:
		while True:
			if os.path.exists(args.csv_path):
				dataframe = _read_log(args.csv_path)
				if dataframe is not None and not dataframe.empty:
					_render(axis_score, axis_plasticity, dataframe, args.window)
					figure.tight_layout()
			plt.pause(args.interval)
	except KeyboardInterrupt:
		print("\nStopped.")


if __name__ == "__main__":
	main()
