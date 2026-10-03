#!/usr/bin/env python
"""Plot per-episode score curves averaged over a study's replicates — one panel per agent, phases marked.

Reads every exp_*/live_{phase}_{agent}.csv a study wrote (one row per episode) and draws, per
agent, the across-replicate mean score per episode (lightly smoothed) with the band holding the
middle 50% of replicates (25th-75th percentile) — the cross-replicate counterpart of a single
experiment's test.png/training.png.

	python src/tools/plot_study_curves.py --phase testing
	python src/tools/plot_study_curves.py --phase training --agents Random ENA-NoSkill ENA-Combined
"""

# General imports
import argparse
import glob
import os
import re

# Libs imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cartpole_ena_performance_outputs")
MASTERY_SCORE = 475.0
SMOOTHING_WINDOW = 10
SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GUIDE_COLOR = "#8a8984"
GRID_COLOR = "#e6e5e1"


def load_replicates(output_dir: str, phase: str, agent: str) -> pd.DataFrame:
	"""Episodes x replicates matrix of scores for one agent and phase, plus the per-episode env name."""
	paths = sorted(glob.glob(os.path.join(output_dir, "exp_*", f"live_{phase}_{agent}.csv")))
	scores, env_ids = {}, None
	for path in paths:
		frame = pd.read_csv(path)
		replicate = os.path.basename(os.path.dirname(path))
		scores[replicate] = frame.set_index("episode")["score"]
		if env_ids is None:
			env_ids = frame.set_index("episode")["env_id"]
	if not scores:
		raise FileNotFoundError(f"No live_{phase}_{agent}.csv under {output_dir}/exp_*")
	matrix = pd.DataFrame(scores).sort_index()
	matrix.attrs["env_ids"] = env_ids
	return matrix


def summarize(matrix: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
	"""Per-episode mean and 25th/75th percentiles across replicates, each smoothed over `window` episodes."""
	summary = pd.DataFrame({"mean": matrix.mean(axis=1), "low": matrix.quantile(0.25, axis=1), "high": matrix.quantile(0.75, axis=1)})
	return summary.rolling(window=window, min_periods=1, center=True).mean()


def phase_boundaries(env_ids: pd.Series) -> list:
	"""[(first_episode, env_name)] for each contiguous phase, in order."""
	starts = env_ids[env_ids != env_ids.shift()]
	return list(starts.items())


def _style_axis(axis, title: str, replicate_count: int) -> None:
	"""Recessive grid and spines, a left-aligned bold title naming the agent and its replicate count."""
	axis.set_title(f"{title}  (n = {replicate_count})", loc="left", fontsize=11, fontweight="bold", color=TEXT_PRIMARY)
	axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
	axis.set_axisbelow(True)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		axis.spines[side].set_color(GUIDE_COLOR)
	axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
	axis.set_ylabel("Score", color=TEXT_SECONDARY, fontsize=9)


# pylint: disable=too-many-locals
def plot(output_dir: str, phase: str, agents: list, save_path: str) -> str:
	"""Draw one panel per agent and save the figure; return the path written."""
	figure, axes = plt.subplots(len(agents), 1, figsize=(12, 2.1 * len(agents) + 1.0), sharex=True, sharey=True, squeeze=False)
	figure.patch.set_facecolor("white")
	for axis, agent in zip(axes[:, 0], agents):
		matrix = load_replicates(output_dir, phase, agent)
		summary = summarize(matrix)
		axis.fill_between(summary.index, summary["low"], summary["high"], color=SERIES_COLOR, alpha=0.18, linewidth=0, label="Middle 50% of runs")
		axis.plot(summary.index, summary["mean"], color=SERIES_COLOR, linewidth=2, label="Mean over runs")
		axis.axhline(MASTERY_SCORE, color=GUIDE_COLOR, linestyle=":", linewidth=1, label=f"Mastery ({MASTERY_SCORE:.0f})")
		for index, (episode, env_name) in enumerate(phase_boundaries(matrix.attrs["env_ids"])):
			if index:
				axis.axvline(episode - 0.5, color=GUIDE_COLOR, linestyle="--", linewidth=1)
			axis.text(episode + 4, 515, env_name, color=TEXT_SECONDARY, fontsize=8, va="bottom")
		axis.set_ylim(-10, 560)
		_style_axis(axis, agent, matrix.shape[1])
	handles, labels = axes[0, 0].get_legend_handles_labels()
	figure.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=3, labelcolor=TEXT_SECONDARY)
	axes[-1, 0].set_xlabel("Episode", color=TEXT_SECONDARY, fontsize=9)
	phase_title = "Test" if phase == "testing" else "Training"
	figure.suptitle(f"{phase_title} score per episode across environment shifts", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
	figure.tight_layout(rect=(0, 0, 1, 0.98))
	figure.savefig(save_path, dpi=300)
	plt.close(figure)
	return save_path


def discover_agents(output_dir: str, phase: str) -> list:
	"""Agent names with a live_{phase}_*.csv in the study's first replicate, Random first."""
	paths = glob.glob(os.path.join(output_dir, "exp_1", f"live_{phase}_*.csv"))
	names = sorted(re.sub(rf"^live_{phase}_|\.csv$", "", os.path.basename(path)) for path in paths)
	return sorted(names, key=lambda name: name != "Random")


def main() -> None:
	"""Parse CLI arguments and draw the figure."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Study directory containing exp_* replicates.")
	parser.add_argument("--phase", choices=["testing", "training"], default="testing")
	parser.add_argument("--agents", nargs="+", default=None, help="Agents to plot, top to bottom (default: all found, Random first).")
	parser.add_argument("--save", default=None, help="Output image path (default: <output-dir>/<phase>_curves.png).")
	args = parser.parse_args()
	agents = args.agents or discover_agents(args.output_dir, args.phase)
	save_path = args.save or os.path.join(args.output_dir, f"{args.phase}_curves.png")
	print(f"Wrote {plot(args.output_dir, args.phase, agents, save_path)}")


if __name__ == "__main__":
	main()
