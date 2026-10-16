#!/usr/bin/env python
"""Plot score curves averaged over a study's replicates — one panel per agent, phases marked."""
# Reads every exp_*/live_{phase}_{agent}.csv a study wrote (one row per episode) and draws, per
# agent, the across-replicate mean score with the band holding the middle 50% of replicates
# (25th-75th percentile), lightly smoothed within each phase (never across a switch) — the
# cross-replicate counterpart of a single experiment's test.png/training.png.
#
# By episode (x_axis "episodes") when every replicate plays the same episodes (every test; training by
# episode counts); by environment steps (x_axis "env_steps", episodes binned by the step at which they end)
# when replicates play different numbers of episodes for the same budget (training at a step budget).
#
#         python src/tools/plot_study_curves.py --phase testing
#         python src/tools/plot_study_curves.py --phase training --agents Random ENA-NoSkill ENA-Combined

# General imports
import argparse
import glob
import os
import re

# Libs imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cartpole_ena_ablation_outputs")
MASTERY_SCORE = 475.0
Y_LIM = (-10, 560)
SMOOTHING_WINDOW = 10
BIN_STEPS = 10_000  # env-steps per bin on the step axis
SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GUIDE_COLOR = "#8a8984"
GRID_COLOR = "#e6e5e1"


def load_replicates(output_dir: str, phase: str, agent: str) -> pd.DataFrame:
	"""Load the episodes x replicates matrix of scores for one agent and phase, plus the per-episode env name."""
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


def load_binned(output_dir: str, phase: str, agent: str, bin_steps: int = BIN_STEPS) -> pd.DataFrame:
	"""Load each replicate's mean score per bin of bin_steps env-steps, with the environment of each bin."""
	# Load the bins x replicates matrix: each replicate's mean score of the episodes ending in each bin of bin_steps env-steps
	# (indexed by the bin's end), plus the environment per bin (attrs["env_ids"], from the first replicate).
	paths = sorted(glob.glob(os.path.join(output_dir, "exp_*", f"live_{phase}_{agent}.csv")))
	scores, env_ids = {}, None
	for path in paths:
		frame = pd.read_csv(path)
		bins = (np.ceil(frame["env_steps"] / bin_steps) * bin_steps).astype(int)
		scores[os.path.basename(os.path.dirname(path))] = frame.groupby(bins)["score"].mean()
		if env_ids is None:
			env_ids = frame.groupby(bins)["env_id"].last()
	if not scores:
		raise FileNotFoundError(f"No live_{phase}_{agent}.csv under {output_dir}/exp_*")
	matrix = pd.DataFrame(scores).sort_index()
	matrix.attrs["env_ids"] = env_ids.reindex(matrix.index).ffill().bfill()
	return matrix


def summarize(matrix: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
	"""Return the mean and interquartile range across replicates per row, smoothed within each phase."""
	# Mean and 25th/75th percentiles across replicates per row, each smoothed over `window` rows within each phase
	# (attrs["env_ids"]; a switch restarts the smoothing).
	summary = pd.DataFrame({"mean": matrix.mean(axis=1), "low": matrix.quantile(0.25, axis=1), "high": matrix.quantile(0.75, axis=1)})
	env_ids = matrix.attrs["env_ids"].reindex(summary.index)
	phases = env_ids.ne(env_ids.shift()).cumsum()
	return summary.groupby(phases, group_keys=False).apply(lambda part: part.rolling(window=window, min_periods=1, center=True).mean())


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


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def plot(
	output_dir: str,
	phase: str,
	agents: list,
	save_path: str,
	mastery_score: float = MASTERY_SCORE,
	ylim: tuple = Y_LIM,
	x_axis: str = "episodes",
	bin_steps: int = BIN_STEPS,
) -> str:
	"""Draw one panel per agent and save the figure; return the path written (CartPole's bar and range by default)."""
	# x_axis "env_steps" bins episodes by environment steps (load_binned).
	if x_axis not in ("episodes", "env_steps"):
		raise ValueError(f"Unknown x_axis {x_axis!r}; expected 'episodes' or 'env_steps'")
	by_steps = x_axis == "env_steps"
	# Environment names sit near the top of the range (515 on CartPole's default range).
	label_y = ylim[1] - (ylim[1] - ylim[0]) * 45 / 570
	figure, axes = plt.subplots(len(agents), 1, figsize=(12, 2.1 * len(agents) + 1.0), sharex=True, sharey=True, squeeze=False)
	figure.patch.set_facecolor("white")
	for axis, agent in zip(axes[:, 0], agents):
		matrix = load_binned(output_dir, phase, agent, bin_steps) if by_steps else load_replicates(output_dir, phase, agent)
		summary = summarize(matrix)
		# A switch line sits between the last row of a phase and the first of the next; names are offset a little.
		half_step, text_offset = (bin_steps / 2, 0.005 * (matrix.index.max() - matrix.index.min())) if by_steps else (0.5, 4)
		axis.fill_between(summary.index, summary["low"], summary["high"], color=SERIES_COLOR, alpha=0.18, linewidth=0, label="Middle 50% of runs")
		axis.plot(summary.index, summary["mean"], color=SERIES_COLOR, linewidth=2, label="Mean over runs")
		axis.axhline(mastery_score, color=GUIDE_COLOR, linestyle=":", linewidth=1, label=f"Threshold ({mastery_score:.0f})")
		for index, (start, env_name) in enumerate(phase_boundaries(matrix.attrs["env_ids"])):
			if index:
				axis.axvline(start - half_step, color=GUIDE_COLOR, linestyle="--", linewidth=1)
			axis.text(start + text_offset, label_y, env_name, color=TEXT_SECONDARY, fontsize=8, va="bottom")
		axis.set_ylim(*ylim)
		_style_axis(axis, agent, matrix.shape[1])
	handles, labels = axes[0, 0].get_legend_handles_labels()
	figure.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=3, labelcolor=TEXT_SECONDARY)
	axes[-1, 0].set_xlabel("Environment steps" if by_steps else "Episode", color=TEXT_SECONDARY, fontsize=9)
	phase_title = "Test" if phase == "testing" else "Training"
	unit = "per environment step bin" if by_steps else "per episode"
	figure.suptitle(f"{phase_title} score {unit} across environment shifts", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
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
	parser.add_argument("--mastery-score", type=float, default=MASTERY_SCORE, help="Dotted 'solved' line (HalfCheetah: its SOLVED_SCORE).")
	parser.add_argument("--ylim", type=float, nargs=2, default=Y_LIM, metavar=("LOW", "HIGH"), help="Score axis range.")
	parser.add_argument("--x-axis", choices=["episodes", "env_steps"], default="episodes", help="env_steps for training at a step budget.")
	args = parser.parse_args()
	agents = args.agents or discover_agents(args.output_dir, args.phase)
	save_path = args.save or os.path.join(args.output_dir, f"{args.phase}_curves.png")
	print(
		f"Wrote {plot(args.output_dir, args.phase, agents, save_path, mastery_score=args.mastery_score, ylim=tuple(args.ylim), x_axis=args.x_axis)}"
	)


if __name__ == "__main__":
	main()
