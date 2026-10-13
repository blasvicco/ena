#!/usr/bin/env python
"""Redraw finished replicates' training.png and test.png from their live CSVs, without re-running anything.

Use it after a chart change: the charts are drawn once at the end of each phase, but every
number they show is in exp_*/live_{training,testing}_{agent}.csv.

	python src/tools/replot_replicate.py --output-dir cartpole_ena_ablation_outputs
"""

# General imports
import argparse
import glob
import os
import re
import sys

# Libs imports
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # pylint: disable=wrong-import-position

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from tools.metrics import plot_academic_comparison, uses_env_detection

# CartPole's defaults, as in cartpole_ena_ablation.py; other domains pass their own.
SUCCESS_THRESHOLD = 475.0
Y_LIM = (-10, 550)
TIME_LINE = re.compile(r"^\s+(.+?): [\d.,]+ seconds")


def agent_order(outputs_text: str) -> list:
	"""Agents in the order the replicate trained them, from outputs.txt's training-time block."""
	block = outputs_text.split("--- Training Performance vs Time ---", 1)[1].split("\n\n", 1)[0]
	return [match.group(1) for match in map(TIME_LINE.match, block.splitlines()) if match]


def known_niches(training: pd.DataFrame):
	"""Niches established in training (what the test chart treats as already known), or None without env detection."""
	if not uses_env_detection(training):
		return None
	return {int(niche_id) for niche_id in training["detected_env_id"].unique() if niche_id != -1}


def replot(exp_dir: str, success_threshold: float = SUCCESS_THRESHOLD, ylim: tuple = Y_LIM, training_x_axis: str = "episodes") -> None:
	"""Rewrite one replicate's training.png and test.png; training_x_axis "env_steps" for studies with an equal step budget."""
	with open(os.path.join(exp_dir, "outputs.txt"), encoding="utf-8") as outputs:
		agents = agent_order(outputs.read())
	training = [pd.read_csv(os.path.join(exp_dir, f"live_training_{agent}.csv")) for agent in agents]
	testing = [pd.read_csv(os.path.join(exp_dir, f"live_testing_{agent}.csv")) for agent in agents]
	common = {"success_threshold": success_threshold, "ylim": ylim}
	plot_academic_comparison(training, agents, save_path=os.path.join(exp_dir, "training.png"), x_axis=training_x_axis, **common)
	plot_academic_comparison(
		testing,
		agents,
		save_path=os.path.join(exp_dir, "test.png"),
		known_niche_ids_by_history=[known_niches(history) for history in training],
		**common,
	)


def main() -> None:
	"""Replot every finished replicate of a study."""
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--output-dir", required=True, help="Study directory holding the exp_* replicates.")
	parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD)
	parser.add_argument("--ylim", type=float, nargs=2, default=Y_LIM)
	parser.add_argument(
		"--training-x-axis",
		choices=["episodes", "env_steps"],
		default="episodes",
		help="env_steps for studies at an equal step budget (the CartPole comparison), so every agent spans the same range.",
	)
	args = parser.parse_args()
	for exp_dir in sorted(glob.glob(os.path.join(args.output_dir, "exp_*"))):
		if not glob.glob(os.path.join(exp_dir, "live_testing_*.csv")):
			print(f"Skipping {exp_dir}: no test data yet")
			continue
		replot(exp_dir, args.success_threshold, tuple(args.ylim), args.training_x_axis)
		print(f"Replotted {exp_dir}")


if __name__ == "__main__":
	main()
