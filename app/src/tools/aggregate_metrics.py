#!/usr/bin/env python
"""Aggregate performance and compute-cost metrics across multiple experiment replicates."""

# General imports
import argparse
import os
import re
import sys

# Libs imports
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
from tools.detection_stats import write_detection_report
from tools.replicates import get_experiment_directories

PERFORMANCE_MARKER = "--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---"
COMPUTE_COST_MARKER = "--- COMPUTE-COST SUMMARY (candidate evaluation, training phase) ---"
ADAPTATION_MARKER = "--- ADAPTATION SUMMARY (training, early-stopped per environment) ---"
PHASES = ["Training", "Compute Cost", "Adaptation", "Test"]
# The summary metrics every performance table ends with (tools/metrics.calculate_paper_metrics).
SUMMARY_HEADERS = ["Reliability (%)", "Stability Gap (↓)", "Efficiency (Score/s)"]
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd, so the
# default target is found regardless of where this script is invoked from.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cartpole_ena_ablation_outputs")


# pylint: disable=too-many-locals
def parse_consolidated_table(lines: list) -> pd.DataFrame:
	"""Parse a performance-summary table (Agent + per-env Avg columns + summary metrics) into a DataFrame."""
	# Columns are separated by at least two spaces, so agent names may hold single spaces.
	headers = ["Agent"] + re.findall(r"(\S+ Avg)", lines[0]) + SUMMARY_HEADERS
	data = [re.split(r"\s{2,}", line.strip()) for line in lines[1:]]
	dataframe = pd.DataFrame(data, columns=headers)
	for column in dataframe.columns:
		if column == "Agent":
			continue
		dataframe[column] = dataframe[column].astype(str).str.replace("%", "").astype(float)

	return dataframe


def parse_compute_cost_table(lines: list) -> pd.DataFrame:
	"""Parse the COMPUTE-COST SUMMARY table (Agent, Avg Eval Depth, Avg Candidates/Gen, Total Candidate Env-Steps)."""
	headers = ["Agent", "Avg Eval Depth", "Avg Candidates/Gen", "Total Candidate Env-Steps"]
	num_metrics = len(headers) - 1

	data = [[" ".join(parts[:-num_metrics])] + parts[-num_metrics:] for parts in (line.split() for line in lines[1:])]

	dataframe = pd.DataFrame(data, columns=headers)
	# Only this table's values carry thousands separators (e.g. "7,745,982").
	dataframe["Total Candidate Env-Steps"] = dataframe["Total Candidate Env-Steps"].astype(str).str.replace(",", "")
	for column in dataframe.columns:
		if column == "Agent":
			continue
		dataframe[column] = dataframe[column].astype(float)

	return dataframe


def parse_adaptation_table(lines: list) -> pd.DataFrame:
	"""Parse the ADAPTATION SUMMARY table: Agent plus whitespace-free numeric columns (thousands separators allowed)."""
	headers = lines[0].split()
	num_metrics = len(headers) - 1
	data = [
		[" ".join(parts[:-num_metrics])] + [float(value.replace(",", "")) for value in parts[-num_metrics:]]
		for parts in (line.split() for line in lines[1:])
	]
	return pd.DataFrame(data, columns=headers)


def _table_after(block: str, marker: str) -> list:
	"""Return the table lines following marker in block, up to the first blank line."""
	lines = []
	for line in block.split(marker, 1)[1].strip("\n").split("\n"):
		if not line.strip():
			break
		lines.append(line.strip())
	return lines


def process_experiment_file(file_path: str) -> tuple:
	"""Split an outputs.txt into its train-performance, compute-cost, test-performance and adaptation tables."""
	with open(file_path, "r", encoding="utf-8") as file_handle:
		_header, train_block, test_block = file_handle.read().split(PERFORMANCE_MARKER)
	train_lines = _table_after(PERFORMANCE_MARKER + train_block, PERFORMANCE_MARKER)
	test_lines = [line.strip() for line in test_block.strip().split("\n") if line.strip()]
	return train_lines, _table_after(train_block, COMPUTE_COST_MARKER), test_lines, _table_after(train_block, ADAPTATION_MARKER)


# pylint: disable=too-many-locals
def write_summary_report(all_results: dict, output_path: str, experiment_count: int) -> None:
	"""Write the mean/std/min/max per agent per metric, across experiments, for every phase."""
	with open(output_path, "w", encoding="utf-8") as output_file:
		output_file.write(f"Averages and Standard Deviations Across {experiment_count} Experiments\n")
		output_file.write("=" * 60 + "\n\n")

		for phase in PHASES:
			output_file.write(f"--- {phase} Phase ---\n")
			combined = pd.concat(all_results[phase], ignore_index=True)
			agents = combined["Agent"].unique()

			for agent in agents:
				output_file.write(f"Agent: {agent}\n")
				df_agent = combined[combined["Agent"] == agent]

				metrics = [col_name for col_name in df_agent.columns if col_name not in ["Agent", "Exp"]]
				for metric_name in metrics:
					values = pd.to_numeric(df_agent[metric_name], errors="coerce").dropna()
					if not values.empty:
						mean_val = values.mean()
						std_val = values.std()
						min_val = values.min()
						max_val = values.max()
						output_file.write(
							f"  {metric_name:.<20}: Mean = {mean_val:8.2f} | " f"Std = {std_val:8.2f} | Min = {min_val:8.2f} | Max = {max_val:8.2f}\n"
						)
				output_file.write("\n")


def compute_significance(values_a: pd.Series, values_b: pd.Series) -> dict:
	"""Mean/diff/SEM/~95% significance between two independent samples, via the normal approximation."""
	# A quick screen for the text reports: SEM = sqrt(std_a^2/n_a + std_b^2/n_b), flagged significant
	# once |diff| exceeds 1.96*SEM (~95%). The paper's statistics (Welch t-test, confidence intervals,
	# Holm) come from paper_results.py, not from this.
	mean_a, mean_b = values_a.mean(), values_b.mean()
	n_a, n_b = len(values_a), len(values_b)
	sem = ((values_a.std() ** 2) / n_a + (values_b.std() ** 2) / n_b) ** 0.5 if n_a > 1 and n_b > 1 else float("nan")
	diff = mean_b - mean_a
	significant = bool(pd.notna(sem) and abs(diff) > 1.96 * sem)
	return {"mean_a": mean_a, "mean_b": mean_b, "diff": diff, "sem": sem, "significant": significant, "n_a": n_a, "n_b": n_b}


def write_comparison_report(all_results: dict, agent_a: str, agent_b: str, output_path: str) -> None:
	"""Write a metric-by-metric delta + ~95% significance comparison between two agents, every phase."""
	with open(output_path, "w", encoding="utf-8") as output_file:
		output_file.write(f"{agent_b} vs {agent_a} — per-metric comparison\n")
		output_file.write("=" * 60 + "\n\n")

		for phase in PHASES:
			combined = pd.concat(all_results[phase], ignore_index=True)
			if agent_a not in combined["Agent"].values or agent_b not in combined["Agent"].values:
				continue

			output_file.write(f"--- {phase} Phase ---\n")
			metrics = [column for column in combined.columns if column not in ["Agent", "Exp"]]
			for metric_name in metrics:
				values_a = pd.to_numeric(combined[combined["Agent"] == agent_a][metric_name], errors="coerce").dropna()
				values_b = pd.to_numeric(combined[combined["Agent"] == agent_b][metric_name], errors="coerce").dropna()
				if values_a.empty or values_b.empty:
					continue
				result = compute_significance(values_a, values_b)
				flag = "significant" if result["significant"] else "not significant"
				output_file.write(
					f"  {metric_name:.<28}: {agent_a}={result['mean_a']:.2f} {agent_b}={result['mean_b']:.2f} "
					f"diff={result['diff']:+.2f} ({flag} @ ~95%, n={result['n_a']}/{result['n_b']})\n"
				)
			output_file.write("\n")

	print(f"Wrote comparison report to {output_path}")


def aggregate(output_dir: str, compare: list = None) -> None:
	"""Aggregate every replicate's metrics in a study directory into its aggregate_summary.txt (and detection_summary.txt)."""
	all_results = collect_results(output_dir)
	report_path = os.path.join(output_dir, "aggregate_summary.txt")
	write_summary_report(all_results, report_path, len(get_experiment_directories(output_dir)))
	print(f"Aggregated {output_dir} into {report_path}")
	print(f"Environment recognition summarized into {write_detection_report(output_dir)}")
	if compare:
		agent_a, agent_b = compare
		write_comparison_report(all_results, agent_a, agent_b, os.path.join(output_dir, f"compare_{agent_a}_vs_{agent_b}.txt"))


def collect_results(output_dir: str) -> dict:
	"""Parse every replicate of a study directory: {phase: [per-replicate DataFrames, each with its Exp label]}."""
	all_results = {phase: [] for phase in PHASES}
	for exp_dir in get_experiment_directories(output_dir):
		tables = process_experiment_file(os.path.join(output_dir, exp_dir, "outputs.txt"))
		parsers = (parse_consolidated_table, parse_compute_cost_table, parse_consolidated_table, parse_adaptation_table)
		for phase, parser, lines in zip(("Training", "Compute Cost", "Test", "Adaptation"), parsers, tables):
			dataframe = parser(lines)
			dataframe["Exp"] = exp_dir
			all_results[phase].append(dataframe)
	return all_results


def main() -> None:
	"""Parse CLI arguments and run the aggregation."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="The study directory holding the exp_* replicates.")
	parser.add_argument(
		"--compare",
		nargs=2,
		metavar=("AGENT_A", "AGENT_B"),
		help="Also write a metric-by-metric delta + significance comparison between these two agents.",
	)
	args = parser.parse_args()
	aggregate(args.output_dir, compare=args.compare)


if __name__ == "__main__":
	main()
