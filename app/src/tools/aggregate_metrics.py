#!/usr/bin/env python
"""Aggregate performance and compute-cost metrics across multiple experiment replicates."""

# General imports
import argparse
import os
import re

# Libs imports
import pandas as pd

PERFORMANCE_MARKER = "--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---"
COMPUTE_COST_MARKER = "--- COMPUTE-COST SUMMARY (candidate evaluation, training phase) ---"
ADAPTATION_MARKER = "--- ADAPTATION SUMMARY (training, early-stopped per environment) ---"
PHASES = ["Training", "Compute Cost", "Adaptation", "Test"]
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd, so the
# default target is found regardless of where this script is invoked from.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cartpole_ena_performance_outputs")


# pylint: disable=too-many-locals
def parse_consolidated_table(lines: list) -> pd.DataFrame:
	"""Parse a performance-summary table (Agent + per-env Avg columns + suffix metrics) into a DataFrame."""
	if not lines:
		return None

	header_line = lines[0]
	# Predefined non-environment headers (ordered as they appear at the end).
	suffix_headers = ["Reliability (%)", "Stability Gap (↓)", "Efficiency (Score/s)"]
	env_headers = re.findall(r"(\S+ Avg)", header_line)
	present_suffixes = [suffix for suffix in suffix_headers if suffix in header_line]

	headers = ["Agent"] + env_headers + present_suffixes
	num_expected_cols = len(headers)

	data = []
	for line in lines[1:]:
		line_text = line.strip()
		if not line_text:
			continue

		# Split by at least 2 spaces to handle names with single spaces.
		parts = re.split(r"\s{2,}", line_text)

		if len(parts) != num_expected_cols:
			# Fixed-width alignment can collapse to single spaces between narrow columns —
			# fall back to positional parsing: suffixes are the last N tokens, envs the next
			# M, and the agent name is whatever whitespace-split tokens remain at the start.
			all_parts = line_text.split()
			num_suffixes = len(present_suffixes)
			num_envs = len(env_headers)

			if len(all_parts) >= num_expected_cols:
				suffix_data = all_parts[-num_suffixes:] if num_suffixes > 0 else []
				env_data = all_parts[-(num_suffixes + num_envs) : -num_suffixes] if num_suffixes > 0 else all_parts[-num_envs:]
				agent_name = " ".join(all_parts[: -(num_suffixes + num_envs)])
				data.append([agent_name] + env_data + suffix_data)
		else:
			data.append(parts)

	dataframe = pd.DataFrame(data, columns=headers)
	for column in dataframe.columns:
		if column == "Agent":
			continue
		dataframe[column] = dataframe[column].astype(str).str.replace("%", "").astype(float)

	return dataframe


def parse_compute_cost_table(lines: list) -> pd.DataFrame:
	"""Parse the COMPUTE-COST SUMMARY table (Agent, Avg Eval Depth, Avg Candidates/Gen, Total Candidate Env-Steps)."""
	if not lines:
		return None

	headers = ["Agent", "Avg Eval Depth", "Avg Candidates/Gen", "Total Candidate Env-Steps"]
	num_metrics = len(headers) - 1

	data = []
	for line in lines[1:]:
		line_text = line.strip()
		if not line_text:
			continue
		all_parts = line_text.split()
		if len(all_parts) <= num_metrics:
			continue
		metric_values = all_parts[-num_metrics:]
		agent_name = " ".join(all_parts[:-num_metrics])
		data.append([agent_name] + metric_values)

	dataframe = pd.DataFrame(data, columns=headers)
	# Only this table's values carry thousands separators (e.g. "7,745,982").
	dataframe["Total Candidate Env-Steps"] = dataframe["Total Candidate Env-Steps"].astype(str).str.replace(",", "")
	for column in dataframe.columns:
		if column == "Agent":
			continue
		dataframe[column] = dataframe[column].astype(float)

	return dataframe


def get_experiment_directories(output_dir: str) -> list:
	"""Find and numerically sort all experiment directories (exp_*) under output_dir."""
	if not os.path.exists(output_dir):
		return []

	directories = [
		directory for directory in os.listdir(output_dir) if directory.startswith("exp_") and os.path.isdir(os.path.join(output_dir, directory))
	]
	directories.sort(key=lambda item: int(re.search(r"\d+", item).group()) if re.search(r"\d+", item) else 0)
	return directories


def parse_adaptation_table(lines: list) -> pd.DataFrame:
	"""Parse the ADAPTATION SUMMARY table: Agent plus whitespace-free numeric columns (thousands separators allowed)."""
	if not lines:
		return None
	headers = lines[0].split()
	num_metrics = len(headers) - 1
	data = []
	for line in lines[1:]:
		parts = line.split()
		if len(parts) <= num_metrics:
			continue
		data.append([" ".join(parts[:-num_metrics])] + [float(value.replace(",", "")) for value in parts[-num_metrics:]])
	return pd.DataFrame(data, columns=headers)


def _table_after(block: str, marker: str) -> list:
	"""The table lines following marker in block, up to the first blank line (empty if absent)."""
	if marker not in block:
		return []
	lines = []
	for line in block.split(marker, 1)[1].strip("\n").split("\n"):
		if not line.strip():
			break
		lines.append(line.strip())
	return lines


def process_experiment_file(file_path: str) -> tuple:
	"""Split an outputs.txt into its train-performance, compute-cost, and test-performance line-groups."""
	try:
		with open(file_path, "r", encoding="utf-8") as file_handle:
			content = file_handle.read()
	except OSError as error:
		print(f"Error reading {file_path}: {error}")
		return None, None, None, None

	sections = content.split(PERFORMANCE_MARKER)
	if len(sections) < 3:
		return None, None, None, None

	train_block, test_block = sections[1], sections[2]

	# The train table itself always ends at the next blank line, regardless of what follows
	# (the compute-cost table for an experiment with ENA agents, or straight to the next
	# phase's own trailer for one without) — stop there rather than taking the whole block,
	# or a compute-cost-free experiment would swallow its own "Test Results" trailer as a
	# bogus extra performance row.
	train_lines = []
	for line in train_block.strip("\n").split("\n"):
		if not line.strip():
			break
		train_lines.append(line.strip())

	# The (ENA-only) compute-cost table, if present, follows the train table before the next
	# phase's own content — extracted independently of train_lines above, same "stop at the
	# first blank line" rule.
	compute_cost_lines = []
	if COMPUTE_COST_MARKER in train_block:
		_, cost_block = train_block.split(COMPUTE_COST_MARKER, 1)
		for line in cost_block.strip("\n").split("\n"):
			if not line.strip():
				break
			compute_cost_lines.append(line.strip())

	test_lines = [line.strip() for line in test_block.strip().split("\n") if line.strip()]
	# The (early-stopped studies only) adaptation table is printed during training, after the compute-cost table.
	adaptation_lines = _table_after(train_block, ADAPTATION_MARKER)

	return train_lines, compute_cost_lines, test_lines, adaptation_lines


# pylint: disable=too-many-locals
def write_summary_report(all_results: dict, output_path: str, experiment_count: int) -> None:
	"""Write the mean/std/min/max per agent per metric, across experiments, for every phase."""
	with open(output_path, "w", encoding="utf-8") as output_file:
		output_file.write(f"Averages and Standard Deviations Across {experiment_count} Experiments\n")
		output_file.write("=" * 60 + "\n\n")

		for phase in PHASES:
			output_file.write(f"--- {phase} Phase ---\n")
			if not all_results.get(phase):
				output_file.write("No data found for this phase.\n\n")
				continue

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
	# Same convention already used conversationally this session and in benchmark_ena.py's own
	# _describe_paired_significance: SEM = sqrt(std_a^2/n_a + std_b^2/n_b), flagged significant
	# once |diff| exceeds 1.96*SEM (~95%) — close enough to a t-test once n is in the 10s-100s.
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
			if not all_results.get(phase):
				continue
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


def parse_exclusions(exclude: list) -> dict:
	"""Turn ["DIR:AGENT", ...] into {normalized DIR: {AGENT, ...}}."""
	exclusions = {}
	for entry in exclude or []:
		directory, separator, agent_name = entry.rpartition(":")
		if not separator or not directory or not agent_name:
			raise ValueError(f"--exclude entries must look like DIR:AGENT, got {entry!r}")
		exclusions.setdefault(os.path.normpath(directory), set()).add(agent_name)
	return exclusions


def _drop_agents(dataframe, agent_names: set):
	"""Remove the given agents' rows from a parsed table (None passes through)."""
	if dataframe is None or not agent_names:
		return dataframe
	return dataframe[~dataframe["Agent"].isin(agent_names)]


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def aggregate(output_dirs: list, compare: list = None, exclude: list = None) -> None:
	"""Aggregate every exp_* replicate's metrics found under one or more output_dirs into one report."""
	# Pooling more than one study's directory (e.g. an older ablation study plus a newer one
	# sharing a row's exact config, so it doesn't need to be re-run) is the whole point of
	# accepting a list here — results still land in the first directory given. exclude
	# ("DIR:AGENT" entries) drops an agent from one directory only — e.g. an older study's row
	# that a newer study re-ran with a changed config.
	exclusions = parse_exclusions(exclude)
	primary_dir = output_dirs[0]
	report_path = os.path.join(primary_dir, "aggregate_summary.txt")
	multi_source = len(output_dirs) > 1

	all_results = {phase: [] for phase in PHASES}
	parsed_count = 0
	total_dirs_found = 0

	for output_dir in output_dirs:
		exp_dirs = get_experiment_directories(output_dir)
		if not exp_dirs:
			print(f"No experiment outputs found in {output_dir}.")
			continue
		total_dirs_found += len(exp_dirs)
		# Both studies independently number exp_1, exp_2, ... — qualify with the source
		# directory once pooling more than one, so the Exp column stays unambiguous.
		exp_label_prefix = f"{os.path.basename(os.path.normpath(output_dir))}/" if multi_source else ""
		excluded_agents = exclusions.get(os.path.normpath(output_dir), set())

		for exp_dir in exp_dirs:
			exp_label = f"{exp_label_prefix}{exp_dir}"
			file_path = os.path.join(output_dir, exp_dir, "outputs.txt")
			if not os.path.exists(file_path):
				print(f"Warning: {exp_label} has no outputs.txt yet — skipped.")
				continue

			train_lines, compute_cost_lines, test_lines, adaptation_lines = process_experiment_file(file_path)
			if train_lines is None:
				print(f"Warning: {exp_label} doesn't have a complete train+test section yet (still running?) — skipped.")
				continue

			# One malformed/truncated (e.g. still-in-progress) experiment file shouldn't abort
			# aggregating the rest — this regex-based parser can fail in many small ways on
			# unexpected input, so it's caught broadly here and the file is skipped with a warning.
			try:
				train_dataframe = _drop_agents(parse_consolidated_table(train_lines), excluded_agents)
				compute_cost_dataframe = _drop_agents(parse_compute_cost_table(compute_cost_lines), excluded_agents)
				test_dataframe = _drop_agents(parse_consolidated_table(test_lines), excluded_agents)
				adaptation_dataframe = _drop_agents(parse_adaptation_table(adaptation_lines), excluded_agents)
				if adaptation_dataframe is not None:
					adaptation_dataframe["Exp"] = exp_label
					all_results["Adaptation"].append(adaptation_dataframe)

				if train_dataframe is not None:
					train_dataframe["Exp"] = exp_label
					all_results["Training"].append(train_dataframe)
				if compute_cost_dataframe is not None:
					compute_cost_dataframe["Exp"] = exp_label
					all_results["Compute Cost"].append(compute_cost_dataframe)
				if test_dataframe is not None:
					test_dataframe["Exp"] = exp_label
					all_results["Test"].append(test_dataframe)
				parsed_count += 1
			except Exception as error:  # pylint: disable=broad-except
				print(f"Warning: Failed to parse {file_path}: {error}")

	if not parsed_count:
		print("No valid experiment data parsed.")
		return

	write_summary_report(all_results, report_path, parsed_count)
	print(f"Successfully aggregated {parsed_count}/{total_dirs_found} experiments into {report_path}")

	if compare:
		agent_a, agent_b = compare
		compare_path = os.path.join(primary_dir, f"compare_{agent_a}_vs_{agent_b}.txt")
		write_comparison_report(all_results, agent_a, agent_b, compare_path)


def main() -> None:
	"""Parse CLI arguments and run the aggregation."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output-dir",
		nargs="+",
		default=[DEFAULT_OUTPUT_DIR],
		help="One or more directories containing exp_* replicate subdirectories; results are pooled across all given.",
	)
	parser.add_argument(
		"--compare",
		nargs=2,
		metavar=("AGENT_A", "AGENT_B"),
		help="Also write a metric-by-metric delta + significance comparison between these two agents.",
	)
	parser.add_argument(
		"--exclude",
		nargs="+",
		default=[],
		metavar="DIR:AGENT",
		help="Drop AGENT's rows from DIR only (e.g. an older study's row that a newer study re-ran).",
	)
	args = parser.parse_args()
	aggregate(args.output_dir, compare=args.compare, exclude=args.exclude)


if __name__ == "__main__":
	main()
