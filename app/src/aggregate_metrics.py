#!/usr/bin/env python
"""Aggregate metrics from multiple experiments"""

# General imports
import os
import re

# Libs imports
import pandas as pd


def parse_consolidated_table(lines):
	"""
	Parses a tabular performance summary into a DataFrame.

	Args:
		lines (list): List of strings representing the table rows.

	Returns:
		pd.DataFrame: A DataFrame containing the parsed metrics, or None if lines is empty.
	"""
	if not lines:
		return None

	# 1. Identify Headers
	header_line = lines[0]

	# Predefined non-environment headers (ordered as they appear at the end)
	suffix_headers = ["Reliability (%)", "Stability Gap (↓)", "Efficiency (Score/s)"]

	# Use regex to find all matches of things like "Something Avg"
	env_headers = re.findall(r'(\S+ Avg)', header_line)

	# Verify which suffix headers are actually present
	present_suffixes = [suffix for suffix in suffix_headers if suffix in header_line]

	headers = ["Agent"] + env_headers + present_suffixes
	num_expected_cols = len(headers)

	data = []
	for line in lines[1:]:
		line_text = line.strip()
		if not line_text:
			continue

		# Split by at least 2 spaces to handle names with single spaces
		parts = re.split(r'\s{2,}', line_text)

		# If split is not perfect (e.g. only single spaces used), use fallback positional logic
		if len(parts) != num_expected_cols:
			all_parts = line_text.split()

			num_suffixes = len(present_suffixes)
			num_envs = len(env_headers)

			if len(all_parts) >= num_expected_cols:
				# Suffixes are the last N
				suffix_data = all_parts[-num_suffixes:] if num_suffixes > 0 else []
				# Envs are the next M before suffixes
				env_data = all_parts[-(num_suffixes + num_envs):-num_suffixes] if num_suffixes > 0 else all_parts[-num_envs:]
				# Agent is everything before
				agent_name = " ".join(all_parts[:-(num_suffixes + num_envs)])
				data.append([agent_name] + env_data + suffix_data)
		else:
			data.append(parts)

	dataframe = pd.DataFrame(data, columns=headers)

	# Convert numeric columns
	for column in dataframe.columns:
		if column == "Agent":
			continue
		# Remove % if present and convert to float
		dataframe[column] = dataframe[column].astype(str).str.replace('%', '').astype(float)

	return dataframe


def get_experiment_directories(output_dir):
	"""
	Finds and sorts all experiment directories (exp_*).

	Args:
		output_dir (str): Path to the outputs directory.

	Returns:
		list: Sorted list of experiment directory names.
	"""
	if not os.path.exists(output_dir):
		return []

	directories = [
		directory for directory in os.listdir(output_dir)
		if directory.startswith('exp_') and os.path.isdir(os.path.join(output_dir, directory))
	]

	# Sort numerically: exp_1, exp_2, ..., exp_10, exp_11
	directories.sort(
		key=lambda item: int(re.search(r'\d+', item).group()) if re.search(r'\d+', item) else 0
	)
	return directories


def process_experiment_file(file_path):
	"""
	Reads an experiment output file and splits it into phase sections.

	Args:
		file_path (str): Path to the outputs.txt file.

	Returns:
		tuple: (train_lines, test_lines) or (None, None) if parsing fails.
	"""
	try:
		with open(file_path, 'r', encoding='utf-8') as file_handle:
			content = file_handle.read()
	except (OSError, IOError) as error:
		print(f"Error reading {file_path}: {error}")
		return None, None

	# Split by the summary marker
	marker = '--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---'
	sections = re.split(marker, content)

	if len(sections) < 3:
		return None, None

	# Section 1 is training table, Section 2 is test table
	train_table_lines = [line.strip() for line in sections[1].strip().split('\n') if line.strip()]
	test_table_lines = [line.strip() for line in sections[2].strip().split('\n') if line.strip()]

	return train_table_lines, test_table_lines


def write_summary_report(all_results, output_path, experiment_count):
	"""
	Generates the final aggregated report.

	Args:
		all_results (dict): Dictionary mapping phase names to lists of DataFrames.
		output_path (str): File path for the final report.
		experiment_count (int): Total number of experiments aggregated.
	"""
	with open(output_path, 'w', encoding='utf-8') as output_file:
		output_file.write(f"Averages and Standard Deviations Across {experiment_count} Experiments\n")
		output_file.write("=" * 60 + "\n\n")

		for phase in ['Training', 'Test']:
			output_file.write(f"--- {phase} Phase ---\n")
			if not all_results[phase]:
				output_file.write("No data found for this phase.\n\n")
				continue

			combined = pd.concat(all_results[phase], ignore_index=True)
			agents = combined['Agent'].unique()

			for agent in agents:
				output_file.write(f"Agent: {agent}\n")
				df_agent = combined[combined['Agent'] == agent]

				# Metrics to aggregate
				metrics = [col_name for col_name in df_agent.columns if col_name not in ['Agent', 'Exp']]
				for metric_name in metrics:
					values = pd.to_numeric(df_agent[metric_name], errors='coerce').dropna()
					if not values.empty:
						mean_val = values.mean()
						std_val = values.std()
						min_val = values.min()
						max_val = values.max()
						output_file.write(
							f"  {metric_name:.<20}: Mean = {mean_val:8.2f} | "
							f"Std = {std_val:8.2f} | Min = {min_val:8.2f} | Max = {max_val:8.2f}\n"
						)
				output_file.write("\n")


def aggregate():
	"""Main execution flow for aggregating metrics."""
	output_dir = os.path.join("app", "outputs") if os.path.exists("app") else "outputs"
	root_output_path = "app/outputs/outputs.txt" if os.path.exists("app") else "outputs/outputs.txt"

	exp_dirs = get_experiment_directories(output_dir)
	if not exp_dirs:
		print(f"No experiment outputs found in {output_dir}.")
		return

	all_results = {'Training': [], 'Test': []}
	found_any = False

	for exp_dir in exp_dirs:
		file_path = os.path.join(output_dir, exp_dir, "outputs.txt")
		if not os.path.exists(file_path):
			continue

		train_lines, test_lines = process_experiment_file(file_path)
		if train_lines is None:
			continue

		try:
			train_dataframe = parse_consolidated_table(train_lines)
			test_dataframe = parse_consolidated_table(test_lines)

			if train_dataframe is not None:
				train_dataframe['Exp'] = exp_dir
				all_results['Training'].append(train_dataframe)
			if test_dataframe is not None:
				test_dataframe['Exp'] = exp_dir
				all_results['Test'].append(test_dataframe)
			found_any = True
		except Exception as error:  # pylint: disable=broad-except
			print(f"Warning: Failed to parse {file_path}: {error}")

	if not found_any:
		print("No valid experiment data parsed.")
		return

	write_summary_report(all_results, root_output_path, len(exp_dirs))
	print(f"Successfully aggregated {len(exp_dirs)} experiments into {root_output_path}")


if __name__ == "__main__":
	aggregate()
