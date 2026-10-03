#!/usr/bin/env python
"""Plot Reliability/Stability Gap variance across experiment replicates as box/strip charts."""

# General imports
import argparse
import os
import re

# Libs imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PERFORMANCE_MARKER = "--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---"
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd, so the
# default target is found regardless of where this script is invoked from.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cartpole_ena_performance_outputs")
# Older runs used long descriptive agent names — shortened here for axis labels; any other
# name (including the current ablation study's) passes through unchanged.
LEGACY_NAME_MAP = {
	"ENA-01 LinearRamp-LinearRamp": "ENA-01",
	"ENA-02 Quadratic-Quadratic": "ENA-02",
}


# pylint: disable=too-many-locals
def parse_txt_file(filepath: str) -> pd.DataFrame:
	"""Parse an outputs.txt's Test-phase performance table into a DataFrame."""
	if not os.path.exists(filepath):
		return None

	with open(filepath, "r", encoding="utf-8") as file_handle:
		content = file_handle.read()

	# The Test summary is the third occurrence of the marker (index 2): preamble, Training, Test.
	sections = content.split(PERFORMANCE_MARKER)
	if len(sections) < 3:
		return None

	lines = [line for line in sections[2].strip().split("\n") if line.strip()]
	if not lines:
		return None

	header_line = lines[0]
	present_suffixes = [suffix for suffix in ["Reliability (%)", "Stability Gap (↓)", "Efficiency (Score/s)"] if suffix in header_line]
	env_headers = re.findall(r"(\S+ Avg)", header_line)

	headers = ["Agent"] + env_headers + present_suffixes
	num_expected_cols = len(headers)

	data = []
	for line in lines[1:]:
		parts = re.split(r"\s{2,}", line.strip())
		if len(parts) != num_expected_cols:
			all_parts = line.split()
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
	dataframe["Agent"] = dataframe["Agent"].apply(lambda name: LEGACY_NAME_MAP.get(name, name))

	for column in dataframe.columns:
		if column != "Agent":
			dataframe[column] = dataframe[column].astype(str).str.replace("%", "").astype(float)

	return dataframe


def get_experiment_directories(output_dir: str) -> list:
	"""Find and numerically sort all experiment directories (exp_*) under output_dir."""
	directories = [
		directory for directory in os.listdir(output_dir) if directory.startswith("exp_") and os.path.isdir(os.path.join(output_dir, directory))
	]
	directories.sort(key=lambda item: int(re.search(r"\d+", item).group()) if re.search(r"\d+", item) else 0)
	return directories


def plot(output_dir: str) -> None:
	"""Aggregate every exp_* replicate's Test-phase performance and plot variance box/bar charts."""
	if not os.path.exists(output_dir):
		print(f"Error: {output_dir} not found.")
		return

	exp_dirs = get_experiment_directories(output_dir)

	records = []
	for exp_dir in exp_dirs:
		file_path = os.path.join(output_dir, exp_dir, "outputs.txt")
		dataframe = parse_txt_file(file_path)
		if dataframe is None:
			print(f"Warning: Could not parse {file_path}")
			continue
		for _, row in dataframe.iterrows():
			records.append(
				{
					"Experiment": exp_dir,
					"Agent": row["Agent"],
					"Reliability (%)": row.get("Reliability (%)", 0),
					"Stability Gap": row.get("Stability Gap (↓)", 0),
				}
			)

	if not records:
		print("No valid data found to plot.")
		return

	final_dataframe = pd.DataFrame(records)

	# 1. Boxplot for Reliability (Variance across all experiments)
	plt.figure(figsize=(10, 7))
	sns.set_style("whitegrid")
	sns.boxplot(data=final_dataframe, x="Agent", y="Reliability (%)", palette="Set2")
	sns.stripplot(data=final_dataframe, x="Agent", y="Reliability (%)", color="black", alpha=0.4, jitter=True)
	plt.title(f"Test Phase Reliability Distribution Across {len(exp_dirs)} Experiments", fontsize=14, fontweight="bold")
	plt.ylabel("Test Phase Reliability (%)", fontweight="bold")
	plt.xlabel("Agent Architecture", fontweight="bold")
	plt.ylim(-5, 105)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "reliability_box.png"), dpi=300)
	plt.close()

	# 2. Barplot for Stability Gap (Mean with 95% Bootstrap Confidence Intervals)
	plt.figure(figsize=(10, 7))
	sns.barplot(data=final_dataframe, x="Agent", y="Stability Gap", capsize=0.1, errorbar=("ci", 95), palette="Set1")
	sns.stripplot(data=final_dataframe, x="Agent", y="Stability Gap", color="black", alpha=0.4, jitter=True)
	plt.title(f"Test Phase Stability Gap (Mean with 95% CI) - {len(exp_dirs)} Experiments", fontsize=14, fontweight="bold")
	plt.ylabel("Score Delta (Pre - Post Switch)", fontweight="bold")
	plt.xlabel("Agent Architecture", fontweight="bold")
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "stability_bar.png"), dpi=300)
	plt.close()

	print(f"Success: Processed {len(exp_dirs)} experiments.")
	print(f"Saved 'reliability_box.png' and 'stability_bar.png' to {output_dir}.")


def main() -> None:
	"""Parse CLI arguments and run the variance plotting."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory containing exp_* replicate subdirectories.")
	args = parser.parse_args()
	plot(args.output_dir)


if __name__ == "__main__":
	main()
