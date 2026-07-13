#!/usr/bin/env python
"""Plot variance."""

# Libs imports
import os
import re

# Libs imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_txt_file(filepath):
	"""Parses the new tabular format using robust logic from aggregate_metrics."""
	if not os.path.exists(filepath):
		return None

	with open(filepath, 'r') as f:
		content = f.read()

	# We focus on the Test section as it's the primary indicator for the paper
	sections = re.split(r'--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---', content)
	if len(sections) < 3:
		return None

	# The third section (index 2) is the Test summary
	lines = [l for l in sections[2].strip().split('\n') if l.strip()]
	if not lines:
		return None

	header_line = lines[0]
	present_suffixes = [s for s in ["Reliability (%)", "Stability Gap (↓)", "Efficiency (Score/s)"] if s in header_line]
	env_headers = re.findall(r'(\S+ Avg)', header_line)

	headers = ["Agent"] + env_headers + present_suffixes
	num_expected_cols = len(headers)

	data = []
	for line in lines[1:]:
		parts = re.split(r'\s{2,}', line.strip())
		if len(parts) != num_expected_cols:
			all_parts = line.split()
			num_suffixes = len(present_suffixes)
			num_envs = len(env_headers)
			if len(all_parts) >= num_expected_cols:
				s_data = all_parts[-num_suffixes:] if num_suffixes > 0 else []
				e_data = all_parts[-(num_suffixes + num_envs):-num_suffixes] if num_suffixes > 0 else all_parts[-num_envs:]
				agent_name = " ".join(all_parts[:-(num_suffixes + num_envs)])
				data.append([agent_name] + e_data + s_data)
		else:
				data.append(parts)

	df = pd.DataFrame(data, columns=headers)

	# Normalize agent names for plotting
	_name_map = {
		'ENA-01 Fuzzy-Fuzzy': 'ENA-01',
		'ENA-02 Quadratic-Quadratic': 'ENA-02',
	}
	df['Agent'] = df['Agent'].apply(lambda x: _name_map.get(x, x))

	# Clean numeric columns
	for col in df.columns:
		if col != "Agent":
			df[col] = df[col].astype(str).str.replace('%', '').astype(float)

	return df

def main():
	# Primary directory search in 'app/outputs' (or 'outputs' if running from app/)
	output_dir = os.path.join("app", "outputs") if os.path.exists("app") else "outputs"
	if not os.path.exists(output_dir):
		# Check current dir if it IS the outputs dir
		if os.path.basename(os.getcwd()) == "outputs":
			output_dir = "."
		else:
			print(f"Error: {output_dir} not found.")
			return

	# Dynamic discovery of all experiment folders (exp_1, exp_2, etc.)
	exp_dirs = [d for d in os.listdir(output_dir) if d.startswith('exp_') and os.path.isdir(os.path.join(output_dir, d))]
	exp_dirs.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

	records = []
	for exp_dir in exp_dirs:
		filepath = os.path.join(output_dir, exp_dir, "outputs.txt")
		df = parse_txt_file(filepath)
		if df is not None:
			for _, row in df.iterrows():
				records.append({
					'Experiment': exp_dir,
					'Agent': row['Agent'],
					'Reliability (%)': row.get('Reliability (%)', 0),
					'Stability Gap': row.get('Stability Gap (↓)', 0)
				})
		else:
			print(f"Warning: Could not parse {filepath}")

	if not records:
		print("No valid data found to plot.")
		return

	df_final = pd.DataFrame(records)

	# 1. Boxplot for Reliability (Variance across all experiments)
	plt.figure(figsize=(10, 7))
	sns.set_style("whitegrid")
	sns.boxplot(data=df_final, x='Agent', y='Reliability (%)', palette='Set2')
	sns.stripplot(data=df_final, x='Agent', y='Reliability (%)', color='black', alpha=0.4, jitter=True)
	plt.title(f'Test Phase Reliability Distribution Across {len(exp_dirs)} Experiments', fontsize=14, fontweight='bold')
	plt.ylabel('Test Phase Reliability (%)', fontweight='bold')
	plt.xlabel('Agent Architecture', fontweight='bold')
	plt.ylim(-5, 105)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, 'reliability_box.png'), dpi=300)
	plt.close()

	# 2. Barplot for Stability Gap (Mean with 95% Bootstrap Confidence Intervals)
	plt.figure(figsize=(10, 7))
	sns.barplot(data=df_final, x='Agent', y='Stability Gap', capsize=.1, errorbar=('ci', 95), palette='Set1')
	sns.stripplot(data=df_final, x='Agent', y='Stability Gap', color='black', alpha=0.4, jitter=True)
	plt.title(f'Test Phase Stability Gap (Mean with 95% CI) - {len(exp_dirs)} Experiments', fontsize=14, fontweight='bold')
	plt.ylabel('Score Delta (Pre - Post Switch)', fontweight='bold')
	plt.xlabel('Agent Architecture', fontweight='bold')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, 'stability_bar.png'), dpi=300)
	plt.close()

	print(f"Success: Processed {len(exp_dirs)} experiments.")
	print("Saved 'reliability_box.png' and 'stability_bar.png'.")

if __name__ == '__main__':
	main()
