"""Metrics module for ENA Experiments"""

# Libs imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# pylint: disable=too-many-locals
def calculate_paper_metrics(
	histories,
	names,
	file=None,
	success_threshold=475.0,
	times=None,
):
	"""Calculate metrics"""
	# Calculates academic metrics including Reliability (Success Rate),
	# Stability Gaps at environment switches, and dynamic environment detection.

	metrics = []
	for dataframe, name in zip(histories, names):
		# Dynamically find where environments change
		switches = dataframe[
			dataframe["env_id"].ne(dataframe["env_id"].shift())
		].index.tolist()

		# 1. Reliability Metric: % of episodes above mastery threshold
		total_episodes = len(dataframe)
		successes = len(dataframe[dataframe["score"] >= success_threshold])
		reliability = (successes / total_episodes) * 100

		# 2. Performance per Environment (Dynamic for any number of worlds)
		env_results = {}
		for env_id in dataframe["env_id"].unique():
			avg = dataframe[dataframe["env_id"] == env_id]["score"].mean()
			env_results[f"{env_id} Avg"] = round(avg, 2)

		# 3. Stability Gap: Measures drop in performance exactly at the first switch
		stability_gap = 0
		if len(switches) > 1:
			first_switch = switches[1]
			# Compare average of 20 episodes before vs 20 episodes after switch
			pre = dataframe.iloc[max(0, first_switch - 20) : first_switch][
				"score"
			].mean()
			post = dataframe.iloc[first_switch : first_switch + 20]["score"].mean()
			stability_gap = pre - post

		# 4. Efficiency: Total Score divided by Wall-Clock Time
		efficiency = 0
		if times and name in times and times[name] > 0:
			total_score = dataframe["score"].sum()
			efficiency = total_score / times[name]

		res = {
			"Agent": name,
			**{k: f"{v:.2f}" for k, v in env_results.items()},
			"Reliability (%)": f"{reliability:.1f}%",
			"Stability Gap (↓)": f"{stability_gap:.2f}",
		}
		if times:
			res["Efficiency (Score/s)"] = f"{efficiency:.2f}"

		metrics.append(res)

	metrics_df = pd.DataFrame(metrics)
	if file:
		print("\n--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---", file=file)
		print(metrics_df.to_string(index=False), file=file)
	return metrics_df


def plot_academic_comparison(histories, names, save_path=None):
	"""
	Creates a publication-quality plot with stability ranges (Std Dev)
	and dynamic environment switch markers.
	"""
	fig, axs = plt.subplots(len(histories), 1, figsize=(15, 14), sharex=True, squeeze=False)
	axs = axs.flatten()
	window = 50

	for index, dataframe in enumerate(histories):
		# Calculate Moving Stats
		mean_score = dataframe["score"].rolling(window=window, min_periods=1).mean()
		std_score = dataframe["score"].rolling(window=window, min_periods=1).std()

		# Shaded area = Reliability / Variance
		axs[index].fill_between(
			dataframe["episode"],
			mean_score - std_score,
			mean_score + std_score,
			alpha=0.2,
			color="tab:blue",
			label="Stability Range",
		)

		# Bold line = Average Trend
		axs[index].plot(
			dataframe["episode"],
			mean_score,
			color="tab:blue",
			linewidth=3,
			label="Mean Performance",
		)

		# Mastery Threshold line (475)
		axs[index].axhline(
			y=475, color="green", linestyle=":", alpha=0.5, label="Mastery (475)"
		)

		# Dynamic Environment Switches
		switches = dataframe[
			dataframe["env_id"].ne(dataframe["env_id"].shift())
		].index.tolist()
		for s_idx in switches:
			if s_idx > 0:
				axs[index].axvline(
					x=s_idx, color="red", linestyle="--", linewidth=2, alpha=0.7
				)
				if index == 0:  # Only label the top plot to keep it clean
					axs[index].text(
						s_idx + 5, 450, "ENV SWITCH", color="red", fontweight="bold"
					)

		axs[index].set_ylabel("Score", fontsize=12)
		axs[index].set_ylim(-10, 550)
		axs[index].set_title(
			f"Agent: {names[index]}", loc="left", fontsize=14, fontweight="bold"
		)
		axs[index].grid(True, alpha=0.2)
		axs[index].legend(loc="upper left")

	plt.suptitle(
		"Neuroevolutionary Adaptation\nAcross Gravitational Shifts", fontsize=16
	)
	plt.xlabel("Total Episodes", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	if save_path:
		fig.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()


def plot_plasticity_analysis(agent, history_df, save_path=None):
	"""Generate the plasticity analysis plot"""
	fig, ax1 = plt.subplots(figsize=(12, 6))
	color_score, color_plast = "tab:blue", "tab:orange"

	# 1. Plot Score
	ax1.set_xlabel("Total Episodes")
	ax1.set_ylabel("Score", color=color_score)
	ax1.plot(history_df["episode"], history_df["score"], alpha=0.15, color=color_score)
	ax1.plot(
		history_df["episode"],
		history_df["score"].rolling(window=20).mean(),
		color=color_score,
		linewidth=2,
	)
	ax1.tick_params(axis="y", labelcolor=color_score)

	# 2. Align Plasticity Data
	# Since plasticity only records during training, we pad it for the test phase
	plast_data = list(agent.plasticity_history)
	if len(plast_data) < len(history_df):
		# Fill the "Zero-Shot" gap with 0.0 plasticity (since no search happened)
		plast_data.extend([0.0] * (len(history_df) - len(plast_data)))

	ax2 = ax1.twinx()
	ax2.set_ylabel("Plasticity (Search Effort)", color=color_plast)
	ax2.plot(
		history_df["episode"],
		plast_data[: len(history_df)],
		alpha=0.8,
		color=color_plast,
		linewidth=2,
	)
	ax2.fill_between(
		history_df["episode"],
		0,
		plast_data[: len(history_df)],
		color=color_plast,
		alpha=0.1,
	)
	ax2.set_ylim(0, 1.1)

	plt.title("ENA Resource Allocation: Performance vs. Search Effort")
	if save_path:
		fig.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()


def plot_specialist_transitions(df_history, pop_size=30, save_path=None):
	"""Generate the specialist transitions plot"""
	# 1. Increase height to (14, 8) to give more vertical room for IDs
	fig, ax1 = plt.subplots(figsize=(14, 8))

	# Color background by environment
	unique_envs = df_history["env_id"].unique()
	colors = ["gray", "orange", "blue", "red"]  # Earth, Mars, Neptune, Jupiter

	for index, env_id in enumerate(unique_envs):
		mask = df_history["env_id"] == env_id
		if not mask.any():
			continue

		start, end = (
			df_history[mask]["episode"].min(),
			df_history[mask]["episode"].max(),
		)
		ax1.axvspan(
			start,
			end,
			alpha=0.07,
			color=colors[index % len(colors)],
		)

		# 2. Position text dynamically at the very top of the limit
		ax1.text(
			(start + end) / 2,
			pop_size + 0.2,
			f"{env_id}",
			alpha=0.6,
			fontweight="bold",
			fontsize=12,
			ha="center",
		)

	# 3. Adjust scatter size 's' slightly down to prevent overlap
	sns.scatterplot(
		alpha=0.7,
		ax=ax1,
		data=df_history,
		hue="specialist_id",
		legend=False,
		palette="tab20",
		s=20,
		x="episode",
		y="specialist_id",
	)

	ax1.set_ylabel("Active Specialist ID", fontweight="bold")
	ax1.set_xlabel("Episode", fontweight="bold")

	# 4. FIXED: Set ticks to show every individual ID for pop_size <= 50
	# If pop_size is very large, use range(0, pop_size + 1, 5)
	tick_step = 1 if pop_size <= 40 else 5
	ax1.set_yticks(range(0, pop_size, tick_step))

	ax1.set_ylim(-0.5, pop_size + 1)  # Extra room at top for labels
	ax1.grid(True, axis="y", alpha=0.2, linestyle="--")  # Emphasize horizontal lines

	plt.title(
		"ENA Behavioral Switching: Selection vs Gravitational Regime",
		pad=20,
		fontsize=14,
	)
	plt.tight_layout()
	if save_path:
		fig.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()
