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
			average_score = dataframe[dataframe["env_id"] == env_id]["score"].mean()
			env_results[f"{env_id} Avg"] = round(average_score, 2)

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

		results = {
			"Agent": name,
			**{k: f"{v:.2f}" for k, v in env_results.items()},
			"Reliability (%)": f"{reliability:.1f}%",
			"Stability Gap (↓)": f"{stability_gap:.2f}",
		}
		if times:
			results["Efficiency (Score/s)"] = f"{efficiency:.2f}"

		metrics.append(results)

	metrics_df = pd.DataFrame(metrics)
	if file:
		print("\n--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---", file=file)
		print(metrics_df.to_string(index=False), file=file)
	return metrics_df


def plot_academic_comparison(
	histories,
	names,
	save_path=None,
	success_threshold=475.0,
	ylim=(-10, 550),
):
	"""
	Creates a publication-quality plot with stability ranges (Std Dev)
	and dynamic environment switch markers.
	"""
	figure, axes = plt.subplots(
		len(histories), 1, figsize=(15, 14), sharex=True, squeeze=False
	)
	axes = axes.flatten()
	window = 50

	for index, dataframe in enumerate(histories):
		# Calculate Moving Stats
		mean_score = dataframe["score"].rolling(window=window, min_periods=1).mean()
		std_score = dataframe["score"].rolling(window=window, min_periods=1).std()

		# Shaded area = Reliability / Variance
		axes[index].fill_between(
			dataframe["episode"],
			mean_score - std_score,
			mean_score + std_score,
			alpha=0.2,
			color="tab:blue",
			label="Stability Range",
		)

		# Bold line = Average Trend
		axes[index].plot(
			dataframe["episode"],
			mean_score,
			color="tab:blue",
			linewidth=3,
			label="Mean Performance",
		)

		# Mastery Threshold line
		axes[index].axhline(
			y=success_threshold,
			alpha=0.5,
			color="green",
			label=f"Mastery ({success_threshold})",
			linestyle=":",
		)

		# 1. Dynamic Environment Switches (Ground Truth)
		switches = dataframe[
			dataframe["env_id"].ne(dataframe["env_id"].shift())
		].index.tolist()
		for switch_index in switches:
			if switch_index > 0:
				axes[index].axvline(
					x=switch_index, alpha=0.5, color="red", linestyle="--", linewidth=2
				)
				if index == 0:
					y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.1
					axes[index].text(
						switch_index + 5,
						y_pos,
						"ENV SWITCH",
						color="red",
						alpha=0.6,
						fontweight="bold",
					)

		# 2. Agent's Internal Detection Points—small orange dots on the score line
		if "detected_env_id" in dataframe.columns:
			detections = dataframe[
				dataframe["detected_env_id"].ne(dataframe["detected_env_id"].shift())
			].index.tolist()
			detect_x = [ep for ep in detections if ep > 0]
			detect_y = [mean_score.iloc[ep] for ep in detect_x]
			if detect_x:
				axes[index].plot(
					detect_x,
					detect_y,
					"o",
					color="orange",
					markersize=8,
					zorder=5,
					label="Env Detected",
				)

		axes[index].set_ylabel("Score", fontsize=12)
		axes[index].set_ylim(ylim[0], ylim[1])
		axes[index].set_title(
			f"Agent: {names[index]}", fontsize=14, fontweight="bold", loc="left"
		)
		axes[index].grid(True, alpha=0.2)
		axes[index].legend(loc="upper left")

	plt.suptitle(
		"Neuroevolutionary Adaptation\nAcross Gravitational Shifts", fontsize=16
	)
	plt.xlabel("Total Episodes", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	if save_path:
		figure.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()


def plot_plasticity_analysis(agent, history_df, save_path=None):
	"""Generate the plasticity analysis plot"""
	figure, axis1 = plt.subplots(figsize=(12, 6))
	color_score, color_plast = "tab:blue", "tab:orange"

	# 1. Plot Score
	axis1.set_xlabel("Total Episodes")
	axis1.set_ylabel("Score", color=color_score)
	axis1.plot(
		history_df["episode"], history_df["score"], alpha=0.15, color=color_score
	)
	axis1.plot(
		history_df["episode"],
		history_df["score"].rolling(window=20).mean(),
		color=color_score,
		linewidth=2,
	)
	axis1.tick_params(axis="y", labelcolor=color_score)

	# 2. Align Plasticity Data
	# Since plasticity only records during training, we pad it for the test phase
	plasticity_data = list(agent.plasticity_history)
	if len(plasticity_data) < len(history_df):
		# Fill the "Zero-Shot" gap with 0.0 plasticity (since no search happened)
		plasticity_data.extend([0.0] * (len(history_df) - len(plasticity_data)))

	axis2 = axis1.twinx()
	axis2.set_ylabel("Plasticity (Search Effort)", color=color_plast)
	axis2.plot(
		history_df["episode"],
		plasticity_data[: len(history_df)],
		alpha=0.8,
		color=color_plast,
		linewidth=2,
	)
	axis2.fill_between(
		history_df["episode"],
		0,
		plasticity_data[: len(history_df)],
		alpha=0.1,
		color=color_plast,
	)
	axis2.set_ylim(0, 1.1)

	plt.title("ENA Resource Allocation: Performance vs. Search Effort")
	if save_path:
		figure.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()


def plot_specialist_transitions(df_history, pop_size=30, save_path=None):
	"""Generate the specialist transitions plot"""
	# 1. Increase height to (14, 8) to give more vertical room for IDs
	figure, axis1 = plt.subplots(figsize=(14, 8))

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
		axis1.axvspan(
			start,
			end,
			alpha=0.07,
			color=colors[index % len(colors)],
		)

		# 2. Position text dynamically at the very top of the limit
		axis1.text(
			(start + end) / 2,
			pop_size + 0.2,
			f"{env_id}",
			alpha=0.6,
			fontsize=12,
			fontweight="bold",
			ha="center",
		)

	# 3. (Detection lines removed — they were too noisy for the specialist chart)

	# 3. Distinguish Gladiators and Learners
	if "is_gladiator" not in df_history.columns:
		df_history["is_gladiator"] = False
		df_history["gladiator_env"] = "Unknown"

	df_history["agent_type"] = df_history.apply(
		lambda row: f"Gladiator: {row['gladiator_env']}"
		if row["is_gladiator"]
		else "Mutant/Learner",
		axis=1,
	)

	sns.scatterplot(
		alpha=0.8,
		ax=axis1,
		data=df_history,
		hue="agent_type",
		s=50,
		x="episode",
		y="specialist_id",
	)
	axis1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

	axis1.set_ylabel("Active Specialist ID", fontweight="bold")
	axis1.set_xlabel("Episode", fontweight="bold")

	# 4. FIXED: Set ticks to show every individual ID for pop_size <= 50
	# If pop_size is very large, use range(0, pop_size + 1, 5)
	tick_step = 1 if pop_size <= 40 else 5
	axis1.set_yticks(range(0, pop_size, tick_step))

	axis1.set_ylim(-0.5, pop_size + 1)  # Extra room at top for labels
	axis1.grid(True, axis="y", alpha=0.2, linestyle="--")  # Emphasize horizontal lines

	plt.title(
		"ENA Behavioral Switching: Selection vs Gravitational Regime",
		pad=20,
		fontsize=14,
	)
	plt.tight_layout()
	if save_path:
		figure.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()
