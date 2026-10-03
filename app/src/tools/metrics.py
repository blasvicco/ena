"""Metrics module for ENA Experiments"""

# Libs imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def _qualitative_colors(count: int) -> list:
	"""Return count visually distinct colors, scaling past any fixed palette size."""
	cmap = plt.get_cmap("tab20" if count > 10 else "tab10")
	return [cmap(index / max(count, 1)) for index in range(count)]


def _classify_niche_events(niche_id_series: pd.Series, known_niche_ids: set = None) -> list:
	"""Classify each niche-id change as 'new' (never seen) or 'switch' (to a known id).

	A niche id of -1 or NaN means "no niche resolved this episode" and is skipped.
	Only value *changes* are emitted — repeated confirmations of the same niche
	produce no event.

	known_niche_ids seeds the "already seen" set — e.g. for a test-phase chart, every
	niche established during training is already known, so it must classify as a
	"switch" the first time it appears in the test series, not a spurious "new niche"
	(training.py's train() never runs during test, so no niche can genuinely be
	created there; without this seed, this function's own per-call "seen" tracking
	would otherwise mislabel the first occurrence, since it has no memory of anything
	that happened before the series it was given).
	"""
	seen = set(known_niche_ids) if known_niche_ids else set()
	events = []
	prev = None
	for episode, niche_id in niche_id_series.items():
		if pd.isna(niche_id) or niche_id == -1:
			continue
		if prev is None:
			seen.add(niche_id)
		elif niche_id != prev:
			events.append((episode, niche_id, "new" if niche_id not in seen else "switch"))
			seen.add(niche_id)
		prev = niche_id
	return events


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
		switches = dataframe[dataframe["env_id"].ne(dataframe["env_id"].shift())].index.tolist()

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
			pre = dataframe.iloc[max(0, first_switch - 20) : first_switch]["score"].mean()
			post = dataframe.iloc[first_switch : first_switch + 20]["score"].mean()
			stability_gap = pre - post

		# 4. Efficiency: Total Score divided by Wall-Clock Time
		efficiency = 0
		if times and name in times and times[name] > 0:
			total_score = dataframe["score"].sum()
			efficiency = total_score / times[name]

		results = {
			"Agent": name,
			**{key: f"{value:.2f}" for key, value in env_results.items()},
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


def estimate_eval_cost(plasticity: float, pop_size: int, min_eval_steps: int, max_eval_steps: int) -> tuple:
	"""Mirror ena.py train()'s num_to_eval/eval_step_cap formula, for display only."""
	# Deliberately duplicated, not imported, to keep this a display-only concern with zero
	# coupling into agents.ena — if ena.py's own formula ever changes, this one should too.
	num_to_eval = max(2, int(pop_size * plasticity))
	eval_step_cap = min_eval_steps + (max_eval_steps - min_eval_steps) * plasticity
	return num_to_eval, eval_step_cap


def calculate_compute_cost(histories: list, names: list, file=None) -> pd.DataFrame:
	"""Summarize each agent's logged per-generation candidate-evaluation compute."""
	rows = []
	for dataframe, name in zip(histories, names):
		if "candidate_env_steps" not in dataframe:
			continue
		depth = dataframe["eval_step_cap"].dropna()
		breadth = dataframe["num_to_eval"].dropna()
		cost = dataframe["candidate_env_steps"].dropna()
		if cost.empty:
			continue
		rows.append(
			{
				"Agent": name,
				"Avg Eval Depth": f"{depth.mean():.1f}",
				"Avg Candidates/Gen": f"{breadth.mean():.1f}",
				"Total Candidate Env-Steps": f"{cost.sum():,.0f}",
			}
		)

	cost_df = pd.DataFrame(rows)
	if file and not cost_df.empty:
		print("\n--- COMPUTE-COST SUMMARY (candidate evaluation, training phase) ---", file=file)
		print(cost_df.to_string(index=False), file=file)
	return cost_df


ADAPTATION_MARKER = "--- ADAPTATION SUMMARY (training, early-stopped per environment) ---"


def calculate_adaptation(histories: list, names: list, file=None) -> pd.DataFrame:
	"""Per agent: env-steps used and solved flag per environment, plus training/evaluation step totals.

	Reads play()'s attrs["phases"]. "<env>:steps" is the steps to the first solve when "<env>:solved"
	is 1, otherwise the steps consumed until the plateau, cap or budget ended it. Train-Env-Steps is
	every training step, including those after the first solve (fixed-budget protocol).
	"""
	rows = []
	for dataframe, name in zip(histories, names):
		phases = dataframe.attrs.get("phases") or []
		if not phases:
			continue
		row = {"Agent": name}
		for phase in phases:
			solved_at = phase.get("solve_steps")
			row[f"{phase['env_id']}:steps"] = f"{solved_at if solved_at is not None else phase['steps']:,}"
			row[f"{phase['env_id']}:solved"] = int(phase["status"] == "solved")
		row["Solved-Phases"] = sum(int(phase["status"] == "solved") for phase in phases)
		row["Train-Env-Steps"] = f"{sum(phase['steps'] for phase in phases):,}"
		row["Eval-Env-Steps"] = f"{sum(phase['eval_steps'] for phase in phases):,}"
		rows.append(row)

	adaptation_df = pd.DataFrame(rows)
	if file and not adaptation_df.empty:
		print(f"\n{ADAPTATION_MARKER}", file=file)
		print(adaptation_df.to_string(index=False), file=file)
	return adaptation_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def plot_academic_comparison(
	histories,
	names,
	save_path=None,
	success_threshold=475.0,
	ylim=(-10, 550),
	known_niche_ids_by_history=None,
):
	"""
	Creates a publication-quality plot with stability ranges (Std Dev)
	and dynamic environment switch markers.

	known_niche_ids_by_history is an optional list, parallel to histories, of niche-id
	sets already established before this chart's own data begins (e.g. from training,
	for a test-phase chart) — see _classify_niche_events for why this matters.
	"""
	figure, axes = plt.subplots(
		len(histories),
		1,
		figsize=(15, 14),
		sharex=True,
		squeeze=False,
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
		switches = dataframe[dataframe["env_id"].ne(dataframe["env_id"].shift())].index.tolist()
		for switch_index in switches:
			if switch_index > 0:
				axes[index].axvline(
					x=switch_index,
					alpha=0.5,
					color="red",
					linestyle="--",
					linewidth=2,
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

		# 2. Real-time shift detection (detect_change() firing) — independent of which
		# niche the agent resolves to, and meaningful in both train and test charts.
		if "env_shift_detected" in dataframe.columns:
			shift_positions = [pos for pos, fired in enumerate(dataframe["env_shift_detected"]) if bool(fired)]
			if shift_positions:
				y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.03
				axes[index].plot(
					shift_positions,
					[y_pos] * len(shift_positions),
					"^",
					color="purple",
					markersize=6,
					zorder=4,
					label="Shift Detected",
				)

		# 3. Niche identity: "new" (never-seen niche) vs. "switch" (reclassified to a
		# different, already-known niche). Fixed colors per kind so the legend swatch
		# always matches every plotted marker — per-niche detail lives in the dedicated
		# niche_transitions_*.png / niche_transition_matrix_*.png charts instead.
		if "detected_env_id" in dataframe.columns:
			known_ids = known_niche_ids_by_history[index] if known_niche_ids_by_history else None
			events = _classify_niche_events(dataframe["detected_env_id"], known_niche_ids=known_ids)
			color_by_kind = {"new": "darkorange", "switch": "black"}

			new_labeled, switch_labeled = False, False
			for pos, _niche_id, kind in events:
				label = None
				if kind == "new" and not new_labeled:
					label, new_labeled = "New Niche", True
				elif kind == "switch" and not switch_labeled:
					label, switch_labeled = "Niche Switch", True
				axes[index].plot(
					pos,
					mean_score.iloc[pos],
					"o" if kind == "new" else "x",
					color=color_by_kind[kind],
					markersize=8,
					zorder=5,
					label=label,
				)

		axes[index].set_ylabel("Score", fontsize=12)
		axes[index].set_ylim(ylim[0], ylim[1])
		axes[index].set_title(
			f"Agent: {names[index]}",
			fontsize=14,
			fontweight="bold",
			loc="left",
		)
		axes[index].grid(True, alpha=0.2)
		axes[index].legend(loc="upper left")

	plt.suptitle("Neuroevolutionary Adaptation\nAcross Gravitational Shifts", fontsize=16)
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
		history_df["episode"],
		history_df["score"],
		alpha=0.15,
		color=color_score,
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

	# Color background by environment — a colormap sized to the actual number of
	# environments in this run, so it never wraps regardless of how many appear.
	unique_envs = df_history["env_id"].unique()
	colors = _qualitative_colors(len(unique_envs))

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
			color=colors[index],
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
		lambda row: f"Gladiator: {row['gladiator_env']}" if row["is_gladiator"] else "Mutant/Learner",
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

	# 3b. Outline gladiator points by their actual recognized niche id (not just their
	# ground-truth env name) — makes niche fragmentation (many niche ids per one true
	# environment) visible directly on this chart. Ring colors are per-niche and change
	# every run, so rather than one legend entry per color, a single generic entry below
	# explains what the ring itself means.
	has_niche_rings = False
	if "internal_env_id" in df_history.columns:
		gladiator_rows = df_history[df_history["is_gladiator"] & (df_history["internal_env_id"] != -1)]
		niche_ids = sorted(gladiator_rows["internal_env_id"].unique())
		if niche_ids:
			has_niche_rings = True
			color_by_niche = dict(zip(niche_ids, _qualitative_colors(len(niche_ids))))
			axis1.scatter(
				gladiator_rows["episode"],
				gladiator_rows["specialist_id"],
				s=110,
				facecolors="none",
				edgecolors=[color_by_niche[nid] for nid in gladiator_rows["internal_env_id"]],
				linewidths=1.5,
				zorder=3,
			)

	handles, labels = axis1.get_legend_handles_labels()
	if has_niche_rings:
		handles.append(
			Line2D(
				[],
				[],
				marker="o",
				linestyle="None",
				markersize=10,
				markerfacecolor="none",
				markeredgecolor="black",
				markeredgewidth=1.5,
				label="Ring: Recognized niche id",
			)
		)
		labels.append("Ring: Recognized niche id")
	axis1.legend(handles=handles, labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

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


def _build_transition_matrix(niche_id_series: pd.Series) -> dict:
	"""Count niche-to-niche transitions: {(from_id, to_id): count} over consecutive changes.

	Both "new niche" and "switch to known niche" changes count as transitions here —
	unlike `_classify_niche_events`, this is about the shape of the transition graph,
	not about labeling a chart marker.
	"""
	counts = {}
	prev = None
	for _, niche_id in niche_id_series.items():
		if pd.isna(niche_id) or niche_id == -1:
			continue
		if prev is not None and niche_id != prev:
			key = (prev, niche_id)
			counts[key] = counts.get(key, 0) + 1
		prev = niche_id
	return counts


def _niche_env_labels(df_history: pd.DataFrame) -> dict:
	"""Best-effort niche id -> ground-truth env name, from whichever rows resolved it."""
	if "detected_env_id" not in df_history.columns or "gladiator_env" not in df_history.columns:
		return {}
	labeled = df_history[(df_history["detected_env_id"] != -1) & df_history["gladiator_env"].notna()]
	if labeled.empty:
		return {}
	return labeled.groupby("detected_env_id")["gladiator_env"].agg(lambda envs: envs.value_counts().idxmax()).to_dict()


def plot_niche_transition_sequence(df_history, save_path=None):
	"""Plot the actual chain of niche reclassifications over the whole run.

	Makes convergence (settles onto one niche and stays) vs. persistent thrashing
	between fragmented niches visually obvious, background-shaded by the ground-truth
	environment so within-env vs. cross-env transitions are easy to tell apart.
	"""
	figure, axis1 = plt.subplots(figsize=(14, 6))

	unique_envs = df_history["env_id"].unique()
	env_colors = _qualitative_colors(len(unique_envs))
	for index, env_id in enumerate(unique_envs):
		mask = df_history["env_id"] == env_id
		if not mask.any():
			continue
		start, end = df_history[mask]["episode"].min(), df_history[mask]["episode"].max()
		axis1.axvspan(start, end, alpha=0.07, color=env_colors[index])
		axis1.text(
			(start + end) / 2,
			axis1.get_ylim()[1],
			f"{env_id}",
			alpha=0.6,
			fontsize=11,
			fontweight="bold",
			ha="center",
		)

	resolved = df_history[df_history["detected_env_id"].notna() & (df_history["detected_env_id"] != -1)]
	axis1.plot(
		resolved["episode"],
		resolved["detected_env_id"],
		"-o",
		color="tab:blue",
		markersize=4,
		linewidth=1,
		alpha=0.8,
	)

	axis1.set_xlabel("Episode", fontweight="bold")
	axis1.set_ylabel("Resolved Niche ID", fontweight="bold")
	axis1.grid(True, axis="y", alpha=0.2, linestyle="--")
	plt.title("Niche Transition Sequence", pad=20, fontsize=14)
	plt.tight_layout()
	if save_path:
		figure.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()


def plot_niche_transition_matrix(df_history, save_path=None):
	"""Plot a niche x niche transition-count heatmap.

	Scales to any niche count (unlike overlapping scatter points), and directly shows
	whether reclassifications cluster within one true environment's fragments (expected
	if niches are merely fragmented) or cross into a different true environment
	(a more serious bug — real cross-environment confusion, not just fragmentation).
	"""
	counts = _build_transition_matrix(df_history["detected_env_id"])
	niche_ids = sorted({nid for pair in counts for nid in pair})

	figure, axis1 = plt.subplots(figsize=(8, 7))
	if not niche_ids:
		axis1.text(0.5, 0.5, "No niche transitions recorded", ha="center", va="center")
	else:
		labels = _niche_env_labels(df_history)
		matrix = [[counts.get((row, col), 0) for col in niche_ids] for row in niche_ids]
		tick_labels = [f"{nid} ({labels.get(nid, 'Unknown')})" for nid in niche_ids]

		sns.heatmap(
			matrix,
			ax=axis1,
			annot=True,
			fmt="d",
			cmap="viridis",
			xticklabels=tick_labels,
			yticklabels=tick_labels,
			cbar_kws={"label": "Transition count"},
		)
		axis1.set_xlabel("To niche", fontweight="bold")
		axis1.set_ylabel("From niche", fontweight="bold")
		plt.setp(axis1.get_xticklabels(), rotation=45, ha="right")

	plt.title("Niche Transition Matrix", pad=20, fontsize=14)
	plt.tight_layout()
	if save_path:
		figure.savefig(save_path, dpi=300)
		plt.close()
	else:
		plt.show()
