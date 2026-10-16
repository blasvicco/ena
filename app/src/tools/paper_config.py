#!/usr/bin/env python
"""The paper's configuration numbers, read from the code that ran: ENA's settings, the environments' physics, the protocols."""
# Every setting the paper states is emitted as a \resdef{cfg/...} macro (or a generated table), so
# the text cannot drift from the code. ENA's settings are read from an ENA-Combined built exactly as
# the studies build it (cartpole_ena_ablation.build_ena_agent); the physics from the environments the
# studies construct; the protocol constants from the study scripts. paper_results.py calls this.
#
#         python src/tools/paper_config.py          # print every entry

# General imports
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App imports
# pylint: disable=wrong-import-position
import cartpole_challenge
import cartpole_ena_ablation
import half_cheetah
from agents.ena import ena as ena_module
from agents.ena import memory as ena_memory
from agents.ena.skills.crossover.driver import uniform_mask
from agents.ena.skills.hall_of_fame.driver.default import Instance as HallOfFame
from agents.ena.skills.mutation.driver import gaussian
from agents.ena.skills.plasticity.driver import fuzzylogic
from agents.ena.skills.selection.driver.bpr import Instance as BPR
from agents.ena.tools import curves, reward_normalizer
from agents.ena.tools.chirp_features import features as chirp_features
from tools import detection_ablation, detection_benchmark
from tools import metrics
from tools import sac_discrete_entropy_check as sac_check


def _number(value) -> str:
	"""A setting as the paper prints it: integers with thousands separators, the rest as short decimals."""
	if float(value).is_integer():
		return f"{int(value):,}"
	return f"{value:g}"


def ena_entries() -> dict:
	"""ENA-Combined's settings as the CartPole studies build it."""
	combined = next(row for row in cartpole_ena_ablation.ABLATION_VARIANTS if row["name"] == "ENA-Combined")
	agent = cartpole_ena_ablation.build_ena_agent(combined, 1)
	# pylint: disable=protected-access
	detection = agent._skills["env_detection"]._instance
	selection = agent._skills["selection"]._instance
	hall_of_fame = agent._skills["hall_of_fame"]._instance
	mutation = agent._skills["mutation"]._instance
	values = {
		"pop": agent.pop_size,
		"smin": agent.min_eval_steps,
		"smax": agent.max_eval_steps,
		"history": agent.memory.reward_history.maxlen,
		"epsilon": selection._exploration_rate,
		"mutation/rate": mutation._mutation_rate,
		"mutation/noise": mutation._mutation_noise,
		"hof/K": hall_of_fame.max_per_niche,
		"hof/reinject": hall_of_fame.max_gladiator_ratio,
		"hof/similarity": HallOfFame.SIMILARITY_THRESHOLD,
		"detect/tau": detection._residual_tolerance,
		"detect/margin": detection._novelty_margin,
		"detect/recent": detection._recent.maxlen,
		"detect/minsamples": detection._min_samples,
		"detect/verify": detection._min_verify_samples,
		"bpr/noise": selection._return_noise,
		"bpr/switch": selection._switch_rate,
		"bpr/prior": selection._prior_strength,
		"bpr/crossprior": selection._cross_prior,
		"bpr/mix": 1 - selection._switch_rate,
		"bpr/ownprior": BPR.OWN_PRIOR,
		"fuzzy/rules": len(fuzzylogic.RULES),
		"fuzzy/trendrecent": ena_memory.TREND_RECENT,
		"fuzzy/trendbefore": ena_memory.TREND_WINDOW - ena_memory.TREND_RECENT,
		"fuzzy/trendmin": ena_memory.TREND_MIN_RETURNS,
		"signal/warmup": curves.WARMUP_RETURNS,
		"signal/percentile": curves.CONFIDENT_PERCENTILE,
		"signal/min": curves.MIN_SIGNAL,
		"signal/slope": 1 - curves.MIN_SIGNAL,
		"trust/low": reward_normalizer.NORMALIZED_RANGE[0],
		"trust/high": reward_normalizer.NORMALIZED_RANGE[1],
		"trust/plow": reward_normalizer.NORMALIZING_PERCENTILES[0],
		"trust/phigh": reward_normalizer.NORMALIZING_PERCENTILES[1],
		"trust/reset": ena_module.TRUST_RESET_SIGNAL,
		"search/mincandidates": ena_module.MIN_CANDIDATES,
		"search/elite": ena_module.ELITE_SHARE * 100,
		"search/replaced": ena_module.REPLACED_SHARE * 100,
		"search/mutants": ena_module.MUTANT_SHARE * 100,
		"search/crossovers": ena_module.CROSSOVER_SHARE * 100,
		# The replaced share splits into mutants, crossovers and the rest fresh; as shares of the whole population too.
		"search/fresh": round((1 - ena_module.MUTANT_SHARE - ena_module.CROSSOVER_SHARE) * 100, 6),
		"search/mutantspop": round(ena_module.REPLACED_SHARE * ena_module.MUTANT_SHARE * 100, 6),
		"search/crossoverspop": round(ena_module.REPLACED_SHARE * ena_module.CROSSOVER_SHARE * 100, 6),
		"search/freshpop": round(ena_module.REPLACED_SHARE * (1 - ena_module.MUTANT_SHARE - ena_module.CROSSOVER_SHARE) * 100, 6),
		"mutation/minnoise": gaussian.MIN_NOISE,
		"crossover/share": uniform_mask.PARENT_SHARE,
		"hof/gatez": HallOfFame.GATE_Z,
		"hof/gaterelief": HallOfFame.GATE_PLASTICITY_RELIEF,
		"hof/gatefloor": HallOfFame.GATE_SPREAD_FLOOR,
		"hof/gatetrust": HallOfFame.GATE_TRUST,
	}
	entries = {f"cfg/ena/{key}": _number(value) for key, value in values.items()}
	# The CuSum alarm threshold is log(gamma); gamma is printed as a power of ten.
	entries["cfg/ena/detect/gammaexp"] = _number(round(detection._cusum._threshold / math.log(10)))
	return entries


def protocol_entries() -> dict:
	"""Episodes, budgets, evaluation checks, thresholds and seeds of the CartPole and HalfCheetah protocols."""
	ablation, challenge, cheetah = cartpole_ena_ablation, cartpole_challenge, half_cheetah
	values = {
		"cartpole/train": ablation.TRAINING_EPISODES_ENA[0],
		"cartpole/test": ablation.TESTING_EPISODES[0],
		"cartpole/solved": ablation.SUCCESS_THRESHOLD,
		"cartpole/seedoffset": ablation.SEED_OFFSET,
		"challenge/sacdratio": challenge.SAC_DISCRETE_TARGET_ENTROPY,
		"sacd/paperratio": sac_check.PAPER_RATIO,
		"sacd/cleanrlratio": sac_check.CLEANRL_RATIO,
		"metrics/stabilitywindow": metrics.STABILITY_WINDOW,
		"challenge/budget": challenge.FIXED_BUDGET_PER_ENV,
		"challenge/checkevery": challenge.CHECK_EVERY,
		"challenge/evalepisodes": challenge.EVAL_EPISODES,
		"cheetah/solved": cheetah.SOLVED_SCORE,
		"cheetah/budget": cheetah.FIXED_BUDGET_PER_ENV,
		"cheetah/test": cheetah.TESTING_EPISODES[0],
	}
	entries = {f"cfg/{key}": _number(value) for key, value in values.items()}
	entries["cfg/cartpole/seedoffsetraw"] = str(ablation.SEED_OFFSET)  # inside formulas: no thousands separator
	entries["cfg/challenge/budgetM"] = _number(challenge.FIXED_BUDGET_PER_ENV / 1e6)
	world = ablation.make_training_worlds()[0]
	entries["cfg/cartpole/maxsteps"] = _number(world.spec.max_episode_steps)
	world.close()
	cheetah_world = cheetah.make_env()
	entries["cfg/cheetah/maxsteps"] = _number(cheetah_world.spec.max_episode_steps)
	cheetah_world.close()
	entries["cfg/cheetah/maxeval"] = _number(cheetah.ENA_SETTINGS["max_eval_steps"])
	return entries


def _row(cells: list) -> str:
	return " & ".join(cells) + " \\\\"


def latex_table(caption: str, label: str, column_spec: str, body: list, fit: str = None) -> str:
	"""Wrap a formatted header and rows in a captioned, labelled LaTeX table, fitted to the column or the page."""
	# A captioned, labelled LaTeX table around body (header and rows, already formatted); fit="column" scales it to the
	# column, fit="page" spans both columns of the page and scales it to their width.
	environment, width = ("table*", "\\textwidth") if fit == "page" else ("table", "\\columnwidth")
	opening = [f"\\begin{{{environment}}}[{'t' if fit == 'page' else 'htbp'}]", f"\\caption{{{caption}}}", "\\begin{center}", "\\footnotesize"]
	if fit:
		opening += ["\\setlength{\\tabcolsep}{3pt}", f"\\begin{{adjustbox}}{{max width={width}}}"]
	closing = ["\\end{tabular}"] + (["\\end{adjustbox}"] if fit else []) + [f"\\label{{{label}}}", "\\end{center}", f"\\end{{{environment}}}"]
	return "\n".join(opening + [f"\\begin{{tabular}}{{{column_spec}}}"] + body + closing) + "\n"


def environment_table(rows: list, headers: list, caption: str, label: str) -> str:
	"""A small LaTeX table, one row per environment."""
	body = ["\\hline", _row([f"\\textbf{{{header}}}" for header in headers]), "\\hline", *[_row(row) for row in rows], "\\hline"]
	return latex_table(caption, label, f"|l|{'c|' * (len(headers) - 1)}", body)


def cartpole_rows(training_worlds: list, test_only_worlds: list) -> list:
	"""(environment, g, pole mass, pole half-length, seen) rows; test-only worlds come from the study's test sequence."""
	rows = []
	for worlds, seen in ((training_worlds, "yes"), (test_only_worlds, "no")):
		for world in worlds:
			physics = world.unwrapped
			rows.append([physics.name, _number(abs(physics.gravity)), _number(physics.masspole), _number(physics.length), seen])
	return rows


def cheetah_rows() -> list:
	"""(environment, g, friction, wind, seen) rows for HalfCheetah's training and never-trained worlds."""
	rows = []
	for worlds, seen in ((half_cheetah.make_training_worlds(), "yes"), (half_cheetah.make_unseen_worlds(), "no")):
		for world in worlds:
			physics = world.unwrapped
			wind = getattr(physics, "wind", (0.0, 0.0, 0.0))
			rows.append([physics.name, _number(abs(physics.gravity)), _number(physics.friction), _number(wind[0]), seen])
			world.close()
	return rows


WAVELET_NAMES = {"haar": "Haar", "mexh": "Mexican hat", "morl": "Morlet"}  # PyWavelets names as the paper prints them
TRANSFORM_NAMES = {"dwt": "DWT", "swt": "SWT", "packet": "wavelet packet", "cwt": "CWT"}


def chirp_entries() -> dict:
	"""HalfCheetah's chirp detector as the study builds it, and the detection benchmark's protocol."""
	agent = half_cheetah.build_ena_agent(1)
	settings = agent._skills["env_detection"]._instance._settings  # pylint: disable=protected-access
	feature_defaults = {name: parameter.default for name, parameter in inspect.signature(chirp_features).parameters.items()}
	features = {**feature_defaults, **settings["features"]}
	case = detection_benchmark.domain_case("half_cheetah")
	entries = {
		"cfg/cheetah/chirp/steps": _number(settings["steps"]),
		"cfg/cheetah/chirp/excitation": settings["excitation"].replace("_", " "),
		"cfg/cheetah/chirp/excitationwavelet": WAVELET_NAMES.get(
			settings["excitation_params"].get("wavelet"), settings["excitation_params"].get("wavelet")
		),
		"cfg/cheetah/chirp/transform": TRANSFORM_NAMES[features["transform"]],
		"cfg/cheetah/chirp/level": _number(features["level"]),
		# Bands per channel: a packet tree's leaves (2^level), or a DWT's details plus its approximation.
		"cfg/cheetah/chirp/bands": _number(2 ** features["level"] if features["transform"] == "packet" else features["level"] + 1),
		"cfg/cheetah/chirp/featurewavelet": WAVELET_NAMES.get(features["wavelet"], features["wavelet"]),
		"cfg/cheetah/chirp/window": _number(settings["window"]),
		"cfg/cheetah/chirp/minchirps": _number(settings["min_chirps"]),
		"cfg/cheetah/chirp/margin": _number(settings["margin"]),
		"cfg/cheetah/chirp/share": _number(round(100 * settings["steps"] / agent.max_episode_steps, 6)),
		"cfg/detbench/episodes": _number(case["episodes_per_world"]),
		"cfg/detbench/block": _number(case["policy_block"]),
		"cfg/detbench/window": _number(case["window"]),
		"cfg/detbench/purity": _number(detection_benchmark.PURITY_TARGET * 100),
		"cfg/detbench/seeds": _number(len(detection_ablation.SEEDS)),
		"cfg/detbench/selectionseeds": _number(len(detection_ablation.SELECTION_SEEDS)),
		"cfg/detbench/confirmseeds": _number(len(detection_ablation.CONFIRMATION_SEEDS)),
		"cfg/detbench/seedfirst": _number(detection_ablation.SEEDS[0]),
		"cfg/detbench/seedlast": _number(detection_ablation.SEEDS[-1]),
		"cfg/detbench/confirmfirst": _number(detection_ablation.CONFIRMATION_SEEDS[0]),
		"cfg/detbench/confirmlast": _number(detection_ablation.CONFIRMATION_SEEDS[-1]),
	}
	return entries


def config_entries() -> dict:
	"""Every configuration macro."""
	return {**ena_entries(), **protocol_entries(), **chirp_entries()}


if __name__ == "__main__":
	for entry_key, entry_value in sorted(config_entries().items()):
		print(f"{entry_key} = {entry_value}")
