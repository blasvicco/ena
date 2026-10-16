#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA compute study: how much compute the failure-driven control of the search saves, and at what performance cost."""

# General imports
import os

# App imports
from cartpole_ena_ablation import ABLATION_VARIANTS, parse_args, run_study, select_variants

# ENA-Combined (Hall of Fame + env detection) at three levels of search effort:
#   full        — fixed maximum effort: every episode evaluates the whole population, at full depth;
#   linear_ramp — the failure-driven control the paper proposes: effort scales with how badly the episode went;
#   zero        — fixed minimum effort: 2 candidates x min_eval_steps per episode.
# Only "full" runs here. The other two are the ablation's own rows of the same configuration
# (REFERENCES): every agent of a replicate is reseeded before it is built, trained and tested, so
# replicate i is the same run in either study, whatever the row is called
# (test_a_variants_run_is_identical_with_or_without_other_agents_before_it). The cost is the
# candidate-evaluation env-steps actually spent in training (machine-independent; wall-clock is not
# comparable, since this run and the ablation ran under different machine load).
# One replicate, by the author's decision: the cost difference is large, and what the paper reports
# from this study is that replicate's measurement, not a statistical test.
_COMBINED = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
COMPUTE_VARIANTS = [{**_COMBINED, "name": "ENA-Combined-PlasticityFull", "plasticity_algorithm": "full"}]
REFERENCES = {"zero": "ENA-HoF+EnvDetection", "linear_ramp": "ENA-Combined"}
NUM_EXPERIMENTS = 1
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartpole_ena_compute_outputs")


if __name__ == "__main__":
	ARGS = parse_args(
		variant_table=COMPUTE_VARIANTS, defaults={"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": False}
	)
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(COMPUTE_VARIANTS, ARGS.variants),
		ARGS.include_random,
		"ENA COMPUTE",
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
		replicate=ARGS.replicate,
	)
