#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA Plasticity-controller study — ENA-Combined with each alternative failure-to-Plasticity controller (30 replicates)."""

# General imports
import os

# App imports
from cartpole_ena_ablation import NUM_EXPERIMENTS, ABLATION_VARIANTS, parse_args, run_study, select_variants

# ENA-Combined (Plasticity + Hall of Fame + env detection, bpr selection, linear_ramp Trust) with
# only the Plasticity driver swapped. The linear_ramp reference is cartpole_ena_ablation.py's own
# ENA-Combined: every agent of a replicate is reseeded before it is built, trained and tested, so
# replicate i of ENA-Combined is the same run here as there and is not repeated (the comparison is
# paired by replicate). Decision rule fixed in docs/PLAN_ena_plasticity_controller.md before results exist.
_COMBINED = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")
CONTROLLER_VARIANTS = [
	{**_COMBINED, "name": "ENA-Combined-QuadraticPlasticity", "plasticity_algorithm": "quadratic"},
	{**_COMBINED, "name": "ENA-Combined-FuzzyPlasticity", "plasticity_algorithm": "fuzzylogic"},
]
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartpole_ena_plasticity_controller_outputs")


if __name__ == "__main__":
	ARGS = parse_args(
		variant_table=CONTROLLER_VARIANTS, defaults={"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": False}
	)
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(CONTROLLER_VARIANTS, ARGS.variants),
		ARGS.include_random,
		"ENA PLASTICITY CONTROLLER",
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
	)
