#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA fuzzy-plasticity study — ENA-Combined with the Mamdani fuzzy Plasticity driver (50 replicates)."""

# General imports
import os

# App imports
from cartpole_ena_performance import NUM_EXPERIMENTS, PERFORMANCE_VARIANTS, parse_args, run_study, select_variants

# ENA-Combined (Plasticity + Hall of Fame + env detection, bpr selection, linear_ramp Trust) with
# only the Plasticity driver swapped from linear_ramp to the Mamdani fuzzylogic controller. Compared
# against cartpole_ena_performance.py's 50 ENA-Combined replicates (same seeds, same protocol), by
# the decision rule fixed in docs/PLAN_ena_fuzzy_logic.md (2.4) before results exist.
FUZZY_VARIANTS = [
	{
		**next(row for row in PERFORMANCE_VARIANTS if row["name"] == "ENA-Combined"),
		"name": "ENA-Combined-FuzzyPlasticity",
		"plasticity_algorithm": "fuzzylogic",
	},
]
# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartpole_ena_performance_real_fuzzylogic_outputs")


if __name__ == "__main__":
	ARGS = parse_args(variant_table=FUZZY_VARIANTS, defaults={"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": False})
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(FUZZY_VARIANTS, ARGS.variants),
		ARGS.include_random,
		"ENA FUZZY PLASTICITY",
		workers=ARGS.workers,
	)
