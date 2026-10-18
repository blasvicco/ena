#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA selection study — ENA-Combined with each simpler test-time retrieval in place of Bayesian policy reuse (30 replicates)."""

# General imports
import os

# App imports
from cartpole_ena_ablation import NUM_EXPERIMENTS, ABLATION_VARIANTS, parse_args, run_study, select_variants

# ENA-Combined (Plasticity + Hall of Fame + env detection, linear_ramp Trust) with only the
# selection driver swapped. Every driver selects identically during training (epsilon-greedy by
# Trust), so each row trains exactly the ablation's ENA-Combined and differs only at test: the
# BPR reference is cartpole_ena_ablation.py's own ENA-Combined, not repeated here (the comparison
# is paired by replicate). Comparisons and reporting rule were fixed before the results existed.
_COMBINED = next(row for row in ABLATION_VARIANTS if row["name"] == "ENA-Combined")

SELECTION_VARIANTS = [
	{**_COMBINED, "name": "ENA-Combined-NicheMatch", "selection_driver": "niche_match"},
	{**_COMBINED, "name": "ENA-Combined-BestTrust", "selection_driver": "best_trust"},
]

# Anchored to this file's own location (one level up from src/), not the caller's cwd.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "cartpole_ena_selection_outputs")


if __name__ == "__main__":
	ARGS = parse_args(
		variant_table=SELECTION_VARIANTS, defaults={"output_dir": OUTPUT_DIR, "num_experiments": NUM_EXPERIMENTS, "include_random": False}
	)
	run_study(
		ARGS.num_experiments,
		ARGS.output_dir,
		select_variants(SELECTION_VARIANTS, ARGS.variants),
		ARGS.include_random,
		"ENA SELECTION",
		workers=ARGS.workers,
		threads_per_worker=ARGS.threads_per_worker,
		replicate=ARGS.replicate,
	)
