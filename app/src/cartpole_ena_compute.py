#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENA compute study — the Plasticity skill's effect on candidate-evaluation cost (zero / linear_ramp / full)."""

# General imports
import os

# App imports
from cartpole_ena_performance import parse_args, run_study, select_variants

# The full agent (Hall of Fame + env detection) with only the plasticity driver varied:
#   zero       — p = 0 always: 2 candidates x 10 steps per training episode (30,000 in total)
#   linear_ramp — adaptive: p rises linearly toward 1 as the episode score falls toward the worst seen
#   full       — p = 1 always: 100 candidates x 500 steps per training episode (~75 million)
# The zero and full costs are deterministic, so one replicate measures them exactly; linear_ramp's
# varies with each run's score trajectory, and its distribution comes from the performance
# study's 50 ENA-Combined replicates (same config as ENA-Combined-PlasticityLinearRamp here). The cost
# metric is the compute-cost table's total candidate env-steps (training phase).
COMPUTE_VARIANTS = [
	{
		"name": "ENA-Combined-PlasticityZero",
		"plasticity_algorithm": "zero",
		"hall_of_fame_driver": "default",
		"recognizer_strategy": "dynamics_model",
	},
	{
		"name": "ENA-Combined-PlasticityLinearRamp",
		"plasticity_algorithm": "linear_ramp",
		"hall_of_fame_driver": "default",
		"recognizer_strategy": "dynamics_model",
	},
	{
		"name": "ENA-Combined-PlasticityFull",
		"plasticity_algorithm": "full",
		"hall_of_fame_driver": "default",
		"recognizer_strategy": "dynamics_model",
	},
]
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
	)
