#!/usr/bin/env python
"""The paper's per-replicate box plots, shared by every results generator: one box per agent, each replicate a dot."""

# Libs imports
import matplotlib.pyplot as plt
import numpy as np


# pylint: disable=too-many-arguments
def replicate_box_figure(
	samples: list, labels: list, ylabel: str, path: str, *, ylim: tuple = None, reference: float = None, rotate: bool = False
) -> str:
	"""One box per sample (its replicates jittered as dots), labelled in order; an optional y range and dotted reference
	line; rotate tilts long labels. Returns the path written."""
	figure, axis = plt.subplots(figsize=(3.5, 2.4))
	axis.boxplot(samples, widths=0.55, showfliers=False, medianprops={"color": "#1f2937", "linewidth": 1.5}, boxprops={"color": "#6b7280"})
	rng = np.random.default_rng(0)
	for position, sample in enumerate(samples, start=1):
		axis.scatter(position + rng.uniform(-0.15, 0.15, len(sample)), sample, s=6, color="#2563eb", alpha=0.55, linewidths=0, zorder=3)
	if reference is not None:
		axis.axhline(reference, color="#9ca3af", linestyle=":", linewidth=0.8)
	if rotate:
		axis.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right", fontsize=7)
	else:
		axis.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
	axis.set_ylabel(ylabel, fontsize=8)
	if ylim is not None:
		axis.set_ylim(*ylim)
	axis.tick_params(axis="y", labelsize=7)
	axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
	axis.set_axisbelow(True)
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	figure.tight_layout()
	figure.savefig(path)
	plt.close(figure)
	return path
