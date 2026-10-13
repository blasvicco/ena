"""Universal Snapshot Projection tool, shared across ENA's skills."""

# Libs imports
import numpy as np


def project_individual_weights(individual: dict, obs_normalizer, max_eval_steps: float) -> None:
	"""Analytically re-scale an individual's first Dense layer to match obs_normalizer's current mean/variance, without retraining."""
	# Needed because an individual's weights are implicitly tuned to whatever normalization
	# was in effect when they were last aligned, and the live normalizer keeps drifting as
	# more states are seen. No-op until obs_normalizer has seen at least 2 * max_eval_steps
	# samples (avoids projecting off noisy early statistics), or if the normalizer hasn't
	# moved since the individual was last aligned.
	if obs_normalizer.count < max_eval_steps * 2:
		return

	mean_old = individual["last_mu"]
	variance_old = individual["last_var"]
	mean_new = obs_normalizer.mean
	variance_new = obs_normalizer.var

	if np.allclose(mean_old, mean_new) and np.allclose(variance_old, variance_new):
		return

	weights = individual["model"].get_weights()
	weight_first_layer = weights[0]

	std_old = np.sqrt(variance_old + 1e-8)
	std_new = np.sqrt(variance_new + 1e-8)

	scale_factors = np.clip(std_new / std_old, 0.1, 10.0)
	shift_factors = (mean_new - mean_old) / std_old

	weights[0] = weight_first_layer * scale_factors[:, np.newaxis]
	weights[1] = weights[1] + np.dot(shift_factors, weight_first_layer)

	individual["model"].set_weights(weights)
	individual["last_mu"] = np.copy(mean_new)
	individual["last_var"] = np.copy(variance_new)
