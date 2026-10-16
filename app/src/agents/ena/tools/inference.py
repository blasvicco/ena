"""Action-inference tool, shared across ENA's skills."""

# Libs imports
import numpy as np
import tensorflow as tf

# App imports
from agents.ena.memory import AgentMemory


def infer_action(individual: dict, state: np.ndarray, memory: AgentMemory, is_continuous: bool) -> np.ndarray:
	"""Normalize state through the individual's frozen or live statistics and extract an action."""
	# Gladiators normalize against the mean/variance snapshot frozen at induction time (their
	# weights were aligned to it); everyone else normalizes against the live obs_normalizer.
	if individual.get("is_gladiator", False):
		mean = individual["last_mu"]
		std = np.sqrt(individual["last_var"] + 1e-8)
		norm_state = (state - mean) / std
	else:
		norm_state = memory.obs_normalizer.normalize(state)

	state_tensor = tf.convert_to_tensor(norm_state.reshape(1, -1), dtype=tf.float32)
	action_output = individual["fast_call"](state_tensor, training=False)
	action_numpy = action_output.numpy()[0]

	return np.clip(action_numpy, -1.0, 1.0) if is_continuous else int(np.argmax(action_numpy))
