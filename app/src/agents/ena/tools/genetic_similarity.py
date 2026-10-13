"""Genetic similarity between two individuals — cosine similarity of their flattened weight vectors."""

# Libs imports
import numpy as np
from numpy.linalg import norm


def genetic_similarity(individual_a: dict, individual_b: dict) -> float:
	"""Cosine similarity between two individuals' weight vectors — 1.0 = genetically identical."""
	vector_a = np.concatenate([weights.flatten() for weights in individual_a["model"].get_weights()])
	vector_b = np.concatenate([weights.flatten() for weights in individual_b["model"].get_weights()])
	return float(np.dot(vector_a, vector_b) / ((norm(vector_a) + 1e-8) * (norm(vector_b) + 1e-8)))
