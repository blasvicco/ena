"""A study directory's replicates: the exp_<n> directories every results tool reads."""

# General imports
import os


def get_experiment_directories(study_dir: str) -> list:
	"""Return the study's replicate directories (exp_<n>), sorted by replicate number."""
	return sorted((name for name in os.listdir(study_dir) if name.startswith("exp_")), key=lambda name: int(name.split("_")[1]))
