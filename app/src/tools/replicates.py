"""Where the studies keep their replicates (app/outputs/), and a study directory's replicates: its exp_<n> directories."""

# General imports
import os

# Every study writes its replicates to app/outputs/<study>_outputs/, and every results tool reads them there.
# Anchored to this file's own location (app/src/tools/ -> app/), not the caller's cwd.
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


def get_experiment_directories(study_dir: str) -> list:
	"""Return the study's replicate directories (exp_<n>), sorted by replicate number."""
	return sorted((name for name in os.listdir(study_dir) if name.startswith("exp_")), key=lambda name: int(name.split("_")[1]))
