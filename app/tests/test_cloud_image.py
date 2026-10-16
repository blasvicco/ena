"""Tests for the cloud image's contents (app/cloud_manifest.txt, copied by hub/cloud/Dockerfile): the study runs from them alone."""

# General imports
import os
import shutil
import subprocess
import sys

# App imports
from helpers import step

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(APP_DIR, "cloud_manifest.txt")
# Imports the study, builds every agent it trains (which loads every skill driver ENA uses) and its test worlds, and
# prints where half_cheetah came from.
PROBE = "\n".join(
	[
		"",
		"import sys",
		"sys.path.insert(0, sys.argv[1])",
		"import half_cheetah",
		'for name in ("ENA-Combined", "PPO", "SAC"):',
		"\thalf_cheetah.build_agent(name, max_agent_threads=1)",
		"assert len(half_cheetah.make_test_worlds(half_cheetah.make_training_worlds())) == 5",
		"print(half_cheetah.__file__)",
		"",
	]
)


def _manifest() -> list:
	"""The manifest's paths (comments and blank lines skipped)."""
	with open(MANIFEST, encoding="utf-8") as manifest:
		return [line.strip() for line in manifest if line.strip() and not line.startswith("#")]


def test_the_study_builds_from_the_cloud_image_files_alone(tmp_path) -> None:
	"""The study builds from the cloud manifest's files alone, in an interpreter that cannot see app/src."""
	# Copy only the manifest's files, then build the study's agents in a fresh interpreter that cannot see app/src: a
	# missing module (a new import, a driver loaded by name) fails here instead of on twenty cloud machines.
	with step("Arrange: the manifest's files only, as the image holds them."):
		source = os.path.join(APP_DIR, "src")
		for path in _manifest():
			origin, target = os.path.join(source, path), tmp_path / "src" / path
			if os.path.isdir(origin):
				shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", "*.md"))
			else:
				target.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy(origin, target)

	with step("Act: build the agents and worlds in an isolated interpreter (no PYTHONPATH, no app/src)."):
		result = subprocess.run([sys.executable, "-I", "-c", PROBE, str(tmp_path / "src")], capture_output=True, text=True, timeout=300, check=False)

	with step("Assert: it ran, and half_cheetah came from the copied files."):
		assert result.returncode == 0, result.stderr[-2000:]
		assert result.stdout.strip().splitlines()[-1].startswith(str(tmp_path))
