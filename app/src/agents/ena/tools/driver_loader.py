"""Build a skill's driver by name: the strategy lookup every skill facade shares."""

# General imports
import importlib


def load_driver(skill: str, driver: str, *args: object, packages: tuple | None = None, **kwargs: object) -> object:
	"""Build `driver` from the first of the skill's driver packages that defines it."""
	# packages defaults to the skill's own driver package; a skill can also draw on shared drivers
	# (Trust and Plasticity on skills/failure_signal). A driver module that exists but fails to import
	# raises its own error instead of reading as an unknown driver.
	for package in packages or (f"agents.ena.skills.{skill}.driver",):
		module_name = f"{package}.{driver}"
		try:
			module = importlib.import_module(module_name)
		except ModuleNotFoundError as error:
			if error.name != module_name:
				raise
			continue
		return module.Instance(*args, **kwargs)
	raise ValueError(f"Unknown {skill} driver: {driver!r}")
