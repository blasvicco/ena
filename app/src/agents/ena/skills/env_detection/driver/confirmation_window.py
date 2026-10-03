"""Environment-detection driver: live rollout with a non-trivial confirmation window."""

# App imports
from agents.ena.skills.env_detection.driver.live import Instance as LiveInstance


class Instance(LiveInstance):
	"""LiveInstance with a non-trivial confirmation window by default (3, instead of 1)."""

	# The gating logic itself lives on the base class so it composes with any
	# driver too — this subclass exists only to name the "live rollout + confirmation window"
	# combination explicitly as its own driver for experiment configs that want exactly that.

	def __init__(self, state_dim, confirmation_window=3, **kwargs):
		super().__init__(state_dim, confirmation_window=confirmation_window, **kwargs)
