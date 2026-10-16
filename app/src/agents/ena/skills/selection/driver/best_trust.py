"""Best-trust selection driver: at test time, always play the highest-trust gladiator, ignoring niches."""

# App imports
from agents.ena.skills.selection.driver.abstract import ADriver


class Instance(ADriver):
	"""Test time: the highest-trust gladiator of any niche; training: epsilon-greedy by trust."""

	# The retrieval floor for the selection ablation: it uses neither the detector nor the test
	# returns, so the same gladiator plays every test episode. Against niche_match it shows whether
	# the niche index matters at retrieval, and not only for organizing the archive.

	def __init__(self, exploration_rate: float = 0.1, **_kwargs) -> None:
		super().__init__(exploration_rate)
		self._best = None  # population index of the highest-trust gladiator

	def evaluate(self, population: list) -> None:
		"""Find the highest-trust gladiator in the population."""
		gladiators = self._gladiators(population)
		self._best = max(gladiators, key=lambda gladiator: gladiator[2])[0] if gladiators else None

	def _select_test(self, population: list, recognizer, memory) -> int:
		"""The highest-trust gladiator, or a random non-gladiator if there is none."""
		return self._best if self._best is not None else self._select_desperate(population)
