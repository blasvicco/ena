"""Niche-match selection driver: at test time, play the highest-trust gladiator of the niche the detector recognizes."""

# App imports
from agents.ena.skills.selection.driver.abstract import ADriver


class Instance(ADriver):
	"""Test time: the detected niche's highest-trust gladiator, else the highest-trust gladiator overall; training: epsilon-greedy by trust."""

	# The simple retrieval Bayesian policy reuse is compared against: it trusts the detector alone.
	# There is no belief and no learning from test returns, so a gladiator that fails in its own
	# niche keeps playing until the detector's niche changes. When the detector recognizes no niche
	# with a gladiator (at the start of a test phase, after a confirmed shift until enough evidence
	# exists, or on novel physics), the highest-trust gladiator of any niche plays.

	def __init__(self, exploration_rate: float = 0.1, **_kwargs) -> None:
		super().__init__(exploration_rate)
		self._gladiators_by_trust = []  # [(population index, home niche, trust)], highest trust first

	def evaluate(self, population: list) -> None:
		"""Rebuild the gladiator library from the population, ranked by trust."""
		self._gladiators_by_trust = sorted(self._gladiators(population), key=lambda gladiator: -gladiator[2])

	def _select_test(self, population: list, recognizer, memory) -> int:
		"""The highest-trust gladiator of the recognized niche, falling back to the highest-trust gladiator overall."""
		if not self._gladiators_by_trust:
			return self._select_desperate(population)
		niche_id = recognizer.current_niche_id if recognizer is not None else -1
		for index, home_niche_id, _trust in self._gladiators_by_trust:
			if home_niche_id == niche_id:
				return index
		return self._gladiators_by_trust[0][0]
