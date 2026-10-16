"""Mamdani fuzzy inference: fuzzify crisp inputs, fire min-AND rules, aggregate by max, defuzzify by centroid."""

# Algorithm: Mamdani fuzzy logic controller (min-max inference, centroid defuzzification)
# Reference: Mamdani & Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller" (1975)
# https://doi.org/10.1016/S0020-7373(75)80002-2
# Fuzzy sets: Zadeh, "Fuzzy sets" (1965), https://doi.org/10.1016/S0019-9958(65)90241-X

# Libs imports
import numpy as np


class Triangle:  # pylint: disable=too-few-public-methods
	"""Triangular membership: 0 at left, rising to 1 at peak, falling to 0 at right."""

	def __init__(self, left: float, peak: float, right: float) -> None:
		self.left, self.peak, self.right = left, peak, right

	def __call__(self, values):
		"""Membership degree of each value."""
		values = np.asarray(values, dtype=np.float64)
		rising = (values - self.left) / (self.peak - self.left)
		falling = (self.right - values) / (self.right - self.peak)
		return np.clip(np.minimum(rising, falling), 0.0, 1.0)


class LeftShoulder:  # pylint: disable=too-few-public-methods
	"""Left shoulder: 1 up to `full`, falling linearly to 0 at `zero`."""

	def __init__(self, full: float, zero: float) -> None:
		self.full, self.zero = full, zero

	def __call__(self, values):
		"""Membership degree of each value."""
		return np.clip((self.zero - np.asarray(values, dtype=np.float64)) / (self.zero - self.full), 0.0, 1.0)


class RightShoulder:  # pylint: disable=too-few-public-methods
	"""Right shoulder: 0 up to `zero`, rising linearly to 1 at `full` and beyond."""

	def __init__(self, zero: float, full: float) -> None:
		self.zero, self.full = zero, full

	def __call__(self, values):
		"""Membership degree of each value."""
		return np.clip((np.asarray(values, dtype=np.float64) - self.zero) / (self.full - self.zero), 0.0, 1.0)


class FuzzyVariable:  # pylint: disable=too-few-public-methods
	"""A linguistic variable: named terms, each a membership function; an output variable also has a universe."""

	def __init__(self, terms: dict, universe: tuple = None, resolution: int = 1001) -> None:
		self.terms = terms
		self.grid = np.linspace(universe[0], universe[1], resolution) if universe is not None else None

	def fuzzify(self, value: float) -> dict:
		"""{term: degree} for one crisp value."""
		return {term: float(function(value)) for term, function in self.terms.items()}


class MamdaniController:
	"""Rules map input terms to an output term; IF ... AND ... is the min of the antecedents' degrees."""

	# Each rule is ({input_name: term, ...}, output_term). The output membership is the max over
	# rules of min(rule strength, output term's membership) (clipping implication), and the crisp
	# result is that aggregate's centroid over the output variable's universe grid.

	def __init__(self, inputs: dict, output: FuzzyVariable, rules: list) -> None:
		if output.grid is None:
			raise ValueError("The output variable needs a universe to defuzzify over")
		self.inputs = inputs
		self.output = output
		self.rules = rules

	def rule_strengths(self, **values: float) -> list:
		"""Each rule's firing strength for the given crisp inputs."""
		degrees = {name: variable.fuzzify(values[name]) for name, variable in self.inputs.items()}
		return [min(degrees[name][term] for name, term in antecedent.items()) for antecedent, _output_term in self.rules]

	def evaluate(self, **values: float) -> float:
		"""The crisp output: centroid of the max-aggregated, strength-clipped output terms."""
		aggregate = np.zeros_like(self.output.grid)
		for strength, (_antecedent, output_term) in zip(self.rule_strengths(**values), self.rules):
			if strength > 0.0:
				aggregate = np.maximum(aggregate, np.minimum(strength, self.output.terms[output_term](self.output.grid)))
		total = float(np.sum(aggregate))
		if total <= 0.0:
			raise ValueError(f"No rule fires for inputs {values}")
		return float(np.sum(self.output.grid * aggregate) / total)
