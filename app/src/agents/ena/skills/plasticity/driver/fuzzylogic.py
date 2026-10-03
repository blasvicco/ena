"""Plasticity driver: Mamdani fuzzy logic controller over relative performance and its trend."""

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.plasticity.driver.abstract import ADriver
from agents.ena.tools.fuzzy_inference import FuzzyVariable, LeftShoulder, MamdaniController, RightShoulder, Triangle

# Inputs:
#   performance p in [0, 1] — the episode score relative to the recent range, normalized exactly as
#     the linear_ramp curve does: (score - worst seen) / (90th percentile - worst seen), clipped;
#   trend d in [-1, 1] — memory.performance_derivative (last-5 mean minus the older baseline mean)
#     divided by the same span, clipped (0 until 20 episodes of history exist).
# Output: plasticity on [0.1, 1.0], the same range as linear_ramp. The output terms are symmetric
# triangles peaking at 0.1 / 0.55 / 1.0 on a universe extended past both ends, so a single fully
# firing rule defuzzifies to exactly its peak instead of being pulled inward by the centroid.
PERFORMANCE = FuzzyVariable({"LOW": LeftShoulder(0.2, 0.5), "MEDIUM": Triangle(0.2, 0.5, 0.8), "HIGH": RightShoulder(0.5, 0.8)})
TREND = FuzzyVariable({"FALLING": LeftShoulder(-0.5, 0.0), "STEADY": Triangle(-0.5, 0.0, 0.5), "RISING": RightShoulder(0.0, 0.5)})
PLASTICITY = FuzzyVariable(
	{"LOW": Triangle(-0.35, 0.1, 0.55), "MEDIUM": Triangle(0.1, 0.55, 1.0), "HIGH": Triangle(0.55, 1.0, 1.45)},
	universe=(-0.35, 1.45),
)
# performance \ trend:  FALLING   STEADY   RISING
#   LOW                 HIGH      HIGH     HIGH
#   MEDIUM              HIGH      MEDIUM   MEDIUM
#   HIGH                MEDIUM    LOW      LOW
RULE_TABLE = {
	"LOW": {"FALLING": "HIGH", "STEADY": "HIGH", "RISING": "HIGH"},
	"MEDIUM": {"FALLING": "HIGH", "STEADY": "MEDIUM", "RISING": "MEDIUM"},
	"HIGH": {"FALLING": "MEDIUM", "STEADY": "LOW", "RISING": "LOW"},
}
RULES = [({"performance": level, "trend": trend}, output) for level, row in RULE_TABLE.items() for trend, output in row.items()]


class Instance(ADriver):  # pylint: disable=too-few-public-methods
	"""Fuzzy plasticity: low or falling performance -> more search effort; high, steady or rising -> less."""

	def __init__(self) -> None:
		self._controller = MamdaniController({"performance": PERFORMANCE, "trend": TREND}, PLASTICITY, RULES)

	def compute(self, reward: float, memory) -> float:
		"""Return the plasticity factor for the given reward."""
		if len(memory.reward_history) < 10:
			return 1.0
		high = float(np.percentile(memory.reward_history, 90))
		span = high - memory.min_seen
		if span < 1e-8:
			performance, trend = (1.0 if reward >= high else 0.0), 0.0
		else:
			performance = float(np.clip((reward - memory.min_seen) / span, 0.0, 1.0))
			trend = float(np.clip(memory.performance_derivative / span, -1.0, 1.0))
		return float(np.clip(self._controller.evaluate(performance=performance, trend=trend), 0.1, 1.0))
