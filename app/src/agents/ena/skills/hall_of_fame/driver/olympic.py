"""Hall of Fame driver: the default archive plus Olympics, periodic re-evaluation of its gladiators on fresh episodes."""

# Algorithm: periodic re-evaluation of archived elites on fresh episodes, against noisy single-rollout fitness.
# References: Such et al., "Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training
# Deep Neural Networks for Reinforcement Learning" (2017), https://arxiv.org/abs/1712.06567 (elite chosen by
# re-evaluation on extra episodes); Justesen, Risi & Mouret, "MAP-Elites for Noisy Domains by Adaptive
# Sampling" (GECCO 2019), https://doi.org/10.1145/3319619.3321904 (archive elites re-sampled).
# Jin & Branke, "Evolutionary Optimization in Uncertain Environments - A Survey" (IEEE TEC 2005).

# General imports
from typing import Callable

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.hall_of_fame.driver.default import Instance as Default

SEED_RANGE = 9999  # Olympiad seeds, drawn from the agent's seeded NumPy generator


class Instance(Default):
	"""The default archive, with Olympiads every olympics_interval environment steps."""

	# A gladiator is admitted on the evidence of one episode; where returns depend strongly on the initial state
	# (HalfCheetah runs or flips), many are lucky. Every olympics_interval steps the current niche's gladiators play
	# olympics_episodes full episodes on shared fresh seeds (ENAgent plays them, in its step budget). Those whose
	# Olympic mean falls below the eviction bar (the population's reference trust) are evicted, then the lowest
	# survivors beyond max_per_niche are relegated; the champion always stays, so a niche never empties. Survivors
	# take their Olympic trust and mean return (BPR's prior at test) and are confirmed. Until it has competed, a
	# gladiator is provisional and does not play at test. Only an Olympiad removes or re-scores a confirmed
	# gladiator (protect_confirmed); qualifier_seats let new winners in beside a niche full of confirmed ones.
	# Records store the statistics their weights are aligned to (aligned_statistics), so a gladiator stays consistent
	# with a normalizer that moves within an episode (HalfCheetah's short window). As in the default archive, the live
	# winner becomes a gladiator that keeps playing during training (provisional until its first Olympiad); an Olympiad
	# that evicts a record also clears its live copy, so mediocre early gladiators cannot anchor the search. Confirmed
	# records are re-introduced at test (load_gladiators) as gladiators (reinject_as_gladiators).

	holds_olympics = True

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self,
		max_per_niche: int,
		max_gladiator_ratio: float,
		brain: Callable,
		action_dim: int,
		state_dim: int,
		olympics_interval: int = 200_000,
		olympics_episodes: int = 3,
		qualifier_seats: int = 2,
		**_kwargs,
	) -> None:
		super().__init__(max_per_niche, max_gladiator_ratio, brain, action_dim, state_dim)
		if olympics_interval <= 0 or olympics_episodes <= 0 or qualifier_seats < 0:
			raise ValueError("olympics_interval and olympics_episodes must be positive, qualifier_seats non-negative")
		self.protect_confirmed = True
		self.aligned_statistics = True
		self.reinject_as_gladiators = True
		self.qualifier_seats = int(qualifier_seats)
		self.interval = int(olympics_interval)
		self.episodes = int(olympics_episodes)
		self._last = 0

	def olympics_due(self, total_env_steps: int) -> bool:
		"""Due once olympics_interval steps have passed since the last Olympiad."""
		return total_env_steps - self._last >= self.interval

	def admits(self, individual: dict) -> bool:
		"""Only gladiators that competed in an Olympiad and survived."""
		return bool(individual.get("olympic_confirmed", False))

	# pylint: disable=too-many-locals
	def hold_olympics(
		self, niche_id: int, play: Callable, trust_of: Callable[[float], float], eviction_bar: float, population: list, total_env_steps: int
	) -> list:
		"""Play the niche's gladiators on shared fresh seeds, judge them by Olympic trust and return one row per competitor."""
		# The niche's gladiators play shared fresh seeds through play(records, seeds) -> [(returns, steps)];
		# trust_of(mean return) gives their Olympic trust. Returns one row per competitor.
		self._last = int(total_env_steps)
		competitors = self.records_in_niche(niche_id)
		if not competitors:
			return []
		results = play(competitors, [int(np.random.randint(0, SEED_RANGE)) for _ in range(self.episodes)])
		olympic_trust = {record["individual_id"]: trust_of(float(np.mean(returns))) for record, (returns, _steps) in zip(competitors, results)}
		evicted = self._judge(olympic_trust, eviction_bar)
		rows = []
		for record, (returns, steps) in zip(competitors, results):
			trust, out = olympic_trust[record["individual_id"]], record["individual_id"] in evicted
			rows.append(self._row(record, niche_id, returns, steps, trust, eviction_bar, out, total_env_steps))
			if out:
				self.evict(record["individual_id"], population)
			else:
				self._confirm(record, trust, float(np.mean(returns)), population)
		return rows

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	@staticmethod
	def _row(record: dict, niche_id: int, returns, steps, olympic_trust: float, eviction_bar: float, evicted: bool, total_env_steps: int) -> dict:
		"""One competitor's Olympiad record (ENAgent.olympics_history)."""
		return {
			"total_env_steps": total_env_steps,
			"niche_id": niche_id,
			"individual_id": record["individual_id"],
			"returns": list(returns),
			"steps": list(steps),
			"admission_trust": record.get("trust"),
			"olympic_trust": olympic_trust,
			"eviction_bar": eviction_bar,
			"evicted": evicted,
		}

	def _judge(self, olympic_trust: dict, eviction_bar: float) -> set:
		"""Everyone below the bar except the champion, then the lowest survivors beyond max_per_niche (relegation)."""
		ranked = sorted(olympic_trust, key=olympic_trust.get, reverse=True)
		evicted = {individual_id for individual_id in ranked[1:] if olympic_trust[individual_id] < eviction_bar}
		survivors = [individual_id for individual_id in ranked if individual_id not in evicted]
		return evicted | set(survivors[max(self.max_per_niche, 1) :])

	@staticmethod
	def _confirm(record: dict, trust: float, olympic_return: float, population: list) -> None:
		"""A survivor takes its Olympic trust and mean return and is confirmed, on the record and its live copy."""
		record["trust"], record["olympic_confirmed"], record["olympic_return"] = trust, True, olympic_return
		for individual in population:
			if individual.get("individual_id") == record["individual_id"] and individual.get("is_gladiator", False):
				individual["trust"], individual["olympic_confirmed"], individual["olympic_return"] = trust, True, olympic_return
