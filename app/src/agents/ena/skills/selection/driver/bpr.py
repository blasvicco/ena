"""Bayesian Policy Reuse selection driver: belief over niches, play the best expected gladiator."""

# General imports
import math

# Libs imports
import numpy as np

# App imports
from agents.ena.skills.selection.driver.heuristic import Instance as HeuristicInstance


# Algorithm: Bayesian Policy Reuse (BPR)
# Reference: Rosman, Hawasly & Ramamoorthy, "Bayesian Policy Reuse", Machine Learning 104 (2016)
# https://doi.org/10.1007/s10994-016-5547-y
# pylint: disable=too-many-instance-attributes
class Instance(HeuristicInstance):
	"""Test time: Bayesian belief over which niche is active; training: the heuristic driver's epsilon-greedy trust selection."""

	# Types are the niches with a gladiator in the population; the policy library is every
	# gladiator in the population. After every test episode the belief is updated by Bayes'
	# rule from two signals: the episode's utility under a Gaussian performance model
	# P(u | niche, policy), and — when the env_detection driver has per-niche dynamics models
	# (dynamics_model) — the episode transitions' log-likelihood under each niche. The next
	# policy is the one with the highest expected utility under the belief (BPR's greedy
	# criterion). The paper assumes a single fixed type per trial; for a non-stationary stream a
	# Fixed-Share switching prior mixes switch_rate of uniform mass back in after every update,
	# so a changed environment can win the belief back instead of being locked out.
	#
	# Utility is score / env_max_steps. The paper builds the performance model offline, by playing
	# every policy on every type; here it is learned online. Its prior says a gladiator reaches
	# utility 1 in its own niche and an uncertain cross_prior elsewhere, worth prior_strength
	# observations; each observed utility is credited to every niche in proportion to its
	# posterior. cross_prior must not be pessimistic: with 0 an own-niche gladiator that fails at
	# test time (observed: a Mars specialist scoring 16/500 in Mars) is never abandoned, since no
	# other policy can look better without first being tried. Ties go to the higher-trust gladiator.

	# pylint: disable=too-many-arguments,too-many-positional-arguments
	def __init__(
		self, exploration_rate=0.1, env_max_steps=1000.0, switch_rate=0.05, return_noise=0.2, prior_strength=1.0, cross_prior=0.5, **_kwargs
	):
		super().__init__(exploration_rate=exploration_rate)
		self._env_max_steps = float(env_max_steps)
		self._switch_rate = switch_rate
		self._return_noise = return_noise
		self._prior_strength = prior_strength
		self._cross_prior = cross_prior

		self._niches = []  # niche ids with a gladiator in the population (the types)
		self._policies = []  # [(population index, individual_id, home niche, trust)] — every gladiator
		self._belief = np.zeros(0)
		self._utility_sum = {}  # (type, individual_id) -> belief-weighted sum of observed utilities
		self._utility_weight = {}  # (type, individual_id) -> belief-weighted number of observations
		self._played = None

	@property
	def belief(self) -> dict:
		"""{niche_id: probability that niche is the active environment}."""
		return dict(zip(self._niches, self._belief.tolist()))

	def select_individual(self, population, recognizer, memory, learn):
		"""Training: epsilon-greedy trust; testing: the gladiator with the highest expected utility under the belief."""
		if learn:
			return self._select_train(population)
		if not self._policies:
			self._played = None
			return self._select_desperate(population)
		self._played = max(self._policies, key=lambda policy: (self._expected_utility(policy), policy[3]))
		return self._played[0]

	def record_episode_result(self, score, memory, recognizer=None):
		"""Bayes-update the belief from this episode's utility (and dynamics likelihoods, if any); never forces a reset."""
		if self._played is None:
			return False
		utility = score / self._env_max_steps
		log_evidence = np.array([self._log_utility_likelihood(utility, niche_id, self._played) for niche_id in self._niches])
		dynamics = recognizer.log_likelihoods(memory) if recognizer is not None else {}
		if all(niche_id in dynamics for niche_id in self._niches):
			log_evidence += np.array([dynamics[niche_id] for niche_id in self._niches])

		log_posterior = np.log(np.maximum(self._belief, 1e-300)) + log_evidence
		posterior = np.exp(log_posterior - np.max(log_posterior))
		posterior /= posterior.sum()

		for niche_id, weight in zip(self._niches, posterior):
			key = (niche_id, self._played[1])
			self._utility_sum[key] = self._utility_sum.get(key, 0.0) + weight * utility
			self._utility_weight[key] = self._utility_weight.get(key, 0.0) + weight

		self._belief = (1.0 - self._switch_rate) * posterior + self._switch_rate / len(self._niches)
		return False

	def evaluate(self, population):
		"""(Re)build the types and policy library from the population's gladiators and reset the belief to uniform."""
		self._policies = [
			(index, ind.get("individual_id", str(index)), ind["internal_env_id"], ind.get("trust", 0.0))
			for index, ind in enumerate(population)
			if ind.get("is_gladiator", False) and ind.get("internal_env_id", -1) >= 0
		]
		self._niches = sorted({policy[2] for policy in self._policies})
		self._belief = np.full(len(self._niches), 1.0 / len(self._niches)) if self._niches else np.zeros(0)
		self._played = None

	# ---------------------------------------------------------------
	# Private
	# ---------------------------------------------------------------

	def _mean_utility(self, niche_id, policy) -> float:
		"""Posterior mean utility of the policy (a gladiator) when niche_id is active."""
		_index, individual_id, home_niche_id, _trust = policy
		prior_mean = 1.0 if niche_id == home_niche_id else self._cross_prior
		key = (niche_id, individual_id)
		weight = self._utility_weight.get(key, 0.0)
		return (self._prior_strength * prior_mean + self._utility_sum.get(key, 0.0)) / (self._prior_strength + weight)

	def _expected_utility(self, policy) -> float:
		"""Belief-weighted expected utility of playing the policy."""
		return float(sum(prob * self._mean_utility(niche_id, policy) for niche_id, prob in zip(self._niches, self._belief)))

	def _log_utility_likelihood(self, utility, niche_id, policy) -> float:
		"""log N(utility; mean utility of the policy in that niche, return_noise^2)."""
		deviation = (utility - self._mean_utility(niche_id, policy)) / self._return_noise
		return -0.5 * deviation * deviation - math.log(self._return_noise * math.sqrt(2.0 * math.pi))
