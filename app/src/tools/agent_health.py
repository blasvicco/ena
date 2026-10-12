#!/usr/bin/env python
"""The internal quantities of an agent that diverged or froze in earlier studies, read during training without side
effects (SAC's sample of its buffer uses its own fixed generator, never the replicate's random stream)."""

# Libs imports
import numpy as np


def agent_health(agent) -> dict:
	"""The internal quantities that diverged or froze in earlier studies, for the agent's type."""
	health = {}
	if hasattr(agent, "log_alpha") and hasattr(agent, "critic_1") and getattr(agent.buffer, "size", 0) >= 256:
		indices = np.random.default_rng(0).integers(0, agent.buffer.size, size=256)
		health["alpha"] = float(np.exp(agent.log_alpha.numpy()))
		health["q_mean"] = float(np.mean(agent.critic_1([agent.buffer.states[indices], agent.buffer.actions[indices]]).numpy()))
	elif hasattr(agent, "log_alpha"):
		health["alpha"] = float(np.exp(agent.log_alpha.numpy()))
	if hasattr(agent, "_scheduled_learning_rate"):
		health["learning_rate"] = float(agent._scheduled_learning_rate())  # pylint: disable=protected-access
	if getattr(agent, "log_std", None) is not None:
		health["policy_std"] = float(np.mean(np.exp(agent.log_std.numpy())))
	return health
