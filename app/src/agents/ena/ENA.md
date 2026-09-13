# ENA — Evolutionary Neuroplastic Agent: Implementation Notes

This document describes the current implementation of the ENA agent as it stands in this package (`app/src/agents/ena/`). It is a code-accurate snapshot, not the paper's algorithm description — see `app/latex/` for the formal presentation.

Per [[project-ena-state]]: the `NeuralSelectionDriver` / "Late Identity Binding" path in this package is the author's most recent rewrite and is considered unstable/not finished (commit `63bf827`, "Failed shit!"). The `HeuristicSelectionDriver` path is the one exercised end-to-end by `cartpole_challenge.py` today (`selection_driver="heuristic"`).

## 1. Big picture

ENA is a population-based, non-gradient RL agent. Instead of backpropagating a single policy, it maintains a **population** of small neural-network "individuals," each carrying a scalar `trust` score. Every episode it **selects** one individual to act, scores it, updates its trust, and then **evolves** the population (mutation, crossover, replacement of the worst performers). Elite individuals that prove durably strong get **frozen** into a `HallOfFame` as "Gladiators," tagged with the environment "niche" they excel in, so they can be recalled later instead of re-discovered from scratch when a previously-seen environment recurs.

Five files implement this:

| File | Class | Role |
|---|---|---|
| `ena.py` | `ENAgent` | Composition root — owns the population, drives selection/training/evolution, implements the `AAgent` contract. |
| `memory.py` | `AgentMemory` | All historical/statistical state (reward history, running normalizers, per-episode counters, REINFORCE buffers), decoupled from `ENAgent`. |
| `recognizer.py` | `EnvironmentRecognizer` | Unsupervised environment-shift **detection** and niche **identification**, via cosine similarity over an observation/transition-statistics vector. |
| `hall_of_fame.py` | `HallOfFame` | Archive of frozen elite specialists ("Gladiators"), gated by a staged induction pipeline. |
| `selection_controller.py` | `SelectionController` (+ `HeuristicSelectionDriver`, `NeuralSelectionDriver`) | Pluggable strategy for picking which population member acts each episode. |

`ENAgent` implements the shared `AAgent` contract (`agents/tools/abstract.py`): `act(env, state)`, `train(**kwargs)`, `set_learning(mode, output_dir=None)`. Its `brain` and `make_env` factories are injected at construction time (strategy pattern, per `STANDARDS.md`).

## 2. `ENAgent` construction (`ena.py:26`)

Key constructor parameters and what they configure:

- `pop_size` — population size (e.g. `100` in `cartpole_challenge.py`'s active config).
- `physics` — list of env attribute names (`gravity`, `masspole`, `length`) the trainer re-reads from `env.unwrapped` each episode to build evaluation environments (`train()`, §6).
- `plasticity_algorithm` / `trust_algorithm` — string names dispatched via `getattr(self, f"_ENAgent__{name}")` to private methods `__zero`, `__full`, `__fuzzylogic`, `__quadratic` (§7). This is explicitly called out in `STANDARDS.md` as the **legacy** dispatch idiom — new variants should be Strategy classes instead, following `selection_controller.py`'s pattern.
- `mutation_rate` / `mutation_noise` — per-weight probability and magnitude of Gaussian mutation noise (§8).
- `max_per_niche` / `max_gladiator_ratio` — forwarded to `HallOfFame` (§5).
- `selection_driver` — `"heuristic"` or `"neural"`, forwarded to `SelectionController` (§4).
- `recognizer_strategy` — `"live"` (default), `"confirmation_window"`, or `"neutral_probe"`; selects the `EnvironmentRecognizer` strategy (§3). `recognizer_threshold` / `recognizer_ema_alpha` / `genetic_threshold` / `genetic_bonus` / `recognizer_confirmation_window` / `recognizer_probe_steps` are forwarded to whichever strategy is selected (the latter two are strategy-specific but harmless to pass regardless — every strategy's constructor accepts `**_kwargs`).
- `env_max_steps` / `max_eval_steps` / `min_eval_steps` — `max_eval_steps` rescales truncated-episode evaluation scores to a full-episode equivalent (`__evaluate_individual_performance`); `min_eval_steps` is the floor of the plasticity-scaled evaluation depth used for candidate evaluation during `train()` (§6).
- `max_threads` — size of the `ThreadPoolExecutor` used to evaluate population candidates in parallel during `train()`.

On init: builds one throwaway env via `make_env()` to read `state_dim`/`action_dim`/`is_continuous`, constructs `AgentMemory`, `EnvironmentRecognizer`, `SelectionController` (context dimension derived from `recognizer.compute_full_vector`), `HallOfFame`, and the initial population — `pop_size` individuals built by `__build_individual()`.

### Individual representation

Each population member is a plain `dict` (not a class) with keys:

```
age, individual_id (uuid4), model, fast_call (tf.function-wrapped model),
trust, is_gladiator, last_mu, last_var,
env_name / internal_env_id (set once inducted as a gladiator)
```

`last_mu`/`last_var` are a snapshot of the observation normalizer's mean/variance at the time the individual's first-layer weights were last aligned — see "Universal Snapshot Projection" (§9).

## 3. `EnvironmentRecognizer` — detect & identify (`recognizer.py`)

`EnvironmentRecognizer` is a thin facade (same pattern as `SelectionController`, §4): constructed as `EnvironmentRecognizer(strategy, state_dim, **kwargs)`, it looks up `strategy` in a `strategies = {"live": LiveRolloutRecognizer, "confirmation_window": ConfirmationWindowRecognizer, "neutral_probe": NeutralProbeRecognizer}` dict and delegates every public call (`update_state`, `compute_physical_vector`, `compute_full_vector`, `detect_change`, `match`, `match_niche`, `current_niche_id`, `known_niche_ids`, `probe`, `reset`) to that instance; an unknown name raises `ValueError`. `cartpole_challenge.py`'s active config uses `recognizer_strategy="neutral_probe"`.

**Vectors.** `compute_physical_vector` concatenates an observation-statistics mean, log(1+variance), and the same for the state-transition delta (`next_state - state`) — a "physics signature" meant to be independent of policy behavior. *Where* those statistics come from is exactly the axis the three strategies below differ on. `compute_full_vector` adds two more scalars on top: normalized performance derivative and recent plasticity, giving a vector sensitive to *distress* as well as physics — used only for `SelectionController`'s context, never for shift detection (see `detect_change` below).

**`LiveRolloutRecognizer`** (the `"live"` strategy, and the base class the other two inherit from) — sources `compute_physical_vector` from `memory.short_term_normalizer`/`memory.transition_normalizer`, i.e. whichever policy is actively playing right now. Cheap, but not policy-independent: a strong generalist's own rollout can make the wrong niche look physically confirmed regardless of the true physics.

**`ConfirmationWindowRecognizer(LiveRolloutRecognizer)`** — no override of the physical vector; exists only to name "live rollout + a non-default confirmation window (3, instead of the base class's immediate-fire default of 1)" as its own explicit strategy. The confirmation-window gating itself lives on the `LiveRolloutRecognizer` base class, parameterized by `confirmation_window`, so it composes with *any* strategy (including `neutral_probe`) rather than being mutually exclusive with it.

**`NeutralProbeRecognizer(LiveRolloutRecognizer)`** — overrides `probe(make_env, physics_env, physics_keys)` (called once per episode, from `ENAgent.act`'s first step, §10) to spin up a short-lived throwaway env built with the same physics as the real one, take `probe_steps` (default 10) random actions (`action_space.sample()` — works uniformly for CartPole's discrete space and HalfCheetah's continuous `Box`), and feed the results into its *own* `_probe_obs_normalizer`/`_probe_transition_normalizer` — separate from `memory`'s. `compute_physical_vector` is overridden to build from these probe normalizers instead of the live rollout, decoupling the physics fingerprint from whichever policy is currently acting. The probe env is closed immediately after and never touches the scored episode's trajectory or step budget. Also overrides `reset()` to flush its own probe normalizers on a detected shift — these live on the strategy instance, not `AgentMemory`, so `memory.reset_recognizer_statistics()` alone never reached them; without this, every environment switch left the probe blending stale pre-switch physics into its readings for up to `probe_steps * 20` samples, right during the window niche identification needs a clean reading most.

**Detection — `detect_change(memory)`** (defined on `LiveRolloutRecognizer`, inherited by both subclasses). Tracks an EMA baseline of `compute_physical_vector` (deliberately *not* `compute_full_vector` — the performance-derivative/plasticity terms are downstream of this method's own side effect via `clear_training_history()`, which forces `__fuzzylogic`'s warmup path to return plasticity `1.0` for several episodes, itself fed right back into the next comparison — using the full vector here previously created a self-sustaining feedback-loop oscillation of spurious firings even while training scores stayed at the maximum the whole time). Each call computes cosine similarity between the live physical vector and the baseline; if similarity drops below `similarity_threshold` (default 0.90) for `confirmation_window` consecutive readings (default 1, i.e. immediate), a shift is declared, a `cooldown` (5 episodes) begins during which the baseline aggressively re-tracks the new regime, and `True` is returned. Otherwise the baseline is smoothly EMA-updated and `False` is returned. Called once per episode-end in `ENAgent.act` (both train and test); the result is also exposed via `ENAgent.last_env_shift_detected`.

**Identification — two paths:**
- `match(memory)` (test-time, cheap): compares the current physical vector against all stored niche prototypes by cosine similarity; returns the best match if above threshold, else `-1` (unknown).
- `match_niche(memory, individual, hall_of_fame, population)` — "**Late Identity Binding**," called only from `HallOfFame.update` when a winner is being considered for induction. If the individual is already a gladiator, its registered niche is trusted outright and its prototype is EMA-updated. Otherwise (a mutant/learner just won), physical cosine similarity to existing prototypes is computed, **boosted by a genetic-similarity bonus** to gladiators sharing that niche (weight-vector cosine similarity via `_calculate_genetic_similarity`, scaled once above `genetic_threshold`), and either assigns the winner to the best-matching niche or provisions a brand-new niche ID.

`known_niche_ids` (the set of niche IDs with a stored physical prototype) is exposed through to `ENAgent.known_niche_ids` — lets a caller (e.g. `metrics.py`'s chart-event classifier) tell a niche merely appearing for the first time in some history slice apart from one that's genuinely new.

## 4. `SelectionController` — pluggable "who acts this episode" (`selection_controller.py`)

`SelectionController` is a thin facade: constructed with a driver name (`"heuristic"` or `"neural"`), it instantiates and delegates every call to the corresponding driver instance. Both drivers implement the same informal interface: `reset_episode`, `select_individual`, `record_episode_result`, `trigger_evaluation`, `set_mode`, `record`, `update`, plus the `_active`/`_active_niche`/`_buffer_size` properties `ena.py`'s `act()` reads directly.

### `HeuristicSelectionDriver` (active in `cartpole_challenge.py` today)

- **Training** (`_select_train`): epsilon-greedy — with probability `exploration_rate` pick a random non-gladiator, otherwise pick `argmax(trust)` over the whole population.
- **Testing** (`_select_test`): a three-state machine (`EVALUATING → LOCKED`, or `DESPERATE` as a fallback):
  1. First tries `recognizer.match(memory)` — if the physics signature matches a known niche, immediately return that niche's best-trust gladiator.
  2. `EVALUATING`: `trigger_evaluation(population)` (called on entering test mode, or when `detect_change` fires) builds a shuffled queue of one gladiator per known niche; the driver plays through the queue one niche per episode, recording each episode's score (`record_episode_result`), and once exhausted locks onto the niche with the best observed score (`LOCKED`), or falls to `DESPERATE` if no gladiators exist.
  3. `LOCKED`: keeps returning the locked niche's best gladiator; if `performance_derivative` (from `AgentMemory`) crashes below `-100` over 20+ episodes, `record_episode_result` signals `force_reset=True`, which `ena.py` uses to clear reward history and re-trigger evaluation (the environment presumably shifted again but wasn't caught by the recognizer, or the gladiator degraded).
  4. `DESPERATE`: picks a random non-gladiator (no known-good option available).
- `record`/`update` are no-ops — this driver does no learning of its own; it's purely rule-based over trust/recognizer signals.

### `NeuralSelectionDriver` (defined, not exercised by the active experiment config)

A contextual-bandit approach: a small softmax MLP (`context_dim → hidden_sizes → num_niches`) maps the recognizer's `compute_full_vector` context to a niche-selection probability distribution, trained via **REINFORCE** (`update()`): sampled `(context, chosen_niche, episode_reward)` tuples are used to compute a mean-baseline advantage and a policy-gradient loss (`-mean(log π(a|s) · advantage)`), applied with Adam.

- Training mixes REINFORCE-driven exploration (`controller_episode_rate` probability, `_action_epsilon` for niche exploration within that) with a trust-argmax fallback the rest of the time; whichever path is taken, if it lands on a gladiator niche, the episode is marked `_active` so `ena.py` records mid-episode context snapshots into `AgentMemory.contexts/actions/rewards` for the eventual REINFORCE update.
- Testing uses the frozen (post-`set_mode(mode=False)`, which triggers one last `update()`) network's masked-softmax output, smoothed 50/50 with the previous step's probabilities, masked to only currently-known niche IDs.
- `_expand_to_niches(new_size)` grows the output layer when a new niche ID is discovered beyond the current output width, copying over old weights and biasing new-niche logits to `-10.0` (initially near-zero probability) rather than rebuilding from scratch.

## 5. `HallOfFame` — Gladiator archive (`hall_of_fame.py`)

Called once per completed training episode from `ENAgent.train` via `hall_of_fame.update(winner, trust, population, memory, recognizer, obs_normalizer)`. A staged pipeline, each stage returning early on a decision:

1. **Memory decay** (`__memory_decay`) — every record's `induction_z_score` decays by `MEMORY_DECAY_RATE = 0.999` each call, so long-untouched records become progressively easier to displace.
2. **Performance gate** (`__performance_gate`) — computes a Z-score of the winner's trust against the *current population's* trust distribution (`std` floored at `0.1` to avoid blow-ups on uniform populations); requires `z_score >= 1.5 * (1 - 0.5 * current_plasticity)` **or** `trust >= 0.5` to proceed at all. Higher current plasticity (more active search) lowers the bar — the agent is more willing to induct while still exploring.
3. **Niche resolution** — `recognizer.match_niche(...)` (Late Identity Binding, §3) assigns/confirms the niche ID for this winner before any HoF decision uses it.
4. **Continual improvement** (`__continual_improvement`) — if the winner's `individual_id` is *already* in the archive: same niche → update in place only if strictly better; different niche → the individual has since evolved elsewhere, so it's **cloned** as a new record for the new niche (if that niche has budget), rather than moved.
5. **Genetic diversity / niche protection** (`__genetic_diversity`) — within the same niche, if the winner's weight-vector cosine similarity to an existing record exceeds `SIMILARITY_THRESHOLD = 0.95` ("strategic twin"), it replaces that twin only if strictly better trust, and the pipeline stops either way (no double-counting near-duplicates).
6. **Open slot induction** (`__slot_induction`) — if the niche has fewer than `max_per_niche` records, save the winner directly as a new record.
7. **Eco-niche protection** (`__eco_niche_protection`) — last resort: if the niche is full, evict the weakest-trust record in that niche if the winner beats it.

`__evict(individual_id, population)` (used by steps 5 and 7 above) removes the archive entry **and** clears the corresponding live population slot's `is_gladiator`/`internal_env_id`/`env_name` tags. Without the population-side clearing, an evicted record left a "zombie gladiator" behind: invisible in `niches_*.txt` but still competing in `SelectionController.trigger_evaluation()` and still protected from replacement in `__evolve_population()`.

`__save` deep-copies the winner's model into a fresh `brain(...)` instance (so the archived weights are independent of the live, still-mutating individual) and immediately flags the *live* population dict as `is_gladiator=True` with its assigned `internal_env_id` — gladiators are then protected from replacement in `__evolve_population` (§8).

`inject_gladiators(population, pop_size, obs_normalizer, restore_trust)` (called by `ENAgent.load_gladiators()`, e.g. before `cartpole_challenge.py`'s testing phase) round-robins one gladiator per known niche per round into the first `floor(pop_size * max_gladiator_ratio)` population slots, until the budget is filled or niches are exhausted.

`save_niche_details(output_dir, agent_name)` writes a `niches_<agent_name>.txt` summary (niche ID, env name, gladiator count) — this is one of the run-artifact files `CLAUDE.md` flags as generated, not source.

## 6. `ENAgent.train()` — the per-episode evolutionary step (`ena.py:275`)

Invoked from the domain runner's `play()` loop whenever `agent.learn` is `True` and an episode just finished (`step_data["done"]`). Sequence:

1. Append `last_episode_score` to `reward_history`; update `max_seen`/`min_seen`.
2. `failure_signal = __get_trust_decay(score)` — dispatches to the configured `trust_algorithm`.
3. Update the acting individual's `trust` via an EMA blend: `trust = (1-failure_signal)*trust + failure_signal*normalized_score` — a bigger failure signal makes trust track the latest score more aggressively (fast punishment, slow reward).
4. `plasticity = __get_plasticity(score)` (dispatches to `plasticity_algorithm`) — appended to `memory.plasticity_history`; this value gates both how many other individuals get evaluated this round and mutation magnitude.
5. `hall_of_fame.update(...)` — see §5.
6. If `failure_signal > 0.1`, `__reset_trust(factor=failure_signal)` blends every **non-gladiator** individual's trust toward the current dynamic reference trust, proportional to the failure signal — a poor result partially "resets confidence" across the population, but gladiators are excluded (they're already protected from replacement in `__evolve_population`, and an unrelated exploratory failure — e.g. the epsilon-greedy random non-gladiator pick in `_select_train` — shouldn't erode an established gladiator's trust either).
7. **Parallel candidate evaluation**: `num_to_eval = max(2, int(pop_size * plasticity))` other individuals are sampled and evaluated in parallel via a `ThreadPoolExecutor(max_workers=max_threads)`, each against a freshly built env cloned from the live env's current physics params (`env_params` read via `physics` attribute list) and a shared `gen_seed` so all candidates face the same conditions this generation. Each evaluation is capped at `eval_step_cap = min_eval_steps + (max_eval_steps - min_eval_steps) * plasticity` steps, applied uniformly (gladiators included, no special-casing) — confident/converged generations (low plasticity) evaluate candidates cheaply, uncertain ones get the full depth for a reliable signal. Their trust scores are updated with the same EMA rule as step 3.
8. `__evolve_population()` — see §8.

More search effort (higher plasticity) → more candidates evaluated, evaluated more deeply, and larger mutation noise; a confidently-performing agent (low plasticity) evaluates as few as 2 others, down to `min_eval_steps` deep, and mutates gently.

## 7. Plasticity / trust-decay algorithm dispatch (`ena.py:514-661`)

Both `plasticity_algorithm` and `trust_algorithm` select one of these private methods by name (legacy string-dispatch, see §2 caveat):

- `__zero(_)` → `0.0` (no decay / no plasticity).
- `__full(_)` → `1.0` (always maximal).
- `__fuzzylogic(reward)` — linear interpolation between the 90th percentile of `reward_history` (→ `0.1`, minimal search) and `memory.min_seen`, the worst score observed this run (→ `1.0`, maximal search); no hard floor below the low end, so plasticity stays strictly monotonic across the whole observed range. (A prior version clamped anything below the 10th percentile to a flat `1.0`; once the agent was consistently near-perfect — a very narrow percentile band — a merely mediocre score and a catastrophic one forced identical maximum-effort search.)
- `__quadratic(reward)` — `((max_seen - reward) / (max_seen - min_seen))²`, clamped to `[0.1, 1.0]` — aggressive, benchmark-relative.

`__get_plasticity` forces `"zero"` whenever `self.learn` is `False` (no search during evaluation). Unknown algorithm names fall back to `1.0` (plasticity) / `0.0` (trust decay) via the `AttributeError` catch — `STANDARDS.md` flags broad exception catching as against convention generally, but this specific narrow `AttributeError` catch is the sanctioned dispatch-fallback pattern it calls out as acceptable.

## 8. `__evolve_population()` — selection/replacement (`ena.py:459`)

Runs every training episode after evaluation:

1. Rank all individuals by `(trust, age)` descending.
2. `replace_count` defaults to `80%` of `pop_size` (clamped to leave at least 2 survivors); the elite pool is the top `20%`.
3. Candidates to replace are the worst `replace_count` individuals, **minus any gladiators** (gladiators are never replaced here).
4. Of the actual replaceable slots: `40%` become **mutants** (`__mutate`: clone an elite's weights, then flip each weight with probability `mutation_rate`, adding Gaussian noise scaled by `mutation_noise` *and* current plasticity — more search effort → more noise), `40%` become **crossovers** (`__crossover`: two random elites' weights mixed per-parameter via a random 50/50 mask, after both parents are first weight-space-aligned via Universal Snapshot Projection, §9), and the remainder are rebuilt **fresh** (`__build_individual` re-orthogonal-initializes in place, reusing the existing Keras model/`tf.function` buffers — the docstring notes this is deliberate, to avoid TF session/graph bloat from repeatedly creating new models).

## 9. Universal Snapshot Projection (`__project_individual_weights`, `ena.py:606`)

Because the observation normalizer's mean/variance keep drifting as more states are seen, an individual's first-layer weights — implicitly tuned to whatever normalization was in effect when it was last aligned — can become stale relative to the *current* normalizer. Before an individual is evaluated or acts, `__project_individual_weights` analytically re-scales its first Dense layer (weight columns scaled by `std_new/std_old` clipped to `[0.1, 10.0]`, bias shifted by `(mean_new - mean_old)/std_old · W`) so its behavior is preserved under the new normalization without any retraining. Skipped for gladiators (frozen, evaluated under their own stored `last_mu`/`last_var` instead — see `act()`'s `is_gladiator` branch) and skipped entirely until the normalizer has seen at least `2 * max_eval_steps` samples (avoids projecting off noisy early statistics).

## 10. `ENAgent.act()` step-by-step (`ena.py:110`)

Called once per environment step by the domain runner's `play()` loop:

1. On the first step of an episode, reset the selection controller's episode state and the step counter, and call `recognizer.probe(make_env, env, physics)` (a no-op unless the `neutral_probe` strategy is active, §3).
2. On that first step only, ask the `SelectionController` to pick `best_individual_idx` (train vs. test path per `self.learn`), then project that individual's weights to the current normalizer (§9).
3. Normalize the current `state` — gladiators use their own frozen `last_mu`/`last_var`; everyone else uses the live `obs_normalizer`. Run the chosen individual's `fast_call` (a `tf.function`-wrapped forward pass) to get an action (argmax for discrete, clipped raw output for continuous).
4. Step the environment; update `memory.update_transition` (physics signature) and running reward bookkeeping.
5. If training and the neural driver is `_active`, periodically (`step == 1` or every 50 steps) record a `(context, niche, reward=0.0)` tuple — reward is a placeholder, backfilled once the episode's final score is known.
6. On episode end (`done`):
   - **Train + neural driver active**: backfill the placeholder rewards for this episode's recorded contexts with the normalized final score, record one final context, and trigger a REINFORCE `update()` once the buffer reaches `_buffer_size`.
   - **Test mode**: update `reward_history`/`max_seen`/`min_seen`, then let the heuristic driver's `record_episode_result` decide if a forced re-evaluation is needed.
   - Always: run `recognizer.detect_change` — on a detected shift, reset the recognizer's memory-owned short-term statistics (`memory.reset_recognizer_statistics()`), the strategy's own state (`recognizer.reset()` — flushes `neutral_probe`'s separate probe normalizers, a no-op for the other two strategies) and the observation normalizer, and (test mode only) re-trigger the evaluation state machine.
   - Test mode only: if `record_episode_result` signaled `force_reset`, clear reward history and re-trigger evaluation too.
   - Reset per-episode counters.

## 11. Supporting state — `AgentMemory` (`memory.py`)

Centralizes everything the other components read/write, decoupled from `ENAgent` itself (per `STANDARDS.md`'s "reuse `AgentMemory`" guidance):

- `obs_normalizer` — a `_RunningNormalizer` (Welford's online mean/variance, capped at `max_eval_steps * 10` samples, used to normalize policy-network inputs).
- `short_term_normalizer` / `transition_normalizer` — separate, faster-adapting normalizers (`max_count = 1/ema_alpha`) feeding the recognizer's physics-signature vectors; reset independently of `obs_normalizer` on a detected environment change.
- `reward_history` (bounded deque, `history_size`), `plasticity_history` (unbounded list), `max_seen`/`min_seen`, `step_rewards` (bounded deque of length 100).
- `contexts`/`actions`/`rewards` — the `NeuralSelectionDriver`'s REINFORCE experience buffers.
- Derived properties: `performance_derivative` (short-term-mean minus a slightly-older baseline mean over `reward_history`, used for stagnation/regression detection), `recent_plasticity`, `live_reward_rate`.

`_RunningNormalizer` (`agents/tools/normalizers.py`) implements a Welford-style running mean/variance with an EMA-style mixing rate once `max_count` is reached (so statistics keep adapting rather than freezing once "full"), clipping normalized output to `±clip` (default 10.0). `_IdentityNormalizer` is a no-op sibling for discrete-observation environments (not used by CartPole/HalfCheetah, which are both continuous-observation).

## 12. Known rough edges (as of this snapshot - do not touch)

- `NeuralSelectionDriver`/"Late Identity Binding" is present and wired but not the default (`selection_driver="heuristic"` in the active `cartpole_challenge.py` config) and is flagged in project memory as unstable — treat it as in-progress, not validated.
- Plasticity/trust-algorithm dispatch (`__get_plasticity`, `__get_trust_decay`) uses the legacy `getattr(self, f"_ENAgent__{name}")` string-dispatch idiom `STANDARDS.md` explicitly says not to extend — any new algorithm variant should instead become a Strategy class analogous to `HeuristicSelectionDriver`/`NeuralSelectionDriver`.
- `physics` defaults to a **mutable list** default argument (`physics=["gravity", "masspole", "length"]` in `ENAgent.__init__`) — harmless today since it's never mutated in place, but worth flagging if future edits start appending to `self.physics`.
- Test-time specialist selection can still lock onto one gladiator across true environments it isn't actually specialized for, even with `neutral_probe` active — the `EVALUATING` cycle's raw-score-with-confirmation winner logic (`SelectionController._select_test`/`record_episode_result`, §4) and how often `detect_change` re-opens that cycle at test time are both implicated; open, not yet root-caused to a single fix.
- Gladiator/niche-count distribution across true environments has been observed uneven in practice — one true environment accumulating far more niches (and, since the archive budget is per-niche-ID via `max_per_niche`, far more total gladiators) than others despite every phase getting an identical training budget. Candidate mechanisms, none yet confirmed as the/a root cause: uneven niche fragmentation across environments (§3's niche-provisioning threshold), `HallOfFame.__continual_improvement`'s cross-niche fork path running before `__slot_induction` so a foreign-niche generalist can claim a fresh niche's slot ahead of an actual specialist, and `__performance_gate`'s Z-score being computed over the *whole* population rather than scoped to the current niche. §13's benchmark tool exists specifically to test these hypotheses fast, without a full experiment run.

## 13. Fast iteration without a full experiment (`tools/benchmark_ena.py`)

`cartpole_challenge.py`'s full experiment (`pop_size=100`, 1500 training + 1000 testing episodes) takes on the order of an hour — too slow for iterating on a single hypothesis about ENA's internal bookkeeping. `tools/benchmark_ena.py` runs the same `ENAgent`/`HallOfFame`/`EnvironmentRecognizer` control flow — via `cartpole_challenge.py`'s own `play()` loop, unmodified — against `tools/synthetic_cartpole_challenge.py`'s `SyntheticCartPoleEnv`: a from-scratch re-implementation of CartPole-v1's governing physics equations (identical constants, termination condition, and reward semantics to gymnasium's `CartPoleEnv`), which skips `gym.make()`'s registry lookup and wrapper-stack construction — the dominant per-episode cost given ENA constructs a fresh env per candidate evaluation (`train()`'s `thread_eval_task`) and per episode-start probe (`NeutralProbeRecognizer.probe`). A small run (`pop_size≈20`, ~100 episodes/phase) finishes in well under a minute and produces a real `niches_*.txt`-style summary, directly comparable in kind (not necessarily magnitude) to a full run's.

Usage: `python tools/benchmark_ena.py --save before.json`, apply a candidate code change, then `python tools/benchmark_ena.py --compare before.json --save after.json` to print a before/after diff of niche/gladiator counts per true environment plus wall time. This is a fast first check that a change moved the mechanism in the intended direction — not a substitute for a full experiment run before calling a fix validated, since the population/episode scale-down makes promotion counts here noisier and not paper-reportable.
