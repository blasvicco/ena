# ENA — Evolutionary Neuroplastic Agent: Implementation Notes

This document describes the current implementation of the ENA agent as it stands in this package (`app/src/agents/ena/`). It is a code-accurate snapshot, not the paper's algorithm description — see `app/latex/` for the formal presentation.

Per [[project-ena-state]]: the `neural` selection driver / "Late Identity Binding" path in this package is the author's most recent rewrite and is considered unstable/not finished (commit `63bf827`, "Failed shit!"). The `heuristic` selection driver is the one exercised end-to-end by `cartpole_challenge.py` today (`selection_driver="heuristic"`).

## 1. Big picture

ENA is a population-based, non-gradient RL agent. Instead of backpropagating a single policy, it maintains a **population** of small neural-network "individuals," each carrying a scalar `trust` score. Every episode it **selects** one individual to act, scores it, updates its trust, and then **evolves** the population (mutation, crossover, replacement of the worst performers). Elite individuals that prove durably strong get **frozen** into a `HallOfFame` as "Gladiators," tagged with the environment "niche" they excel in, so they can be recalled later instead of re-discovered from scratch when a previously-seen environment recurs.

Every one of those mechanisms — Mutation, Crossover, Trust, Plasticity, Hall of Fame, environment detection, and individual selection — is implemented as a **skill**: an abstract driver interface plus one or more interchangeable driver implementations, resolved by name at construction time. `ena.py` itself only orchestrates *when* each skill gets called; it holds no algorithm logic of its own beyond a few core primitives every skill depends on (building a fresh individual, running one evaluation rollout).

## 2. Directory layout

```
agents/ena/
├── ena.py                  — ENAgent: composition root, orchestration only
├── memory.py                — AgentMemory: shared statistics store (not yet a skill, see §11)
├── skills/
│   ├── mutation/             main.py (facade) + driver/{abstract,gaussian}.py
│   ├── crossover/            main.py + driver/{abstract,uniform_mask}.py
│   ├── trust/                 main.py + driver/{abstract,zero,full,linear_ramp,quadratic}.py
│   ├── plasticity/             main.py + driver/{abstract,zero,full,linear_ramp,quadratic}.py
│   ├── hall_of_fame/            main.py + driver/{abstract,default,none}.py
│   ├── env_detection/            main.py + driver/{abstract,live,confirmation_window,dynamics_model,none}.py
│   └── selection/                 main.py + driver/{abstract,heuristic,neural,bpr}.py
└── tools/
    ├── curves.py              — zero/full/linear_ramp/quadratic formulas shared by trust & plasticity
    ├── genetic_similarity.py   — cosine similarity of two individuals' weight vectors
    ├── linear_dynamics.py      — LinearGaussianDynamics: ridge least-squares transition model (dynamics_model driver)
    ├── pm_cusum.py             — Predictive-Mixture CuSum change detector (dynamics_model driver)
    ├── inference.py            — normalize + forward-pass + action extraction (§10)
    ├── reward_normalizer.py    — percentile-based reward scaling
    └── weight_projection.py     — Universal Snapshot Projection (§9)
```

Every skill follows the same shape, borrowed from a sibling project's `modules/<name>/` driver-factory convention:

- `main.py` defines a **facade** class (e.g. `Mutation`, `HallOfFame`, `EnvironmentRecognizer`, `SelectionController`) whose `__init__(self, driver: str, **kwargs)` resolves `driver` via `importlib.import_module(f"agents.ena.skills.<skill>.driver.{driver}")` and constructs `module.Instance(**kwargs)`, storing it as `self._instance`. Every public method just delegates to `self._instance`. Unlike the reference project's version of this pattern, **driver instances are never cached across calls** — each carries per-agent state (EMA baselines, archive contents, REINFORCE buffers), so caching by driver name would leak state between agents or between skill instances.
- `driver/abstract.py` defines an `ADriver(abc.ABC)` with `@abstractmethod`s for the skill's public contract.
- `driver/<name>.py` defines `class Instance(ADriver)` — the concrete implementation. The driver's own file name *is* its identity; the class itself is always just called `Instance`.
- A driver named `none` (where present) is the skill's off-switch: every method is a no-op or returns an inert constant. This is what makes each skill "easy to enable or disable" — disabling is just selecting a different, already-existing driver, never a special code path in `ena.py`.

`ENAgent.__init__` builds one facade per skill and stores all seven under a single `self._skills` dict, keyed by name: `self._skills["env_detection"]`, `self._skills["selection"]`, `self._skills["hall_of_fame"]`, `self._skills["mutation"]`, `self._skills["crossover"]`, `self._skills["trust"]`, `self._skills["plasticity"]`. `ena.py` never holds a skill under any other attribute name.

## 3. `ENAgent` construction (`ena.py:32`)

Key constructor parameters and what they configure:

- `pop_size` — population size (e.g. `100` in `cartpole_challenge.py`'s active config).
- `physics` — list of env attribute names (`gravity`, `masspole`, `length`) the trainer re-reads from `env.unwrapped` each episode to build evaluation environments (`train()`, §7).
- `plasticity_algorithm` / `trust_algorithm` — driver name forwarded to the Plasticity/Trust skills (§6): `"zero"`, `"full"`, `"linear_ramp"`, or `"quadratic"`.
- `mutation_rate` / `mutation_noise` — forwarded to the Mutation skill's `gaussian` driver (§5).
- `hall_of_fame_driver` — `"default"` (the real archive) or `"none"` (disables specialist archiving/recall entirely — everything else, e.g. exploration/mutation, still runs); forwarded with `max_per_niche`/`max_gladiator_ratio` (§8).
- `selection_driver` — `"heuristic"` or `"neural"`, forwarded to the Selection skill (§4).
- `recognizer_strategy` — `"live"` (default), `"confirmation_window"`, `"dynamics_model"`, or `"none"` (disables environment detection — always resolves to a single universal niche); selects the environment-detection skill's driver (§3.1). `recognizer_threshold` / `recognizer_ema_alpha` / `genetic_threshold` / `genetic_bonus` / `recognizer_confirmation_window` / `recognizer_residual_tolerance` / `recognizer_novelty_margin` / `recognizer_cusum_false_alarm_steps` are forwarded to whichever driver is selected (driver-specific ones are harmless to pass regardless — every driver's constructor accepts `**_kwargs`).
- `env_max_steps` / `max_eval_steps` / `min_eval_steps` — `max_eval_steps` rescales truncated-episode evaluation scores to a full-episode equivalent (`__evaluate_individual_performance`); `min_eval_steps` is the floor of the plasticity-scaled evaluation depth used for candidate evaluation during `train()` (§7).
- `max_threads` — size of the `ThreadPoolExecutor` used to evaluate population candidates in parallel during `train()`.

On init: builds one throwaway env via `make_env()` to read `state_dim`/`action_dim`/`is_continuous`, constructs `AgentMemory`, then all seven `self._skills` facades in dependency order (environment detection first, since the selection facade's context dimension is derived from its `compute_full_vector`; Hall of Fame after `action_dim`/`state_dim` are known), and finally the initial population — `pop_size` individuals built by `__build_individual()`.

### Individual representation

Each population member is a plain `dict` (not a class) with keys:

```
age, individual_id (uuid4), model, fast_call (tf.function-wrapped model),
trust, is_gladiator, last_mu, last_var,
env_name / internal_env_id (set once inducted as a gladiator)
```

`last_mu`/`last_var` are a snapshot of the observation normalizer's mean/variance at the time the individual's first-layer weights were last aligned — see "Universal Snapshot Projection" (§9).

## 3.1. `env_detection` skill — detect & identify (`skills/env_detection/`)

The facade class is still named `EnvironmentRecognizer` (constructed as `EnvironmentRecognizer(driver, state_dim, **kwargs)`) and delegates every public call (`update`, `compute_physical_vector`, `compute_full_vector`, `detect_change`, `match`, `match_niche`, `current_niche_id`, `known_niche_ids`, `start_episode`, `observe`, `log_likelihoods`, `reset`) to `self._instance`; an unknown driver name raises `ValueError`. `cartpole_challenge.py`, `cartpole_ena_performance.py` and `cartpole_ena_compute.py` use `recognizer_strategy="dynamics_model"`.

**Vectors.** `compute_physical_vector` concatenates an observation-statistics mean, log(1+variance), and the same for the state-transition delta (`next_state - state`) — a "physics signature" meant to be independent of policy behavior. *Where* those statistics come from is exactly the axis the drivers below differ on. `compute_full_vector` adds two more scalars on top: normalized performance derivative and recent plasticity, giving a vector sensitive to *distress* as well as physics — used only for the Selection skill's neural driver context, never for shift detection (see `detect_change` below).

**`live`** (`driver/live.py`, the base class the other drivers inherit from) — sources `compute_physical_vector` from `memory.short_term_normalizer`/`memory.transition_normalizer`, i.e. whichever policy is actively playing right now. Cheap, but not policy-independent: a strong generalist's own rollout can make the wrong niche look physically confirmed regardless of the true physics.

**`confirmation_window`** (`driver/confirmation_window.py`) — no override of the physical vector; exists only to name "live rollout + a non-default confirmation window (3, instead of the base class's immediate-fire default of 1)" as its own explicit driver. The confirmation-window gating itself lives on the `live` driver's base class, parameterized by `confirmation_window`, so it composes with any `live`-based driver rather than being mutually exclusive with it.

**`dynamics_model`** (`driver/dynamics_model.py`, standalone — not a `live` subclass) — replaces the statistics fingerprint with **one `LinearGaussianDynamics` model per niche** (`tools/linear_dynamics.py`), predicting `next_state - state` from `[state, action]` (one-hot for discrete actions, `[action, 1]` for continuous) by closed-form ridge least squares, with a diagonal Gaussian residual (variance floored at `var_floor`, default `1e-6`). Evidence is every real transition of every episode, handed over by `ENAgent.act` through `observe(state, action, next_state, memory)`; conditioning on the action makes it policy-independent without a probe env. The lineage is RL-CD (da Silva et al., 2006): contexts are recognized by which model predicts best.
- **Detection**: PM-CuSum (`tools/pm_cusum.py`; Halme, Poor & Koivunen, 2026) runs per step on the current niche's standardized residuals divided by `residual_tolerance` (default 3). Two deviations from the paper's Gaussian example, both documented in the tool: each sliding window also estimates a variance (a wrong gain on a zero-mean state changes the residual variance, not its mean), and that variance is **one-sided** (never below the pre-change's) — a wrong model can only fit worse, and CartPole's exactly-predictable kinematic dimensions otherwise read as a variance *decrease*. The threshold is `log(cusum_false_alarm_steps)` (default `1e8`). `detect_change()` returns whether it fired during the episode. Calibrated on CartPole with two different controllers: zero false alarms over ~37k own-env steps; distinct physics caught in 2-3 steps, the near-identical Earth/Neptune/Brown-dwarf pairs in 7-152.
- **Identification**: mean per-step log-likelihood of the recent transitions (up to `recent_size`, default 500, since the last reset) under each mature niche model (≥ `min_samples`, default 100). A niche is recognized if it scores within `novelty_margin` (default 2) nats/step of a model freshly fitted on those same transitions, otherwise the physics are novel. `start_episode()` re-identifies the active niche every episode; `match()` returns `(niche_id or -1, exp(-gap))`.
- **Late Identity Binding** (`match_niche`): resolve the recent transitions to a niche and refine its model with the transitions not yet learned from, or provision a new niche fitted on them. Gladiators are resolved the same way, so foreign-physics success never touches the home niche's model (the same guarantee the `live` driver's gladiator branch gives, below). A niche provisioned from fewer than `min_samples` transitions keeps absorbing its regime's data instead of being compared against, so it can't be duplicated before it's judged.
- `log_likelihoods(memory)` returns `{niche_id: total log-likelihood of this episode's transitions}` — the dynamics signal the `bpr` selection driver consumes (§4).
- **Detect, then verify**: a PM-CuSum alarm is only a candidate. Once `min_verify_samples` (default 50) post-alarm transitions exist, `detect_change()` runs the identification test on them against the alarm's niche. Still explained → a coverage gap, not new physics (early in training the model is fitted on short failing episodes and a better policy visits new states): the transitions are folded into that niche's model and CuSum restarts. Not explained → shift. While an alarm is pending, neither `start_episode()` nor `match_niche()` identifies or provisions on the recent buffer, which may straddle the change.
- **Consistent tolerance**: every likelihood used for identification, verification and `log_likelihoods()` is evaluated with the residual variance widened by `residual_tolerance²` — the same widening detection uses. Otherwise identification is `residual_tolerance²` times stricter than detection and a niche's model rejects its own physics on states it saw little of (observed at test time). Nearly identical physics (CartPole Earth vs. Neptune) then share a niche.
- `reset()` (on a detected shift) drops the old regime's evidence but **keeps a confirmed shift's post-alarm transitions** as the new regime's first evidence, then restarts PM-CuSum; the active niche is `-1` until re-identified. Discarding them made the next induction re-attach the evidence-less regime to the niche just ruled out, whose model alarmed again: a shift on every episode. A current niche with fewer than `min_samples` transitions is never re-identified by `start_episode()`; it keeps growing through `match_niche()`.

`start_episode()`, `observe(...)` and `log_likelihoods(memory)` are part of every driver's interface; they are no-ops (and `{}`) on the `live`-based drivers and `none`.

**`none`** (`driver/none.py`) — the disable switch: `detect_change` always `False`, `match`/`match_niche` always resolve to niche `0`, `current_niche_id`/`known_niche_ids` report only niche `0`. Combined with the `none` Hall of Fame driver, this collapses specialist archiving to pure global elitism — gladiators get remembered, but never tied to which physics they were good for. `compute_physical_vector`/`compute_full_vector` still return zero vectors of the *same shape* the other drivers would (`4*state_dim` and `4*state_dim+2`), so `context_dim` stays consistent regardless of which driver is chosen.

**Detection — `detect_change(memory)`** (defined on the `live` driver, inherited by all others). Tracks an EMA baseline of `compute_physical_vector` (deliberately *not* `compute_full_vector` — the performance-derivative/plasticity terms are downstream of this method's own side effect via `clear_training_history()`, which forces the linear_ramp curve's warmup path to return `1.0` for several episodes, itself fed right back into the next comparison — using the full vector here previously created a self-sustaining feedback-loop oscillation of spurious firings even while training scores stayed at the maximum the whole time). No reading is taken until the driver's fingerprint is *ready* — the normalizers `compute_physical_vector` reads (memory's short-term/transition normalizers) have re-accumulated a full `max_count` window since their last reset. After a firing the baseline is cleared and re-established from the first ready reading of the new regime; using the vector read from just-reset normalizers as the baseline made their refill look like another shift, a self-sustaining loop of spurious firings. Each call computes cosine similarity between the live physical vector and the baseline; if similarity drops below `similarity_threshold` (default 0.90) for `confirmation_window` consecutive readings, a shift is declared, a `cooldown` (5 episodes) begins during which the baseline aggressively re-tracks the new regime, and `True` is returned. Otherwise the baseline is smoothly EMA-updated and `False` is returned. Called once per episode-end in `ENAgent.act` (both train and test); the result is also exposed via `ENAgent.last_env_shift_detected`.

**Identification — two paths:**
- `match(memory)` (test-time, cheap): compares the current physical vector against all stored niche prototypes by cosine similarity; returns the best match if above threshold, else `-1` (unknown).
- `match_niche(memory, individual, hall_of_fame, population)` — "**Late Identity Binding**," called only from `HallOfFame.update` when a winner is being considered for induction. If the individual is already a gladiator and the current physics match its home prototype (cosine similarity above the threshold), it resolves to its home niche and that prototype is EMA-updated. If the physics don't match home, the gladiator is resolved on physics alone (no genetic bonus, home niche excluded) to the best-matching other niche or a brand-new one, and the home prototype is left untouched — so a gladiator also succeeding under foreign physics can never drag its home niche's fingerprint there. Otherwise (a mutant/learner just won), physical cosine similarity to existing prototypes is computed, **boosted by a genetic-similarity bonus** to gladiators sharing that niche (weight-vector cosine similarity, scaled once above `genetic_threshold`), and either assigns the winner to the best-matching niche or provisions a brand-new niche ID.

`known_niche_ids` is exposed through to `ENAgent.known_niche_ids` — lets a caller (e.g. `metrics.py`'s chart-event classifier) tell a niche merely appearing for the first time in some history slice apart from one that's genuinely new.

## 4. `selection` skill — pluggable "who acts this episode" (`skills/selection/`)

The facade class is still named `SelectionController` (constructed as `SelectionController(driver, **kwargs)`) and delegates every call to `self._instance`. Both drivers implement the same interface: `reset_episode`, `select_individual`, `record_episode_result`, `evaluate`, `set_mode`, `record`, `update`, `on_step`, `on_learn_episode_end`. The last two encapsulate what used to be `ena.py`'s own reach into the driver's `_active`/`_active_niche`/`_buffer_size` state: `on_step(memory, env_detection, learn)` records a mid-episode REINFORCE context sample when the driver is currently exploring (a no-op for `heuristic`), and `on_learn_episode_end(memory, env_detection, env_max_steps)` backfills the episode's final reward into buffered contexts and triggers an `update()` once the buffer fills (also a no-op for `heuristic`) — both called from the facade's own `on_step`/`on_episode_end`, so `_active`/`_active_niche`/`_buffer_size` never need to leave the `neural` driver itself.

### `heuristic` (`driver/heuristic.py`, active in `cartpole_challenge.py` today)

- **Training** (`_select_train`): epsilon-greedy — with probability `exploration_rate` pick a random non-gladiator, otherwise pick `argmax(trust)` over the whole population.
- **Testing** (`_select_test`): a three-state machine (`EVALUATING → LOCKED`, or `DESPERATE` as a fallback):
  1. First tries `recognizer.match(memory)` — if the physics signature matches a known niche, immediately return that niche's best-trust gladiator.
  2. `EVALUATING`: `evaluate(population)` (called on entering test mode, or when `detect_change` fires) builds a shuffled queue of one gladiator per known niche; the driver plays through the queue one niche per episode, recording each episode's score (`record_episode_result`), and once exhausted locks onto the niche with the best observed score (`LOCKED`), or falls to `DESPERATE` if no gladiators exist.
  3. `LOCKED`: keeps returning the locked niche's best gladiator; if `performance_derivative` (from `AgentMemory`) crashes below `-100` over 20+ episodes, `record_episode_result` signals `force_reset=True`, which `ena.py` uses to clear reward history and re-trigger evaluation (the environment presumably shifted again but wasn't caught by the recognizer, or the gladiator degraded).
  4. `DESPERATE`: picks a random non-gladiator (no known-good option available).
- `record`/`update` are no-ops — this driver does no learning of its own; it's purely rule-based over trust/recognizer signals.

### `bpr` (`driver/bpr.py`) — Bayesian Policy Reuse (Rosman, Hawasly & Ramamoorthy, 2016)

Subclasses `heuristic`: training is the same epsilon-greedy trust selection; only test time differs, replacing the EVALUATING/LOCKED state machine.
- **Types and policies**: the types are the niches with a gladiator in the population, and every gladiator is a policy (rebuilt, with a uniform belief, by `evaluate(population)` — on entering test mode and on every detected shift). Ties go to the higher-trust gladiator.
- **Selection**: the policy with the highest expected utility under the belief (BPR's greedy criterion), utility = `score / env_max_steps`.
- **Belief update** (`record_episode_result`): Bayes' rule from the episode's utility under a Gaussian performance model `P(u | niche, policy)` (noise `return_noise`, default 0.2), times — when the `env_detection` driver provides them for every niche (`dynamics_model`) — the episode transitions' per-niche log-likelihoods. Without them (`live`, `confirmation_window`, `none`) it is return-only BPR, as in the original paper.
- **Performance model**, learned online (the paper builds it offline by playing every policy on every type): prior mean utility 1 for a gladiator in its own niche, an uncertain `cross_prior` (default 0.5) elsewhere, worth `prior_strength` pseudo-observations (default 1); every observed utility is credited to each niche in proportion to its posterior. `cross_prior` must not be pessimistic: at 0, an own-niche gladiator failing at test time is never abandoned, since no alternative can look better without first being tried.
- **Non-stationarity**: the paper assumes one fixed type per trial; here a Fixed-Share switching prior mixes `switch_rate` (default 0.05) of uniform mass back in after every update, so a changed environment can win the belief back. Never forces a reset.

### `neural` (`driver/neural.py`, defined, not exercised by the active experiment config)

A contextual-bandit approach: a small softmax MLP (`context_dim → hidden_sizes → num_niches`) maps the recognizer's `compute_full_vector` context to a niche-selection probability distribution, trained via **REINFORCE** (`update()`): sampled `(context, chosen_niche, episode_reward)` tuples are used to compute a mean-baseline advantage and a policy-gradient loss (`-mean(log π(a|s) · advantage)`), applied with Adam.

- Training mixes REINFORCE-driven exploration (`controller_episode_rate` probability, `_action_epsilon` for niche exploration within that) with a trust-argmax fallback the rest of the time; whichever path is taken, if it lands on a gladiator niche, the episode is marked `_active` so the driver's own `on_step`/`on_learn_episode_end` (§4) record mid-episode context snapshots into `AgentMemory.contexts/actions/rewards` for the eventual REINFORCE update.
- Testing uses the frozen (post-`set_mode(mode=False)`, which triggers one last `update()`) network's masked-softmax output, smoothed 50/50 with the previous step's probabilities, masked to only currently-known niche IDs.
- `_expand_to_niches(new_size)` grows the output layer when a new niche ID is discovered beyond the current output width, copying over old weights and biasing new-niche logits to `-10.0` (initially near-zero probability) rather than rebuilding from scratch.

## 5. `mutation` and `crossover` skills (`skills/mutation/`, `skills/crossover/`)

**Mutation** (`driver/gaussian.py`, the only driver today) — `apply(parent, target, plasticity, default_trust)`: clones `parent`'s weights into `target`, flips each weight with probability `mutation_rate`, and adds Gaussian noise scaled by `mutation_noise * plasticity` (more search effort → more noise). `target`'s trust is set to `default_trust` (computed by `ena.py` via `self._skills["trust"].reference(memory, env_max_steps)` and passed in, once per generation) and it inherits the parent's `last_mu`/`last_var` USP alignment state unchanged (the child's weights are still in the parent's normalization space until projected during evaluation).

**Crossover** (`driver/uniform_mask.py`, the only driver today) — `apply(parent_1, parent_2, target, obs_normalizer, max_eval_steps, default_trust)`: first aligns both parents to the current normalizer via the weight-projection tool (§9 — parents may have drifted apart in normalization space since they were last evaluated), then mixes their weights per-parameter via a random 50/50 mask. `target`'s `last_mu`/`last_var` are set to the *current* normalizer (since both parents were just projected to it, the child inherits that alignment directly, unlike Mutation's children).

Both are called from `ENAgent.__evolve_population()` (§8), which computes `plasticity`/`default_trust` once per generation and passes them to every mutation/crossover call that generation.

## 6. `trust` and `plasticity` skills, and the shared curve tool (`skills/trust/`, `skills/plasticity/`, `tools/curves.py`)

Trust and Plasticity are separate skills — one gates trust decay (how fast an individual's trust tracks its latest score), the other gates search effort (breadth and depth of candidate evaluation, mutation noise magnitude) — but their four drivers (`zero`, `full`, `linear_ramp`, `quadratic`) are each a thin wrapper over the **same shared formula** in `tools/curves.py`, since the underlying math is identical:

- `curves.zero(reward, memory)` → always `0.0`.
- `curves.full(reward, memory)` → always `1.0`.
- `curves.linear_ramp(reward, memory)` — percentile-scaled linear ramp: `0.1` at or above the 90th percentile of recent scores, rising linearly to `1.0` at `memory.min_seen` (the worst score seen this run); `1.0` during the first 10 episodes of history. (Previously named `fuzzylogic`; renamed because it is a single clipped linear function, not a fuzzy inference system — see `docs/PLAN_ena_fuzzy_logic.md`.) No hard floor below the low end: an earlier version clamped anything below the 10th percentile to a flat `1.0`, so once the agent was consistently near-perfect (a very narrow percentile band), a merely mediocre score and a catastrophic one both forced identical maximum-effort search. Interpolating all the way to `min_seen` keeps the value proportional across the whole observed range.
- `curves.quadratic(reward, memory)` — aggressive: `((max_seen - reward) / (max_seen - min_seen))²`, clamped to `[0.1, 1.0]`.

`ena.py` calls `self._skills["trust"].decay(reward, self.memory)` directly — no `ena.py`-side hook. `self._skills["plasticity"].compute(reward, self.memory, self.learn)` is likewise called directly; the facade itself (not `ena.py`, and not duplicated per driver) forces the result to `0.0` whenever `learn` is `False` — no exploratory search effort during evaluation. The CartPole studies use `"linear_ramp"` for Trust in every ENA row and for Plasticity wherever the Plasticity skill is on. The Trust facade also owns `reference(memory, env_max_steps)` (the population's dynamic reference trust — the same formula regardless of driver, used to seed fresh/reset individuals) and `reset_population(population, memory, env_max_steps, factor)` (blends every non-gladiator's trust toward that reference) — both are facade-level methods, not delegated to `self._instance`, since neither varies by driver.

Percentile-based reward scaling used by the dynamic reference trust (and available to any skill that needs it) is its own tool, `tools/reward_normalizer.py`'s `normalize_reward(reward, memory, env_max_steps)`: below 10 samples, scales against `env_max_steps` for warmup; otherwise linearly rescales relative to the 5th/95th percentile of `memory.reward_history`, deliberately not clipped to `[0, 1]` so an individual that breaks the current record still shows as strictly better.

## 7. `ENAgent.train()` — the per-episode evolutionary step (`ena.py:211`)

Invoked from the domain runner's `play()` loop whenever `agent.learn` is `True` and an episode just finished (`step_data["done"]`). Sequence:

1. Append `last_episode_score` to `reward_history`; update `max_seen`/`min_seen`.
2. `failure_signal = self._skills["trust"].decay(score, memory)` (§6).
3. Update the acting individual's `trust` via an EMA blend: `trust = (1-failure_signal)*trust + failure_signal*normalized_score` — a bigger failure signal makes trust track the latest score more aggressively (fast punishment, slow reward).
4. `plasticity = self._skills["plasticity"].compute(score, memory, learn)` (§6); appended to `memory.plasticity_history`. This value gates both how many other individuals get evaluated this round and mutation magnitude.
5. `hall_of_fame.update(...)` — see §8.
6. If `failure_signal > 0.1`, `self._skills["trust"].reset_population(population, memory, env_max_steps, factor=failure_signal)` blends every **non-gladiator** individual's trust toward the current dynamic reference trust, proportional to the failure signal — a poor result partially "resets confidence" across the population, but gladiators are excluded (they're already protected from replacement in `__evolve_population`, and an unrelated exploratory failure — e.g. the epsilon-greedy random non-gladiator pick in the heuristic selection driver — shouldn't erode an established gladiator's trust either).
7. **Parallel candidate evaluation**: `num_to_eval = max(2, int(pop_size * plasticity))` other individuals are sampled and evaluated in parallel via a `ThreadPoolExecutor(max_workers=max_threads)`, each against a freshly built env cloned from the live env's current physics params (`env_params` read via `physics` attribute list) and a shared `gen_seed` so all candidates face the same conditions this generation. Each evaluation is capped at `eval_step_cap = min_eval_steps + (max_eval_steps - min_eval_steps) * plasticity` steps, applied uniformly (gladiators included, no special-casing) — confident/converged generations (low plasticity) evaluate candidates cheaply, uncertain ones get the full depth for a reliable signal. Their trust scores are updated with the same EMA rule as step 3.
8. `__evolve_population()` — see §8.

More search effort (higher plasticity) → more candidates evaluated, evaluated more deeply, and larger mutation noise; a confidently-performing agent (low plasticity) evaluates as few as 2 others, down to `min_eval_steps` deep, and mutates gently.

## 8. `hall_of_fame` skill — Gladiator archive (`skills/hall_of_fame/`)

Facade class `HallOfFame`, constructed as `HallOfFame(driver, **kwargs)`. Called once per completed training episode from `ENAgent.train` via `self._skills["hall_of_fame"].update(winner, trust, population, memory, env_detection, obs_normalizer)`.

**`default`** (`driver/default.py`) — a staged pipeline, each stage returning early on a decision:

1. **Memory decay** (`__memory_decay`) — every record's `induction_z_score` decays by `MEMORY_DECAY_RATE = 0.999` each call, so long-untouched records become progressively easier to displace.
2. **Performance gate** (`__performance_gate`) — computes a Z-score of the winner's trust against the *current population's* trust distribution (`std` floored at `0.1` to avoid blow-ups on uniform populations); requires `z_score >= 1.5 * (1 - 0.5 * current_plasticity)` **or** `trust >= 0.5` to proceed at all. Higher current plasticity (more active search) lowers the bar — the agent is more willing to induct while still exploring.
3. **Niche resolution** — `recognizer.match_niche(...)` (Late Identity Binding, §3.1) assigns/confirms the niche ID for this winner before any HoF decision uses it.
4. **Continual improvement** (`__continual_improvement`) — if the winner's `individual_id` is *already* in the archive: same niche → update in place only if strictly better; different niche → the gladiator also performs well under this niche's physics, so it's **cloned** as a new record for that niche (if it has budget), labeled with the current environment's name, rather than moved; the original record and live gladiator keep their niche and label. If that niche already holds a genetic twin (typically an earlier clone of the same gladiator), the twin is refreshed instead when the new trust is higher, so repeated foreign wins don't fill the niche with copies of one individual.

A gladiator's `env_name` is set once, at induction, from the `env_name` argument `ena.py`'s `train()` passes to `HallOfFame.update` (the environment the winner just succeeded in); it is never rewritten afterwards, so the label always means "the environment this specialist was archived for", not "the last environment it played".
5. **Genetic diversity / niche protection** (`__genetic_diversity`) — within the same niche, if the winner's weight-vector cosine similarity to an existing record exceeds `SIMILARITY_THRESHOLD = 0.95` ("strategic twin"), it replaces that twin only if strictly better trust, and the pipeline stops either way (no double-counting near-duplicates).
6. **Open slot induction** (`__slot_induction`) — if the niche has fewer than `max_per_niche` records, save the winner directly as a new record.
7. **Eco-niche protection** (`__eco_niche_protection`) — last resort: if the niche is full, evict the weakest-trust record in that niche if the winner beats it.

`__evict(individual_id, population)` (used by steps 5 and 7 above) removes the archive entry **and** clears the corresponding live population slot's `is_gladiator`/`internal_env_id`/`env_name` tags. Without the population-side clearing, an evicted record left a "zombie gladiator" behind: invisible in `niches_*.txt` but still competing in the selection skill's `evaluate()` and still protected from replacement in `__evolve_population()`.

`__save` deep-copies the winner's model into a fresh `brain(...)` instance (so the archived weights are independent of the live, still-mutating individual) and immediately flags the *live* population dict as `is_gladiator=True` with its assigned `internal_env_id` — gladiators are then protected from replacement in `__evolve_population` (§9's sibling, population replacement in `ena.py`).

`inject_gladiators(population, pop_size, obs_normalizer, restore_trust)` (called by `ENAgent.load_gladiators()`, e.g. before `cartpole_challenge.py`'s testing phase) round-robins one gladiator per known niche per round into the first `floor(pop_size * max_gladiator_ratio)` population slots, until the budget is filled or niches are exhausted.

`save_niche_details(output_dir, agent_name)` writes a `niches_<agent_name>.txt` summary (niche ID, env name, gladiator count) — this is one of the run-artifact files `CLAUDE.md` flags as generated, not source.

**`none`** (`driver/none.py`) — the disable switch: `update`/`inject_gladiators` are no-ops, `__bool__` is always `False`, `save_niche_details` writes the same "No gladiators inducted" summary the default driver writes when its archive is empty.

## 9. Universal Snapshot Projection tool (`tools/weight_projection.py`)

Because the observation normalizer's mean/variance keep drifting as more states are seen, an individual's first-layer weights — implicitly tuned to whatever normalization was in effect when it was last aligned — can become stale relative to the *current* normalizer. `project_individual_weights(individual, obs_normalizer, max_eval_steps)` analytically re-scales its first Dense layer (weight columns scaled by `std_new/std_old` clipped to `[0.1, 10.0]`, bias shifted by `(mean_new - mean_old)/std_old · W`) so its behavior is preserved under the new normalization without any retraining. Applies uniformly to gladiators too, even though their *inputs* are normalized differently (see `tools/inference.py`'s `is_gladiator` branch, called from `act()`, which uses their own frozen `last_mu`/`last_var` instead of the live normalizer): `__evaluate_individual_performance` always normalizes an individual's input with the live `obs_normalizer` regardless of `is_gladiator` (it does not go through `infer_action` — a deliberate difference, not an oversight), so leaving a gladiator's *weights* un-projected would evaluate them with inputs aligned to one normalizer but weights aligned to another. Skipped entirely until the normalizer has seen at least `2 * max_eval_steps` samples (avoids projecting off noisy early statistics). Called from three places: `ENAgent.act()` (for the acting individual), `__evaluate_individual_performance` (for every evaluated candidate), and the Crossover skill's `uniform_mask` driver (for both parents before mixing their weights).

## 10. `ENAgent.act()` step-by-step (`ena.py:129`)

Called once per environment step by the domain runner's `play()` loop. `act()` itself is a short sequence of delegated steps; the mechanism-specific logic lives behind three private helpers and one tool function:

1. `__start_episode(env)` (first step of an episode only): resets the selection skill's episode state, zeroes the step counter, and calls `env_detection.start_episode()` (re-identification for `dynamics_model`, a no-op otherwise, §3.1).
2. `__select_individual()` (first step of an episode only): asks the selection facade to pick `best_individual_idx` (train vs. test path per `self.learn`), then projects that individual's weights to the current normalizer (§9).
3. `tools/inference.py`'s `infer_action(individual, state, memory, is_continuous)`: normalizes `state` — gladiators use their own frozen `last_mu`/`last_var`; everyone else uses the live `obs_normalizer` (updated just before this call, and fed to `env_detection.update` too) — and runs the chosen individual's `fast_call` (a `tf.function`-wrapped forward pass) to get an action (argmax for discrete, clipped raw output for continuous).
4. Step the environment; `memory.record_step(state, next_state, reward)` updates the transition-dynamics normalizer (physics signature) and running episode-reward bookkeeping in one call, and `env_detection.observe(state, action, next_state, memory)` hands the real transition to the driver (the `dynamics_model` driver's evidence and PM-CuSum input; a no-op otherwise).
5. `self._skills["selection"].on_step(memory, env_detection, learn)` — delegates the mid-episode REINFORCE recording decision entirely to the active driver (§4); a no-op unless the `neural` driver is both selected and currently exploring this episode.
6. `__end_episode()`, only on episode end (`done`):
   - Records `last_episode_score`, then `selection.on_episode_end(memory, learn, population, env_detection, env_max_steps)` — in learn mode, delegates reward-backfill and REINFORCE `update()` triggering entirely to the driver (§4); in test mode, updates `reward_history`/`max_seen`/`min_seen` and lets `record_episode_result` decide whether to force a re-evaluation reset (clearing reward history and re-triggering evaluation if so).
   - `env_detection.evaluate(memory, learn)` — runs `detect_change`; on a detected shift, resets the recognizer's memory-owned short-term statistics (`memory.reset_recognizer_statistics()`), the driver's own state (`reset()` — drops `dynamics_model`'s old-regime transition buffers, a no-op for the other drivers), the observation normalizer, and (learn mode only) `memory.clear_training_history()`. Returns whether a shift fired.
   - If a shift fired and the agent is in test mode, `ena.py` itself calls `selection.evaluate(population)` — the one cross-skill call `env_detection` deliberately doesn't own itself, since it would otherwise need to know about the selection skill.
   - Reset per-episode counters.

## 11. Supporting state — `AgentMemory` (`memory.py`)

Centralizes everything the other components read/write, decoupled from `ENAgent` itself. **Not currently a skill** — it's a shared data store other skills read/write (reward history, normalizers, REINFORCE buffers), not a swappable *behavior* the way Mutation or Trust are. Making it one with a `default` driver would be mechanical, but there's no second implementation motivating the abstraction yet; do it only if a real need shows up (e.g. a bounded-memory or decay-weighted variant), not speculatively.

- `record_step(state, next_state, reward)` — called once per `act()` step; consolidates `update_transition` (physics-signature tracking) and running episode-reward bookkeeping (`current_episode_reward`, `step_rewards`) into one call.
- `obs_normalizer` — a `_RunningNormalizer` (Welford's online mean/variance, capped at `max_eval_steps * 10` samples, used to normalize policy-network inputs).
- `short_term_normalizer` / `transition_normalizer` — separate, faster-adapting normalizers (`max_count = 1/ema_alpha`) feeding the `live`/`confirmation_window` env-detection drivers' physics-signature vectors; reset independently of `obs_normalizer` on a detected environment change.
- `reward_history` (bounded deque, `history_size`), `plasticity_history` (unbounded list), `max_seen`/`min_seen`, `step_rewards` (bounded deque of length 100).
- `contexts`/`actions`/`rewards` — the `neural` selection driver's REINFORCE experience buffers.
- Derived properties: `performance_derivative` (short-term-mean minus a slightly-older baseline mean over `reward_history`, used for stagnation/regression detection), `recent_plasticity`, `live_reward_rate`.

`_RunningNormalizer` (`agents/tools/normalizers.py` — shared across *all* agents, not ENA-specific, hence living one level up from `agents/ena/tools/`) implements a Welford-style running mean/variance with an EMA-style mixing rate once `max_count` is reached (so statistics keep adapting rather than freezing once "full"), clipping normalized output to `±clip` (default 10.0). `_IdentityNormalizer` is a no-op sibling for discrete-observation environments (not used by CartPole/HalfCheetah, which are both continuous-observation).

## 12. Known rough edges (as of this snapshot - do not touch)

- The `neural` selection driver / "Late Identity Binding" is present and wired but not the default (`selection_driver="heuristic"` in the active `cartpole_challenge.py` config) and is flagged in project memory as unstable — treat it as in-progress, not validated.
- `physics` defaults to a **mutable list** default argument (`physics=["gravity", "masspole", "length"]` in `ENAgent.__init__`) — harmless today since it's never mutated in place, but worth flagging if future edits start appending to `self.physics`.
- With the `heuristic` selection driver, test-time selection still underuses correct niche identification (its `EVALUATING` cycle locks on one noisy episode's raw score) — measured: `dynamics_model` + `heuristic` identified every test phase correctly but scored 415/500 mean vs. 490 with `bpr` (docs/PLAN_ena_cartpole_fixes.md, Stage 12/13 results). Prefer `bpr`.
- Niche fragmentation (13-22 niches for 3 training environments) was a property of the removed statistics-fingerprint drivers; `dynamics_model` yields one niche per training environment (3-5 over 3 seeds). The `hall_of_fame_driver`/`recognizer_strategy="none"` switches exist to let a factorial ablation isolate how much each mechanism (env-detection, Hall of Fame, Plasticity) contributes.
- §13's benchmark tool exists specifically to test hypotheses about any of the above fast, without a full experiment run.

## 13. Fast iteration without a full experiment (`tools/benchmark_ena.py`)

`cartpole_challenge.py`'s full experiment (`pop_size=100`, 1500 training + 1000 testing episodes) takes on the order of an hour — too slow for iterating on a single hypothesis about ENA's internal bookkeeping. `tools/benchmark_ena.py` runs the same `ENAgent`/Hall-of-Fame/environment-detection control flow — via `cartpole_challenge.py`'s own `play()` loop, unmodified — against `tools/synthetic_cartpole.py`'s `SyntheticCartPoleEnv`: a from-scratch re-implementation of CartPole-v1's governing physics equations (identical constants, termination condition, and reward semantics to gymnasium's `CartPoleEnv`), which skips `gym.make()`'s registry lookup and wrapper-stack construction — the dominant per-episode cost given ENA constructs a fresh env per candidate evaluation (`train()`'s `thread_eval_task`). A small run (`pop_size≈20`, ~100 episodes/phase) finishes in well under a minute and produces a real `niches_*.txt`-style summary, directly comparable in kind (not necessarily magnitude) to a full run's.

Since a single seed's dynamics are not reproducible even with `random`/`np.random`/`tf.random` all fixed (TensorFlow's own internal op-level parallelism introduces floating-point non-associativity that compounds over hundreds of episodes into completely different outcomes), the tool runs several seeds and reports mean±std per environment, and can run a paired significance check between two saved runs on matching seeds.

Usage: `python tools/benchmark_ena.py --save before.json`, apply a candidate code change, then `python tools/benchmark_ena.py --compare before.json --save after.json` to print a before/after diff of niche/gladiator counts per true environment plus wall time. This is a fast first check that a change moved the mechanism in the intended direction — not a substitute for a full experiment run before calling a fix validated, since the population/episode scale-down makes promotion counts here noisier and not paper-reportable.
