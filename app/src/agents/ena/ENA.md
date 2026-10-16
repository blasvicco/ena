# ENA — Evolutionary Neuroplastic Agent: Implementation Notes

This document describes the current implementation of the ENA agent as it stands in this package (`app/src/agents/ena/`). It is a code-accurate snapshot, not the paper's algorithm description — see `app/latex/` for the formal presentation.

Specialist selection uses Bayesian Policy Reuse (`bpr`); `niche_match` and `best_trust` are simpler test-time drivers used only by the selection study (`cartpole_ena_selection.py`). Environment detection uses per-niche dynamics models (`dynamics_model`).

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
│   ├── failure_signal/        driver/{abstract,zero,full,linear_ramp,quadratic}.py — shared by trust & plasticity
│   ├── trust/                 main.py (facade; drivers from failure_signal)
│   ├── plasticity/             main.py + driver/fuzzylogic.py (and failure_signal's drivers)
│   ├── hall_of_fame/            main.py + driver/{abstract,default,olympic,none}.py
│   ├── env_detection/            main.py + driver/{abstract,dynamics_model,chirp,none}.py
│   └── selection/                 main.py + driver/{abstract,bpr,niche_match,best_trust}.py
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

- `main.py` defines a **facade** class (e.g. `Mutation`, `HallOfFame`, `EnvironmentRecognizer`, `SelectionController`) whose `__init__(self, driver: str, **kwargs)` builds its driver with the shared `load_driver(skill, driver, ...)` (`tools/driver_loader.py`): it imports `agents.ena.skills.<skill>.driver.<driver>` (or the first of the skill's `packages` that defines it), constructs `module.Instance(**kwargs)`, and stores it as `self._instance`. A name no package defines raises `ValueError("Unknown <skill> driver: ...")`; a driver module that exists but fails to import raises its own error. Every public method just delegates to `self._instance`. Unlike the reference project's version of this pattern, **driver instances are never cached across calls** — each carries per-agent state (archive contents, dynamics models, the selection belief), so caching by driver name would leak state between agents or between skill instances.
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
- `olympics_interval`, `olympics_episodes`, `olympics_qualifier_seats` — the `olympic` Hall of Fame driver's Olympiads: environment steps between them, full episodes per competitor, provisional seats per niche beyond `max_per_niche` (§8.1). The other drivers ignore them.
- `hall_of_fame_driver` — `"default"` (the real archive) or `"none"` (disables specialist archiving/recall entirely — everything else, e.g. exploration/mutation, still runs); forwarded with `max_per_niche`/`max_gladiator_ratio` (§8).
- `selection_driver` — `"bpr"` (default), `"niche_match"` or `"best_trust"`, forwarded to the Selection skill with `exploration_rate` and `return_scale` (§4).
- `recognizer_strategy` — `"dynamics_model"` (default) or `"none"` (disables environment detection — always resolves to a single universal niche); selects the environment-detection driver (§3.1). `recognizer_residual_tolerance` / `recognizer_novelty_margin` / `recognizer_cusum_false_alarm_steps` are forwarded to it.
- `max_episode_steps` / `return_scale` — the episode length and the return that counts as solving the task, read from the environment's spec when not given (CartPole-v1: 500 / 475; HalfCheetah-v5: 1000 / 4800); construction fails if neither is available. A truncated candidate evaluation is extrapolated to `max_episode_steps` (`__evaluate_individual_performance`); `return_scale` is BPR's utility scale (§4) and the warmup scale of return normalization (§6).
- `max_eval_steps` / `min_eval_steps` — the range of the plasticity-scaled candidate-evaluation depth during `train()` (§7).
- `obs_normalizer_max_count` — how many recent samples the observation normalizer averages over (§9); default `max_eval_steps * 10` (HalfCheetah: 2,000 at its `max_eval_steps` of 200). The statistics follow recent states, so a gladiator's frozen snapshot can go stale; the `olympic` driver stores the statistics each gladiator's weights are aligned to (`aligned_statistics`, §8.1), so a short window is safe there.
- `max_threads` — size of the `ThreadPoolExecutor` used to evaluate population candidates in parallel during `train()`.

On init: builds one throwaway env via `make_env()` to read `state_dim`/`action_dim`/`is_continuous`, constructs `AgentMemory`, then all seven `self._skills` facades (Hall of Fame after `action_dim`/`state_dim` are known), and finally the initial population — `pop_size` individuals built by `__build_individual()`.

### Individual representation

Each population member is a plain `dict` (not a class) with keys:

```
age, individual_id (uuid4), model, fast_call (tf.function-wrapped model),
trust, is_gladiator, last_mu, last_var,
env_name / internal_env_id (set once inducted as a gladiator)
```

`last_mu`/`last_var` are a snapshot of the observation normalizer's mean/variance at the time the individual's first-layer weights were last aligned — see "Universal Snapshot Projection" (§9).

## 3.1. `env_detection` skill — detect & identify (`skills/env_detection/`)

The facade `EnvironmentRecognizer(driver, state_dim, **kwargs)` delegates `detect_change`, `match`, `match_niche`, `current_niche_id`, `known_niche_ids`, `start_episode`, `observe`, `log_likelihoods` and `reset` to its driver; an unknown driver name raises `ValueError`. Its own `evaluate(memory, learn)` runs `detect_change` once per episode end and, on a shift, calls the driver's `reset()`, resets the observation normalizer, and (training only) clears the return history (`memory.clear_training_history()`).

**`dynamics_model`** (`driver/dynamics_model.py`) — **one `LinearGaussianDynamics` model per niche** (`tools/linear_dynamics.py`), predicting `next_state - state` from `[state, action]` (one-hot for discrete actions, `[action, 1]` for continuous) by closed-form ridge least squares, with a diagonal Gaussian residual (variance floored at `var_floor`, default `1e-6`). Evidence is every real transition of every episode, handed over by `ENAgent.act` through `observe(state, action, next_state, memory)`; conditioning on the action makes it policy-independent. The lineage is RL-CD (da Silva et al., 2006): contexts are recognized by which model predicts best.
- **Detection**: PM-CuSum (`tools/pm_cusum.py`; Halme, Poor & Koivunen, 2026) runs per step on the current niche's standardized residuals divided by `residual_tolerance` (default 3). Two deviations from the paper's Gaussian example, both documented in the tool: each sliding window also estimates a variance (a wrong gain on a zero-mean state changes the residual variance, not its mean), and that variance is **one-sided** (never below the pre-change's). The threshold is `log(cusum_false_alarm_steps)` (default `1e8`). `detect_change()` returns whether a shift was confirmed during the episode. Calibrated on CartPole with two different controllers: zero false alarms over ~37k own-env steps; distinct physics caught in 2-3 steps, the near-identical Earth/Neptune/Brown-dwarf pairs in 7-152.
- **Identification**: mean per-step log-likelihood of the recent transitions (up to `recent_size`, default 500, since the last reset) under each mature niche model (≥ `min_samples`, default 100). A niche is recognized if it scores within `novelty_margin` (default 2) nats/step of a model freshly fitted on those same transitions, otherwise the physics are novel. `start_episode()` re-identifies the active niche every episode; `match()` returns `(niche_id or -1, exp(-gap))`.
- **Late Identity Binding** (`match_niche`): resolve the recent transitions to a niche and refine its model with the transitions not yet learned from, or provision a new niche fitted on them. Gladiators are resolved the same way, so foreign-physics success never touches the home niche's model. A niche provisioned from fewer than `min_samples` transitions keeps absorbing its regime's data instead of being compared against, so it can't be duplicated before it's judged.
- `log_likelihoods(memory)` returns `{niche_id: total log-likelihood of this episode's transitions}` — the dynamics signal the `bpr` selection driver consumes (§4).
- **Detect, then verify**: a PM-CuSum alarm is only a candidate. Once `min_verify_samples` (default 50) post-alarm transitions exist, `detect_change(memory, learn)` runs the identification test on them against the alarm's niche. Still explained → a coverage gap, not new physics: in training the transitions are folded into that niche's model, and CuSum restarts. At test time (`learn=False`) the models are never updated: the test is frozen. Not explained → shift. While an alarm is pending, neither `start_episode()` nor `match_niche()` identifies or provisions on the recent buffer, which may straddle the change.
- **Consistent tolerance**: every likelihood used for identification, verification and `log_likelihoods()` is evaluated with the residual variance widened by `residual_tolerance²` — the same widening detection uses. Nearly identical physics (CartPole Earth vs. Neptune) then share a niche.
- `reset()` (on a detected shift) drops the old regime's evidence but **keeps a confirmed shift's post-alarm transitions** as the new regime's first evidence, then restarts PM-CuSum; the active niche is `-1` until re-identified.

**`chirp`** (`driver/chirp.py`) — detection from a fixed probing excitation, independent of the acting individual. It is HalfCheetah's detector (`half_cheetah.ENA_DETECTION`); CartPole keeps `dynamics_model`.
- **Why:** `dynamics_model`'s evidence is the acting individual's transitions. In a contact-rich body every new behavior reaches unseen contact states, which read as new physics: 27 changes and 23 niches for 3 environments on the detection benchmark (`tools/detection_benchmark.py`), against 2–3 with one fixed gait.
- **Chirp:** `chirp_actions`, a fixed `steps`-long action sequence from `tools/chirp_signals.py` (`excitation`, `excitation_params`; default a 64-step Mexican-hat wavelet train, the wavelet's waveform at dyadic scales).
  - ENAgent plays it at every episode end (`__run_chirp`) in the real environment, from a fresh reset, restoring the environment's random stream afterwards so episode starts are unchanged.
  - Its steps count as candidate steps; the observations go to `observe_chirp`.
- **Features:** `tools/chirp_features.py` on the response (observations minus the start observation): default Haar wavelet packets at level 3 (dyadic, non-overlapping, 8 bands) with one log-energy per channel and band; also `dwt`, `swt` (overlapping), `cwt` and raw.
- **Detection and identification:**
  - Each niche keeps its chirps' mean feature vector; the per-feature variance is pooled across niches.
  - A change is confirmed when the mean of the last `window` chirps is more than `margin` from the current niche: the RMS over features of the deviation in standard errors, about 1 under the same physics.
  - Its most recent chirps then identify a judged niche (at least `min_chirps`) within the margin, or create a new niche in training (never at test).
- **The rest:** `log_likelihoods` gives each judged niche's Gaussian log-likelihood of the latest chirp (BPR's dynamics evidence). `observe`/`start_episode` are no-ops.
- **Settings:** one dict, ENAgent's `recognizer_chirp`, over the driver's `DEFAULTS`, key by key (unknown keys rejected). A given `features` or `excitation_params` dict *replaces* the default whole: missing keys fall back to the feature or signal function's own defaults, so a configuration is fully determined by what it passes. Changing `DEFAULTS` cannot change an ablation result. The default (Mexican-hat wavelet train, Haar wavelet packets at level 3) comes from `tools/detection_ablation.py`.
- **Interface:** the facade and `ADriver` expose `chirp_actions` (None for passive drivers) and `observe_chirp`.

**`none`** (`driver/none.py`) — the disable switch: `detect_change` always `False`, `match`/`match_niche` always resolve to niche `0`, `current_niche_id`/`known_niche_ids` report only niche `0`. Combined with the `none` Hall of Fame driver, this collapses specialist archiving to pure global elitism.

`known_niche_ids` is exposed through to `ENAgent.known_niche_ids` — lets a caller (e.g. `metrics.py`'s chart-event classifier) tell a niche merely appearing for the first time in some history slice apart from one that's genuinely new.

## 4. `selection` skill — who acts this episode (`skills/selection/`)

The facade `SelectionController(driver, **kwargs)` delegates `select_individual`, `record_episode_result` and `evaluate` to its driver; `on_episode_end(memory, learn, env_detection)` passes each test episode's result to `record_episode_result` (training episodes don't update the selector).

**Training selection is shared** (`driver/abstract.py`): epsilon-greedy — with probability `exploration_rate` a random non-gladiator, otherwise `argmax(trust)` over the whole population. Drivers differ only in `_select_test`, so a seeded training run is identical whichever driver is configured. The base class also provides the no-gladiator fallback (a random non-gladiator) and a no-op `record_episode_result` for drivers that don't learn at test.

### `bpr` (`driver/bpr.py`) — Bayesian Policy Reuse (Rosman, Hawasly & Ramamoorthy, 2016)

- **Types and policies** (test time): the types are the niches with a gladiator in the population, and every gladiator is a policy (rebuilt, with a uniform belief, by `evaluate(population)` — on entering test mode and on every detected shift). Ties go to the higher-trust gladiator. With no gladiator at all, a random non-gladiator plays.
- **Selection**: the policy with the highest expected utility under the belief (BPR's greedy criterion), utility = `score / return_scale`, so solving the task reads as utility 1 (the own-niche prior).
- **Belief update** (`record_episode_result`): Bayes' rule from the episode's utility under a Gaussian performance model `P(u | niche, policy)` (noise `return_noise`, default 0.2), times — when the `env_detection` driver provides them for every niche (`dynamics_model`) — the episode transitions' per-niche log-likelihoods. Without them (`none`) it is return-only BPR, as in the original paper.
- **Performance model**, learned online (the paper builds it offline by playing every policy on every type): prior mean utility 1 for a gladiator in its own niche, an uncertain `cross_prior` (default 0.5) elsewhere, worth `prior_strength` pseudo-observations (default 1); every observed utility is credited to each niche in proportion to its posterior. `cross_prior` must not be pessimistic: at 0, an own-niche gladiator failing at test time is never abandoned, since no alternative can look better without first being tried.
- **Non-stationarity**: the paper assumes one fixed type per trial; here a Fixed-Share switching prior mixes `switch_rate` (default 0.05) of uniform mass back in after every update, so a changed environment can win the belief back.

### `niche_match` (`driver/niche_match.py`) — detector-only retrieval

At test, plays the highest-trust gladiator of `recognizer.current_niche_id`, read at every episode; when that niche has no gladiator (start of a phase, after a confirmed shift until the detector has enough evidence, novel physics), the highest-trust gladiator overall. No belief and no learning from test returns: a gladiator failing in its own niche keeps playing while the detector stays there. It is BPR without the return evidence, for the selection ablation.

### `best_trust` (`driver/best_trust.py`) — retrieval floor

At test, always plays the highest-trust gladiator, ignoring both the detector and the returns. Against `niche_match` it shows whether the niche index matters at retrieval.

## 5. `mutation` and `crossover` skills (`skills/mutation/`, `skills/crossover/`)

**Mutation** (`driver/gaussian.py`, the only driver today) — `apply(parent, target, plasticity, default_trust)`: clones `parent`'s weights into `target`, flips each weight with probability `mutation_rate`, and adds Gaussian noise scaled by `mutation_noise * plasticity` (more search effort → more noise). `target`'s trust is set to `default_trust` (computed by `ena.py` via `self._skills["trust"].reference(memory, return_scale)` and passed in, once per generation) and it inherits the parent's `last_mu`/`last_var` USP alignment state unchanged (the child's weights are still in the parent's normalization space until projected during evaluation).

**Crossover** (`driver/uniform_mask.py`, the only driver today) — `apply(parent_1, parent_2, target, obs_normalizer, max_eval_steps, default_trust)`: first aligns both parents to the current normalizer via the weight-projection tool (§9 — parents may have drifted apart in normalization space since they were last evaluated), then mixes their weights per-parameter via a random 50/50 mask. `target`'s `last_mu`/`last_var` are set to the *current* normalizer (since both parents were just projected to it, the child inherits that alignment directly, unlike Mutation's children).

Both are called from `ENAgent.__evolve_population()` (§8), which computes `plasticity`/`default_trust` once per generation and passes them to every mutation/crossover call that generation.

## 6. `trust` and `plasticity` skills, and their shared failure-signal drivers (`skills/trust/`, `skills/plasticity/`, `skills/failure_signal/`, `tools/curves.py`)

Trust and Plasticity are separate skills — one gates trust decay (how fast an individual's trust tracks its latest score), the other gates search effort (breadth and depth of candidate evaluation, mutation noise magnitude) — but they share one set of drivers, `skills/failure_signal/driver/` (`zero`, `full`, `linear_ramp`, `quadratic`), each a thin wrapper whose `value(reward, memory)` returns the **same shared formula** in `tools/curves.py`, since the underlying math is identical. Plasticity also has its own `fuzzylogic` driver (a Mamdani controller), looked up in its own package first; Trust has no other driver. The formulas:

- `curves.zero(reward, memory)` → always `0.0`.
- `curves.full(reward, memory)` → always `1.0`.
- `curves.linear_ramp(reward, memory)` — percentile-scaled linear ramp: `0.1` at or above the 90th percentile of recent scores, rising linearly to `1.0` at `memory.min_seen` (the worst score seen this run); `1.0` during the first 10 episodes of history. (Previously named `fuzzylogic`; renamed because it is a single clipped linear function, not a fuzzy inference system.) No hard floor below the low end: an earlier version clamped anything below the 10th percentile to a flat `1.0`, so once the agent was consistently near-perfect (a very narrow percentile band), a merely mediocre score and a catastrophic one both forced identical maximum-effort search. Interpolating all the way to `min_seen` keeps the value proportional across the whole observed range.
- `curves.quadratic(reward, memory)` — aggressive: `((max_seen - reward) / (max_seen - min_seen))²`, clamped to `[0.1, 1.0]`.

`ena.py` calls `self._skills["trust"].decay(reward, self.memory)` directly — no `ena.py`-side hook. `self._skills["plasticity"].compute(reward, self.memory, self.learn)` is likewise called directly; the facade itself (not `ena.py`, and not duplicated per driver) forces the result to `0.0` whenever `learn` is `False` — no exploratory search effort during evaluation. The CartPole studies use `"linear_ramp"` for Trust in every ENA row and for Plasticity wherever the Plasticity skill is on. The Trust facade also owns `reference(memory, return_scale)` (the population's dynamic reference trust — the same formula regardless of driver, used to seed fresh/reset individuals) and `reset_population(population, memory, return_scale, factor)` (blends every non-gladiator's trust toward that reference) — both are facade-level methods, not delegated to `self._instance`, since neither varies by driver.

Percentile-based reward scaling used by the dynamic reference trust (and available to any skill that needs it) is its own tool, `tools/reward_normalizer.py`'s `normalize_reward(reward, memory, return_scale)`: below 10 samples, scales against `return_scale` for warmup; otherwise linearly rescales relative to the 5th/95th percentile of `memory.reward_history`, deliberately not clipped to `[0, 1]` so an individual that breaks the current record still shows as strictly better.

## 7. `ENAgent.train()` — the per-episode evolutionary step (`ena.py:211`)

Invoked from the domain runner's `play()` loop whenever `agent.learn` is `True` and an episode just finished (`step_data["done"]`). Sequence:

1. Append `last_episode_score` to `reward_history`; update `max_seen`/`min_seen`.
2. `failure_signal = self._skills["trust"].decay(score, memory)` (§6).
3. Update the acting individual's `trust` via an EMA blend: `trust = (1-failure_signal)*trust + failure_signal*normalized_score` — a bigger failure signal makes trust track the latest score more aggressively (fast punishment, slow reward).
4. `plasticity = self._skills["plasticity"].compute(score, memory, learn)` (§6); appended to `memory.plasticity_history`. This value gates both how many other individuals get evaluated this round and mutation magnitude.
5. `hall_of_fame.update(...)` — see §8.
6. If `failure_signal > 0.1`, `self._skills["trust"].reset_population(population, memory, return_scale, factor=failure_signal)` blends every **non-gladiator** individual's trust toward the current dynamic reference trust, proportional to the failure signal — a poor result partially "resets confidence" across the population, but gladiators are excluded (they're already protected from replacement in `__evolve_population`, and an unrelated exploratory failure — e.g. the epsilon-greedy random non-gladiator pick during training — shouldn't erode an established gladiator's trust either).
6b. If an Olympiad is due (`hall_of_fame.olympics_due(total_env_steps)`; only the `olympic` driver ever is) and the eviction bar exists (at least `REFERENCE_MIN_RETURNS` training returns; before that the reference trust is a −1 placeholder), `__hold_olympics(env)` runs it (§8.1); otherwise it is postponed to the next episode.
7. **Parallel candidate evaluation**: `num_to_eval = max(2, int(pop_size * plasticity))` other individuals are sampled and evaluated in parallel via a `ThreadPoolExecutor(max_workers=max_threads)`, each against a freshly built env cloned from the live env's current physics params (`env_params` read via `physics` attribute list) and a shared `gen_seed` so all candidates face the same conditions this generation. Each evaluation is capped at `eval_step_cap = min_eval_steps + (max_eval_steps - min_eval_steps) * plasticity` steps, applied uniformly (gladiators included, no special-casing) — confident/converged generations (low plasticity) evaluate candidates cheaply, uncertain ones get the full depth for a reliable signal. A candidate whose evaluation is cut off before the episode ends has its score extrapolated to the full episode (`score * max_episode_steps / eval_step_cap`). Their trust scores are updated with the same EMA rule as step 3. `candidate_env_steps` counts the steps actually spent; the runner logs its per-episode increase.
8. `__evolve_population()` — see §8.

More search effort (higher plasticity) → more candidates evaluated, evaluated more deeply, and larger mutation noise; a confidently-performing agent (low plasticity) evaluates as few as 2 others, down to `min_eval_steps` deep, and mutates gently.

## 8. `hall_of_fame` skill — Gladiator archive (`skills/hall_of_fame/`)

Facade class `HallOfFame`, constructed as `HallOfFame(driver, **kwargs)`. Called once per completed training episode from `ENAgent.train` via `self._skills["hall_of_fame"].update(winner, trust, population, memory, env_detection, obs_normalizer)`.

**`default`** (`driver/default.py`) — a staged pipeline, each stage returning early on a decision:

1. **Performance gate** (`__passes_performance_gate`) — computes a Z-score of the winner's trust against the *current population's* trust distribution (`std` floored at `0.1` to avoid blow-ups on uniform populations); requires `z_score >= 1.5 * (1 - 0.5 * current_plasticity)` **or** `trust >= 0.5` to proceed at all. Higher current plasticity (more active search) lowers the bar — the agent is more willing to induct while still exploring.
2. **Niche resolution** — `recognizer.match_niche(...)` (Late Identity Binding, §3.1) assigns/confirms the niche ID for this winner before any HoF decision uses it.
3. **Continual improvement** (`__continual_improvement`) — if the winner's `individual_id` is *already* in the archive: same niche → update in place only if strictly better; different niche → the gladiator also performs well under this niche's physics, so it's **cloned** as a new record for that niche (if it has budget), labeled with the current environment's name, rather than moved; the original record and live gladiator keep their niche and label. If that niche already holds a genetic twin (typically an earlier clone of the same gladiator), the twin is refreshed instead when the new trust is higher, so repeated foreign wins don't fill the niche with copies of one individual.

A gladiator's `env_name` is set once, at induction, from the `env_name` argument `ena.py`'s `train()` passes to `HallOfFame.update` (the environment the winner just succeeded in); it is never rewritten afterwards, so the label always means "the environment this specialist was archived for", not "the last environment it played".
4. **Genetic diversity / niche protection** (`__genetic_diversity`) — within the same niche, if the winner's weight-vector cosine similarity to an existing record exceeds `SIMILARITY_THRESHOLD = 0.95` ("strategic twin"), it replaces that twin only if strictly better trust, and the pipeline stops either way (no double-counting near-duplicates).
5. **Open slot induction** (`__slot_induction`) — if the niche has fewer than `max_per_niche` records, save the winner directly as a new record.
6. **Eco-niche protection** (`__eco_niche_protection`) — last resort: if the niche is full, evict the weakest-trust record in that niche if the winner beats it.

`__evict(individual_id, population)` (used by steps 4 and 6 above) removes the archive entry **and** clears the corresponding live population slot's `is_gladiator`/`internal_env_id`/`env_name` tags. Without the population-side clearing, an evicted record left a "zombie gladiator" behind: invisible in `niches_*.txt` but still competing in the selection skill's `evaluate()` and still protected from replacement in `__evolve_population()`.

`__save` deep-copies the winner's model into a fresh `brain(...)` instance (so the archived weights are independent of the live, still-mutating individual) and immediately flags the *live* population dict as `is_gladiator=True` with its assigned `internal_env_id` — gladiators are then protected from replacement in `__evolve_population` (§9's sibling, population replacement in `ena.py`).

`inject_gladiators(population, pop_size, obs_normalizer, restore_trust, admit=None)` (called by `ENAgent.load_gladiators()`, e.g. before `cartpole_challenge.py`'s testing phase) round-robins one gladiator per known niche per round into the first `floor(pop_size * max_gladiator_ratio)` population slots, until the budget is filled or niches are exhausted. With `admit` (the Olympics' `admits`, when enabled) only the records it accepts are re-introduced; each copy carries the record's `olympic_confirmed` flag.

`records_in_niche(niche_id)` lists one niche's records (the Olympics' competitors); `evict(individual_id, population)` removes a record through the same path as the archive's own replacement, clearing the live copy's gladiator tags.

`save_niche_details(output_dir, agent_name)` writes a `niches_<agent_name>.txt` summary (niche ID, env name, gladiator count) — this is one of the run-artifact files `CLAUDE.md` flags as generated, not source.

**`none`** (`driver/none.py`) — the disable switch: `update`/`inject_gladiators` are no-ops, `__bool__` is always `False`, `save_niche_details` writes the same "No gladiators inducted" summary the default driver writes when its archive is empty.

## 8.1. `olympic` Hall of Fame driver — the default archive plus Olympiads (`skills/hall_of_fame/driver/olympic.py`)

A gladiator is admitted on the evidence of one episode. Where a policy's return depends strongly on the initial state (HalfCheetah runs or flips), many admissions are lucky, and test time plays them. The Olympics re-evaluate the archive periodically: elite re-evaluation (Such et al. 2017) applied to the archive only (Justesen et al. 2019 re-sample archive elites in noisy MAP-Elites).

`hall_of_fame_driver="olympic"` extends `default` with Olympiads. The driver decides and judges; ENAgent only plays. Every `olympics_interval` environment steps (`total_env_steps`, checked after each training episode), `ENAgent.__hold_olympics(env)` calls `hall_of_fame.hold_olympics(niche, play, trust_of, eviction_bar, population, total_env_steps)`, which:
1. takes the current niche's records (`hall_of_fame.records_in_niche`); other niches' gladiators cannot be judged in this physics;
2. draws `olympics_episodes` seeds shared by every competitor (common random numbers), fresh at each Olympiad, from the agent's seeded NumPy RNG;
3. has each competitor play those full episodes through ENA's `play` callback, in a clone of the current environment, as test time plays a gladiator (`__run_gladiator`: its frozen `last_mu`/`last_var` normalization via `infer_action`, no projection, so the archive is not modified); the steps count as `candidate_env_steps`, the agent's budget;
4. turns each competitor's mean return into trust (ENA's `trust_of`: `normalize_reward`) and evicts those below the eviction bar — the population's reference trust (`trust.reference`: a gladiator must beat what the population typically does) — except the champion, so a niche never empties;
5. re-scores survivors (record and live copy) with their Olympic trust and marks them `olympic_confirmed`.

**Only an Olympiad removes or re-scores a confirmed gladiator.** The `olympic` driver sets `protect_confirmed`, which the `default` archive's own steps respect. A new winner's trust rests on one episode, a confirmed gladiator's on its Olympic mean, so the archive's own steps would otherwise let a lucky winner displace it:
- continual improvement no longer overwrites a confirmed record's weights or trust, including the cross-niche twin refresh;
- a genetic twin of a confirmed record joins beside it as provisional if the niche has a free slot, and is dropped otherwise;
- eco-niche protection replaces only the weakest *provisional* record, and drops the winner if every record is confirmed;
- a newly saved or evicted live copy has `olympic_confirmed` cleared, so no stale flag admits it.

**Qualifier seats** (`olympics_qualifier_seats`, default 2): a niche full of confirmed gladiators would otherwise freeze, dropping every newer and better winner.
- The Hall of Fame (`qualifier_seats`, active only with `protect_confirmed`) lets a niche hold up to `max_per_niche + qualifier_seats` records; new winners always enter as provisional.
- Each Olympiad relegates the lowest survivors beyond `max_per_niche`, even if they cleared the bar. So confirmed gladiators never exceed `max_per_niche`, and a confirmed gladiator never holds a qualifier seat.
- Provisional entrants may also use empty confirmed seats.

**Olympic prior for BPR.** Each survivor's raw Olympic mean return is stored as `olympic_return`, on the record and the live copy, and re-injected copies carry it. BPR (§4) uses it as the gladiator's prior utility in its own niche (`olympic_return / return_scale`) instead of the solved return (`OWN_PRIOR` = 1); the foreign-niche prior is `cross_prior` times that own prior.
- **Why:** BPR's performance model is measured offline in the original paper, and the Olympics measure exactly that.
- **The problem it fixes:** where the solved return is out of reach (HalfCheetah), a prior of 1 makes every untried gladiator look better than anything observed, so the test spends its episodes trying them all.
- **Olympics off:** without an Olympic result, the priors are as before.

**Gladiators play during training, as with the `default` driver.** The live winner becomes a gladiator of its niche (provisional until its first Olympiad), keeps playing and stays protected from replacement; an Olympiad that evicts a record clears its live copy, so a mediocre early gladiator cannot anchor the search for long. Re-injected records are gladiators (`reinject_as_gladiators`: every record is one); the `default` driver copies the record's own flag, as every CartPole run did.

A gladiator that has not competed yet is **provisional**: with the `olympic` driver (`holds_olympics`), `load_gladiators()` re-introduces only confirmed records (`admits`) and `set_learning(False)` strips gladiator status from unconfirmed live copies, so test time plays only gladiators that competed. It also restores each confirmed live copy's trust from its record (`hall_of_fame.record`). During training, a live copy's trust keeps moving with single episodes and candidate evaluations, which is deliberate: after an environment switch a failing gladiator must lose trust, so training selection can move on. Test-time selection therefore starts from the Olympic ranking. Every competitor's returns, steps, admission and Olympic trust, bar and verdict are appended to `ENAgent.olympics_history`.

**Without the Olympics** (the `default` and `none` drivers): `holds_olympics` is False, an Olympiad is never due, and every gladiator is admitted at test. That is ENA exactly as before, proved on the CartPole ablation's seeded replicate 1 (first 100 episodes byte-identical).

## 9. Universal Snapshot Projection tool (`tools/weight_projection.py`)

Because the observation normalizer's mean/variance keep drifting as more states are seen, an individual's first-layer weights — implicitly tuned to whatever normalization was in effect when it was last aligned — can become stale relative to the *current* normalizer. `project_individual_weights(individual, obs_normalizer, max_eval_steps)` analytically re-scales its first Dense layer (weight columns scaled by `std_new/std_old` clipped to `[0.1, 10.0]`, bias shifted by `(mean_new - mean_old)/std_old · W`) so its behavior is preserved under the new normalization without any retraining. Applies uniformly to gladiators too, even though their *inputs* are normalized differently (see `tools/inference.py`'s `is_gladiator` branch, called from `act()`, which uses their own frozen `last_mu`/`last_var` instead of the live normalizer): `__evaluate_individual_performance` always normalizes an individual's input with the live `obs_normalizer` regardless of `is_gladiator` (it does not go through `infer_action` — a deliberate difference, not an oversight), so leaving a gladiator's *weights* un-projected would evaluate them with inputs aligned to one normalizer but weights aligned to another. Skipped entirely until the normalizer has seen at least `2 * max_eval_steps` samples (avoids projecting off noisy early statistics). Called from three places: `ENAgent.act()` (for the acting individual), `__evaluate_individual_performance` (for every evaluated candidate), and the Crossover skill's `uniform_mask` driver (for both parents before mixing their weights).

## 10. `ENAgent.act()` step-by-step (`ena.py:129`)

Called once per environment step by the domain runner's `play()` loop. `act()` itself is a short sequence of delegated steps; the mechanism-specific logic lives behind three private helpers and one tool function:

1. `__start_episode(env)` (first step of an episode only): zeroes the step counter and calls `env_detection.start_episode()` (re-identification for `dynamics_model`, a no-op otherwise, §3.1).
2. `__select_individual()` (first step of an episode only): asks the selection facade to pick `best_individual_idx` (train vs. test path per `self.learn`), then projects that individual's weights to the current normalizer (§9).
3. `tools/inference.py`'s `infer_action(individual, state, memory, is_continuous)`: normalizes `state` — gladiators use their own frozen `last_mu`/`last_var`; everyone else uses the live `obs_normalizer` (updated just before this call) — and runs the chosen individual's `fast_call` (a `tf.function`-wrapped forward pass) to get an action (argmax for discrete, clipped raw output for continuous).
4. Step the environment; `memory.record_step(reward)` accumulates the episode's return, and `env_detection.observe(state, action, next_state, memory)` hands the real transition to the driver (the `dynamics_model` driver's evidence and PM-CuSum input; a no-op otherwise).
5. `__end_episode()`, only on episode end (`done`):
   - Records `last_episode_score`, then `selection.on_episode_end(memory, learn, env_detection)` — at test time, passes the episode's result to the driver's `record_episode_result` (§4); training episodes don't update the selector.
   - `env_detection.evaluate(memory, learn)` — runs `detect_change`; on a detected shift, resets the driver's own state (`reset()` — `dynamics_model` drops its old-regime transition buffers), the observation normalizer, and (learn mode only) `memory.clear_training_history()`. Returns whether a shift fired.
   - If a shift fired and the agent is in test mode, `ena.py` itself calls `selection.evaluate(population)` — the one cross-skill call `env_detection` deliberately doesn't own itself, since it would otherwise need to know about the selection skill.
   - Reset per-episode counters.

## 11. Supporting state — `AgentMemory` (`memory.py`)

Centralizes everything the other components read/write, decoupled from `ENAgent` itself. **Not currently a skill** — it's a shared data store other skills read/write (return history, the observation normalizer), not a swappable *behavior* the way Mutation or Trust are. Making it one with a `default` driver would be mechanical, but there's no second implementation motivating the abstraction yet; do it only if a real need shows up (e.g. a bounded-memory or decay-weighted variant), not speculatively.

- `record_step(reward)` — called once per `act()` step; accumulates `current_episode_reward`.
- `obs_normalizer` — a `_RunningNormalizer` (Welford's online mean/variance, capped at `max_eval_steps * 10` samples, used to normalize policy-network inputs).
- `reward_history` (bounded deque, `history_size`), `plasticity_history` (unbounded list), `max_seen`/`min_seen`.
- Derived property: `performance_derivative` (short-term mean minus a slightly older baseline mean over `reward_history`), read by the `fuzzylogic` Plasticity driver.

`_RunningNormalizer` (`agents/tools/normalizers.py` — shared across *all* agents, not ENA-specific, hence living one level up from `agents/ena/tools/`) implements a Welford-style running mean/variance with an EMA-style mixing rate once `max_count` is reached (so statistics keep adapting rather than freezing once "full"), clipping normalized output to `±clip` (default 10.0). `_IdentityNormalizer` is a no-op sibling for discrete-observation environments (not used by CartPole/HalfCheetah, which are both continuous-observation).

## 12. Known rough edges (as of this snapshot - do not touch)

- `physics` defaults to a **mutable list** default argument (`physics=["gravity", "masspole", "length"]` in `ENAgent.__init__`) — harmless today since it's never mutated in place, but worth flagging if future edits start appending to `self.physics`.
- The `hall_of_fame_driver`/`recognizer_strategy="none"` switches exist to let a factorial ablation isolate how much each mechanism (env-detection, Hall of Fame, Plasticity) contributes.
