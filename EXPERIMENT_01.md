# EXPERIMENT_01 — `app/src/cartpole_challenge.py`

This document describes, step by step, what running `cartpole_challenge.py` actually does. It covers the CartPole domain runner as currently checked in — see [[project-ena-state]]-equivalent caveat below: only the ENA-01 (Fuzzy-Fuzzy) agent is active today; DQN and PPO are wired in but commented out of the run.

## 0. Purpose

`cartpole_challenge.py` is a domain runner: it builds one or more agents, trains them on a sequence of non-stationary CartPole physics regimes ("worlds"), tests them (including on unseen/extrapolated physics), computes academic comparison metrics, and writes logs + plots to a per-experiment output directory. It is the CartPole counterpart to `half_cheetah.py` (same lifecycle, different domain).

## 1. Top-level constants

| Constant | Value | Meaning |
|---|---|---|
| `NUM_EXPERIMENTS` | `1` | Number of independent, seeded experiment repetitions to run. |
| `MAX_WORKERS` | `os.cpu_count() - 2` | Process pool size for running experiments in parallel (leaves 2 cores free). |
| `GPUS` | `tf.config.list_physical_devices("GPU")` | Detected GPUs, logged at startup. |
| `TESTING_EPISODES` | `[200, 200, 200, 200, 200]` | Episodes per test world (5 worlds — see §4). |
| `TRAINING_EPISODES_BASELINE` | `[1500, 1500, 1500]` | Episodes per training world for DQN/PPO/SAC baselines (currently unused — those agents are commented out). |
| `TRAINING_EPISODES_ENA` | `[500, 500, 500]` | Episodes per training world for ENA agents. |

## 2. Brain factories

Three Keras model-building functions are defined; each agent is constructed with one of these injected as its `brain` factory (per the `AAgent` strategy pattern — see `STANDARDS.md`):

- `dqn_brain(action_dim, state_dim, hidden=(64, 64))` — plain MLP Q-network, ReLU hidden layers, linear output over actions.
- `ena_brain(action_dim, state_dim)` — small `Sequential` network (Dense(4) → Dense(2) → Dense(action_dim)), orthogonal-initialized. Deliberately tiny since ENA relies on population-level search rather than gradient descent on a large network.
- `ppo_actor_brain(...)` / `ppo_critic_brain(...)` — MLP actor and critic with `tanh` hidden activations and orthogonal initialization (actor output gain `0.01`, critic output gain `1.0`), following standard PPO initialization conventions.

## 3. `make_env(gravity, masspole, name, length, render_mode)`

Builds a `CartPole-v1` Gymnasium environment and overrides its physics parameters (`gravity`, `masspole`, `length`), then recomputes the internal derived quantities the physics engine depends on (`total_mass`, `polemass_length`, and `masspole_length` if present). Each call produces a differently-"shaped" CartPole world, tagged with a human-readable `name` for logging/plotting.

## 4. `run_experiment(experiment_id, output_dir, queue)` — the experiment lifecycle

This is the core of the script, executed once per experiment (per `NUM_EXPERIMENTS`):

1. **Prepare output directory** — creates `output_dir` and opens `outputs.txt` for the experiment's text log (metrics tables, timing summary).
2. **Build the training worlds** — three fixed physics regimes, presented in this order:
   - `Mars-Light` (gravity 3.7, masspole 0.05, length 0.1)
   - `Jupiter-Heavy` (gravity 24.8, masspole 0.5, length 1.5)
   - `Earth-Standard` (gravity 9.8, masspole 0.1, length 0.5)
   This sequence is what makes the domain non-stationary: the agent trains through a shift, then a second shift.
3. **Instantiate agents.** Currently only one is active:
   - `ena_agent_01` = `ENAgent(...)`, named `"ENA-01 Fuzzy-Fuzzy"`, configured with `pop_size=100`, `plasticity_algorithm="fuzzylogic"`, `trust_algorithm="fuzzylogic"`, `selection_driver="heuristic"`, `mutation_rate=0.05`, `max_per_niche=5`, `max_gladiator_ratio=0.5`, tracking `physics=["gravity", "masspole", "length"]` as the recognizer's feature set. It's given `TRAINING_EPISODES_ENA` (`[500, 500, 500]`, one count per training world).
   - A second ENA variant (`ena_agent_02`, quadratic plasticity/trust), a `DQNAgent`, and a `PPOAgent` are defined in the file but commented out — not part of the active run.
4. **Training phase** — `agent_training(...)`:
   - For each active agent: `agent.set_learning(True)`, then `play(agent, worlds, episodes)` runs it through all three training worlds in order, timed with `time.time()`.
   - Writes a timing summary and calls `calculate_paper_metrics(...)` (success threshold `475.0`, the CartPole-v1 "solved" score) to `outputs.txt` under a `Training:` heading.
   - Calls `plot_academic_comparison(...)` → `<output_dir>/training.png`.
   - For any ENA agent, additionally renders `plot_plasticity_analysis(...)` → `plasticity_<agent_name>.png` and `plot_specialist_transitions(...)` → `specialist_transitions_TRAINING_<agent_name>.png`.
5. **Switch to testing episode counts** — `ena_agent_01.set_episodes(TESTING_EPISODES)`.
6. **Load gladiators** — `ena_agent_01.load_gladiators()` restores the agent's Hall of Fame of frozen elite specialists (built up during training) before evaluation.
7. **Testing phase** — `agent_testing(...)`:
   - Builds `test_worlds` as five worlds: the two training extremes plus Earth, interleaved with two **held-out** regimes the agent never trained on:
     - `worlds[0]` = Mars-Light
     - `mystery_env` = `Neptune` (gravity 11.15) — interpolation test, inside the training gravity range but a physics point never seen.
     - `worlds[1]` = Jupiter-Heavy
     - `extrap_env` = `Brown-dwarf` (gravity 50.0) — extrapolation test, gravity far outside anything trained on.
     - `worlds[2]` = Earth-Standard
   - For each agent: `agent.set_learning(False, output_dir=output_dir)` freezes the neural weights — per the code comment, only the archive selector's Trust score still updates at test time — then `play(...)` runs `TESTING_EPISODES` (200 each) per world.
   - Writes `calculate_paper_metrics(...)` results to `outputs.txt` under `--- Experiment N Test Results ---`.
   - Renders `plot_academic_comparison(...)` → `test.png`, and for ENA agents, `plot_specialist_transitions(...)` → `specialist_transitions_<agent_name>.png`.
8. **Signal completion** via `post_update(..., done=True)`.

## 5. `play(agent, environments, episodes_per_env)` — the episode loop

Shared by both training and testing. For every environment, run the given number of episodes:

1. Reset the environment, cast the initial state to `float32`.
2. Step loop: `agent.act(env, state)` returns `(next_state, reward, terminated, truncated, action)` — the agent owns its own action-selection and environment-stepping.
3. If `agent.learn` is `True`, call `agent.train(env=..., step_data={...})` with the transition — DQN consumes this per-step; ENA batches it for later use in specialist selection/mutation.
4. Accumulate `episode_score`, advance `state`.
5. After each episode, append a row to a `history` list: `env_id`, running `episode` index, `score`, plus ENA-specific bookkeeping (`specialist_id`, `detected_env_id`, `is_gladiator`, `gladiator_env`) pulled via `getattr` so non-ENA agents don't need these attributes.

Returns the full history as a `pandas.DataFrame` — this is what feeds `calculate_paper_metrics` and every plotting function.

## 6. Metrics computed (`tools/metrics.py::calculate_paper_metrics`)

For each agent's history DataFrame:

- **Reliability (%)** — fraction of episodes scoring ≥ `success_threshold` (475.0).
- **Per-environment average score** — mean score, grouped by `env_id`.
- **Stability Gap (↓)** — mean score over the 20 episodes immediately before the *first* environment switch minus the mean score over the 20 episodes immediately after; measures how badly performance drops when physics changes.
- **Efficiency (Score/s)** — total score divided by wall-clock training/testing time, if timing data was supplied.

Results are printed as a table to `outputs.txt`.

## 7. Plots produced, per experiment output directory

| File | Function | When |
|---|---|---|
| `training.png` | `plot_academic_comparison` | After training |
| `test.png` | `plot_academic_comparison` | After testing |
| `plasticity_<agent>.png` | `plot_plasticity_analysis` | After training, ENA agents only — score vs. plasticity/search-effort over time |
| `specialist_transitions_TRAINING_<agent>.png` | `plot_specialist_transitions` | After training, ENA agents only |
| `specialist_transitions_<agent>.png` | `plot_specialist_transitions` | After testing, ENA agents only |

`plot_academic_comparison` draws, per agent, a rolling mean/std-dev score band, a horizontal mastery-threshold line, vertical dashed lines at each true environment switch, and markers for when the agent's own recognizer detected a new vs. previously-seen environment. `plot_specialist_transitions` shows which population member (`specialist_id`) was active over time, color-coded by whether it's a frozen "Gladiator" (Hall of Fame member) or an active mutant/learner, against a background shaded by the ground-truth environment.

## 8. Process/threading model (`__main__`)

1. Logs GPU availability.
2. Opens a `multiprocessing.Manager` and creates a `Queue` for cross-process progress reporting.
3. Starts a background thread running `ProgressReporter` (`tools/visualization.py`), which renders a live `rich` progress bar per experiment plus overall memory usage, fed by messages agents post via `post_update(...)`.
4. Spins up a `ProcessPoolExecutor` with `MAX_WORKERS` workers. For each experiment id in `range(NUM_EXPERIMENTS)`:
   - If `cartpole_outputs/exp_<n>/outputs.txt` already exists and is non-empty, the experiment is skipped (treated as already run) and reported as such.
   - Otherwise `run_experiment_worker(experiment_id, queue)` is submitted to the pool.
5. `run_experiment_worker` creates `cartpole_outputs/exp_<n>/`, opens `crash.log`, enables `faulthandler` against it (so segfaults/native crashes are captured), and calls `run_experiment(...)` inside a `try/except`. Any exception's traceback is written to `crash.log` and also pushed to the reporting queue as an error status; the worker's `finally` closes the crash file regardless of outcome.
6. Once all futures complete (`concurrent.futures.as_completed`), the queue is sent a `None` sentinel to stop the reporter thread, which is joined with a 2-second timeout.
7. Final summary lines are printed pointing to the `cartpole_outputs/` (technically the per-experiment `output_dir`) for logs and plots.

## 9. Output layout produced by a run

```
cartpole_outputs/
└── exp_1/
    ├── outputs.txt                              # training + testing metrics tables, timing
    ├── crash.log                                 # empty unless the experiment raised/crashed
    ├── training.png
    ├── test.png
    ├── plasticity_ENA-01 Fuzzy-Fuzzy.png
    ├── specialist_transitions_TRAINING_ENA-01 Fuzzy-Fuzzy.png
    └── specialist_transitions_ENA-01 Fuzzy-Fuzzy.png
```

These directories are run artifacts (already `.gitignore`d as `*outputs`) — don't hand-edit or commit them, per `CLAUDE.md`.
