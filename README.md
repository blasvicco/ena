# ENA — Evolutionary Neuroplastic Agent

> A gradient-free agent for non-stationary reinforcement learning that detects changes in the environment's physics and reuses the specialists it has archived for them.

---

## Abstract

Traditional Reinforcement Learning (RL) agents often struggle to adapt rapidly in non-stationary environments, suffering catastrophic forgetting when facing sudden physical shifts. Drawing inspiration from biological meta-learning and evolutionary diversity, this preliminary study introduces the Evolutionary Neuroplastic Agent (ENA). The ENA dynamically maintains an active population of multiple lightweight neural network policies (a "pool of specialists") instead of a monolithic model. By continuously evaluating each specialist iteratively through defined *Trust* and *Plasticity* heuristics, the ENA autonomously triggers localized evolutionary search algorithms and substitutes specialists during environmental changes. We empirically evaluate two mechanisms for governing these shifts: a Quadratic Ratio formulation and a normalized Fuzzy-logic heuristic model. Evaluating this framework in a continuously shifting 1D CartPole domain transitioning across sequential gravities (Earth, Mars, Jupiter), we execute an online meta-learning testing protocol comparing frozen neural behaviors sequentially across those bodies tracking into a hidden fourth phase (Neptune). We demonstrate that the Fuzzy-logic framework manages inter-environment stability gaps smoothly. However, the study identifies critical limitations: the ENA exhibits massive standard-deviation dependencies on initial population states, confirming that robust adaptation within tightly constrained genetic boundaries remains highly vulnerable to initial parameter clustering. Our findings position the ENA as a promising proof-of-concept for dynamic Quality-Diversity adaptation, warranting further benchmarking against standard deep RL alternatives in complex domains.

---

## What This Project Is

ENA evolves a population of small neural-network policies without gradients and keeps proven specialists in a Hall of Fame, organized by the physics it detects. It is evaluated on two non-stationary domains, training on three environments in sequence without being told when they change, then tested frozen on five, two of them never seen:

- **CartPole** — training on Mars, Jupiter, Earth (gravity, pole mass and length change); the test adds Neptune and Brown-dwarf.
- **HalfCheetah** — training on Mars-Slippery, Jupiter-Rough, Earth (gravity and ground friction); the test adds Ice-Floor and High-Wind.

The studies:
- **Skill ablation** (`cartpole_ena_ablation.py`): each skill (Plasticity, Hall of Fame, environment detection) switched on and off, 30 replicates.
- **Compute cost of Plasticity** (`cartpole_ena_compute.py`), and the **Plasticity controller** study (`cartpole_ena_plasticity_controller.py`): the linear ramp against quadratic and Mamdani fuzzy controllers, 30 replicates.
- **Comparison with gradient-based agents** at an equal budget of environment steps: DQN, PPO and SAC-Discrete on CartPole (`cartpole_challenge.py`), PPO and SAC on HalfCheetah (`half_cheetah.py`).

Metrics: test **reliability** (% of episodes at the success threshold), mean return per environment, **stability gap** at an environment switch, **steps to first solve** per environment, and **compute** (environment steps, wall-clock time).

---

## Repository Structure

```
app/
├── src/
│   ├── agents/
│   │   ├── tools/
│   │   │   ├── abstract.py              # AAgent — shared agent contract (act/train/set_learning)
│   │   │   ├── baseline.py              # ABaseline — shared contract for single-model RL baselines
│   │   │   └── normalizers.py           # Running/identity observation normalizers
│   │   ├── ena/
│   │   │   ├── ena.py                   # ENAgent — the Evolutionary Neuroplastic Agent
│   │   │   ├── memory.py                # AgentMemory — return/plasticity history, observation normalizer
│   │   │   ├── ENA.md                   # Implementation notes
│   │   │   ├── skills/                  # One facade (main.py) + drivers (driver/) per mechanism:
│   │   │   │                            #   trust, plasticity, mutation, crossover, hall_of_fame,
│   │   │   │                            #   env_detection (dynamics_model), selection (bpr)
│   │   │   └── tools/                   # Dynamics models, PM-CuSum, curves, weight projection, ...
│   │   ├── dqn.py                       # DQN baseline
│   │   ├── ppo.py                       # PPO-clip baseline (discrete + continuous)
│   │   ├── sac.py                       # SAC baseline (continuous)
│   │   ├── sac_discrete.py              # SAC-Discrete baseline
│   │   └── random.py                    # Random-action floor
│   ├── cartpole_ena_ablation.py      # CartPole skill ablation (shared study runner)
│   ├── cartpole_ena_compute.py          # CartPole compute cost of Plasticity
│   ├── cartpole_ena_plasticity_controller.py # CartPole Plasticity controller: linear vs. quadratic vs. fuzzy
│   ├── cartpole_challenge.py            # CartPole: ENA vs. DQN / PPO / SAC-Discrete
│   ├── half_cheetah.py                  # HalfCheetah: ENA vs. PPO / SAC
│   └── tools/
│       ├── metrics.py                   # Per-replicate metrics (reliability, stability gap, compute, adaptation) + plots
│       ├── early_stopping.py            # Evaluation / stopping rule shared by the challenge protocols
│       ├── aggregate_metrics.py         # Aggregates metrics across a study's replicates
│       ├── detection_stats.py           # Environment-recognition numbers per phase (part of aggregation)
│       ├── replot_replicate.py          # Redraws a replicate's training/test charts from its logs
│       ├── paper_results.py             # Generates the paper's numbers, tables and figures
│       ├── paper_config.py              # The paper's configuration numbers, read from the code that ran
│       ├── sac_discrete_entropy_check.py # Measures SAC-Discrete's temperature per target-entropy ratio
│       ├── plot_study_curves.py         # Per-episode curves across replicates
│       ├── live_data.py                 # Reads the live_*.csv logs
│       ├── live_dashboard.py            # Streamlit live convergence dashboard
│       ├── live_monitor.py              # Matplotlib live convergence chart
│       ├── visualization.py             # Multi-experiment progress reporting (terminal bar + ETA)
│       └── benchmark_half_cheetah.py    # HalfCheetah throughput and study cost estimate
├── tests/                               # pytest suite (mirrors src/)
├── latex/                               # Paper source (build.sh, package_arxiv.sh, generated/)
├── shell/                               # Format + lint and test helper scripts
├── requirements.txt                     # Pinned Python dependencies
└── .pylintrc                            # Lint configuration (see STANDARDS.md)
docs/                                    # Plans, and the cloud runbook (CLOUD_RUN.md)
hub/
├── python/Dockerfile                    # Experiment environment
└── cloud/sync_results.sh                # Mirrors a study's outputs to object storage
docker-compose.yml                       # ena_experiment and paper (TeX Live) services
```

See [STANDARDS.md](STANDARDS.md) for coding conventions and a closer look at how these pieces fit together.

---

## How to Run

### Option A — Docker (recommended)

**1. Build and start the container**
```bash
docker compose up -d
```

**2. Open a shell inside it**
```bash
docker compose exec ena_experiment bash
```

The `app/` directory is bind-mounted to `/home/app` inside the container, so edits on the host are immediately reflected. Port `8501` (Streamlit's default) is mapped to the host, for the live dashboard described below.

**3. Run a study** (each takes `--help`; `--workers` sets parallel replicates)
```bash
python src/cartpole_ena_ablation.py      # skill ablation
python src/cartpole_challenge.py            # ENA vs. DQN / PPO / SAC-Discrete
python src/half_cheetah.py                  # ENA vs. PPO / SAC
```
Studies run in batches: re-running with a larger `--num-experiments` skips replicates that are already complete. For cloud runs see `docs/CLOUD_RUN.md`.

### Option B — Local Python

**1. Install dependencies**
```bash
pip install -r app/requirements.txt
```

**2. Run an experiment**
```bash
python app/src/cartpole_challenge.py
# or
python app/src/half_cheetah.py
```

**3. Aggregate results across repeated runs**
```bash
python app/src/tools/aggregate_metrics.py
```
Reads per-experiment output folders and computes mean, standard deviation, min, and max for every reported metric.

**4. Generate the paper's numbers, tables and figures**
```bash
python app/src/tools/paper_results.py
```
Writes `app/latex/generated/` and the figures in `app/latex/images/` from every study's outputs.

---

## Live Convergence Monitoring

Every training run writes `live_training_{agent_name}.csv` to the experiment's output directory incrementally, one row per episode (score, plasticity, detected niche, ...), flushed immediately — so you can watch convergence while a run is still in progress instead of waiting for it to finish. The terminal progress bar also shows a live ETA per phase, e.g. `Training ENA-Combined (ETA 6m 22s)`, estimated from the average time per episode elapsed so far.

Two ways to watch the CSV update live, run as a **separate process** from the training run (read-only, no risk to reproducibility):

**Streamlit dashboard (recommended — auto-refreshing, in a browser):**
```bash
streamlit run app/src/tools/live_dashboard.py
```
Run this either directly on the host (simplest — `app/` is bind-mounted so the CSV paths are visible there too) or inside the container (`docker compose exec ena_experiment streamlit run src/tools/live_dashboard.py --server.headless true`, then open `http://localhost:8501` on the host — port `8501` is already mapped in `docker-compose.yml`). Pick a study, a replicate and an agent in the sidebar; the page refreshes every 5 seconds and downsamples long runs.

**Matplotlib monitor (needs a real display — run on the host, not inside the container):**
```bash
python app/src/tools/live_monitor.py app/cartpole_ena_ablation_outputs/exp_1/live_training_ENA-Combined.csv
```
Polls the given CSV and keeps an updating chart on screen until interrupted (Ctrl+C). This will not work inside a headless docker container — matplotlib's interactive mode needs a display backend.

---

## Scripts

| Script | What it does |
|---|---|
| `src/cartpole_ena_ablation.py` | CartPole skill ablation; also the shared study runner the other studies reuse (training, frozen test, live logs, metrics) |
| `src/cartpole_ena_compute.py` | Compute cost of Plasticity: one ENA-Combined run at full search effort, against the ablation's replicate 1 of the same configuration at zero effort (ENA-HoF+EnvDetection) and with the ramp (ENA-Combined) |
| `src/cartpole_ena_plasticity_controller.py` | ENA-Combined with the quadratic and the Mamdani fuzzy Plasticity controllers; the linear-ramp reference is the ablation's ENA-Combined (same replicates, paired) |
| `src/cartpole_challenge.py` | ENA vs. DQN, PPO, SAC-Discrete under `--protocol fixed-budget` (default) or `first-solve` |
| `src/half_cheetah.py` | Same comparison on HalfCheetah vs. PPO and SAC (protocol parameters provisional, see `docs/PLAN_half_cheetah.md`) |
| `src/tools/aggregate_metrics.py` | Mean / std / min / max of every metric over a study's replicates, plus pairwise comparisons; also writes `detection_summary.txt` |
| `src/tools/paper_config.py` | Every setting the paper states (ENA, physics, protocols) read from the built agents, environments and study scripts; called by `paper_results.py` |
| `src/tools/sac_discrete_entropy_check.py` | Trains the comparison's SAC-Discrete per target-entropy ratio and logs its temperature and critic values (the evidence for the 0.5 ratio) |
| `src/tools/detection_stats.py` | Environment recognition per phase for agents with env detection: unrecognized episodes, confirmed changes, episodes to the first one, Hall of Fame niches per environment |
| `src/tools/replot_replicate.py` | Redraws finished replicates' `training.png` / `test.png` from their live CSVs (after a chart change) |
| `src/tools/paper_results.py` | Every number, table and figure the paper quotes |
| `src/tools/plot_study_curves.py` | Per-episode return across replicates, one panel per agent |
| `src/tools/live_dashboard.py`, `live_monitor.py` | Live convergence views (see above) |
| `src/tools/benchmark_half_cheetah.py` | Measures throughput on the current machine and estimates the HalfCheetah study's time and cost |

---

## Paper

The paper's LaTeX source is in `app/latex/` (IEEE two-column). Build it in Docker:

```bash
docker compose up -d --no-deps paper
docker compose exec paper ./build.sh           # -> app/latex/build/main.pdf
docker compose exec paper ./package_arxiv.sh   # -> app/latex/build/arxiv.tar.gz (refuses while TODOs remain)
```

Numbers, tables and the results figures are generated by `src/tools/paper_results.py`; never edit `app/latex/generated/` by hand.

---

## Future Work

HalfCheetah results (planned, `docs/PLAN_half_cheetah.md`); nonlinear, still gradient-free dynamics models for richer domains; evolution strategies as ENA's search operator; a learned specialist selector; and larger populations on accelerators.
