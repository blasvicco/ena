# ENA — Evolutionary Neuroplastic Agent

> A preliminary study into evolutionary, non-gradient adaptation across non-stationary physical environments.

[Open In Colab](https://colab.research.google.com/drive/1i8LY95hCwqbk4fcmAK12YT0lxH5TDzrW?usp=sharing)

---

## Abstract

Traditional Reinforcement Learning (RL) agents often struggle to adapt rapidly in non-stationary environments, suffering catastrophic forgetting when facing sudden physical shifts. Drawing inspiration from biological meta-learning and evolutionary diversity, this preliminary study introduces the Evolutionary Neuroplastic Agent (ENA). The ENA dynamically maintains an active population of multiple lightweight neural network policies (a "pool of specialists") instead of a monolithic model. By continuously evaluating each specialist iteratively through defined *Trust* and *Plasticity* heuristics, the ENA autonomously triggers localized evolutionary search algorithms and substitutes specialists during environmental changes. We empirically evaluate two mechanisms for governing these shifts: a Quadratic Ratio formulation and a normalized Fuzzy-logic heuristic model. Evaluating this framework in a continuously shifting 1D CartPole domain transitioning across sequential gravities (Earth, Mars, Jupiter), we execute an online meta-learning testing protocol comparing frozen neural behaviors sequentially across those bodies tracking into a hidden fourth phase (Neptune). We demonstrate that the Fuzzy-logic framework manages inter-environment stability gaps smoothly. However, the study identifies critical limitations: the ENA exhibits massive standard-deviation dependencies on initial population states, confirming that robust adaptation within tightly constrained genetic boundaries remains highly vulnerable to initial parameter clustering. Our findings position the ENA as a promising proof-of-concept for dynamic Quality-Diversity adaptation, warranting further benchmarking against standard deep RL alternatives in complex domains.

---

## What This Project Is

ENA is evaluated head-to-head against standard deep RL baselines — **DQN**, **PPO**, and **SAC** — across two non-stationary domains:

- **CartPole** — 1D pole-balancing under sequentially shifting gravity (Earth → Mars → Jupiter), tested against a hidden Neptune phase.
- **HalfCheetah** — continuous-control locomotion under shifting physics (gravity, friction, wind), extending the evaluation beyond a single toy benchmark into higher-dimensional continuous action spaces.

Each domain runner trains every agent (ENA plus the three baselines) under the same shifting-environment protocol and reports comparable metrics: **Reliability** (%), per-environment average score, **Stability Gap** (performance drop at an environment switch), and **Efficiency** (score per wall-clock second).

---

## Repository Structure

```
app/
├── src/
│   ├── agents/
│   │   ├── tools/
│   │   │   ├── abstract.py             # AAgent — shared agent contract (act/train/set_learning)
│   │   │   ├── baseline.py             # ABaseline — shared contract for single-model RL baselines
│   │   │   └── normalizers.py          # Running/identity observation normalizers
│   │   ├── ena/
│   │   │   ├── ena.py                  # ENAgent — the Evolutionary Neuroplastic Agent
│   │   │   ├── hall_of_fame.py         # Archive of elite frozen specialists ("Gladiators")
│   │   │   ├── memory.py               # AgentMemory — reward/plasticity history & running stats
│   │   │   ├── recognizer.py           # EnvironmentRecognizer — detects & identifies env shifts
│   │   │   └── selection_controller.py # Specialist-selection strategies (heuristic + neural)
│   │   ├── dqn.py                      # DQNAgent — Deep Q-Network baseline
│   │   ├── ppo.py                      # PPOAgent — PPO-clip baseline (discrete + continuous)
│   │   └── sac.py                      # SACAgent — Soft Actor-Critic baseline (continuous)
│   ├── cartpole.py                     # CartPole experiment runner
│   ├── half_cheetah.py                 # HalfCheetah experiment runner
│   └── tools/
│       ├── metrics.py                  # Reliability / Stability Gap / Efficiency computation + plots
│       ├── visualization.py            # Live multi-experiment progress reporting
│       ├── aggregate_metrics.py        # Aggregates metrics across repeated experiment runs
│       └── plot_variance.py            # Publication-quality variance plots across runs
├── latex/                          # Full LaTeX source for the paper
├── requirements.txt                # Python dependencies
└── .pylintrc                       # Lint configuration (see STANDARDS.md)
hub/
└── python/Dockerfile               # Docker environment for running experiments
docker-compose.yml                  # Container orchestration for local development
```

See [STANDARDS.md](STANDARDS.md) for coding conventions and a closer look at how these pieces fit together.

---

## How to Run

### Option A — Google Colab

Click the badge at the top of this page. No local setup required.

### Option B — Docker (recommended for local development)

**1. Build and start the container**
```bash
docker compose up -d
```

**2. Open a shell inside it**
```bash
docker compose exec ena_experiment bash
```

The `app/` directory is bind-mounted to `/home/app` inside the container, so edits on the host are immediately reflected.

**3. Run an experiment**
```bash
python src/cartpole.py
# or
python src/half_cheetah.py
```

### Option C — Local Python

**1. Install dependencies**
```bash
pip install -r app/requirements.txt
```

**2. Run an experiment**
```bash
python app/src/cartpole.py
# or
python app/src/half_cheetah.py
```

**3. Aggregate results across repeated runs**
```bash
python app/src/tools/aggregate_metrics.py
```
Reads per-experiment output folders and computes mean, standard deviation, min, and max for every reported metric.

**4. Generate variance plots**
```bash
python app/src/tools/plot_variance.py
```
Produces `reliability_box.png` and `stability_bar.png` — the figures used in the paper.

---

## Scripts

### `src/cartpole.py`
Runs the ENA agent (and, where enabled, the DQN/PPO baselines) on CartPole under shifting gravity (Earth, Mars, Jupiter), then tests against training worlds plus held-out Neptune/Brown-dwarf phases. Writes per-experiment results, plots, and logs to an output directory.

### `src/half_cheetah.py`
Runs the same comparative protocol on MuJoCo HalfCheetah, varying gravity/friction/wind instead of pole physics, with SAC/PPO baselines for continuous-control comparison.

### `src/tools/metrics.py`
Computes the paper's core metrics (Reliability, Stability Gap, Efficiency) from raw episode histories and renders the academic comparison, plasticity, and specialist-transition figures.

### `src/tools/aggregate_metrics.py`
Aggregates metrics across all repeated experiment runs into a single summary report.

### `src/tools/plot_variance.py`
Produces the reliability and stability-gap variance figures used in the paper, directly from raw experiment output.

---

## Paper

The full IEEE two-column paper is available as LaTeX source under `app/latex/`. To compile:

1. Zip the `app/latex/` folder
2. Upload to [Overleaf](https://www.overleaf.com) and compile with `pdflatex`

All figures referenced in the paper are pre-rendered and stored in `app/latex/images/`.

---

## Future Work

Future work centers on strengthening the evidentiary basis of the ENA results: running the full DQN/PPO/SAC baseline comparison across both CartPole and HalfCheetah, scaling generation counts and population pools into deeper continuous domains, and tightening statistical rigor around the initialization-sensitivity findings. The long-term aim is to validate ENA's rapid-adaptation mechanisms against established baseline DRL frameworks under a broader, more demanding evaluation protocol.
