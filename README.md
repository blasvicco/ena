# ENA — Evolutionary Neuroplastic Agent

> A preliminary study into evolutionary, non-gradient adaptation across non-stationary physical environments.

[![Open In Colab](https://colab.research.google.com/drive/1i8LY95hCwqbk4fcmAK12YT0lxH5TDzrW?usp=sharing)]

---

## Abstract

Traditional Reinforcement Learning (RL) agents often struggle to adapt rapidly in non-stationary environments, suffering catastrophic forgetting when facing sudden physical shifts. Drawing inspiration from biological meta-learning and evolutionary diversity, this preliminary study introduces the Evolutionary Neuroplastic Agent (ENA). The ENA dynamically maintains an active population of multiple lightweight neural network policies (a "pool of specialists") instead of a monolithic model. By continuously evaluating each specialist iteratively through defined *Trust* and *Plasticity* heuristics, the ENA autonomously triggers localized evolutionary search algorithms and substitutes specialists during environmental changes. We empirically evaluate two mechanisms for governing these shifts: a Quadratic Ratio formulation and a normalized Fuzzy-logic heuristic model. Evaluating this framework in a continuously shifting 1D CartPole domain transitioning across sequential gravities (Earth, Mars, Jupiter), we execute an online meta-learning testing protocol comparing frozen neural behaviors sequentially across those bodies tracking into a hidden fourth phase (Neptune). We demonstrate that the Fuzzy-logic framework manages inter-environment stability gaps smoothly. However, the study identifies critical limitations: the ENA exhibits massive standard-deviation dependencies on initial population states, confirming that robust adaptation within tightly constrained genetic boundaries remains highly vulnerable to initial parameter clustering. Our findings position the ENA as a promising proof-of-concept for dynamic Quality-Diversity adaptation, warranting further benchmarking against standard deep RL alternatives in complex domains.

---

## Repository Structure

```
app/
├── ena_test.py            # Main experiment script (10 independent runs)
├── aggregate_metrics.py   # Aggregates results across exp_01/ – exp_10/
├── plot_variance.py       # Generates reliability and stability plots
├── latex/                 # Full LaTeX source for the paper
└── requirements.txt       # Python dependencies
hub/
└── python/Dockerfile      # Docker environment for helper scripts
```

---

## How to Run

### Option A — Google Colab (recommended)

Click the badge at the top of this page. No local setup required. The full experiment was originally executed on Colab using free GPU resources.

### Option B — Local

**1. Install dependencies**
```bash
pip install -r app/requirements.txt
```

**2. Run the 10 independent experiments**
```bash
python app/ena_test.py
```
Each of the 10 runs creates its own fresh agent instances with independent random population seeds. Save the console output from each run into separate folders named `exp_01/outputs.txt`, `exp_02/outputs.txt`, and so on.

**3. Aggregate results across all 10 experiments**
```bash
python app/aggregate_metrics.py
```
Reads all `exp_##/outputs.txt` files and computes mean, standard deviation, min, and max for every reported metric.

**4. Generate variance plots**
```bash
python app/plot_variance.py
```
Produces `reliability_box.png` and `stability_bar.png` — the figures used in the paper.

---

## Scripts

### `ena_test.py`

Main experiment script. Runs 10 independent training + testing cycles across all three ENA configurations (ENA-01, ENA-02, ENA-03). Each cycle initializes fresh agents with independent random population seeds, ensuring full statistical independence between runs.

**Dependencies:** `gymnasium`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `tensorflow`

---

### `aggregate_metrics.py`

Reads per-experiment result files and computes aggregate statistics. Expects results stored in folders named `exp_01/` through `exp_10/`, each containing an `outputs.txt` file in the following format:

```
Outputs
Training:
--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---
       Agent  Env_0 Avg  Env_1 Avg  Env_2 Avg Reliability (%)  Stability Gap (↓)  Efficiency (Score/s)
ena_agent_01     311.73     468.60     353.81           63.5%             -124.2                 40.10
ena_agent_02     478.69     468.44     355.88           81.2%               26.3                 67.44

Test
--- ACADEMIC PERFORMANCE & RELIABILITY SUMMARY ---
       Agent  Env_0 Avg  Env_1 Avg  Env_2 Avg  Env_3 Avg Reliability (%)  Stability Gap (↓)
ena_agent_01     493.45      500.0     463.30      500.0           97.2%                0.0
ena_agent_02     490.19      500.0     500.00      500.0           99.2%                0.0
ena_agent_03     493.30      500.0     483.56      500.0           98.4%                0.0
```

---

### `plot_variance.py`

Generates two publication-quality figures directly from the raw `exp_##/outputs.txt` files:

- **`reliability_box.png`** — box plot of test-phase reliability distribution across 10 runs per configuration
- **`stability_bar.png`** — bar chart of stability gap (mean ± std) per configuration

---

## Paper

The full IEEE two-column paper is available as LaTeX source under `app/latex/`. To compile:

1. Zip the `app/latex/` folder
2. Upload to [Overleaf](https://www.overleaf.com) and compile with `pdflatex`

All figures referenced in the paper are pre-rendered and stored in `app/latex/images/`.

---

## Future Work

Future work will expand the analysis beyond the preliminary 1D kinematic CartPole benchmark by scaling generation counts and population pools sequentially into deeper continuous domains. By mathematically eroding the initialization bottleneck, we plan to directly benchmark ENA's overarching rapid-learning mechanisms continuously against established baseline DRL frameworks (such as PPO or Deep Q-Networks) to validate absolute operational supremacy.
