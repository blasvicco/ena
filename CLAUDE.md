# CLAUDE.md

This is a research codebase backing an academic paper on the **Evolutionary Neuroplastic Agent (ENA)** — a population-based, non-gradient RL agent evaluated against DQN/PPO/SAC baselines across non-stationary CartPole and HalfCheetah domains.

See [STANDARDS.md](STANDARDS.md) for coding conventions and a full description of how the codebase is organized. Read it before writing or modifying Python code here.

## Dev workflow

Use Docker rather than installing dependencies on the host:
```bash
docker compose up -d
docker compose exec ena_experiment bash
```
`app/` is bind-mounted to `/home/app`, so edits made on the host are immediately visible inside the container. Run experiment scripts (`src/cartpole_challenge.py`, `src/half_cheetah.py`, and the paper's CartPole studies: `src/cartpole_ena_ablation.py` — 30-replicate skill ablation — `src/cartpole_ena_compute.py` — compute saved by failure-driven Plasticity: one run at full effort, against the ablation's replicate 1 at zero and ramp effort — and `src/cartpole_ena_plasticity_controller.py` — linear-ramp Plasticity vs. quadratic and Mamdani fuzzy, paired against the ablation's ENA-Combined — and `src/cartpole_ena_selection.py` — Bayesian policy reuse vs. the simpler `niche_match` and `best_trust` test-time retrieval, paired against the ablation's ENA-Combined) from inside the container.

Every training run writes `live_training_{agent_name}.csv` to its output directory incrementally (one row per episode), and the terminal progress bar shows a live per-phase ETA.

## Cloud run

The HalfCheetah study runs in AWS from `infra/` (AWS CDK in TypeScript, run from the `ena_infra` toolbox service): one on-demand machine per replicate (`half_cheetah.py --replicate k`), results in S3, everything destroyable. The image is `hub/cloud/Dockerfile`, which ships only the files listed in `app/cloud_manifest.txt` (`tests/test_cloud_image.py` checks the study builds from them alone). The procedure is in `infra/README.md`; every AWS step needs the author's go-ahead.

## Extending the codebase

- New agents must implement the `AAgent` contract (`agents/tools/abstract.py`); single-model RL baselines should further follow `ABaseline` (`agents/tools/baseline.py`). Reuse `AgentMemory`, `normalizers.py` (`agents/tools/normalizers.py`), and `EnvironmentRecognizer` rather than re-implementing equivalent bookkeeping.
- `EnvironmentRecognizer` (`agents/ena/skills/env_detection/main.py`) is a pluggable facade, same pattern as `SelectionController` (`agents/ena/skills/selection/main.py`): concrete drivers in `driver/` (`dynamics_model`, `chirp`, `none`) are selected by the `recognizer_strategy` string. `dynamics_model` (per-niche linear dynamics models + PM-CuSum) is the one the CartPole experiments use, paired with the `bpr` selection driver; `chirp` (a fixed multiresolution excitation played at each episode end, its response's wavelet features compared per niche) is HalfCheetah's. Evaluate a detector offline with `tools/detection_benchmark.py` / `tools/detection_ablation.py` before any training run. Add new environment-recognition approaches as another driver there, not as a special case in `ena.py`.
- New domain runners should stay structurally parallel to `cartpole_challenge.py`/`half_cheetah.py` (same experiment lifecycle: build agents, train, test against held-out phases, write results to an output directory).
- This project is tied to an academic paper, in two formats: `app/latex_ieee/` (IEEEtran; the arXiv version, arXiv:2611.21598, built at tag `v1`) and `app/latex_tmlr/` (the anonymous TMLR submission). `src/tools/paper_results.py --paper latex_ieee|latex_tmlr` writes each folder's numbers, tables and figures; only the caption case differs. The TMLR folder keeps everything that identifies the authors in `deanonymized/`, and its supplement (built by `package_tmlr.sh` from the committed tree) ships `README.md`, `STANDARDS.md`, `docker-compose.yml` and `app/` minus the paper folders, so those must never name the authors, the repository or the arXiv ID. Preserve reproducibility (seeded, independent experiment runs) and keep the academic-reference header convention when touching an RL algorithm implementation.

## Things to leave alone

- Generated experiment output directories (`app/outputs/*_outputs*/`, `crash.log`, `niches_*.txt`) are run artifacts, not source — don't hand-edit them. They're already covered by `.gitignore` (`*outputs`).
- Don't commit local-only working files that aren't already tracked in git (e.g. scratch PDFs, personal notes) unless the user explicitly asks you to.
