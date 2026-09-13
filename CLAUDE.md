# CLAUDE.md

This is a research codebase backing an academic paper on the **Evolutionary Neuroplastic Agent (ENA)** — a population-based, non-gradient RL agent evaluated against DQN/PPO/SAC baselines across non-stationary CartPole and HalfCheetah domains.

See [STANDARDS.md](STANDARDS.md) for coding conventions and a full description of how the codebase is organized. Read it before writing or modifying Python code here.

## Dev workflow

Use Docker rather than installing dependencies on the host:
```bash
docker compose up -d
docker compose exec ena_experiment bash
```
`app/` is bind-mounted to `/home/app`, so edits made on the host are immediately visible inside the container. Run experiment scripts (`src/cartpole_challenge.py`, `src/half_cheetah.py`) from inside the container.

Every training run writes `live_training_{agent_name}.csv` to its output directory incrementally (one row per episode), and the terminal progress bar shows a live per-phase ETA. To watch convergence while a run is in progress: `streamlit run src/tools/live_dashboard.py` (port `8501` is mapped to the host in `docker-compose.yml`) or, on the host directly (not inside the container — it needs a real display), `python src/tools/live_monitor.py <path/to/live_training_*.csv>`.

## Extending the codebase

- New agents must implement the `AAgent` contract (`agents/tools/abstract.py`); single-model RL baselines should further follow `ABaseline` (`agents/tools/baseline.py`). Reuse `AgentMemory`, `normalizers.py` (`agents/tools/normalizers.py`), and `EnvironmentRecognizer` rather than re-implementing equivalent bookkeeping.
- `EnvironmentRecognizer` (`agents/ena/recognizer.py`) is a pluggable facade, same pattern as `SelectionController`: concrete strategies (`LiveRolloutRecognizer`, `ConfirmationWindowRecognizer`, `NeutralProbeRecognizer`) are selected by a `recognizer_strategy` string and composed via shared kwargs (e.g. `confirmation_window` works with any strategy). Add new environment-recognition approaches as another strategy class here, not as a special case in `ena.py`.
- New domain runners should stay structurally parallel to `cartpole_challenge.py`/`half_cheetah.py` (same experiment lifecycle: build agents, train, test against held-out phases, write results to an output directory).
- This project is tied to an academic paper (`app/latex/`). Preserve reproducibility (seeded, independent experiment runs) and keep the academic-reference header convention when touching an RL algorithm implementation.

## Things to leave alone

- Generated experiment output directories (`*_outputs*/`, `crash.log`, `niches_*.txt`) are run artifacts, not source — don't hand-edit them. They're already covered by `.gitignore` (`*outputs`).
- Don't commit local-only working files that aren't already tracked in git (e.g. scratch PDFs, personal notes) unless the user explicitly asks you to.
