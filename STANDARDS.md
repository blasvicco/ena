# Standards

This document describes how the codebase is organized and the coding conventions expected for Python code in this project.

## Project Structure

```
app/
├── src/
│   ├── agents/     # Agent definitions (dqn.py, ppo.py, sac.py), plus:
│   │   ├── tools/  # General agent-shared contract & utilities (abstract.py, baseline.py, normalizers.py)
│   │   └── ena/    # ENA-exclusive components (ena.py, memory.py, recognizer.py, hall_of_fame.py, selection_controller.py)
│   ├── cartpole.py, half_cheetah.py  # Domain runners — the only *.py directly under src/
│   └── tools/      # Metrics, visualization, and output post-processing scripts
├── latex/          # Paper source (independent of the codebase)
├── shell/          # do.sh — format (Black) + lint (Pylint) helper
├── tests/          # pytest suite, structurally parallel to src/
├── requirements.txt
└── .pylintrc
hub/python/Dockerfile
docker-compose.yml
```

### `app/src/agents/`

All agents share a common contract defined in `agents/tools/abstract.py` (`AAgent`): a `brain` factory (builds the underlying Keras model) and a `make_env` factory (builds the Gym-like environment) are injected at construction time, and every agent implements `act`, `train`, and `set_learning`. This is a strategy pattern — swapping `brain`/`make_env` retargets an agent to a new architecture or domain without touching its algorithm.

Each `.py` file directly under `agents/` is an agent definition only (`dqn.py`, `ppo.py`, `sac.py`; ENA's definition lives at `agents/ena/ena.py`, see below). Support code is split into two subfolders by scope:

`agents/tools/` — building blocks shared across multiple agents:
- `abstract.py` (`AAgent`) — the shared agent contract described above.
- `baseline.py` (`ABaseline`) — shared base for single-model RL baselines.
- `normalizers.py` — running/identity observation normalizers shared across agents.

`agents/ena/` — components exclusive to the ENA agent:
- `ena.py` (`ENAgent`) — the ENA agent itself; composition root wiring together the pieces below.
- `memory.py` (`AgentMemory`) — reward/plasticity history and running statistics, decoupled from `ENAgent`.
- `recognizer.py` (`EnvironmentRecognizer`) — detects environment shifts and identifies the current niche.
- `hall_of_fame.py` (`HallOfFame`) — archive of elite frozen specialists ("Gladiators"), gated by a staged pipeline (performance, continual improvement, genetic diversity, slot induction).
- `selection_controller.py` — specialist-selection strategies (`HeuristicSelectionDriver`, `NeuralSelectionDriver`) behind a common `SelectionController` facade.

`agents/dqn.py`, `agents/ppo.py`, `agents/sac.py` — DQN, PPO-clip, and SAC baseline implementations, built on `agents/tools/baseline.py`.

### `app/src/`

Domain runners (`cartpole.py`, `half_cheetah.py`) each build agent instances, run training + testing episodes across a shifting-physics protocol, and write results to a per-experiment output directory. They are the only `.py` files directly under `src/`; everything else lives in `src/tools/`: `metrics.py` computes the reported metrics (Reliability, Stability Gap, Efficiency) and renders the comparison figures; `visualization.py` handles live progress reporting; `aggregate_metrics.py` and `plot_variance.py` post-process results across repeated runs.

### Docker setup

`docker-compose.yml` defines a single `ena_experiment` service built from `hub/python/Dockerfile` (`python:3.11-slim`, with `build-essential`/`gcc` for native extensions and the OpenCV/MuJoCo system libraries required by `gymnasium[mujoco]`). `app/` is bind-mounted to `/home/app` so host edits are immediately visible in the container. The container itself doesn't run an experiment on start — it's kept alive with `tail -f /dev/null` so it can be entered via `docker compose exec ena_experiment bash` and used as an interactive dev environment. Python dependencies (`app/requirements.txt`) are installed at image build time, not on every container start.

---

## Python Coding Standards

These are the standards new and modified code should meet. Where existing code falls short, bring it up to standard in the lines you're already touching rather than leaving it as-is.

### Formatting
- Formatter: `black-with-tabs` — tabs instead of spaces for indentation.
- Linter: `pylint` against `app/.pylintrc`.
- `app/shell/do.sh <file.py> [<file.py> ...]` formats a file with Black then lints it with Pylint — run it on every modified file before committing.

### Rules & naming conventions
- Line Length: maximum 150 characters.
- Module Size: maximum 1000 lines per module.
- Indentation: Tabs (`\t`).
- Complexity limits: max arguments per function 5, max branches 12, max locals 15, max statements 50, max returns 6.
- `snake_case` for variables, arguments, attributes, functions, methods, and modules.
- `PascalCase` for classes.
- `UPPER_CASE` for constants.
- Allowed short names: `Run`, `_`.
- Prohibited names: `foo`, `bar`, `baz`, `toto`, `tutu`, `tata`, `i`, `j`, `k`.
- Unused or dummy variables must be prefixed with an underscore (e.g. `_unused`, `dummy`).

### Type hints
Required on all new public function/method signatures — parameters and return type. When modifying an existing untyped function, add hints to its signature as part of the change.

### Docstrings
One-line, imperative-mood docstring on every public class and function (e.g. `"""Compute the stability gap for a single agent."""`). RL algorithm implementations (`dqn.py`, `ppo.py`, `sac.py`, and similar) keep the existing academic header convention — algorithm name, original paper/author/year, and a link — directly above the class definition.

### Abstraction
Base classes must inherit `abc.ABC` and use `@abstractmethod` consistently, so abstract classes are actually non-instantiable — not just documented as abstract. When adding a new form of branching (e.g. a new mode or algorithm variant), prefer an explicit Strategy class (as in `selection_controller.py`) over string-built `getattr(self, f"_{mode}")` dispatch. The existing dispatch idiom in `ena.py`/`ppo.py` is legacy — don't extend it with new branches.

### Privacy convention
Use a single leading underscore for internal-use members. Reserve double-underscore (name-mangled) members for cases where mangling itself is the intent (e.g. avoiding subclass collisions in a deep composition root like `ena.py`) — not as a generic "more private" marker.

### Imports
Group imports into three tiers with a comment header for each:
```python
# General imports
import os

# Libs imports
import numpy as np

# App imports
from agents.abstract import AAgent
```

### Error handling
Catch specific exceptions only (e.g. `except AttributeError:` for a known dispatch fallback). Don't use broad `except Exception:` to swallow errors — let unexpected failures surface.

### Linting
`pylint` must pass against `app/.pylintrc` before committing, scoring 10.00/10. Notable overridden defaults worth knowing about: tab indentation, 150-character line limit, minimum 3-character names (with `Run`/`_` exempted), `max-args=5`, `max-statements=50`, `max-nested-blocks=5`.

### Tests
Real tests belong under a `tests/` directory using `pytest` with actual assertions. The `test_*.py` scripts currently at `app/` root are exploratory diagnostics (no assertions, executed at import time) — treat them as scratch scripts, not as the testing convention, and don't model new tests on them.

Structure every test in three labelled blocks using the `step(...)` context manager from `app/tests/helpers.py`:
```python
with step("Arrange: ..."):
	...
with step("Act: ..."):
	...
with step("Assert: ..."):
	...
```
Prefer `@pytest.mark.parametrize` over near-duplicate test functions whenever cases differ only by input/expected values — collapse them into one parametrized test rather than declaring one function per case.
