# Standards

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
Base classes must inherit `abc.ABC` and use `@abstractmethod` consistently, so abstract classes are actually non-instantiable — not just documented as abstract. When adding a new form of branching (e.g. a new mode or algorithm variant), prefer an explicit Strategy class (as in `selection_controller.py` and `recognizer.py`) over string-built `getattr(self, f"_{mode}")` dispatch. The existing dispatch idiom in `ena.py`/`ppo.py` is legacy — don't extend it with new branches. Where a strategy has independent, composable axes (e.g. `recognizer.py`'s data source vs. confirmation window), implement the shared axis on the common base class, parameterized, rather than duplicating it across mutually-exclusive subclasses.

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
