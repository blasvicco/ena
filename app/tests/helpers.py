"""Shared test-only utilities."""

# General imports
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def step(_label: str) -> Iterator[None]:
	"""Mark an Arrange/Act/Assert block for readability; the label is not used at runtime."""
	yield
