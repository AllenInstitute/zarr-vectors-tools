"""Shared pytest fixtures for zarr-vectors-tools tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy random generator for reproducible test data."""
    return np.random.default_rng(42)
