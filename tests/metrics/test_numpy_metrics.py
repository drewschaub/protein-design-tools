# tests/metrics/test_numpy_metrics.py
"""Reference values for the pure-NumPy metric implementations."""

import numpy as np
import pytest

from protein_design_tools.metrics import (
    compute_gdt_numpy,
    compute_lddt_numpy,
    compute_rmsd_numpy,
    compute_tmscore_numpy,
    rmsd,
)

L = 120
SHIFT = np.array([2.0, 2.0, 2.0])  # 2 Å along each axis
DIST = np.sqrt(12.0)  # so every point moves by 2*sqrt(3) = 3.464 Å
D0 = 1.24 * (L - 15) ** (1 / 3) - 1.8  # TM-score d0, 4.05 Å at L = 120
EXPECTED_TM = 1.0 / (1.0 + (DIST / D0) ** 2)


@pytest.fixture
def coords():
    return np.random.default_rng(0).uniform(-30.0, 30.0, size=(L, 3))


def test_identical_coordinates(coords):
    assert compute_rmsd_numpy(coords, coords) == pytest.approx(0.0)
    assert compute_gdt_numpy(coords, coords) == pytest.approx(100.0)
    assert compute_lddt_numpy(coords, coords) == pytest.approx(100.0)
    assert compute_tmscore_numpy(coords, coords) == pytest.approx(1.0)


def test_uniform_shift(coords):
    shifted = coords + SHIFT
    assert compute_rmsd_numpy(coords, shifted) == pytest.approx(DIST)
    # 3.464 Å passes the 4 Å and 8 Å thresholds and fails 1 Å and 2 Å -> 50 %
    assert compute_gdt_numpy(coords, shifted) == pytest.approx(50.0)
    # a rigid translation preserves every intra-structure distance
    assert compute_lddt_numpy(coords, shifted) == pytest.approx(100.0)
    tm = compute_tmscore_numpy(coords, shifted)
    assert tm == pytest.approx(EXPECTED_TM)
    assert tm == pytest.approx(0.5775, abs=1e-4)  # 1 / (1 + (3.464 / 4.05)^2)


def test_rmsd_dispatcher_numpy(coords):
    assert rmsd(coords, coords + SHIFT) == pytest.approx(DIST)
