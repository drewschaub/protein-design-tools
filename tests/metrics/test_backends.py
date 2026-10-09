# tests/metrics/test_backends.py
"""PyTorch and JAX implementations; skipped when the backend is not installed."""

import numpy as np
import pytest

from protein_design_tools.metrics import (
    compute_gdt_jax,
    compute_gdt_pytorch,
    compute_lddt_jax,
    compute_lddt_pytorch,
    compute_rmsd_jax,
    compute_rmsd_pytorch,
    compute_tmscore_jax,
    compute_tmscore_pytorch,
    rmsd,
)

L = 120
DIST = np.sqrt(12.0)  # |(2, 2, 2)|
D0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
EXPECTED_TM = 1.0 / (1.0 + (DIST / D0) ** 2)


@pytest.fixture
def pq():
    P = np.random.default_rng(0).uniform(-30.0, 30.0, size=(L, 3))
    return P, P + 2.0


def test_pytorch_metrics(pq):
    torch = pytest.importorskip("torch")
    P, Q = (torch.as_tensor(a) for a in pq)
    assert float(compute_rmsd_pytorch(P, Q)) == pytest.approx(DIST)
    assert float(compute_gdt_pytorch(P, Q)) == pytest.approx(50.0)
    assert float(compute_tmscore_pytorch(P, Q)) == pytest.approx(EXPECTED_TM)
    half = compute_tmscore_pytorch(P[:60], Q[:60], L_ref=L)
    assert float(half) == pytest.approx(EXPECTED_TM / 2)
    assert float(rmsd(P, Q)) == pytest.approx(DIST)  # dispatcher picks PyTorch


@pytest.mark.xfail(strict=True, reason="pre-existing: torch.intersect1d does not exist")
def test_pytorch_lddt(pq):
    torch = pytest.importorskip("torch")
    P, Q = (torch.as_tensor(a) for a in pq)
    assert float(compute_lddt_pytorch(P, Q)) == pytest.approx(100.0)


def test_jax_metrics(pq):
    jnp = pytest.importorskip("jax.numpy")
    P, Q = (jnp.asarray(a) for a in pq)
    assert float(compute_rmsd_jax(P, Q)) == pytest.approx(DIST, abs=1e-4)
    assert float(compute_gdt_jax(P, Q)) == pytest.approx(50.0, abs=1e-4)
    assert float(compute_tmscore_jax(P, Q)) == pytest.approx(EXPECTED_TM, abs=1e-4)
    half = compute_tmscore_jax(P[:60], Q[:60], L_ref=L)
    assert float(half) == pytest.approx(EXPECTED_TM / 2, abs=1e-4)
    assert float(rmsd(P, Q)) == pytest.approx(DIST, abs=1e-4)  # dispatcher -> JAX


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: jnp.where inside jit/vmap needs a static size "
    "(ConcretizationTypeError)",
)
def test_jax_lddt(pq):
    jnp = pytest.importorskip("jax.numpy")
    P, Q = (jnp.asarray(a) for a in pq)
    assert float(compute_lddt_jax(P, Q)) == pytest.approx(100.0, abs=1e-4)
