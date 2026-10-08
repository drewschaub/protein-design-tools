# tests/metrics/test_optional_backends.py
"""
The regression that matters: the NumPy metrics must import and run when
PyTorch and JAX are absent, and the backend-specific functions must then fail
with an ImportError naming the extra to install.

The backends are made "absent" by setting ``sys.modules[name] = None``, which
makes ``import name`` raise ImportError.  That runs in a subprocess so this
process' own imports are untouched, and it works whether or not the backends
are really installed.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ABSENT_BACKENDS_SCRIPT = r"""
import sys

for name in ("torch", "jax", "jax.numpy", "requests"):
    sys.modules[name] = None

import numpy as np
from protein_design_tools.metrics import (
    compute_gdt_jax,
    compute_gdt_numpy,
    compute_gdt_pytorch,
    compute_lddt_jax,
    compute_lddt_numpy,
    compute_lddt_pytorch,
    compute_rmsd_jax,
    compute_rmsd_numpy,
    compute_rmsd_pytorch,
    compute_tmscore_jax,
    compute_tmscore_numpy,
    compute_tmscore_pytorch,
    rmsd,
)

L = 120
P = np.random.default_rng(0).uniform(-30.0, 30.0, size=(L, 3))
Q = P + 2.0  # every point moves by sqrt(12) = 3.464 A
dist = 12 ** 0.5
d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8

assert compute_tmscore_numpy(P, P) == 1.0
assert abs(compute_tmscore_numpy(P, Q) - 1 / (1 + (dist / d0) ** 2)) < 1e-12
assert abs(compute_rmsd_numpy(P, Q) - dist) < 1e-12
assert compute_gdt_numpy(P, Q) == 50.0
assert compute_lddt_numpy(P, Q) == 100.0
assert abs(rmsd(P, Q) - dist) < 1e-12

for fn, extra in [
    (compute_rmsd_pytorch, "torch"),
    (compute_gdt_pytorch, "torch"),
    (compute_lddt_pytorch, "torch"),
    (compute_tmscore_pytorch, "torch"),
    (compute_rmsd_jax, "jax"),
    (compute_gdt_jax, "jax"),
    (compute_lddt_jax, "jax"),
    (compute_tmscore_jax, "jax"),
]:
    try:
        fn(P, Q)
    except ImportError as exc:
        assert f"protein-design-tools[{extra}]" in str(exc), (fn.__name__, exc)
    else:
        raise AssertionError(f"{fn.__name__} did not raise ImportError")

print("OK")
"""


def test_numpy_metrics_work_without_torch_jax_or_requests():
    result = subprocess.run(
        [sys.executable, "-c", ABSENT_BACKENDS_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
