# protein_design_tools/metrics/__init__.py

from .rmsd import (
    compute_rmsd_numpy,
    compute_rmsd_pytorch,
    compute_rmsd_jax,
)
from .gdt import (
    compute_gdt_numpy,
    compute_gdt_pytorch,
    compute_gdt_jax,
)
from .lddt import (
    compute_lddt_numpy,
    compute_lddt_pytorch,
    compute_lddt_jax,
)
from .tmscore import (
    compute_tmscore_numpy,
    compute_tmscore_pytorch,
    compute_tmscore_jax,
)

__all__ = [
    "compute_rmsd_numpy",
    "compute_rmsd_pytorch",
    "compute_rmsd_jax",
    "compute_gdt_numpy",
    "compute_gdt_pytorch",
    "compute_gdt_jax",
    "compute_lddt_numpy",
    "compute_lddt_pytorch",
    "compute_lddt_jax",
    "compute_tmscore_numpy",
    "compute_tmscore_pytorch",
    "compute_tmscore_jax",
]

import numpy as _np

from .._optional import torch as _torch


def rmsd(P, Q):
    """
    Convenience dispatcher that chooses NumPy / PyTorch / JAX implementation
    based on input types.

    Inputs must already be superposed and in 1:1 correspondence; see
    :mod:`protein_design_tools.alignment.superpose`.
    """
    if isinstance(P, _np.ndarray):
        return compute_rmsd_numpy(P, Q)
    if _torch is not None and isinstance(P, _torch.Tensor):
        return compute_rmsd_pytorch(P, Q)
    return compute_rmsd_jax(P, Q)  # falls back to JAX


__all__.extend(["rmsd"])
