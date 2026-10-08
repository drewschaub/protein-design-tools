# protein_design_tools/metrics/rmsd.py

from __future__ import annotations

import numpy as np

from .._optional import jit, jnp, require_jax, require_torch, torch


@jit
def compute_rmsd_jax(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute RMSD between two NxD JAX arrays using JIT compilation.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : jnp.ndarray
        Mobile points, shape (N, D)
    Q : jnp.ndarray
        Target points, shape (N, D)

    Returns
    -------
    jnp.ndarray
        RMSD between P and Q
    """
    require_jax()
    assert P.shape == Q.shape
    return jnp.sqrt(jnp.mean(jnp.sum((P - Q) ** 2, axis=1)))


def compute_rmsd_numpy(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two NxD NumPy arrays.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (N, D)
    Q : np.ndarray
        Target points, shape (N, D)

    Returns
    -------
    float
        RMSD between P and Q
    """
    assert P.shape == Q.shape
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def compute_rmsd_pytorch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSD between two NxD PyTorch tensors.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : torch.Tensor
        Mobile points, shape (N, D)
    Q : torch.Tensor
        Target points, shape (N, D)

    Returns
    -------
    float
        RMSD between P and Q
    """
    require_torch()
    assert P.shape == Q.shape
    return torch.sqrt(torch.mean(torch.sum((P - Q) ** 2, dim=1)))
