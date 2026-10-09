# protein_design_tools/metrics/tmscore.py

from __future__ import annotations

from typing import Optional

import numpy as np

from .._optional import jit, jnp, require_jax, require_torch, torch


def tm_d0(L: int) -> float:
    """
    TM-score distance scale ``d0(L)`` as TM-align sets it for the final score
    (``parameter_set4final`` in TMalign.cpp): 0.5 Å for ``L <= 21``, otherwise
    ``max(0.5, 1.24 * cbrt(L - 15) - 1.8)``.
    """
    if L <= 21:
        return 0.5
    return max(0.5, 1.24 * (L - 15) ** (1 / 3) - 1.8)


@jit
def compute_tmscore_jax(
    P: jnp.ndarray, Q: jnp.ndarray, L_ref: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute TM-score between two NxD JAX arrays using JIT compilation.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : jnp.ndarray
        Mobile points, shape (N, D)
    Q : jnp.ndarray
        Target points, shape (N, D)
    L_ref : int, optional
        Length of the reference protein, which normalises the score and sets
        d0.  Defaults to N, which is right only when the N pairs cover the
        whole reference; when they are an aligned subset, pass the reference
        chain's full length.

    Returns
    -------
    jnp.ndarray
        TM-score between P and Q
    """
    require_jax()
    L = P.shape[0] if L_ref is None else L_ref
    d0 = jnp.where(L <= 21, 0.5, jnp.maximum(0.5, 1.24 * jnp.cbrt(L - 15.0) - 1.8))
    distances = jnp.linalg.norm(P - Q, axis=1)
    tm_scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return jnp.sum(tm_scores) / L


def compute_tmscore_numpy(
    P: np.ndarray, Q: np.ndarray, L_ref: Optional[int] = None
) -> float:
    """
    Compute TM-score between two NxD NumPy arrays.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (N, D)
    Q : np.ndarray
        Target points, shape (N, D)
    L_ref : int, optional
        Length of the reference protein, which normalises the score and sets
        d0.  Defaults to N, which is right only when the N pairs cover the
        whole reference; when they are an aligned subset, pass the reference
        chain's full length.

    Returns
    -------
    float
        TM-score between P and Q
    """
    L = P.shape[0] if L_ref is None else L_ref
    d0 = tm_d0(L)
    distances = np.linalg.norm(P - Q, axis=1)
    tm_scores = 1 / (1 + (distances / d0) ** 2)
    return np.sum(tm_scores) / L


def compute_tmscore_pytorch(
    P: torch.Tensor, Q: torch.Tensor, L_ref: Optional[int] = None
) -> torch.Tensor:
    """
    Compute TM-score between two NxD PyTorch tensors.

    Inputs must already be superposed and in 1:1 correspondence (row i of P
    pairs with row i of Q); no alignment is performed here.  See
    :mod:`protein_design_tools.alignment.superpose`.

    Parameters
    ----------
    P : torch.Tensor
        Mobile points, shape (N, D)
    Q : torch.Tensor
        Target points, shape (N, D)
    L_ref : int, optional
        Length of the reference protein, which normalises the score and sets
        d0.  Defaults to N, which is right only when the N pairs cover the
        whole reference; when they are an aligned subset, pass the reference
        chain's full length.

    Returns
    -------
    torch.Tensor
        TM-score between P and Q
    """
    require_torch()
    L = P.shape[0] if L_ref is None else L_ref
    d0 = tm_d0(L)
    distances = torch.norm(P - Q, dim=1)
    tm_scores = 1 / (1 + (distances / d0) ** 2)
    return torch.sum(tm_scores) / L
