# protein_design_tools/metrics/rmsd.py

import numpy as np
import torch
import jax.numpy as jnp
from jax import jit


@jit
def compute_rmsd_jax(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute RMSD between two NxD JAX arrays using JIT compilation.

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
    assert P.shape == Q.shape
    return jnp.sqrt(jnp.mean(jnp.sum((P - Q) ** 2, axis=1)))


def compute_rmsd_numpy(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two NxD NumPy arrays.

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
    assert P.shape == Q.shape
    return torch.sqrt(torch.mean(torch.sum((P - Q) ** 2, dim=1)))
