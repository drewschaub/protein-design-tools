# protein_design_tools/metrics/gdt.py

import numpy as np
import torch
import jax.numpy as jnp
from jax import jit


@jit
def compute_gdt_jax(
    P: jnp.ndarray, Q: jnp.ndarray, thresholds=[1, 2, 4, 8]
) -> jnp.ndarray:
    """
    Compute GDT-TS between two NxD JAX arrays using JIT compilation.

    Parameters
    ----------
    P : jnp.ndarray
        Mobile points, shape (N, D)
    Q : jnp.ndarray
        Target points, shape (N, D)
    thresholds : list of float
        Distance thresholds in Å (default: [1, 2, 4, 8])

    Returns
    -------
    jnp.ndarray
        GDT-TS score between P and Q
    """
    assert P.shape == Q.shape
    N = P.shape[0]
    distances = jnp.linalg.norm(P - Q, axis=1)
    percentages = []
    for t in thresholds:
        count = jnp.sum(distances <= t)
        percentage = (count / N) * 100.0
        percentages.append(percentage)
    gdt_ts = jnp.mean(jnp.array(percentages))
    return gdt_ts


def compute_gdt_numpy(P: np.ndarray, Q: np.ndarray, thresholds=[1, 2, 4, 8]) -> float:
    """
    Compute GDT-TS between two NxD NumPy arrays.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (N, D)
    Q : np.ndarray
        Target points, shape (N, D)
    thresholds : list of float
        Distance thresholds in Å (default: [1, 2, 4, 8])

    Returns
    -------
    float
        GDT-TS score between P and Q
    """
    N = P.shape[0]
    distances = np.linalg.norm(P - Q, axis=1)
    percentages = []
    for t in thresholds:
        count = np.sum(distances <= t)
        percentage = (count / N) * 100
        percentages.append(percentage)
    gdt_ts = np.mean(percentages)
    return gdt_ts


def compute_gdt_pytorch(
    P: torch.Tensor, Q: torch.Tensor, thresholds=[1, 2, 4, 8]
) -> torch.Tensor:
    """
    Compute GDT-TS between two NxD PyTorch tensors.

    Parameters
    ----------
    P : torch.Tensor
        Mobile points, shape (N, D)
    Q : torch.Tensor
        Target points, shape (N, D)
    thresholds : list of float
        Distance thresholds in Å (default: [1, 2, 4, 8])

    Returns
    -------
    torch.Tensor
        GDT-TS score between P and Q
    """
    N = P.shape[0]
    distances = torch.norm(P - Q, dim=1)
    percentages = []
    for t in thresholds:
        count = torch.sum(distances <= t).float()
        percentage = (count / N) * 100
        percentages.append(percentage)
    gdt_ts = torch.mean(torch.stack(percentages))
    return gdt_ts
