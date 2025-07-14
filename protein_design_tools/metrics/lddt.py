# protein_design_tools/metrics/lddt.py

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import jit


@jit
def compute_lddt_jax(P: jnp.ndarray, Q: jnp.ndarray, cutoff=8.0) -> jnp.ndarray:
    """
    Compute a simplified LDDT between two NxD JAX arrays using JIT compilation.

    Parameters
    ----------
    P : jnp.ndarray
        Mobile points, shape (N, D)
    Q : jnp.ndarray
        Target points, shape (N, D)
    cutoff : float
        Distance cutoff to consider neighbors (default: 8.0 Å)

    Returns
    -------
    jnp.ndarray
        Simplified LDDT score between P and Q (percentage)
    """
    N = P.shape[0]

    def compute_lddt_for_residue(i, acc):
        distances_p = jnp.linalg.norm(P[i] - P, axis=1)
        neighbors_p = jnp.where((distances_p <= cutoff) & (distances_p > 0))[0]

        distances_q = jnp.linalg.norm(Q[i] - Q, axis=1)
        neighbors_q = jnp.where((distances_q <= cutoff) & (distances_q > 0))[0]

        # Compute overlap
        overlap = jnp.intersect1d(neighbors_p, neighbors_q)
        lddt = (
            (overlap.size / neighbors_p.size) * 100.0 if neighbors_p.size > 0 else 100.0
        )
        return acc.at[i].set(lddt)

    lddt_scores = jax.vmap(compute_lddt_for_residue)(jnp.arange(N), jnp.zeros(N))
    lddt_avg = jnp.mean(lddt_scores)
    return lddt_avg


def compute_lddt_numpy(P: np.ndarray, Q: np.ndarray, cutoff=8.0) -> float:
    """
    Compute a simplified LDDT between two NxD NumPy arrays.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (N, D)
    Q : np.ndarray
        Target points, shape (N, D)
    cutoff : float
        Distance cutoff to consider neighbors (default: 8.0 Å)

    Returns
    -------
    float
        Simplified LDDT score between P and Q (percentage)
    """
    N = P.shape[0]
    lddt_scores = []
    for i in range(N):
        # Compute distances in P
        distances_p = np.linalg.norm(P[i] - P, axis=1)
        neighbors_p = np.where((distances_p <= cutoff) & (distances_p > 0))[0]

        # Compute distances in Q
        distances_q = np.linalg.norm(Q[i] - Q, axis=1)
        neighbors_q = np.where((distances_q <= cutoff) & (distances_q > 0))[0]

        # Compute overlap
        overlap = np.intersect1d(neighbors_p, neighbors_q)
        lddt = (
            (len(overlap) / len(neighbors_p)) * 100 if len(neighbors_p) > 0 else 100.0
        )
        lddt_scores.append(lddt)

    lddt_avg = np.mean(lddt_scores)
    return lddt_avg


def compute_lddt_pytorch(P: torch.Tensor, Q: torch.Tensor, cutoff=8.0) -> torch.Tensor:
    """
    Compute a simplified LDDT between two NxD PyTorch tensors.

    Parameters
    ----------
    P : torch.Tensor
        Mobile points, shape (N, D)
    Q : torch.Tensor
        Target points, shape (N, D)
    cutoff : float
        Distance cutoff to consider neighbors (default: 8.0 Å)

    Returns
    -------
    torch.Tensor
        Simplified LDDT score between P and Q (percentage)
    """
    N = P.shape[0]
    lddt_scores = []
    for i in range(N):
        # Compute distances in P
        distances_p = torch.norm(P[i] - P, dim=1)
        neighbors_p = torch.nonzero(
            (distances_p <= cutoff) & (distances_p > 0)
        ).squeeze()

        # Compute distances in Q
        distances_q = torch.norm(Q[i] - Q, dim=1)
        neighbors_q = torch.nonzero(
            (distances_q <= cutoff) & (distances_q > 0)
        ).squeeze()

        # Compute overlap
        overlap = torch.intersect1d(neighbors_p, neighbors_q)
        lddt = (
            (overlap.numel() / neighbors_p.numel()) * 100
            if neighbors_p.numel() > 0
            else 100.0
        )
        lddt_scores.append(lddt)

    lddt_avg = torch.mean(torch.stack(lddt_scores))
    return lddt_avg
