import numpy as np
import tensorflow.experimental.numpy as tnp
import torch
import jax.numpy as jnp
from jax import jit


@jit
def compute_tm_score_jax(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute TM-score between two NxD JAX arrays using JIT compilation.

    Parameters
    ----------
    P : jnp.ndarray
        Mobile points, shape (N, D)
    Q : jnp.ndarray
        Target points, shape (N, D)

    Returns
    -------
    jnp.ndarray
        TM-score between P and Q
    """
    L_ref = P.shape[0]
    d0 = 1.24 * (L_ref - 15) ** (1 / 3) - 1.8
    d0 = jnp.maximum(d0, 1.0)  # Ensure d0 is positive
    distances = jnp.linalg.norm(P - Q, axis=1)
    tm_scores = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = jnp.sum(tm_scores) / L_ref
    return tm_score


def compute_tm_score_numpy(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute TM-score between two NxD NumPy arrays.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (N, D)
    Q : np.ndarray
        Target points, shape (N, D)

    Returns
    -------
    float
        TM-score between P and Q
    """
    L_ref = P.shape[0]
    d0 = 1.24 * (L_ref - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 1.0)  # Ensure d0 is positive
    distances = np.linalg.norm(P - Q, axis=1)
    tm_scores = 1 / (1 + (distances / d0) ** 2)
    tm_score = np.sum(tm_scores) / L_ref
    return tm_score


def compute_tm_score_pytorch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute TM-score between two NxD PyTorch tensors.

    Parameters
    ----------
    P : torch.Tensor
        Mobile points, shape (N, D)
    Q : torch.Tensor
        Target points, shape (N, D)

    Returns
    -------
    torch.Tensor
        TM-score between P and Q
    """
    L_ref = P.shape[0]
    d0 = 1.24 * (L_ref - 15) ** (1 / 3) - 1.8
    d0 = torch.clamp(torch.tensor(d0), min=1.0).to(P.device)
    distances = torch.norm(P - Q, dim=1)
    tm_scores = 1 / (1 + (distances / d0) ** 2)
    tm_score = torch.sum(tm_scores) / L_ref
    return tm_score


def compute_tm_score_tensorflow(P: tnp.ndarray, Q: tnp.ndarray) -> tnp.ndarray:
    """
    Compute TM-score between two NxD TensorFlow tensors using tf.experimental.numpy.

    Parameters
    ----------
    P : tnp.ndarray
        Mobile points, shape (N, D)
    Q : tnp.ndarray
        Target points, shape (N, D)

    Returns
    -------
    tnp.ndarray
        TM-score between P and Q
    """
    L_ref = tnp.array(P.shape[0], dtype=P.dtype)
    d0 = 1.24 * tnp.power(L_ref - 15.0, 1 / 3) - 1.8
    d0 = tnp.maximum(d0, 1.0)  # Ensure d0 is positive
    distances = tnp.linalg.norm(P - Q, axis=1)
    tm_scores = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = tnp.sum(tm_scores) / L_ref
    return tm_score
