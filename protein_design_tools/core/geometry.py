# protein_design_tools/core/geometry.py
"""Rigid-body geometry shared by the alignment code."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotation ``R`` and translation ``t`` minimising ``|R @ P[i] + t - Q[i]|``
    over the paired points (Kabsch via SVD; reflections are rejected).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - cP).T @ (Q - cQ))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cQ - R @ cP
