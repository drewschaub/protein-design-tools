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
    tm_d0,
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
    "tm_d0",
]

import numpy as _np

from .._optional import torch as _torch
from ..alignment.correspond import paired_coordinates as _paired_coordinates
from ..alignment.tmalign import tm_superpose as _tm_superpose
from ..core.protein_structure import ProteinStructure as _ProteinStructure
from ..core.selection import select as _select


def _prepare(P, Q, over, pairs, atom):
    """Two structures -> paired (N, 3) arrays; two arrays pass through."""
    is_structure = (isinstance(P, _ProteinStructure), isinstance(Q, _ProteinStructure))
    if all(is_structure):
        P, Q, used = _paired_coordinates(P, Q, pairs=pairs, over=over, atom=atom)
        if not used:
            raise ValueError(
                f"no paired residues carry atom {atom!r}; check 'over' and 'pairs'"
            )
        return P, Q
    if any(is_structure):
        raise TypeError("P and Q must both be ProteinStructure or both be arrays")
    if over is not None or pairs is not None:
        raise TypeError("'over' and 'pairs' apply only to ProteinStructure inputs")
    return P, Q


def _dispatch(P, Q, numpy_fn, torch_fn, jax_fn, **kwargs):
    """Pick the implementation from the array type: NumPy, PyTorch, else JAX."""
    if isinstance(P, _np.ndarray):
        return numpy_fn(P, Q, **kwargs)
    if _torch is not None and isinstance(P, _torch.Tensor):
        return torch_fn(P, Q, **kwargs)
    return jax_fn(P, Q, **kwargs)


def rmsd(P, Q, *, over=None, pairs=None, atom="CA"):
    """
    RMSD between two coordinate arrays or two structures.

    Arrays ``(N, D)`` must already be superposed and in 1:1 correspondence
    (row i of P pairs with row i of Q); the implementation is chosen from the
    array type (NumPy, PyTorch or JAX).  See
    :mod:`protein_design_tools.alignment.superpose`.

    Structures are compared through one ``atom`` per paired residue.  ``pairs``
    defaults to residues with the same chain ID, number and insertion code
    (pass :func:`~protein_design_tools.alignment.correspond.correspond` output
    otherwise), and ``over`` is a selection on ``Q``, the reference, that
    restricts which pairs count, e.g. fit on chain A and then
    ``rmsd(fit, ref, over="B")``.
    """
    P, Q = _prepare(P, Q, over, pairs, atom)
    return _dispatch(P, Q, compute_rmsd_numpy, compute_rmsd_pytorch, compute_rmsd_jax)


def tmscore(P, Q, *, over=None, pairs=None, atom="CA", L_ref=None, optimize=False):
    """
    TM-score between two coordinate arrays or two structures.

    Takes the same inputs as :func:`rmsd`.  ``L_ref`` normalises the score and
    sets d0; for arrays it defaults to N, for structures to the number of
    reference residues in ``over`` (or, given only ``pairs``, in the reference
    chains those pairs touch).

    By default the score is taken in the frame the inputs are in.  With
    ``optimize=True`` it is the TM-score proper: the maximum over all rigid
    superpositions of the paired points, found with the TM-score search
    (:func:`~protein_design_tools.alignment.tmalign.tm_superpose`, NumPy).
    """
    if pairs is not None:
        pairs = list(pairs)
    if L_ref is None and isinstance(Q, _ProteinStructure):
        if over is not None:
            L_ref = len(_select(Q, over))
        elif pairs is not None:
            chains = sorted({key_ref[0] for _, key_ref in pairs})
            L_ref = len(_select(Q, chains)) if chains else 0
        else:
            L_ref = len(_select(Q))
    P, Q = _prepare(P, Q, over, pairs, atom)
    if optimize:
        score, _, _ = _tm_superpose(
            _np.asarray(P, dtype=float), _np.asarray(Q, dtype=float), L_ref
        )
        return score
    return _dispatch(
        P,
        Q,
        compute_tmscore_numpy,
        compute_tmscore_pytorch,
        compute_tmscore_jax,
        L_ref=L_ref,
    )


def gdt(P, Q, *, over=None, pairs=None, atom="CA", thresholds=(1, 2, 4, 8)):
    """GDT-TS between two coordinate arrays or two structures; see :func:`rmsd`."""
    P, Q = _prepare(P, Q, over, pairs, atom)
    return _dispatch(
        P,
        Q,
        compute_gdt_numpy,
        compute_gdt_pytorch,
        compute_gdt_jax,
        thresholds=list(thresholds),
    )


def lddt(P, Q, *, over=None, pairs=None, atom="CA", cutoff=8.0):
    """
    Simplified lDDT between two coordinate arrays or two structures; see
    :func:`rmsd` for the inputs.  lDDT compares intra-structure distances, so
    the inputs need to be paired but not superposed.
    """
    P, Q = _prepare(P, Q, over, pairs, atom)
    return _dispatch(
        P, Q, compute_lddt_numpy, compute_lddt_pytorch, compute_lddt_jax, cutoff=cutoff
    )


__all__.extend(["rmsd", "tmscore", "gdt", "lddt"])
