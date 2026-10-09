# protein_design_tools/alignment/superpose.py
"""
Rigid-body superposition.

:func:`superpose` fits ``mobile`` onto ``ref`` over paired residues (Kabsch)
and returns a :class:`Transform`; the fit can be restricted to a selection of
the reference (``on=``) while the transform still moves every atom, so one
part can be aligned and another measured.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from ..core.geometry import kabsch
from ..core.protein_structure import ProteinStructure
from ..core.selection import Spec
from .correspond import Pair, paired_coordinates


@dataclass
class Transform:
    """
    Rigid transform ``x -> rotation @ x + translation`` that superposes a
    mobile structure onto a reference, as returned by :func:`superpose`.

    ``rmsd`` and ``n`` describe the fit over the residue pairs that were used
    (kept in ``pairs``); applying the transform moves every atom.
    """

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    n: int
    pairs: List[Pair] = field(default_factory=list, repr=False)

    @property
    def matrix(self) -> np.ndarray:
        """The same transform as a 4x4 homogeneous matrix."""
        M = np.eye(4)
        M[:3, :3] = self.rotation
        M[:3, 3] = self.translation
        return M

    def apply_to(self, coords: np.ndarray) -> np.ndarray:
        """Transform an ``(N, 3)`` array of coordinates."""
        return np.asarray(coords, dtype=float) @ self.rotation.T + self.translation

    def apply(
        self, structure: ProteinStructure, inplace: bool = False
    ) -> ProteinStructure:
        """
        Move every atom of ``structure``.  Returns a transformed deep copy, or
        ``structure`` itself when ``inplace`` is set.
        """
        target = structure if inplace else copy.deepcopy(structure)
        atoms = [
            atom
            for chain in target.chains
            for residue in chain.residues
            for atom in residue.atoms
        ]
        if atoms:
            xyz = self.apply_to(np.array([[a.x, a.y, a.z] for a in atoms]))
            for atom, (x, y, z) in zip(atoms, xyz):
                atom.x, atom.y, atom.z = float(x), float(y), float(z)
        return target


def superpose(
    mobile: ProteinStructure,
    ref: ProteinStructure,
    on: Spec = None,
    pairs: Optional[Iterable[Pair]] = None,
    atom: str = "CA",
) -> Transform:
    """
    Least-squares superposition of ``mobile`` onto ``ref`` (Kabsch).

    One ``atom`` per paired residue is fitted.  ``pairs`` defaults to residues
    with the same chain ID, number and insertion code; pass the result of
    :func:`~protein_design_tools.alignment.correspond.correspond` when the two
    structures are numbered differently or have insertions/deletions.  ``on``
    is a selection on ``ref`` that restricts the fit without restricting what
    the returned transform moves, so you can align on one part and measure
    another::

        fit = superpose(model, ref, on="A").apply(model)  # fit on chain A
        rmsd(fit, ref, over="B")                          # judge chain B there

    Raises
    ------
    ValueError
        If fewer than three paired residues carry ``atom``.
    """
    P, Q, used = paired_coordinates(mobile, ref, pairs=pairs, over=on, atom=atom)
    if len(used) < 3:
        raise ValueError(
            f"need at least 3 paired residues with atom {atom!r} to superpose, "
            f"found {len(used)}; if the structures are numbered differently, "
            "pass pairs=correspond(mobile, ref)"
        )
    R, t = kabsch(P, Q)
    fit_rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return Transform(rotation=R, translation=t, rmsd=fit_rmsd, n=len(used), pairs=used)
