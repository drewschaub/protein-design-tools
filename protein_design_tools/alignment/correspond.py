# protein_design_tools/alignment/correspond.py
"""
Residue correspondence between two structures.

:func:`correspond` pairs residues by global sequence alignment (BLOSUM62,
affine gaps, free end gaps, so it survives point mutations and indels), by
residue number, or by position, and returns the pairs as residue keys.
:func:`paired_coordinates` turns pairs into two ``(N, 3)`` arrays in the same
order, which is what superposition and every metric consume.  NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ..core.chain import Chain
from ..core.protein_structure import ProteinStructure
from ..core.selection import (
    ResidueKey,
    Spec,
    atom_coordinates,
    residue_index,
    residue_key,
    select,
)
from ._blosum62 import BLOSUM62, INDEX
from .tmalign import tm_align

#: ``(key_in_a, key_in_b)``
Pair = Tuple[ResidueKey, ResidueKey]

_NEG = float("-inf")
_X = INDEX["X"]
# Alignment states: residues paired, a-residue against a gap, gap against b-residue.
_MATCH, _GAP_A, _GAP_B = 0, 1, 2


def needleman_wunsch(
    seq_a: str,
    seq_b: str,
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    penalize_end_gaps: bool = False,
) -> Tuple[str, str, float]:
    """
    Global alignment of two sequences (Gotoh's affine-gap Needleman-Wunsch)
    scored with BLOSUM62.

    A gap of length k costs ``gap_open + (k - 1) * gap_extend``.  End gaps are
    free unless ``penalize_end_gaps`` is set, so a tag or a disordered terminus
    present in only one sequence does not distort the rest of the alignment.
    Letters outside the BLOSUM62 alphabet score as ``X``.

    Returns the two gapped sequences and the alignment score.  Pure Python,
    O(len(seq_a) * len(seq_b)) time and memory; if it ever becomes the
    bottleneck, vectorise the row recurrences with NumPy.
    """
    a = [INDEX.get(c, _X) for c in seq_a.upper()]
    b = [INDEX.get(c, _X) for c in seq_b.upper()]
    n, m = len(a), len(b)

    def grid(fill):
        return [[fill] * (m + 1) for _ in range(n + 1)]

    # Best score of an alignment of a[:i], b[:j] ending in each state, and the
    # state of the cell it came from.
    M, A, B = grid(_NEG), grid(_NEG), grid(_NEG)
    pM, pA, pB = grid(_MATCH), grid(_MATCH), grid(_MATCH)
    M[0][0] = 0.0
    for i in range(1, n + 1):
        A[i][0] = gap_open + (i - 1) * gap_extend if penalize_end_gaps else 0.0
    for j in range(1, m + 1):
        B[0][j] = gap_open + (j - 1) * gap_extend if penalize_end_gaps else 0.0

    for i in range(1, n + 1):
        row = BLOSUM62[a[i - 1]]
        M_i, A_i, B_i = M[i], A[i], B[i]
        pM_i, pA_i, pB_i = pM[i], pA[i], pB[i]
        M_p, A_p, B_p = M[i - 1], A[i - 1], B[i - 1]
        for j in range(1, m + 1):
            # pair a[i-1] with b[j-1]
            best, ptr = M_p[j - 1], _MATCH
            if A_p[j - 1] > best:
                best, ptr = A_p[j - 1], _GAP_A
            if B_p[j - 1] > best:
                best, ptr = B_p[j - 1], _GAP_B
            M_i[j] = best + row[b[j - 1]]
            pM_i[j] = ptr
            # a[i-1] against a gap
            best, ptr = M_p[j] + gap_open, _MATCH
            if A_p[j] + gap_extend > best:
                best, ptr = A_p[j] + gap_extend, _GAP_A
            if B_p[j] + gap_open > best:
                best, ptr = B_p[j] + gap_open, _GAP_B
            A_i[j] = best
            pA_i[j] = ptr
            # gap against b[j-1]
            best, ptr = M_i[j - 1] + gap_open, _MATCH
            if A_i[j - 1] + gap_open > best:
                best, ptr = A_i[j - 1] + gap_open, _GAP_A
            if B_i[j - 1] + gap_extend > best:
                best, ptr = B_i[j - 1] + gap_extend, _GAP_B
            B_i[j] = best
            pB_i[j] = ptr

    # Where the alignment ends: the corner, or with free end gaps the best
    # cell of the last row or column (the rest is an unpaired tail).
    grids = ((_MATCH, M), (_GAP_A, A), (_GAP_B, B))
    i, j, state, best = n, m, _MATCH, M[n][m]
    for st, g in grids[1:]:
        if g[n][m] > best:
            best, state = g[n][m], st
    if not penalize_end_gaps:
        for jj in range(m):
            for st, g in grids:
                if g[n][jj] > best:
                    best, i, j, state = g[n][jj], n, jj, st
        for ii in range(n):
            for st, g in grids:
                if g[ii][m] > best:
                    best, i, j, state = g[ii][m], ii, m, st

    out_a: List[str] = []
    out_b: List[str] = []
    for k in range(m, j, -1):
        out_a.append("-")
        out_b.append(seq_b[k - 1])
    for k in range(n, i, -1):
        out_a.append(seq_a[k - 1])
        out_b.append("-")
    while i > 0 or j > 0:
        if i == 0:
            out_a.append("-")
            out_b.append(seq_b[j - 1])
            j -= 1
        elif j == 0:
            out_a.append(seq_a[i - 1])
            out_b.append("-")
            i -= 1
        elif state == _MATCH:
            out_a.append(seq_a[i - 1])
            out_b.append(seq_b[j - 1])
            state = pM[i][j]
            i -= 1
            j -= 1
        elif state == _GAP_A:
            out_a.append(seq_a[i - 1])
            out_b.append("-")
            state = pA[i][j]
            i -= 1
        else:
            out_a.append("-")
            out_b.append(seq_b[j - 1])
            state = pB[i][j]
            j -= 1
    return "".join(reversed(out_a)), "".join(reversed(out_b)), best


@dataclass
class Correspondence:
    """
    Ordered residue pairs ``(key_in_a, key_in_b)`` between two structures.

    Iterating or ``len()`` gives the pairs; ``a`` and ``b`` are the two key
    lists; ``identity`` is the fraction of pairs with the same residue name.
    For ``method="sequence"``, ``alignments`` holds the gapped sequences per
    chain pair and ``score`` the summed alignment score; for
    ``method="structure"`` they hold TM-align's gapped alignment and the
    TM-score normalised by ``b``'s length.
    """

    pairs: List[Pair]
    method: str
    identity: float = 0.0
    score: Optional[float] = None
    alignments: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def a(self) -> List[ResidueKey]:
        return [key_a for key_a, _ in self.pairs]

    @property
    def b(self) -> List[ResidueKey]:
        return [key_b for _, key_b in self.pairs]


def correspond(
    a: ProteinStructure,
    b: ProteinStructure,
    chain_a: Optional[str] = None,
    chain_b: Optional[str] = None,
    method: str = "sequence",
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    penalize_end_gaps: bool = False,
    fast: bool = False,
) -> Correspondence:
    """
    Pair the residues of ``a`` with the residues of ``b``.

    Parameters
    ----------
    a, b : ProteinStructure
    chain_a, chain_b : str, optional
        Pair this chain of ``a`` with this chain of ``b`` (one defaults to the
        other).  With neither given, every chain ID present in both structures
        is paired with its namesake, in ``a``'s chain order.
    method : {"sequence", "number", "index"}
        ``"sequence"``: global alignment of the one-letter sequences with
        BLOSUM62 and affine gaps (see :func:`needleman_wunsch`); survives point
        mutations and insertions/deletions.  ``"number"``: residues with the
        same number and insertion code.  ``"index"``: the i-th residue of one
        chain with the i-th of the other.  ``"structure"``: TM-align on the
        C-alpha traces (see :mod:`protein_design_tools.alignment.tmalign`),
        which needs no sequence similarity at all.
    gap_open, gap_extend, penalize_end_gaps
        Passed to :func:`needleman_wunsch` for ``method="sequence"``.
    fast : bool
        For ``method="structure"``: TM-align's faster, slightly less thorough
        search.

    Raises
    ------
    KeyError
        If a named chain is missing.
    ValueError
        For an unknown ``method``.
    """
    if method not in ("sequence", "structure", "number", "index"):
        raise ValueError(
            f"unknown method {method!r}; expected 'sequence', 'structure', "
            "'number' or 'index'"
        )
    pairs: List[Pair] = []
    alignments: Dict[Tuple[str, str], Tuple[str, str]] = {}
    score = 0.0
    for ca, cb in _chain_pairs(a, b, chain_a, chain_b):
        if method == "sequence":
            aln_a, aln_b, chain_score = needleman_wunsch(
                _sequence(ca), _sequence(cb), gap_open, gap_extend, penalize_end_gaps
            )
            alignments[(ca.name, cb.name)] = (aln_a, aln_b)
            score += chain_score
            ia = ib = 0
            for x, y in zip(aln_a, aln_b):
                if x != "-" and y != "-":
                    pairs.append(
                        (
                            residue_key(ca.name, ca.residues[ia]),
                            residue_key(cb.name, cb.residues[ib]),
                        )
                    )
                ia += x != "-"
                ib += y != "-"
        elif method == "structure":
            xa, ra = _ca_trace(ca)
            xb, rb = _ca_trace(cb)
            res = tm_align(
                xa,
                xb,
                [r.one_letter_code or "X" for r in ra],
                [r.one_letter_code or "X" for r in rb],
                fast=fast,
            )
            alignments[(ca.name, cb.name)] = (res.aligned_x, res.aligned_y)
            score += res.tm_norm_y
            for i, j in res.pairs:
                pairs.append((residue_key(ca.name, ra[i]), residue_key(cb.name, rb[j])))
        elif method == "number":
            by_number = {(r.res_seq, r.i_code or ""): r for r in cb.residues}
            for r in ca.residues:
                match = by_number.get((r.res_seq, r.i_code or ""))
                if match is not None:
                    pairs.append((residue_key(ca.name, r), residue_key(cb.name, match)))
        else:
            for ra, rb in zip(ca.residues, cb.residues):
                pairs.append((residue_key(ca.name, ra), residue_key(cb.name, rb)))

    index_a, index_b = residue_index(a), residue_index(b)
    same = sum(index_a[ka].name == index_b[kb].name for ka, kb in pairs)
    identity = same / len(pairs) if pairs else 0.0
    return Correspondence(
        pairs=pairs,
        method=method,
        identity=identity,
        score=score if method in ("sequence", "structure") else None,
        alignments=alignments,
    )


def paired_coordinates(
    mobile: ProteinStructure,
    ref: ProteinStructure,
    pairs: Optional[Iterable[Pair]] = None,
    over: Spec = None,
    atom: str = "CA",
) -> Tuple[np.ndarray, np.ndarray, List[Pair]]:
    """
    Coordinates of one ``atom`` per paired residue, as two ``(N, 3)`` arrays in
    the same order (``mobile`` first), plus the pairs actually used.

    ``pairs`` defaults to residues with the same chain ID, number and
    insertion code (``correspond(mobile, ref, method="number")``); pass
    :func:`correspond` output for structures that are numbered differently.
    ``over`` is a selection on ``ref`` (see
    :func:`~protein_design_tools.core.selection.select`); only pairs whose
    reference residue it contains are kept.  Pairs in which either residue
    lacks ``atom`` are dropped.

    Raises
    ------
    KeyError
        If a pair names a residue that is not in its structure.
    """
    if pairs is None:
        pairs = correspond(mobile, ref, method="number").pairs
    pairs = list(pairs)
    if over is not None:
        keep = set(select(ref, over))
        pairs = [pair for pair in pairs if pair[1] in keep]

    index_m, index_r = residue_index(mobile), residue_index(ref)
    P: List[Tuple[float, float, float]] = []
    Q: List[Tuple[float, float, float]] = []
    used: List[Pair] = []
    for key_m, key_r in pairs:
        residue_m, residue_r = index_m.get(key_m), index_r.get(key_r)
        if residue_m is None:
            raise KeyError(f"residue {key_m} is not in structure {mobile.name!r}")
        if residue_r is None:
            raise KeyError(f"residue {key_r} is not in structure {ref.name!r}")
        xyz_m, xyz_r = atom_coordinates(residue_m, atom), atom_coordinates(
            residue_r, atom
        )
        if xyz_m is None or xyz_r is None:
            continue
        P.append(xyz_m)
        Q.append(xyz_r)
        used.append((key_m, key_r))
    return (
        np.array(P, dtype=float).reshape(-1, 3),
        np.array(Q, dtype=float).reshape(-1, 3),
        used,
    )


def _chain_pairs(
    a: ProteinStructure,
    b: ProteinStructure,
    chain_a: Optional[str],
    chain_b: Optional[str],
) -> List[Tuple[Chain, Chain]]:
    chains_a = {chain.name: chain for chain in a.chains}
    chains_b = {chain.name: chain for chain in b.chains}
    if chain_a is None and chain_b is None:
        return [
            (chain, chains_b[chain.name])
            for chain in a.chains
            if chain.name in chains_b
        ]
    name_a = chain_a if chain_a is not None else chain_b
    name_b = chain_b if chain_b is not None else chain_a
    if name_a not in chains_a:
        raise KeyError(f"chain {name_a!r} not in structure {a.name!r}")
    if name_b not in chains_b:
        raise KeyError(f"chain {name_b!r} not in structure {b.name!r}")
    return [(chains_a[name_a], chains_b[name_b])]


def _sequence(chain: Chain) -> str:
    return "".join(residue.one_letter_code or "X" for residue in chain.residues)


def _ca_trace(chain: Chain):
    """C-alpha coordinates of the residues that have one, and those residues."""
    residues = [r for r in chain.residues if atom_coordinates(r, "CA") is not None]
    coords = [atom_coordinates(r, "CA") for r in residues]
    return np.array(coords, dtype=float).reshape(-1, 3), residues
