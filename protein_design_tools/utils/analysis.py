# protein_design_tools/utils/analysis.py

import numpy as np
from typing import List, Tuple, Optional, Dict
from ..core.chain import Chain
from ..core.protein_structure import ProteinStructure


def _nw_align(seq1: str, seq2: str) -> tuple[str, str]:
    """
    Minimal Needleman-Wunsch global alignment (identity scoring, gap = -1).

    Returns the two aligned strings (with '-').
    """
    m, n = len(seq1), len(seq2)
    # DP matrix
    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        score[i][0] = -i
    for j in range(1, n + 1):
        score[0][j] = -j

    # fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + (1 if seq1[i - 1] == seq2[j - 1] else -1)
            delete = score[i - 1][j] - 1
            insert = score[i][j - 1] - 1
            score[i][j] = max(match, delete, insert)

    # traceback
    aln1, aln2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and score[i][j]
            == score[i - 1][j - 1] + (1 if seq1[i - 1] == seq2[j - 1] else -1)
        ):
            aln1.append(seq1[i - 1])
            aln2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] - 1:
            aln1.append(seq1[i - 1])
            aln2.append("-")
            i -= 1
        else:
            aln1.append("-")
            aln2.append(seq2[j - 1])
            j -= 1
    return "".join(reversed(aln1)), "".join(reversed(aln2))


def _raw_overlap(
    chain1: Chain,
    chain2: Chain,
    match_res_names: bool = False,
) -> list[tuple[int, str, str]]:
    """
    Internal: residue-ID intersection between two Chain objects.
    Returns a sorted list of (res_seq, i_code, res_name1).
    """

    def key(r):  # tuple used for set operations
        return (r.res_seq, r.i_code, r.name if match_res_names else "")

    set1 = {key(r) for r in chain1.residues}
    set2 = {key(r) for r in chain2.residues}

    hits = sorted(set1 & set2, key=lambda x: (x[0], x[1]))
    out = []
    for res_seq, i_code, _ in hits:
        name = next(
            r.name
            for r in chain1.residues
            if r.res_seq == res_seq and r.i_code == i_code
        )
        out.append((res_seq, i_code, name))
    return out


def build_residue_map(seq_ref: str, seq_model: str) -> dict[int, int]:
    """
    Map residue numbers in the reference sequence → residue numbers in the model
    by global identity alignment.

    Parameters
    ----------
    seq_ref, seq_model : str
        Full amino-acid sequences (1-letter codes).

    Returns
    -------
    dict
        Keys are 1-based residue indices in `seq_ref`,
        values are 1-based indices in `seq_model`.
    """
    aln_ref, aln_mod = _nw_align(seq_ref, seq_model)
    ref_i = mod_i = 0
    mapping = {}
    for a, b in zip(aln_ref, aln_mod):
        if a != "-":
            ref_i += 1
        if b != "-":
            mod_i += 1
        if a != "-" and b != "-":
            mapping[ref_i] = mod_i
    return mapping


def coords_from_overlap(
    struct: ProteinStructure,
    chain_id: str,
    overlap: list[tuple[int, str, str]],
    atom_name: str = "CA",
) -> np.ndarray:
    """
    Given a ProteinStructure, a chain, and an overlap list of
    (res_seq, i_code, res_name), return an (N×3) array of
    [x,y,z] for `atom_name` in *exact* overlap order.
    """
    # 1) find chain
    chain = next((c for c in struct.chains if c.name == chain_id), None)
    if chain is None:
        return np.zeros((0, 3), dtype=float)

    pts = []
    for res_seq, i_code, _ in overlap:
        res = next(
            (r for r in chain.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        if not res:
            continue
        atom = next((a for a in res.atoms if a.name == atom_name), None)
        if atom:
            pts.append((atom.x, atom.y, atom.z))

    return np.array(pts, dtype=float)


def filter_overlap_by_atom(
    ref: ProteinStructure,
    mob: ProteinStructure,
    overlap: list[tuple[int, str, str]],
    chain_ref: str = "A",
    chain_mob: str = "A",
    atom_name: str = "CA",
) -> list[tuple[int, str, str]]:
    """
    Keep only those (res_seq, i_code, res_name) from *overlap* for which **both**
    structures contain *atom_name*.

    The routine is agnostic to residue renumbering: if the first residue of the
    model is 1 while the crystal starts at 13, supply an *overlap* list that
    already accounts for that shift (see `build_residue_map`).

    Parameters
    ----------
    ref, mob : ProteinStructure
        Reference and mobile structures.
    overlap : list of tuple
        Triplets ``(res_seq, i_code, res_name)`` describing candidate residues
        in *ref*.  Only those for which the corresponding residue in *mob*
        exists **and** both carry *atom_name* are kept.
    chain_ref, chain_mob : str, optional
        Chain identifiers in the two structures (default: "A").
    atom_name : str, optional
        Atom to require in both residues (default: "CA").

    Returns
    -------
    list of tuple
        Filtered overlap list, same tuple format as input.
    """
    c_ref = next((ch for ch in ref.chains if ch.name == chain_ref), None)
    c_mob = next((ch for ch in mob.chains if ch.name == chain_mob), None)
    if not c_ref or not c_mob:
        return []

    def has_atom(res, an):  # small helper
        return any(a.name == an for a in res.atoms)

    ok = []
    for res_seq, i_code, res_name in overlap:
        r_ref = next(
            (r for r in c_ref.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        r_mob = next(
            (r for r in c_mob.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        if (
            r_ref
            and r_mob
            and has_atom(r_ref, atom_name)
            and has_atom(r_mob, atom_name)
        ):
            ok.append((res_seq, i_code, res_name))
    return ok


def find_overlapping_residues(
    protein1: ProteinStructure,
    chain_id1: str,
    protein2: ProteinStructure,
    chain_id2: str,
    match_res_names: bool = False,
    index_map1_to_2: Optional[Dict[int, int]] = None,
) -> List[Tuple[int, str, str]]:
    """
    Find overlapping residues between two ProteinStructures for specific chains.

    Parameters
    ----------
    protein1 : ProteinStructure
        The first protein structure object.
    chain_id1 : str
        The chain ID in protein1 to consider (e.g. 'A').
    protein2 : ProteinStructure
        The second protein structure object.
    chain_id2 : str
        The chain ID in protein2 to consider (e.g. 'H').
    match_res_names : bool, optional
        If True, also require residue names (e.g., 'ALA') to match.
        If False (default), only match by residue number + insertion code.
    index_map1_to_2 : dict, optional
        Mapping *protein1* residue numbers ➜ *protein2* residue numbers.
        Create it with :func:`build_residue_map` when the two structures use
        different numbering (e.g. crystal begins at 13, model at 1).
        If supplied, `chain_id2` is ignored and `chain_id1` is used for both
        structures.

    Notes
    -----
    • When *index_map1_to_2* is given the function transparently renumbers
      *protein2* on the fly, so downstream code can work in the crystal’s
      numbering scheme.
    • Falls back to legacy behaviour when the mapping is *None*.
    """
    chain1 = next((ch for ch in protein1.chains if ch.name == chain_id1), None)
    chain2 = next((ch for ch in protein2.chains if ch.name == chain_id2), None)
    if not chain1 or not chain2:
        return []

    if index_map1_to_2:
        # Build a synthetic chain with renumbered residues for protein2
        from copy import deepcopy

        chain2 = deepcopy(chain2)  # local copy
        for r in chain2.residues:
            if r.res_seq in index_map1_to_2.values():
                # inverse lookup: model# -> crystal#
                new_num = next(k for k, v in index_map1_to_2.items() if v == r.res_seq)
                r.res_seq = new_num

    return _raw_overlap(chain1, chain2, match_res_names)
