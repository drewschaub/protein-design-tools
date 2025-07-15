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
        # ignore insertion codes for matching
        return (r.res_seq, r.name if match_res_names else "")

    set1 = {key(r) for r in chain1.residues}
    set2 = {key(r) for r in chain2.residues}

    hits = sorted(set1 & set2, key=lambda x: x[0])
    out: List[tuple[int, str, str]] = []
    for res_seq, _ in hits:
        name = next(r.name for r in chain1.residues if r.res_seq == res_seq)
        out.append((res_seq, "", name))
    return out


def build_residue_map(seq_ref: str, seq_model: str) -> Dict[int, int]:
    """
    Map residue numbers in the reference sequence → residue numbers in the model
    by global identity alignment.
    """
    aln_ref, aln_mod = _nw_align(seq_ref, seq_model)
    ref_i = mod_i = 0
    mapping: Dict[int, int] = {}
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
    chain = next((c for c in struct.chains if c.name == chain_id), None)
    if chain is None:
        return np.zeros((0, 3), dtype=float)

    pts: List[tuple[float, float, float]] = []
    for res_seq, _, _ in overlap:
        res = next(
            (r for r in chain.residues if r.res_seq == res_seq),
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
    Keep only those (res_seq, i_code, res_name) from *overlap* for which both
    structures contain *atom_name*.
    """
    c_ref = next((ch for ch in ref.chains if ch.name == chain_ref), None)
    c_mob = next((ch for ch in mob.chains if ch.name == chain_mob), None)
    if not c_ref or not c_mob:
        return []

    def has_atom(res, an):
        return any(a.name == an for a in res.atoms)

    ok: List[tuple[int, str, str]] = []
    for res_seq, _, res_name in overlap:
        r_ref = next(
            (r for r in c_ref.residues if r.res_seq == res_seq),
            None,
        )
        r_mob = next(
            (r for r in c_mob.residues if r.res_seq == res_seq),
            None,
        )
        if (
            r_ref
            and r_mob
            and has_atom(r_ref, atom_name)
            and has_atom(r_mob, atom_name)
        ):
            ok.append((res_seq, "", res_name))
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
    """
    chain1 = next((ch for ch in protein1.chains if ch.name == chain_id1), None)
    chain2 = next((ch for ch in protein2.chains if ch.name == chain_id2), None)
    if not chain1 or not chain2:
        return []

    if index_map1_to_2:
        from copy import deepcopy

        chain2 = deepcopy(chain2)
        for r in chain2.residues:
            if r.res_seq in index_map1_to_2.values():
                new_num = next(k for k, v in index_map1_to_2.items() if v == r.res_seq)
                r.res_seq = new_num

    return _raw_overlap(chain1, chain2, match_res_names)


def debug_pair_table(
    coords_t: np.ndarray,
    coords_m: np.ndarray,
    labels: list[tuple[int, str]],
    model: ProteinStructure = None,
    after: bool = False,
) -> None:
    """
    Print a tabular summary of distances between two coordinate sets,
    and show the corresponding resname in model for verification.

    Parameters
    ----------
    coords_t : np.ndarray
        Template coordinates (N×3).
    coords_m : np.ndarray
        Model coordinates (N×3).
    labels : List of (res_seq, res_name)
        Each tuple: residue number and name in template.
    model : ProteinStructure, optional
        Model structure, used to look up resname for each res_seq.
    after : bool
        If True, indicate after-alignment.
    """
    title = "after Kabsch" if after else "before Kabsch"
    print(f"  idx | temp res   | model res  |  match? |   d(Å) ({title})")
    print("  ----+------------+-----------+---------+-------------")
    for i, ((x_t, y_t, z_t), (x_m, y_m, z_m)) in enumerate(zip(coords_t, coords_m)):
        res_seq, t_name = labels[i]
        # Try to find this residue in the model (by number only)
        m_name = "???"
        match = "?"
        if model is not None:
            chain = model.chains[0]
            r = next((r for r in chain.residues if r.res_seq == res_seq), None)
            if r:
                m_name = r.name
                match = "✓" if m_name == t_name else "✗"
        d = np.linalg.norm(np.array([x_t, y_t, z_t]) - np.array([x_m, y_m, z_m]))
        print(
            f"{i:5d} | {res_seq:4d} {t_name:>3}"
            f"   | {res_seq:4d} {m_name:>3}"
            f"  |   {match}   | {d:8.3f}"
        )


def align_sequences(seq_t: str, seq_m: str) -> dict[int, int]:
    """
    Simple global identity alignment (like Biopython globalxx) without deps.
    Returns mapping {template_resno -> model_resno} (1-based).
    """
    # initialize DP table
    n, m = len(seq_t), len(seq_m)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # fill
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = 1 if seq_t[i - 1] == seq_m[j - 1] else 0
            dp[i][j] = max(dp[i - 1][j - 1] + match, dp[i - 1][j], dp[i][j - 1])
    # traceback
    aln_t, aln_m = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j]
            == dp[i - 1][j - 1] + (1 if seq_t[i - 1] == seq_m[j - 1] else 0)
        ):
            aln_t.append(seq_t[i - 1])
            aln_m.append(seq_m[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j]:
            aln_t.append(seq_t[i - 1])
            aln_m.append("-")
            i -= 1
        else:
            aln_t.append("-")
            aln_m.append(seq_m[j - 1])
            j -= 1
    aln_t = "".join(reversed(aln_t))
    aln_m = "".join(reversed(aln_m))
    # build index map
    idx_map = {}
    it = im = 0
    for a_t, a_m in zip(aln_t, aln_m):
        if a_t != "-":
            it += 1
        if a_m != "-":
            im += 1
        if a_t != "-" and a_m != "-":
            idx_map[it] = im
    return idx_map
