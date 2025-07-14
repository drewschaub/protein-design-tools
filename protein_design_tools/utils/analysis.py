# protein_design_tools/utils/analysis.py

from typing import List, Tuple
from ..core.protein_structure import ProteinStructure

def filter_overlap_by_atom(
    ref: ProteinStructure,
    mob: ProteinStructure,
    overlap: list[tuple[int, str, str]],
    chain_ref: str = "A",
    chain_mob: str = "A",
    atom_name: str = "CA",
) -> list[tuple[int, str, str]]:
    """
    Filter an overlap list to only those residues where both structures have a given atom.

    Parameters
    ----------
    ref : ProteinStructure
        Reference protein structure.
    mob : ProteinStructure
        Mobile (to-be-aligned) protein structure.
    overlap : list of tuple
        List of residue identifiers produced by `find_overlapping_residues()`,
        each tuple `(res_seq, i_code, res_name)`.
    chain_ref : str, optional
        Chain ID in `ref` to consider (default: "A").
    chain_mob : str, optional
        Chain ID in `mob` to consider (default: "A").
    atom_name : str, optional
        Atom name to require in both structures (e.g. "CA", "N", "O") (default: "CA").

    Returns
    -------
    list of tuple
        A filtered list of `(res_seq, i_code, res_name)` containing only those
        residues for which **both** `ref` and `mob` have an atom named `atom_name`
        in the specified chains.

    Notes
    -----
    - Uses residue sequence number and insertion code from the original `overlap`.
    - If a residue is missing in either chain, or lacks the specified atom,
      it is dropped.
    - Useful to ensure equal‐length coordinate arrays before Kabsch superposition.
    """
    c_ref = next(ch for ch in ref.chains if ch.name == chain_ref)
    c_mob = next(ch for ch in mob.chains if ch.name == chain_mob)

    def has_atom(residue, aname):
        return any(a.name == aname for a in residue.atoms)

    filtered = []
    for res_seq, i_code, res_name in overlap:
        r_ref = next(
            (r for r in c_ref.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        r_mob = next(
            (r for r in c_mob.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        if r_ref and r_mob and has_atom(r_ref, atom_name) and has_atom(r_mob, atom_name):
            filtered.append((res_seq, i_code, res_name))
    return filtered


def find_overlapping_residues(
    protein1: ProteinStructure,
    chain_id1: str,
    protein2: ProteinStructure,
    chain_id2: str,
    match_res_names: bool = False,
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

    Returns
    -------
    List[Tuple[int, str, str]]
        A sorted list of overlapping residue tuples:
        (res_seq, i_code, res_name) for residues that exist in both chains.

    Notes
    -----
    - Overlap is based on residue sequence number + insertion code, and optionally
      residue name if match_res_names=True.
    - If either chain is not found, returns an empty list.
    - You can adapt this for more complex matching logic (like checking one-letter codes
      or alignment-based residue mapping).
    """
    # Retrieve the specified chains
    chain1 = next((ch for ch in protein1.chains if ch.name == chain_id1), None)
    chain2 = next((ch for ch in protein2.chains if ch.name == chain_id2), None)
    if not chain1 or not chain2:
        return []  # If either chain doesn't exist, no overlap

    # Build sets for each chain
    # Each residue is identified by (res_seq, i_code, res_name).
    # If match_res_names=False, we might ignore res_name or store it separately.
    residues1 = set()
    for r in chain1.residues:
        key = (r.res_seq, r.i_code, r.name if match_res_names else "")
        residues1.add(key)

    residues2 = set()
    for r in chain2.residues:
        key = (r.res_seq, r.i_code, r.name if match_res_names else "")
        residues2.add(key)

    # Intersection
    overlap = residues1.intersection(residues2)

    # Build sorted list
    # If we used empty string for the 3rd element when match_res_names=False,
    # we might want to retrieve actual r.name from each chain.
    # But for simplicity, we store as is.
    # We'll sort primarily by res_seq, then by i_code
    overlap_list = sorted(overlap, key=lambda x: (x[0], x[1]))  # (res_seq, i_code)

    # If match_res_names=False, that 3rd element might be "".
    # If it's non-empty, it means they matched names as well.

    # Return a standardized structure: (res_seq, i_code, res_name).
    # If we didn't store res_name, put a placeholder '???' or look it up
    # from chain1 or chain2. We'll do a small tweak here to handle both:
    results = []
    for res_seq, i_code, name_field in overlap_list:
        if match_res_names:
            # We have actual residue name
            res_name = name_field
        else:
            # We didn't store the name, let's just fetch from chain1 or chain2
            # for convenience. If it doesn't exist or differ, we can store "???".
            # We'll do a quick search:
            # This is optional. You might just return no name or an empty string.
            residue_in_chain1 = next(
                (
                    r
                    for r in chain1.residues
                    if r.res_seq == res_seq and r.i_code == i_code
                ),
                None,
            )
            res_name = residue_in_chain1.name if residue_in_chain1 else "???"

        results.append((res_seq, i_code, res_name))

    return results
