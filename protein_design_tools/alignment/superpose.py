# protein_design_tools/alignment/superpose.py

from typing import Optional, Dict, Union, List, Tuple
import numpy as np

from ..core.protein_structure import ProteinStructure


def superpose_structures(
    mobile: ProteinStructure,
    target: ProteinStructure,
    atom_type: str = "CA",
    selection: Optional[Dict[str, Union[List[int], List[range]]]] = None,
    method: str = "kabsch",
    overlapping_residues: Optional[List[Tuple[int, str, str]]] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Superpose (align) the 'mobile' structure onto the 'target' structure using
    the specified alignment method (currently only 'kabsch'), optionally restricting
    to a list of overlapping residues.

    Parameters
    ----------
    mobile, target : ProteinStructure
        Structures to be aligned.
    atom_type : str
        Which atom type to use (e.g. "CA").
    selection : Dict[str, ...], optional
        Additional selection of residues.
    method : str
        Alignment method (only 'kabsch' supported).
    overlapping_residues : List[Tuple[int, str, str]], optional
        Overlapping residues for alignment, e.g. from find_overlapping_residues().

    Returns
    -------
    transform_matrix : np.ndarray
        A 4×4 homogeneous transformation matrix representing the rotation and
        translation that superposes 'mobile' onto 'target'.
    """
    if method.lower() == "kabsch":
        return _superpose_kabsch(
            mobile,
            target,
            atom_type,
            selection,
            overlapping_residues,
            debug=debug,  #  👈 forward the flag
        )
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _superpose_kabsch(
    mobile: ProteinStructure,
    target: ProteinStructure,
    atom_type: str,
    selection: Optional[Dict[str, Union[List[int], List[range]]]],
    overlapping_residues: Optional[List[Tuple[int, str, str]]],
    debug: bool = False,
) -> np.ndarray:
    """
    Internal function: Perform Kabsch superposition of 'mobile' onto 'target' for
    matching residues/atoms. Returns a 4x4 homogeneous transform.
    """

    # Build coords from overlap list, skipping missing atoms
    def _coords_from_overlap(
        struct: ProteinStructure,
        chain_id: str,
        overlap: list[tuple[int, str, str]],
        atom_name: str,
        debug: bool = False,   # new argument
    ) -> np.ndarray:
        # find the chain
        chain = next((c for c in struct.chains if c.name == chain_id), None)
        if chain is None:
            if debug:
                print(f"[DEBUG] _coords_from_overlap: Chain {chain_id} not found in {struct.name or struct}")
            return np.empty((0, 3))
        pts = []
        missing = []
        for res_seq, i_code, _ in overlap:
            res = next((r for r in chain.residues if r.res_seq == res_seq and r.i_code == i_code), None)
            if not res:
                missing.append(res_seq)
                continue
            atom = next((a for a in res.atoms if a.name == atom_name), None)
            if atom:
                pts.append([atom.x, atom.y, atom.z])
        if debug and missing:
            print(f"[DEBUG] _coords_from_overlap: Missing {atom_name} in residues {missing[:10]}{'...' if len(missing)>10 else ''} ({len(missing)} total)")
        return np.asarray(pts)


    if overlapping_residues is None:
        raise ValueError("Must supply overlapping_residues to _superpose_kabsch")

    # get CA/backbone/etc coords in the same order for both structures
    chain_id_ref = chain_id_mob = "A"  # or pull from function args if you generalize
    coords_t = _coords_from_overlap(
        target, chain_id_ref, overlapping_residues, atom_type, debug=debug
    )
    coords_m = _coords_from_overlap(
        mobile, chain_id_mob, overlapping_residues, atom_type, debug=debug
    )

    # cutoff if too few
    if coords_t.shape[0] < 3 or coords_m.shape[0] < 3:
        raise ValueError(
            f"Need ≥3 common {atom_type} atoms; found "
            f"{coords_t.shape[0]} vs {coords_m.shape[0]}."
        )

    # ensure same length
    n = min(len(coords_t), len(coords_m))
    coords_target = coords_t[:n]
    coords_mobile = coords_m[:n]

    if debug:
        print(f"[DEBUG] {atom_type} overlap: {coords_target.shape[0]} atoms")
        from protein_design_tools.utils.analysis import debug_pair_table

        labels = [(r[0], r[2]) for r in overlapping_residues]
        debug_pair_table(coords_target, coords_mobile, labels)

    # 1) Compute centroids
    centroid_t = np.mean(coords_target, axis=0)
    centroid_m = np.mean(coords_mobile, axis=0)

    # 2) Center
    T = coords_target - centroid_t
    M = coords_mobile - centroid_m

    # 3) Covariance
    H = M.T @ T
    U, S, Vt = np.linalg.svd(H)

    if debug:
        print(f"[DEBUG] SVD singular values: {S}")

    # 4) Rotation
    R = Vt.T @ U.T

    # 5) Reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6) Translation
    t = centroid_t - (R @ centroid_m)

    if debug:
        # apply R & t to the mobile coords and show post‐fit distances
        coords_m_fit = (R @ coords_mobile.T).T + t
        print("[DEBUG] distances _after_ Kabsch:")
        debug_pair_table(coords_target, coords_m_fit, labels)

    # 7) Build 4×4 transform
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix


def apply_transform(structure: ProteinStructure, transform: np.ndarray) -> None:
    """
    Apply a 4×4 homogeneous transformation matrix in-place to update the coordinates
    of all atoms in the given ProteinStructure.

    Parameters
    ----------
    structure : ProteinStructure
        The protein structure to modify.
    transform : np.ndarray
        A 4×4 homogeneous rotation+translation matrix.
    """
    if transform.shape != (4, 4):
        raise ValueError("Expected a 4x4 homogeneous transformation matrix.")

    for chain in structure.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                hom_coords = np.array([atom.x, atom.y, atom.z, 1.0])
                new_coords = transform @ hom_coords
                atom.x = new_coords[0]
                atom.y = new_coords[1]
                atom.z = new_coords[2]
