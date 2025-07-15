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
    to a list of overlapping residues.  If no overlap list is provided, we simply
    grab all atoms of the given type in chain 'A' and align them by index.
    """
    if method.lower() != "kabsch":
        raise ValueError(f"Unknown alignment method: {method}")

    # if no explicit overlap, align all atoms of type `atom_type` by order
    if overlapping_residues is None:
        # grab Nx3 arrays of coordinates
        coords_t = target.get_coordinates(atom_type=atom_type)
        coords_m = mobile.get_coordinates(atom_type=atom_type)
        if coords_t.shape[0] < 3 or coords_m.shape[0] < 3:
            raise ValueError(
                f"Need ≥3 {atom_type} atoms to align; found "
                f"{coords_t.shape[0]} vs {coords_m.shape[0]}"
            )
        # truncate to same length
        n = min(len(coords_t), len(coords_m))
        P = coords_t[:n]
        Q = coords_m[:n]

        # standard Kabsch
        cP = P.mean(axis=0)
        cQ = Q.mean(axis=0)
        X = P - cP
        Y = Q - cQ
        H = Y.T @ X
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = cP - R @ cQ

        M = np.eye(4, dtype=float)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    # otherwise use the existing overlap-based routine
    return _superpose_kabsch(
        mobile, target, atom_type, selection, overlapping_residues, debug=debug
    )


def _superpose_kabsch(
    mobile: ProteinStructure,
    target: ProteinStructure,
    atom_type: str,
    selection: Optional[Dict[str, Union[List[int], List[range]]]],
    overlapping_residues: Optional[List[Tuple[int, str, str]]],
    debug: bool = False,
) -> np.ndarray:
    """
    Internal function: Perform Kabsch superposition of 'mobile' onto 'target'
    for matching residues/atoms. Returns a 4×4 homogeneous transform.
    """

    def _coords_from_overlap(
        struct: ProteinStructure,
        chain_id: str,
        overlap: List[Tuple[int, str, str]],
        atom_name: str,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Build an (N×3) array of atom_name coords for this struct,
        using overlap tuples (ref_seq, i_code, mob_seq).
        """
        chain = next((c for c in struct.chains if c.name == chain_id), None)
        if chain is None:
            if debug:
                msg = (
                    "[DEBUG] _coords_from_overlap: no chain "
                    f"{chain_id} in {struct.name}"
                )
                print(msg)
            return np.empty((0, 3))

        pts = []
        missing = []
        for ref_seq, i_code, mob_seq in overlap:
            # pick the right residue number:
            # - for target struct, use ref_seq
            # - for mobile struct, use mob_seq
            want_seq = mob_seq if struct is mobile else ref_seq

            # match insertion code if provided
            res = next(
                (
                    r
                    for r in chain.residues
                    if r.res_seq == want_seq
                    and (not i_code or (r.i_code or "") == i_code)
                ),
                None,
            )
            if not res:
                missing.append(want_seq)
                continue

            atom = next((a for a in res.atoms if a.name == atom_name), None)
            if atom:
                pts.append([atom.x, atom.y, atom.z])

        if debug and missing:
            to_show = missing[:10]
            more = "..." if len(missing) > 10 else ""
            print(
                "[DEBUG] _coords_from_overlap: Missing "
                f"{atom_name} on residues {to_show}{more} "
                f"({len(missing)} total)"
            )

        return np.asarray(pts, dtype=float)

    if overlapping_residues is None:
        raise ValueError("Must supply overlapping_residues to _superpose_kabsch")

    # fixed chain IDs here; could be parameterized later
    chain_id_ref = chain_id_mob = "A"

    # build coordinate arrays
    coords_t = _coords_from_overlap(
        target, chain_id_ref, overlapping_residues, atom_type, debug=debug
    )
    coords_m = _coords_from_overlap(
        mobile, chain_id_mob, overlapping_residues, atom_type, debug=debug
    )

    # need at least 3 matching points
    if coords_t.shape[0] < 3 or coords_m.shape[0] < 3:
        msg = (
            f"Need ≥3 common {atom_type} atoms; found "
            f"{coords_t.shape[0]} vs {coords_m.shape[0]}."
        )
        raise ValueError(msg)

    # truncate to equal length
    n = min(len(coords_t), len(coords_m))
    P = coords_t[:n]
    Q = coords_m[:n]

    if debug:
        print(f"[DEBUG] {atom_type} overlap count: {P.shape[0]}")

    # 1) centroids
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)

    # 2) center
    X = P - cP
    Y = Q - cQ

    # 3) covariance
    H = Y.T @ X
    U, S, Vt = np.linalg.svd(H)
    if debug:
        print(f"[DEBUG] SVD singular values: {S}")

    # 4) rotation
    R = Vt.T @ U.T
    # 5) reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6) translation
    t = cP - R @ cQ

    if debug:
        # optional distance check after fit
        fit = (R @ Q.T).T + t
        from protein_design_tools.utils.analysis import debug_pair_table

        # labels not needed here, just distances
        debug_pair_table(P, fit, [(int(r[0]), int(r[2])) for r in overlapping_residues])

    # build 4×4 homogeneous transform
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    M[:3, 3] = t

    return M


def apply_transform(structure: ProteinStructure, transform: np.ndarray) -> None:
    """
    Apply a 4×4 homogeneous transformation matrix in-place to
    update all atom coordinates in the given ProteinStructure.
    """
    if transform.shape != (4, 4):
        raise ValueError("Expected a 4×4 homogeneous transform matrix.")

    for chain in structure.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                x, y, z = atom.x, atom.y, atom.z
                new = transform @ np.array([x, y, z, 1.0])
                atom.x, atom.y, atom.z = new[:3]
