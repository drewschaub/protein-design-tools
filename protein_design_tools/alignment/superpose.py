# protein_design_tools/alignment/superpose.py

from typing import Optional, Dict, Union, List
import numpy as np

from ..core.protein_structure import ProteinStructure
from ..utils.helpers import parse_residue_selection

def superpose_structures(
    mobile: ProteinStructure,
    target: ProteinStructure,
    atom_type: str = "CA",
    selection: Optional[Dict[str, Union[List[int], List[range]]]] = None,
    method: str = "kabsch",
) -> np.ndarray:
    """
    Superpose (align) the 'mobile' structure onto the 'target' structure using
    the specified alignment method (currently only 'kabsch').

    Returns
    -------
    transform_matrix : np.ndarray
        A 4×4 homogeneous transformation matrix representing the rotation and translation
        that superposes 'mobile' onto 'target'.
    """
    if method.lower() == "kabsch":
        return _superpose_kabsch(mobile, target, atom_type, selection)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _superpose_kabsch(
    mobile: ProteinStructure,
    target: ProteinStructure,
    atom_type: str,
    selection: Optional[Dict[str, Union[List[int], List[range]]]]
) -> np.ndarray:
    """
    Internal function: Perform Kabsch superposition of 'mobile' onto 'target' for matching
    residues/atoms. Returns a 4x4 homogeneous transform.
    """
    parsed_selection = parse_residue_selection(selection) if selection else None

    coords_target = target.get_coordinates(atom_type=atom_type, selection=parsed_selection)
    coords_mobile = mobile.get_coordinates(atom_type=atom_type, selection=parsed_selection)

    if coords_target.shape != coords_mobile.shape or coords_target.size == 0:
        raise ValueError(
            "Mismatch in coordinate arrays or zero-sized. "
            "Ensure correct residue overlap and atom_type."
        )

    # 1) Compute centroids
    centroid_t = np.mean(coords_target, axis=0)
    centroid_m = np.mean(coords_mobile, axis=0)

    # 2) Center
    T = coords_target - centroid_t
    M = coords_mobile - centroid_m

    # 3) Covariance
    H = M.T @ T
    U, S, Vt = np.linalg.svd(H)

    # 4) Rotation
    R = Vt.T @ U.T

    # 5) Reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6) Translation
    t = centroid_t - (R @ centroid_m)

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
