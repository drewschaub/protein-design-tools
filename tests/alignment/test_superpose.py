# tests/alignment/test_superpose.py

import pytest
import numpy as np

from protein_design_tools.core.atom import Atom
from protein_design_tools.core.residue import Residue
from protein_design_tools.core.chain import Chain
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.alignment.superpose import (
    superpose_structures,
    apply_transform,
)


def test_superpose_kabsch():
    mobile = ProteinStructure()
    target = ProteinStructure()

    # Create chains
    mobile_chain = Chain("A")
    target_chain = Chain("A")
    mobile.chains.append(mobile_chain)
    target.chains.append(target_chain)

    # Add a few residues with CA atoms
    for i in range(3):
        # Provide i_code as an empty string if your constructor is: Residue(res_seq, name, i_code)
        mobile_res = Residue(i + 1, "ALA", "")
        target_res = Residue(i + 1, "ALA", "")

        # Create a CA Atom for each
        # Just put random coords
        atom_mobile = Atom(
            atom_id=i + 1,
            name="CA",
            alt_loc="",
            x=float(i),  # simple coords
            y=float(i + 0.5),
            z=float(i + 1),
            occupancy=1.0,
            temp_factor=0.0,
            segment_id="",
            element="C",
            charge="",
        )
        atom_target = Atom(
            atom_id=i + 1,
            name="CA",
            alt_loc="",
            x=float(i + 2),  # offset to create difference
            y=float(i + 2.5),
            z=float(i + 3),
            occupancy=1.0,
            temp_factor=0.0,
            segment_id="",
            element="C",
            charge="",
        )

        mobile_res.atoms.append(atom_mobile)
        target_res.atoms.append(atom_target)

        mobile_chain.residues.append(mobile_res)
        target_chain.residues.append(target_res)

    # Now we have a mobile and target structure each with 3 residues, each having a single CA.

    # 1) compute transformation via kabsch on CA
    transform = superpose_structures(
        mobile, target, atom_type="CA", selection=None, method="kabsch"
    )

    # 2) apply transform to mobile in place
    apply_transform(mobile, transform)

    # 3) extract final coords and compare to target
    final_mobile_coords = mobile.get_coordinates(atom_type="CA")
    final_target_coords = target.get_coordinates(atom_type="CA")

    # RMSD
    diff = final_mobile_coords - final_target_coords
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    print("final_mobile_coords:\n", final_mobile_coords)
    print("final_target_coords:\n", final_target_coords)
    print("RMSD:", rmsd)

    assert rmsd < 1e-5, f"Kabsch superpose RMSD too high: {rmsd}"
