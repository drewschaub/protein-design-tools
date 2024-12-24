# tests/core/test_residue.py

import pytest
from protein_design_tools.core.residue import Residue
from protein_design_tools.core.atom import Atom


def test_residue_one_letter_code():
    residue_ala = Residue(name="ALA", res_seq=1, i_code="")
    assert residue_ala.one_letter_code == "A"

    residue_arg = Residue(name="ARG", res_seq=2, i_code="")
    assert residue_arg.one_letter_code == "R"

    residue_unknown = Residue(name="XYZ", res_seq=3, i_code="")
    assert residue_unknown.one_letter_code is None  # Undefined mapping


def test_residue_atoms():
    residue = Residue(name="SER", res_seq=4, i_code="")
    atom_ca = Atom(
        atom_id=1,
        name="CA",
        alt_loc="",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        temp_factor=0.0,
        segment_id="",
        element="C",
        charge="",
    )
    residue.atoms.append(atom_ca)

    assert len(residue.atoms) == 1
    assert residue.atoms[0].name == "CA"
