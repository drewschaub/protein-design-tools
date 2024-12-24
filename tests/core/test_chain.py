# tests/core/test_chain.py

import pytest
from protein_design_tools.core.chain import Chain
from protein_design_tools.core.residue import Residue
from protein_design_tools.core.atom import Atom


def test_chain_residues():
    chain = Chain(name="A")
    residue1 = Residue(name="MET", res_seq=1, i_code="")
    residue2 = Residue(name="GLY", res_seq=2, i_code="A")

    chain.residues.extend([residue1, residue2])

    assert len(chain.residues) == 2
    assert chain.residues[0].name == "MET"
    assert chain.residues[1].res_seq == 2
    assert chain.residues[1].i_code == "A"
