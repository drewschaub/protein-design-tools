# tests/core/test_protein_structure.py

import pytest
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.core.chain import Chain
from protein_design_tools.core.residue import Residue
from protein_design_tools.core.atom import Atom


def test_protein_structure_sequence():
    structure = ProteinStructure(name="TestProtein")
    chain_a = Chain(name="A")
    residue1 = Residue(name="ALA", res_seq=1, i_code="")
    residue2 = Residue(name="ARG", res_seq=2, i_code="")
    chain_a.residues.extend([residue1, residue2])

    structure.chains.append(chain_a)

    sequences = structure.get_sequence_dict()
    assert sequences == {"A": "AR"}


def test_protein_structure_multiple_chains():
    structure = ProteinStructure(name="MultiChainProtein")
    chain_a = Chain(name="A")
    chain_b = Chain(name="B")

    residue_a1 = Residue(name="GLY", res_seq=1, i_code="")
    residue_a2 = Residue(name="SER", res_seq=2, i_code="")
    chain_a.residues.extend([residue_a1, residue_a2])

    residue_b1 = Residue(name="LYS", res_seq=1, i_code="")
    residue_b2 = Residue(name="MET", res_seq=2, i_code="")
    chain_b.residues.extend([residue_b1, residue_b2])

    structure.chains.extend([chain_a, chain_b])

    sequences = structure.get_sequence_dict()
    assert sequences == {
        "A": "GS",
        "B": "KM",
    }  # Corrected to include 'GLY' as 'G' and 'SER' as 'S'
