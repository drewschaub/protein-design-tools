from protein_structure import ProteinStructure
from protein_structure_utils import get_radgy
from pathlib import Path

# Test the ProteinStructure class
protein = ProteinStructure()
protein.read_pdb("example.pdb", name="test")

# Print the protein structure
for chain in protein.chains:
    print(f"Chain {chain.chain_name}")
    for residue in chain.residues:
        print(f"Residue {residue.res_name} {residue.res_seq}")
        for atom in residue.atoms:
            print(f"Atom {atom.atom_name} at {atom.x}, {atom.y}, {atom.z}")
