from protein_structure import ProteinStructure
import protein_structure_utils
from pathlib import Path

# Test the ProteinStructure class
protein = ProteinStructure()
protein.read_pdb("example.pdb", name="test")

# Print the protein structure
for chain in protein.chains:
    print(f"Chain {chain.name}")
    for residue in chain.residues:
        print(f"Residue {residue.name} {residue.res_seq}")
        for atom in residue.atoms:
            print(f"Atom {atom.name} at {atom.x}, {atom.y}, {atom.z}")

protein = ProteinStructure()
protein.read_pdb("example.pdb")

# Display the amino acid sequence of the protein
# Get the sequence of each chain in the protein
sequence_dict = protein.get_sequence_dict()
for chain_id, sequence in sequence_dict.items():
    print(f"Chain {chain_id}: {sequence}")

# What is the radius of gyration of the backbone of chain A in our protein structure?
rgA = protein_structure_utils.get_radgyr(protein, chains='A', atom_type="backbone")
print(f"protein structure chain A rg : {rgA:.4f}")

# What is the radius of gyration of the backbone of an ideal alanine helix?
ideal_helix_seq_length = len(sequence_dict['A'])
rg_ideal_helix = protein_structure_utils.get_radgyr_alanine_helix(ideal_helix_seq_length, atom_type="backbone")
print(f"ideal alanine helix rg : {rg_ideal_helix:.4f}")

# What is the radius of gyration ratio of the protein structure to an ideal alanine helix?
rg_ratio = protein_structure_utils.get_radgyr_ratio(protein, chains='A', atom_type="backbone")
print(f"rg ratio : {rg_ratio:.4f}")