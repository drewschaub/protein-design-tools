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

# Get the radius of gyration for all atoms in the protein
radius_of_gyration = get_radgy(protein, atom_type="all")
print(f"Radius of gyration: {radius_of_gyration:.8f} Å")

# Get the radius of gyration for the backbone atoms
radius_of_gyration = get_radgy(protein, atom_type="backbone")
print(f"Radius of gyration (backbone): {radius_of_gyration:.8f} Å")

# Calculate the radius of gyration of all the alanine helices in ~/Downloads/alanines/
p = Path("../../../../Downloads/alanines/")

alanine_rg_dict_backbones = {}
alanine_rg_dict_all = {}
for i in range(0, 300):
    try:
        residue_count = i+2

        file = p / f"{i}_alanines.pdb"
        protein = ProteinStructure()
        protein.read_pdb(file, name=file.stem)
        radius_of_gyration = get_radgy(protein, atom_type="all")
        print(file)
        print(f"Radius of gyration for {residue_count} (all): {radius_of_gyration:.8f} Å")
        alanine_rg_dict_all[residue_count] = radius_of_gyration

        radius_of_gyration = get_radgy(protein, atom_type="backbone")
        print(f"Radius of gyration for {residue_count} (backbone): {radius_of_gyration:.8f} Å")

        alanine_rg_dict_backbones[residue_count] = radius_of_gyration

    except FileNotFoundError:
        continue

print(alanine_rg_dict_all)
print(alanine_rg_dict_backbones)
