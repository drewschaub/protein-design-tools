![Banner](assets/banner.png)

<br>
<br>

A library of tools for protein design.

## Installation

Describe how to install your package. For example:

```bash
pip install protein-design-tools
```

## Usage
Provide some examples of how to use your package. For example:

### Calculate the radius of gyration of a protein structure's backbone

```python
from protein_design_tools import protein_structure, protein_structure_utils

protein = protein_structure.ProteinStructure()
protein.read_pdb("example.pdb")

# Display the amino acid sequence of the protein
# Get the sequence of each chain in the protein
sequence_dict = protein.get_sequence_dict()
for chain_id, sequence in sequence_dict.items():
    print(f"Chain {chain_id}: {sequence}")

# What is the radius of gyration of the backbone of the protein structure?
rg = protein_structure_utils.get_radgyr(protein, atom_type="backbone")
print(f"protein structure rg : {rg:.4f}")

# What is the radius of gyration of the backbone of an ideal alanine helix?
ideal_helix_seq_length = 75
rg_ideal_helix = protein_structure_utils.get_radgyr_alanine_helix(ideal_helix_seq_length)
print(f"ideal alanine helix rg : {rg_ideal_helix:.4f}")

# What is the radius of gyration ratio of the protein structure to an ideal alanine helix?
rg_ratio = protein_structure_utils.get_radgyr_ratio(protein, atom_type="backbone")
print(f"rg ratio of the protein structure to an ideal alanine helix : {rg_ratio:.4f}")
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Include information about the license. For example:

MIT
