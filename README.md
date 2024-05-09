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

# Get the sequence of each chain in the protein
sequence_dict = protein.get_sequence()
for chain_id, sequence in sequence_dict.items():
    print(f"Chain {chain_id}: {sequence}")

# Calculate the radius of gyration
rg = protein_structure_utils.get_radgy(protein)
print(f"The radius of gyration of the protein structure is : {rg}")
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Include information about the license. For example:

MIT
