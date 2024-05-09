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
protein_structure = ProteinStructure()
protein_structure.read_pdb("your.pdb")
radgyr = get_radgy(protein_structure, atom_type="backbone")

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Include information about the license. For example:

MIT
