# Protein-Design Tools

![Banner](assets/banner.png)

[![PyPI version](https://badge.fury.io/py/protein-design-tools.svg)](https://badge.fury.io/py/protein-design-tools)
![License](https://img.shields.io/badge/license-MIT-blue.svg) 
![Python Version](https://img.shields.io/pypi/pyversions/protein-design-tools)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Reading Protein Structures](#reading-protein-structures)
  - [Analyzing Sequences](#analyzing-sequences)
  - [Computing Structural Metrics](#computing-structural-metrics)
  - [Generating Idealized Structures](#generating-idealized-structures)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

**Protein-Design Tools** is a Python library tailored for structural bioinformatics, with a specific focus on protein design and engineering. It provides a suite of tools for analyzing and manipulating protein structures, enabling researchers and practitioners to perform complex structural comparisons, design new proteins, and engineer existing ones with ease.

Whether you're conducting research in protein folding, designing novel enzymes, or engineering therapeutic proteins, Protein-Design Tools offers the functionalities you need to advance your projects.

## Features

### **Protein Structure Representation**
- **Core Classes**:
  - `ProteinStructure`: Represents the entire protein structure.
  - `Chain`: Represents individual chains within the protein.
  - `Residue`: Represents residues within chains.
  - `Atom`: Represents individual atoms within residues.
- **File Parsing**:
  - **PDB Support**: Parse and read PDB files seamlessly.
  - **CIF Support**: Future support planned for CIF files.
- **Programmatic Construction**:
  - Build idealized protein structures (e.g., alpha helices) programmatically.

### **Structural Metrics**
Calculate structural metrics across multiple computational frameworks for flexibility and performance optimization:
- **RMSD (Root Mean Square Deviation)**: Measure the average distance between atoms of superimposed proteins.
- **TM-score**: Assess structural similarity normalized by protein length.
- **GDT-TS (Global Distance Test - Total Score)**: Evaluate global structural similarity using multiple distance thresholds.
- **LDDT (Local Distance Difference Test)**: Measure local structural accuracy.

### **Utilities**
- **Radius of Gyration**: Compute the radius of gyration for protein structures to assess compactness.
- **Sequence Analysis**: Extract and manipulate amino acid sequences from structures.

### **Input/Output Support**
- **File Operations**:
  - Read and write protein structures in PDB format.
  - Write FASTA sequences derived from 3D structure files.
- **Data Export**:
  - Export coordinates and other structural data in various formats, including HDF5.

### **Extensible Architecture**
- **Modular Design**: Easily add new metrics, file formats, and functionalities without disrupting existing components.
- **Multiple Frameworks**: Leverage the strengths of NumPy, PyTorch, and JAX for computational tasks.

## Installation

### 1. Choose the right requirements file

To keep the repo platform-agnostic, dependencies are split into small files in  
`requirements/`. Pick the one that matches your hardware/accelerator:

| File | When to use it | Key extra deps |
|------|----------------|----------------|
| **`requirements/cpu.txt`**   | CPU-only | `jax[cpu]` |
| **`requirements/cuda12.txt`**| NVIDIA GPU, CUDA 12 toolchain | `jax[cuda12]` (installs a CUDA-enabled `jaxlib` wheel) |
| **`requirements/tpu.txt`**   | Google Cloud TPU VMs | `jax[tpu]` + `libtpu` link |

All three files include `-r requirements/base.txt`, which lists NumPy 1.26,  
PyTorch ( CPU wheel by default ), FreeSASA, etc.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\activate.bat      # Windows CMD
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell
```

### 3. Install

CPU-only:

```bash
pip install -r requirements/cpu.txt
```

NVIDIA GPU:

```bash
pip install -r requirements/cuda12.txt
```

TPU VM:

```bash
pip install -r requirements/tpu.txt
```

### 4. Verify

```python
import numpy, torch, jax, jaxlib, freesasa
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("JAX :", jax.__version__,   "| jaxlib:", jaxlib.__version__)
```


## Quick Start

Here's a quick example to get you started with Protein-Design Tools:

```python
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.io.pdb_io import read_pdb
from protein_design_tools.metrics import compute_rmsd_numpy, compute_gdt_pytorch
from protein_design_tools.utils.coordinate_utils import get_coordinates, get_masses

# Reading a PDB file
protein = read_pdb("path/to/file.pdb", chains=['A', 'B'], name="Sample_Protein")

# Getting sequences
sequences = protein.get_sequence_dict()
print(sequences)

# Getting coordinates of all backbone atoms in chain A
coords = get_coordinates(protein, atom_type="backbone", chains={'A': range(1, 21)})

# Getting masses of all non-hydrogen atoms
masses = get_masses(protein, atom_type="non-hydrogen")
```

Several structural metrics are available, which are accessible across multiple computational frameworks
```python
from protein_design_tools.metrics import compute_rmsd_numpy, compute_gdt_pytorch
# Computing RMSD using NumPy
import numpy as np
import torch

P = np.random.rand(1000, 3)
Q = np.random.rand(1000, 3)
rmsd = compute_rmsd_numpy(P, Q)
print(f"RMSD (NumPy): {rmsd:.4f}")

# Computing GDT-TS using PyTorch
P_pt = torch.tensor(P)
Q_pt = torch.tensor(Q)
gdt = compute_gdt_pytorch(P_pt, Q_pt)
print(f"GDT-TS (PyTorch): {gdt:.2f}")
```

## Detailed Usage

### Reading Protein Structures

Protein-Design Tools supports reading and parsing protein structures from PDB files. Future updates will include CIF file support.

```python
from protein_design_tools.io.pdb_io import read_pdb

# Read all chains
protein = read_pdb("path/to/file.pdb")

# Read specific chains
protein = read_pdb("path/to/file.pdb", chains=['A', 'B'], name="My_Protein")
```

### Analyzing Sequences

Extract amino acid sequences from the protein structure.

```python
# Get the sequence of each chain in the protein
sequence_dict = protein.get_sequence_dict()
for chain_id, sequence in sequence_dict.items():
    print(f"Chain {chain_id}: {sequence}")
```

### Computing Structural Metrics

Leverage multiple frameworks to compute various structural metrics.

```python
from protein_design_tools.metrics import compute_rmsd_numpy, compute_gdt_pytorch

# Example data
import numpy as np
import torch

P = np.random.rand(1000, 3)
Q = np.random.rand(1000, 3)

# Compute RMSD using NumPy
rmsd = compute_rmsd_numpy(P, Q)
print(f"RMSD (NumPy): {rmsd:.4f}")

# Compute GDT-TS using PyTorch
P_pt = torch.tensor(P)
Q_pt = torch.tensor(Q)
gdt = compute_gdt_pytorch(P_pt, Q_pt)
print(f"GDT-TS (PyTorch): {gdt:.2f}")
```

### Generating Idealized Structures

Create idealized protein structures programmatically, such as an alpha helix.

```python
from protein_design_tools.io.builder import build_ideal_alpha_helix

# Build an idealized alpha helix with 10 residues
ideal_helix = build_ideal_alpha_helix(sequence_length=10, chain_id='A', start_res_seq=1)

# Display sequence
sequence_dict = ideal_helix.get_sequence_dict()
print(sequence_dict)
```

## Examples

### Calculating the Radius of Gyration

Calculate the radius of gyration for a protein and compare it to an idealized alpha helix.

```python
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.io.pdb_io import read_pdb
from protein_design_tools.metrics import compute_radgyr, compute_radgyr_ratio

# Read the protein structure
protein = read_pdb("example.pdb")

# Display the amino acid sequence of the protein
sequence_dict = protein.get_sequence_dict()
for chain_id, sequence in sequence_dict.items():
    print(f"Chain {chain_id}: {sequence}")

# Calculate the radius of gyration of the backbone of chain A
rgA = compute_radgyr(protein, chains={'A'}, atom_type="backbone")
print(f"Protein Structure Chain A Radius of Gyration: {rgA:.4f}")

# Calculate the radius of gyration of an ideal alanine helix
ideal_helix_seq_length = len(sequence_dict['A'])
rg_ideal_helix = compute_radgyr_alanine_helix(ideal_helix_seq_length, atom_type="backbone")
print(f"Ideal Alanine Helix Radius of Gyration: {rg_ideal_helix:.4f}")

# Calculate the radius of gyration ratio
rg_ratio = compute_radgyr_ratio(protein, chains={'A'}, atom_type="backbone")
print(f"Radius of Gyration Ratio: {rg_ratio:.4f}")
```

### Comparing Protein Structures Using TM-score

Assess the structural similarity between two protein structures.

```python
from protein_design_tools.metrics import compute_tmscore_numpy

# Assume P and Q are numpy arrays of shape (N, D) representing atom coordinates
P = np.random.rand(1000, 3)
Q = np.random.rand(1000, 3)

# Compute TM-score using NumPy
tm_score = compute_tmscore_numpy(P, Q)
print(f"TM-score (NumPy): {tm_score:.4f}")
```

### Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or adding new features, your help is greatly appreciated.

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page.
2. **Clone Your Fork**:
    ```bash
    git clone https://github.com/your-username/protein-design-tools.git
    ```
3. **Create a New Branch**:
    ```bash
    git checkout -b feature/YourFeatureName
    ```
4. **Make Your Changes**: Implement your feature or fix.
5. **Commit Your Changes**:
    ```bash
    git commit -m "Add feature: YourFeatureName"
    ```
6. **Push to Your Fork**:
    ```bash
    git push origin feature/YourFeatureName
    ```
7. **Create a Pull Request**: Go to the original repository and create a pull request from your fork.

For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

- Follow PEP8 style guidelines.
- Write clear and concise docstrings for all functions and classes.
- Include unit tests for new features or bug fixes.
- Ensure that existing tests pass before submitting a pull request. 

## License

This project is licensed under the MIT License.

## Contact

For any questions, suggestions, or contributions, please reach out:
- **Author**: Andrew Schaub
- **Linkedin**: https://www.linkedin.com/in/andrewjschaub
- **GitHub**: https://github.com/drewschaub/protein-design-tools

---

Thank you for using Protein-Design Tools! We hope it serves as a valuable resource in your structural bioinformatics and protein engineering endeavors.
