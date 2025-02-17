# protein_design_tools/io/pdb.py

import gzip
from pathlib import Path
from typing import Optional, List
from ..core.protein_structure import ProteinStructure
from ..core.chain import Chain
from ..core.residue import Residue
from ..core.atom import Atom
import requests


def fetch_pdb(
    pdb_id: str,
    file_path: Optional[str] = None,
    chains: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> ProteinStructure:
    """
    Fetch a PDB file from RCSB PDB by its ID and optionally save it to a file.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure to fetch.
    file_path : str, optional
        Path to save the downloaded PDB file. If None, the file is not saved.
    chains : list of str, optional
        The chain identifiers to read. If None, all chains are read.
    name : str, optional
        The name of the protein structure.

    Returns
    -------
    ProteinStructure
        The parsed protein structure.

    """
    structure = ProteinStructure(name=name)

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        if file_path:
            with open(file_path, "w") as file:
                file.write(response.text)
        return _parse_pdb_content(response.text.splitlines(), chains, structure)
    else:
        print(f"Attempted URL: {url}")
        raise ValueError(
            f"Failed to fetch PDB ID {pdb_id}: HTTP status {response.status_code}. "
            f"No .pdb file exists for this entry. Consider using the `fetch_cif` "
            f"method for CIF format files."
        )


def read_pdb(
    file_path: str, chains: Optional[List[str]] = None, name: Optional[str] = None
) -> ProteinStructure:
    """
    Read a PDB file and return a ProteinStructure object.

    Parameters
    ----------
    file_path : str or file-like object
        The path to the PDB file or a file-like object containing PDB content.
    chains : list of str, optional
        The chain identifiers to read. If None, all chains are read.
    name : str, optional
        The name of the protein structure.

    Returns
    -------
    ProteinStructure
        The parsed protein structure.

    """
    structure = ProteinStructure(name=name)

    if isinstance(file_path, (str, Path)):
        p = Path(file_path)
        if p.suffix.lower() not in [".pdb", ".gz"]:
            raise ValueError("File must have a .pdb or .pdb.gz extension.")
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "rt") as f:
                content = f.readlines()
        else:
            content = p.read_text().splitlines()
    else:
        # Assume file_path is a file-like object
        content = file_path.read().splitlines()

    return _parse_pdb_content(content, chains, structure)


def _parse_pdb_content(
    content: List[str], chains: Optional[List[str]], structure: ProteinStructure
) -> ProteinStructure:
    """
    Parse PDB content and populate the ProteinStructure object.

    Parameters
    ----------
    content : list of str
        The PDB content split into lines.
    chains : list of str, optional
        The chain identifiers to read. If None, all chains are read.
    structure : ProteinStructure
        The ProteinStructure object to populate.

    Returns
    -------
    ProteinStructure
        The populated ProteinStructure object.

    """
    for line in content:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_id = int(line[6:11].strip())
            name = line[12:16].strip()
            alt_loc = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = int(line[22:26].strip())
            i_code = line[26].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occupancy = float(line[54:60].strip())
            temp_factor = float(line[60:66].strip())
            segment_id = line[72:76].strip()
            element = line[76:78].strip()
            charge = line[78:80].strip()

            # Filter chains if specified
            if chains and chain_id not in chains:
                continue

            # Find or create the chain
            chain = next((c for c in structure.chains if c.name == chain_id), None)
            if not chain:
                chain = Chain(name=chain_id)
                structure.chains.append(chain)

            # Find or create the residue
            residue = next(
                (
                    r
                    for r in chain.residues
                    if r.res_seq == res_seq and r.i_code == i_code
                ),
                None,
            )
            if not residue:
                residue = Residue(name=res_name, res_seq=res_seq, i_code=i_code)
                chain.residues.append(residue)

            # Create and add the atom
            atom = Atom(
                atom_id=atom_id,
                name=name,
                alt_loc=alt_loc,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                temp_factor=temp_factor,
                segment_id=segment_id,
                element=element,
                charge=charge,
            )
            residue.atoms.append(atom)

    return structure
