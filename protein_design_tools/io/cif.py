# protein_design_tools/io/cif.py
"""
mmCIF (PDBx) I/O utilities.

Only the atom-level categories are parsed; metadata blocks (symmetry, secondary
structure, etc.) are ignored because the core modelling workflows (Boltz,
RFdiffusion, AlphaFold2 validation, …) only require atomic coordinates.

Public API
----------
fetch_cif(pdb_id, file_path=None, chains=None, name=None) -> ProteinStructure
read_cif(file_path, chains=None, name=None)               -> ProteinStructure
"""
from __future__ import annotations

import gzip
import shlex
from pathlib import Path
from typing import List, Optional

import requests

from ..core.protein_structure import ProteinStructure
from ..core.chain import Chain
from ..core.residue import Residue
from ..core.atom import Atom


def fetch_cif(
    pdb_id: str,
    file_path: Optional[str] = None,
    chains: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> ProteinStructure:
    """
    Fetch an mmCIF file from RCSB PDB and return it as a :class:`ProteinStructure`

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

    Notes
    -----
    * RCSB stores canonical mmCIF files at
      ``https://files.rcsb.org/download/{pdb_id}.cif``; no gzip wrapper.
    * For entries deposited *only* as gzip, we transparently gunzip in-memory.
    """
    structure = ProteinStructure(name=name)

    base_url = f"https://files.rcsb.org/download/{pdb_id}"
    tried: list[str] = []

    # Try to fetch the plan mmCIF
    url = f"{base_url}.cif"
    tried.append(url)
    r = requests.get(url)
    if r.ok:
        text = r.text
    else:
        # Try gzip fallback
        url_gz = f"{base_url}.cif.gz"
        tried.append(url_gz)
        r = requests.get(url_gz)
        if not r.ok:
            raise ValueError(
                f"Failed to fetch CIF for {pdb_id}. Tried: {', '.join(tried)} "
                f"(HTTP status {r.status_code})."
            )
        text = gzip.decompress(r.content).decode()

    if file_path:
        Path(file_path).write_text(text)

    return _parse_cif_content(text.splitlines(), chains, structure)


def read_cif(
    file_path: str, chains: Optional[List[str]] = None, name: Optional[str] = None
) -> ProteinStructure:
    """
    Read a local ``.cif`` or ``.cif.gz`` file (or file-like) into a
    :class:`ProteinStructure`.

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
        if p.suffix.lower() not in [".cif", ".gz"]:
            raise ValueError("File must have a .cif or .cif.gz extension.")
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "rt") as f:
                content = f.readlines()
        else:
            content = p.read_text().splitlines()
    else:  # file-like
        content = file_path.read().splitlines()

    return _parse_cif_content(content, chains, structure)


def _parse_cif_content(
    content: List[str],
    chains: Optional[List[str]],
    structure: ProteinStructure,
) -> ProteinStructure:
    """
    Minimalistic mmCIF parser that extracts coordinates from the ``loop_
    _atom_site`` table.  It deliberately avoids external dependencies to keep
    the library lightweight.

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
    field_names: list[str] = []
    chains_by_name: dict[str, Chain] = {}

    def _add_atom(rec: dict[str, str]) -> None:

        # In mmCIF . or ? means “unknown / not applicable” for any data item,
        # so the parser must cope with those placeholders.
        def _to_int(s: str, d: int = 0) -> int:
            return d if s in {".", "?", ""} else int(s)

        def _to_float(s: str, d: float = 0.0) -> float:
            return d if s in {".", "?", ""} else float(s)

        # prefer canonical (auth) ID, fall back to label ID
        label_chain = rec.get("chain_id", "")
        auth_chain = rec.get("auth_chain", "")
        chain_id = auth_chain or label_chain

        if chains and (label_chain not in chains and auth_chain not in chains):
            return

        # Canonical chain bookkeeping
        chain = chains_by_name.get(chain_id)
        if chain is None:
            chain = Chain(name=chain_id)
            chains_by_name[chain_id] = chain
            structure.chains.append(chain)

        # Residue - Use label_seq_id when present, else auth_seq_id
        raw_seq = rec.get("res_seq") or rec.get("auth_seq", ".")
        res_seq = _to_int(raw_seq, 0)
        i_code = rec.get("i_code", "").strip() or ""
        residue = next(
            (r for r in chain.residues if r.res_seq == res_seq and r.i_code == i_code),
            None,
        )
        if not residue:
            residue = Residue(
                name=rec.get("res_name", "").strip(),
                res_seq=res_seq,
                i_code=i_code,
            )
            chain.residues.append(residue)

        # Atom
        residue.atoms.append(
            Atom(
                atom_id=int(rec.get("atom_id", 0) or 0),
                name=rec.get("name", "").strip(),
                alt_loc=rec.get("alt_loc", "").strip(),
                x=float(rec.get("x", 0.0) or 0.0),
                y=float(rec.get("y", 0.0) or 0.0),
                z=float(rec.get("z", 0.0) or 0.0),
                occupancy=_to_float(rec.get("occupancy", ".")),
                temp_factor=_to_float(rec.get("b_factor", ".")),
                segment_id=rec.get("segment_id", "").strip(),
                element=rec.get("element", "").strip(),
                charge=rec.get("charge", "").strip(),
            )
        )

    # main-scan
    i = 0
    n = len(content)

    while i < n:
        line = content[i].strip()

        # Start of a loop
        if line == "loop_":
            # Capture subsequent data item names
            j = i + 1
            field_names.clear()
            while j < n and content[j].lstrip().startswith("_"):
                field_names.append(content[j].strip())
                j += 1

            # Interested only in atom_site loops
            if not any(f.startswith("_atom_site.") for f in field_names):
                i = j
                continue

            # Pre-compute indices for the fields we need
            idx = {
                key: field_names.index(field)
                for key, field in {
                    "atom_id": "_atom_site.id",
                    "name": "_atom_site.label_atom_id",
                    "alt_loc": "_atom_site.label_alt_id",
                    "res_name": "_atom_site.label_comp_id",
                    "chain_id": "_atom_site.label_asym_id",  # generic
                    "auth_chain": "_atom_site.auth_asym_id",  # canonical
                    "auth_seq": "_atom_site.auth_seq_id",
                    "i_code": "_atom_site.pdbx_PDB_ins_code",
                    "x": "_atom_site.Cartn_x",
                    "y": "_atom_site.Cartn_y",
                    "z": "_atom_site.Cartn_z",
                    "occupancy": "_atom_site.occupancy",
                    "b_factor": "_atom_site.B_iso_or_equiv",
                    "segment_id": "_atom_site.pdbx_segment_id",
                    "element": "_atom_site.type_symbol",
                    "charge": "_atom_site.pdbx_formal_charge",
                }.items()
                if field in field_names
            }

            # Consume data rows until the next category/loop_.
            i = j
            while i < n and not content[i].lstrip().startswith(("loop_", "_")):
                raw = shlex.split(content[i].rstrip())
                if len(raw) == len(field_names):  # well-formed row
                    _add_atom({k: raw[v] for k, v in idx.items()})
                i += 1
            continue

        i += 1

    return structure
