# protein_design_tools/core/residue.py

from dataclasses import dataclass, field
from typing import List, Optional
from .atom import Atom

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass
class Residue:
    """Represents a residue in a protein structure."""

    name: str
    res_seq: int
    i_code: str
    atoms: List[Atom] = field(default_factory=list)

    @property
    def one_letter_code(self) -> Optional[str]:
        """Convert three-letter amino acid code to one-letter code."""
        return THREE_TO_ONE.get(self.name.upper())
