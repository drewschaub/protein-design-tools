# protein_design_tools/core/chain.py

from dataclasses import dataclass, field
from typing import List
from .residue import Residue


@dataclass
class Chain:
    """Represents a chain in a protein structure."""

    name: str
    residues: List[Residue] = field(default_factory=list)
