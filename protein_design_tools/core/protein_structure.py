# protein_design_tools/core/protein_structure.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .chain import Chain


@dataclass
class ProteinStructure:
    """Represents a protein structure and its components."""

    name: Optional[str] = None
    chains: List[Chain] = field(default_factory=list)

    def get_sequence_dict(self) -> Dict[str, str]:
        """
        Retrieve the amino acid sequence for each chain.

        Returns
        -------
        dict
            A dictionary mapping chain names to their amino acid sequences.
        """
        sequences = {}
        for chain in self.chains:
            sequence = "".join(
                [
                    res.one_letter_code if res.one_letter_code else "X"
                    for res in chain.residues
                ]
            )
            sequences[chain.name] = sequence
        return sequences
