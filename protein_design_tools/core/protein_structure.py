# protein_design_tools/core/protein_structure.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List
from .chain import Chain
from ..utils.helpers import parse_residue_selection


@dataclass
class ProteinStructure:
    """Represents a protein structure and its components."""

    name: Optional[str] = None
    chains: List[Chain] = field(default_factory=list)

    def get_sequence_dict(self) -> Dict[str, str]:
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

    def get_coordinates(
        self,
        atom_type: str = "all",
        selection: Optional[Dict[str, Union[List[int], List[range]]]] = None,
    ) -> np.ndarray:
        """
        Retrieve coordinates based on atom type and residue selection.

        Parameters
        ----------
        atom_type : str, optional
            Type of atoms to retrieve ("all", "backbone", "CA", "non-hydrogen").
            Defaults to "all".
        selection : dict, optional
            Dictionary specifying residue selection per chain,
            e.g., {'A': [1,2,3], 'B': range(10, 20)}.

        Returns
        -------
        np.ndarray
            Array of coordinates.
        """

        coordinates = []
        parsed_selection = parse_residue_selection(selection) if selection else {}

        for chain in self.chains:
            if selection and chain.name not in parsed_selection:
                continue
            selected_residues = parsed_selection.get(
                chain.name, [res.res_seq for res in chain.residues]
            )

            for residue in chain.residues:
                if residue.res_seq not in selected_residues:
                    continue
                for atom in residue.atoms:
                    if atom_type == "all":
                        coordinates.append([atom.x, atom.y, atom.z])
                    elif atom_type == "backbone" and atom.name in ["N", "CA", "C", "O"]:
                        coordinates.append([atom.x, atom.y, atom.z])
                    elif atom_type == "CA" and atom.name == "CA":
                        coordinates.append([atom.x, atom.y, atom.z])
                    elif atom_type == "non-hydrogen" and atom.element != "H":
                        coordinates.append([atom.x, atom.y, atom.z])
        return np.array(coordinates)
