# protein_design_tools/core/protein_structure.py

from __future__ import annotations
from typing import ClassVar, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np

from .chain import Chain
from .residue import Residue
from ..utils.helpers import parse_residue_selection


@dataclass
class ProteinStructure:
    """Represents a protein structure and its components."""

    STANDARD_RESIDUES: ClassVar[set[str]] = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

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
        chain: Optional[str] = None,
        chains: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Retrieve coordinates based on atom type and residue selection.

        Parameters
        ----------
        atom_type : str, optional
            Type of atoms to retrieve. One of:
            - 'all' : All atoms
            - 'backbone' : Only backbone atoms (N, CA, C, O)
            - 'CA' : Only alpha carbons
            - 'non-hydrogen' : All non-hydrogen atoms
            Defaults to 'all'.
        selection : dict, optional
            Dictionary specifying residue selection per chain, e.g.
            { 'A': [1, 2, 3], 'B': range(10, 20) }.
            If provided, only those residues are considered for each chain.
        chain : str, optional
            A single chain ID to retrieve all residues from. If set, `chains`
            is ignored. For example, `chain='A'`.
        chains : list of str, optional
            Multiple chain IDs to retrieve all residues from. e.g. `chains=['A','B']`.
            If `chain` is also provided, `chain` takes precedence.

        Returns
        -------
        np.ndarray
            Array of coordinates.

        """

        # 1) Merge chain(s) info into a selection dictionary if needed
        #    so that each specified chain -> all residue IDs
        chain_selection = {}

        # If user specified chain=..., override chains
        if chain is not None:
            # single chain
            chain_selection = {chain: None}  # None -> all residues in that chain
        elif chains is not None:
            for ch_id in chains:
                chain_selection[ch_id] = None

        # 2) If user also provided `selection`, unify with chain_selection
        #    - If chain_selection is empty, we just parse selection as usual
        #    - If chain_selection is not empty, it overrides or merges logic
        parsed_selection = parse_residue_selection(selection) if selection else {}

        # If chain_selection is specified, we build a combined dict
        # The logic here: if chain X is in chain_selection, that means "all" residues
        # unless user also gave `selection` for that chain -> intersection
        if chain_selection:
            combined_selection = {}
            for c in self.chains:
                if c.name in chain_selection:
                    # If chain_selection[c.name] is None => all residues
                    # if selection also has c.name => intersection
                    if c.name in parsed_selection:
                        # intersection
                        selection_res = parsed_selection[c.name]
                        chain_selection_res = [r.res_seq for r in c.residues]  # all
                        # intersection of the two sets
                        inter = set(selection_res).intersection(chain_selection_res)
                        combined_selection[c.name] = list(inter)
                    else:
                        # chain specified, but not in parsed_selection => take all
                        all_res = [r.res_seq for r in c.residues]
                        combined_selection[c.name] = all_res
                else:
                    # chain not specified => skip
                    pass
            # Now combined_selection is the new parsed_selection
            parsed_selection = {
                ch: sorted(reslist) for ch, reslist in combined_selection.items()
            }
        else:
            # no chain(s) param, so rely solely on the parsed_selection
            # i.e., if user gave selection = {'A': [1,2,3], ...} we use that
            # if user gave no selection, parsed_selection is empty => no restriction
            pass

        # 3) Gather coordinates
        coordinates = []
        for c in self.chains:
            # If parsed_selection is non-empty and c.name isn't in it => skip chain
            if parsed_selection and c.name not in parsed_selection:
                continue

            # If c.name is in parsed_selection => we have a list of residues
            selected_residues = parsed_selection.get(
                c.name, [r.res_seq for r in c.residues]
            )

            for residue in c.residues:
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

    def remove_hydrogens(self) -> None:
        """
        Delete all atoms whose *element* field is 'H' (case-insensitive).
        """
        for chain in self.chains:
            for residue in chain.residues:
                residue.atoms[:] = [
                    a for a in residue.atoms if a.element.upper() != "H"
                ]

    def remove_residues_by_name(self, residue_name: str) -> None:
        """
        Remove all residues from this ProteinStructure that have the specified res name.

        Parameters
        ----------
        residue_name : str
            Residue name to be removed (e.g. 'HOH' for water).

        """
        for chain in self.chains:
            # Keep only residues whose name != residue_name
            chain.residues = [r for r in chain.residues if r.name != residue_name]

    def remove_residues_if(self, condition: Callable[[Residue], bool]) -> None:
        """
        Remove residues in-place if they match a given condition (predicate).

        Parameters
        ----------
        condition : Callable[[Residue], bool]
            A function that takes a Residue object and returns True if
            that residue should be removed.

        Examples
        --------
        # Remove all water residues (HOH):
        protein.remove_residues_if(lambda r: r.name == "HOH")

        # Remove all non-standard amino acids:
        standard_aa = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        }
        protein.remove_residues_if(lambda r: r.name not in standard_aa)
        """
        for chain in self.chains:
            new_list = []
            for residue in chain.residues:
                # If condition(...) is True, we remove that residue
                if not condition(residue):
                    new_list.append(residue)
            chain.residues = new_list

    def remove_water(self) -> None:
        """
        Remove all water residues (often named 'HOH') in-place.
        """
        self.remove_residues_if(lambda r: r.name == "HOH")

    def remove_non_standard_residues(self) -> None:
        """
        Remove all residues not in the standard 20 amino acids set.
        """
        standard_aa = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
        self.remove_residues_if(lambda r: r.name not in standard_aa)

    def sort_residues(self) -> None:
        """Ensure every chain’s residue list is ordered by (res_seq, i_code)."""
        for ch in self.chains:
            ch.residues.sort(key=lambda r: (r.res_seq, r.i_code))
