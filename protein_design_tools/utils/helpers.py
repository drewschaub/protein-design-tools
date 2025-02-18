# protein_design_tools/utils/helpers.py

from typing import Union, List, Dict


def parse_residue_selection(
    selection: Union[Dict[str, List[Union[int, range]]], None],
) -> Dict[str, List[int]]:
    """
    Parse residue selection input into a standardized dictionary.

    Parameters
    ----------
    selection : dict or None
        Residue selection, e.g.,
        {'A': [1, 2, 3, 50, 60], 'B': [range(1, 21), 50, 60]} or None.

    Returns
    -------
    dict
        Parsed residue selection with chain IDs as keys and lists of residue numbers as
        values.
    """
    if selection is None:
        return {}

    parsed_selection = {}
    for chain, residues in selection.items():
        parsed_residues = []
        for res in residues:
            if isinstance(res, range):
                parsed_residues.extend(list(res))
            elif isinstance(res, int):
                parsed_residues.append(res)
            else:
                raise ValueError(f"Invalid residue specification: {res}")
        parsed_selection[chain] = parsed_residues
    return parsed_selection
