# protein_design_tools/core/selection.py
"""
Residue keys and selections.

Every residue is identified by its key ``(chain_id, res_seq, i_code)``.
:func:`select` turns the selection specs accepted across the package into an
ordered list of keys, and :func:`coordinates` fetches one atom per key as an
``(N, 3)`` array.  Superposition, correspondence and the structure-level
metrics all speak in these keys, so a selection made once can be reused
everywhere.
"""

from __future__ import annotations

import numbers
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .protein_structure import ProteinStructure
from .residue import Residue

#: ``(chain_id, res_seq, i_code)``; the insertion code is ``""`` when absent.
ResidueKey = Tuple[str, int, str]

#: What :func:`select` accepts: ``None`` (everything), a chain ID, several
#: chain IDs, ``{chain_id: None | residue numbers / ranges}``, or residue keys.
Spec = Union[
    None, str, Sequence[str], Dict[str, Optional[Iterable]], Iterable[ResidueKey]
]


def residue_key(chain_id: str, residue: Residue) -> ResidueKey:
    """Key of ``residue`` in chain ``chain_id``."""
    return (chain_id, residue.res_seq, residue.i_code or "")


def residue_index(structure: ProteinStructure) -> Dict[ResidueKey, Residue]:
    """Map every residue key of ``structure`` to its Residue."""
    index: Dict[ResidueKey, Residue] = {}
    for chain in structure.chains:
        for residue in chain.residues:
            index.setdefault(residue_key(chain.name, residue), residue)
    return index


def select(structure: ProteinStructure, spec: Spec = None) -> List[ResidueKey]:
    """
    Resolve ``spec`` against ``structure`` into residue keys in structure order.

    Parameters
    ----------
    structure : ProteinStructure
    spec : see :data:`Spec`
        ``None`` selects every residue; a chain ID or a sequence of chain IDs
        selects whole chains; ``{"A": range(10, 50), "B": None}`` selects
        residue numbers per chain (``None`` = the whole chain; a number matches
        every insertion code); a residue key or an iterable of keys is returned
        as given, without checking that the residues exist.

    Raises
    ------
    KeyError
        If a named chain is not in ``structure``.
    """
    if spec is None:
        return [
            residue_key(chain.name, residue)
            for chain in structure.chains
            for residue in chain.residues
        ]
    if _is_key(spec):
        return [_as_key(spec)]
    if isinstance(spec, str):
        wanted: Dict[str, Optional[Set[int]]] = {spec: None}
    elif isinstance(spec, dict):
        wanted = {
            chain: _residue_numbers(spec_numbers)
            for chain, spec_numbers in spec.items()
        }
    else:
        items = list(spec)
        if not all(isinstance(item, str) for item in items):
            return [_as_key(item) for item in items]
        wanted = {chain: None for chain in items}

    present = {chain.name for chain in structure.chains}
    missing = [chain for chain in wanted if chain not in present]
    if missing:
        raise KeyError(f"chain(s) {missing} not in structure {structure.name!r}")

    keys: List[ResidueKey] = []
    for chain in structure.chains:
        if chain.name not in wanted:
            continue
        allowed = wanted[chain.name]
        for residue in chain.residues:
            if allowed is None or residue.res_seq in allowed:
                keys.append(residue_key(chain.name, residue))
    return keys


def coordinates(
    structure: ProteinStructure, keys: Iterable[ResidueKey], atom: str = "CA"
) -> np.ndarray:
    """
    ``(N, 3)`` coordinates of ``atom`` for each key, in the order given.

    Raises
    ------
    KeyError
        If a residue is missing from ``structure`` or has no atom ``atom``.
    """
    index = residue_index(structure)
    keys = list(keys)
    out = np.empty((len(keys), 3), dtype=float)
    for i, key in enumerate(keys):
        residue = index.get(key)
        if residue is None:
            raise KeyError(f"residue {key} is not in structure {structure.name!r}")
        xyz = atom_coordinates(residue, atom)
        if xyz is None:
            raise KeyError(f"residue {key} has no atom {atom!r}")
        out[i] = xyz
    return out


def atom_coordinates(
    residue: Residue, atom: str
) -> Optional[Tuple[float, float, float]]:
    """``(x, y, z)`` of the first atom named ``atom`` in ``residue``, or None."""
    for a in residue.atoms:
        if a.name == atom:
            return (a.x, a.y, a.z)
    return None


def _is_key(spec) -> bool:
    return (
        isinstance(spec, tuple)
        and len(spec) == 3
        and isinstance(spec[0], str)
        and isinstance(spec[1], numbers.Integral)
    )


def _as_key(item) -> ResidueKey:
    chain_id, res_seq, i_code = item
    return (chain_id, int(res_seq), i_code or "")


def _residue_numbers(spec_numbers) -> Optional[Set[int]]:
    if spec_numbers is None:
        return None
    if isinstance(spec_numbers, numbers.Integral):
        return {int(spec_numbers)}
    out: Set[int] = set()
    for item in spec_numbers:
        if isinstance(item, range):
            out.update(item)
        elif isinstance(item, numbers.Integral):
            out.add(int(item))
        else:
            raise ValueError(f"invalid residue specification: {item!r}")
    return out
