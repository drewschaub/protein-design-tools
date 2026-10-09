# tests/core/test_selection.py

import numpy as np
import pytest

from protein_design_tools.core.selection import coordinates, residue_index, select
from tests.helpers import make_chain, make_structure, random_coords


@pytest.fixture
def structure():
    rng = np.random.default_rng(0)
    a = make_chain("A", "ACDEF", random_coords(rng, 5))
    h = make_chain(
        "H", "GHIK", random_coords(rng, 4), numbering=[99, 100, (100, "A"), (100, "B")]
    )
    return make_structure(a, h, name="test")


H_KEYS = [("H", 99, ""), ("H", 100, ""), ("H", 100, "A"), ("H", 100, "B")]


def test_select_everything_in_structure_order(structure):
    keys = select(structure)
    assert keys[:5] == [("A", i, "") for i in range(1, 6)]
    assert keys[5:] == H_KEYS


def test_select_chains(structure):
    assert select(structure, "H") == H_KEYS
    # structure order, not spec order
    assert select(structure, ["H", "A"]) == select(structure)
    assert select(structure, ("A", "H")) == select(structure)


def test_select_residue_numbers_match_every_insertion_code(structure):
    assert select(structure, {"A": range(2, 4), "H": [100]}) == [
        ("A", 2, ""),
        ("A", 3, ""),
        ("H", 100, ""),
        ("H", 100, "A"),
        ("H", 100, "B"),
    ]
    assert select(structure, {"H": None}) == H_KEYS
    assert select(structure, {"A": 3}) == [("A", 3, "")]
    assert select(structure, {"A": [range(1, 3), 5]}) == [
        ("A", 1, ""),
        ("A", 2, ""),
        ("A", 5, ""),
    ]


def test_select_keys_pass_through(structure):
    keys = [("H", 100, "B"), ("A", 1, "")]
    assert select(structure, keys) == keys  # order kept
    assert select(structure, ("A", 3, "")) == [("A", 3, "")]  # a single key
    assert select(structure, [("Z", 9, None)]) == [("Z", 9, "")]  # not checked


def test_select_unknown_chain(structure):
    with pytest.raises(KeyError):
        select(structure, "Z")
    with pytest.raises(KeyError):
        select(structure, {"A": None, "Z": [1]})


def test_select_invalid_residue_spec(structure):
    with pytest.raises(ValueError):
        select(structure, {"A": ["1"]})


def test_coordinates(structure):
    xyz = coordinates(structure, select(structure, "A"), atom="CA")
    assert xyz.shape == (5, 3)
    atom = structure.chains[0].residues[2].atoms[0]
    assert tuple(xyz[2]) == (atom.x, atom.y, atom.z)
    assert coordinates(structure, [], atom="CA").shape == (0, 3)


def test_coordinates_missing_residue_or_atom(structure):
    with pytest.raises(KeyError, match="not in structure"):
        coordinates(structure, [("A", 42, "")])
    with pytest.raises(KeyError, match="no atom 'CB'"):
        coordinates(structure, [("A", 1, "")], atom="CB")


def test_residue_index(structure):
    index = residue_index(structure)
    assert len(index) == 9
    assert index[("H", 100, "A")].name == "ILE"
