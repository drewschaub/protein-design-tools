# tests/core/test_atom.py

import pytest
from protein_design_tools.core.atom import Atom


def test_atom_mass():
    # Test known atomic weights
    atom_h = Atom(
        atom_id=1,
        name="H",
        alt_loc="",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        temp_factor=0.0,
        segment_id="",
        element="H",
        charge="",
    )
    assert atom_h.mass == pytest.approx(1.00794, rel=1e-5)

    atom_c = Atom(
        atom_id=2,
        name="C",
        alt_loc="",
        x=1.0,
        y=1.0,
        z=1.0,
        occupancy=1.0,
        temp_factor=0.0,
        segment_id="",
        element="C",
        charge="",
    )
    assert atom_c.mass == pytest.approx(12.0107, rel=1e-4)

    atom_unknown = Atom(
        atom_id=3,
        name="X",
        alt_loc="",
        x=2.0,
        y=2.0,
        z=2.0,
        occupancy=1.0,
        temp_factor=0.0,
        segment_id="",
        element="X",
        charge="",
    )
    assert atom_unknown.mass == 0.0  # Default for unknown elements
