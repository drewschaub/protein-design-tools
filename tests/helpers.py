# tests/helpers.py
"""Builders for the small synthetic structures used across the test suite."""

import copy

import numpy as np

from protein_design_tools.core.atom import Atom
from protein_design_tools.core.chain import Chain
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.core.residue import THREE_TO_ONE, Residue

ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


def make_atom(name, xyz, atom_id=1):
    x, y, z = (float(v) for v in xyz)
    return Atom(
        atom_id=atom_id,
        name=name,
        alt_loc="",
        x=x,
        y=y,
        z=z,
        occupancy=1.0,
        temp_factor=0.0,
        segment_id="",
        element=name[0],
        charge="",
    )


def make_chain(chain_id, sequence, coords, numbering=None, atom="CA"):
    """
    One residue per letter of ``sequence`` with a single ``atom`` at
    ``coords[i]``.  ``numbering`` lists residue numbers or
    ``(number, insertion_code)`` tuples; the default is 1, 2, 3, ...
    """
    chain = Chain(name=chain_id)
    for i, (aa, xyz) in enumerate(zip(sequence, coords)):
        number = numbering[i] if numbering is not None else i + 1
        res_seq, i_code = number if isinstance(number, tuple) else (number, "")
        residue = Residue(
            name=ONE_TO_THREE.get(aa, "UNK"), res_seq=res_seq, i_code=i_code
        )
        residue.atoms.append(make_atom(atom, xyz, atom_id=i + 1))
        chain.residues.append(residue)
    return chain


def make_structure(*chains, name=None):
    return ProteinStructure(name=name, chains=list(chains))


def random_coords(rng, n, scale=20.0):
    return rng.uniform(-scale, scale, size=(n, 3))


def random_rigid(rng):
    """A random proper rotation and translation."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.uniform(-10.0, 10.0, size=3)


def ca_coords(structure, chain_id, atom="CA"):
    chain = next(c for c in structure.chains if c.name == chain_id)
    return np.array(
        [[a.x, a.y, a.z] for r in chain.residues for a in r.atoms if a.name == atom]
    )


def moved(structure, rotation, translation):
    """Deep copy with every atom moved by ``x -> rotation @ x + translation``."""
    out = copy.deepcopy(structure)
    for chain in out.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                x, y, z = rotation @ np.array([atom.x, atom.y, atom.z]) + translation
                atom.x, atom.y, atom.z = float(x), float(y), float(z)
    return out


def shift_chain(structure, chain_id, delta):
    """Deep copy with one chain translated by ``delta``."""
    out = copy.deepcopy(structure)
    chain = next(c for c in out.chains if c.name == chain_id)
    for residue in chain.residues:
        for atom in residue.atoms:
            atom.x, atom.y, atom.z = (
                float(v) for v in np.array([atom.x, atom.y, atom.z]) + delta
            )
    return out
