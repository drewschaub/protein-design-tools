# protein_design_tools/core/atom.py

from dataclasses import dataclass

# Atomic weights from:
# https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii
ATOMIC_WEIGHTS = {
    "H": 1.00794075405578,
    "C": 12.0107358967352,
    "N": 14.0067032114458,
    "O": 15.9994049243183,
    "S": 32.0647874061271,
    # Add the rest of the elements here
}


@dataclass
class Atom:
    """Represents an atom in a protein structure."""

    atom_id: int
    name: str
    alt_loc: str
    x: float
    y: float
    z: float
    occupancy: float
    temp_factor: float
    segment_id: str
    element: str
    charge: str

    @property
    def mass(self) -> float:
        """Calculate the mass of the atom based on its element."""
        return ATOMIC_WEIGHTS.get(self.element, 0.0)
