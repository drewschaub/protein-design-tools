import unittest

import sys
sys.path.append('../protein_design_tools')
from protein_structure import ProteinStructure
from protein_structure_utils import get_coordinates

class TestProteinStructureUtils(unittest.TestCase):
    def setUp(self):
        self.protein = ProteinStructure()
        self.protein.read_pdb("example.pdb", name="test")

    def test_get_coordinates_all(self):
        all_coordinates = get_coordinates(self.protein, atom_type="all")
        # Add assertions here to check the result
        # For example:
        self.assertIsNotNone(all_coordinates)
        # Add more assertions based on your expected results

    def test_get_coordinates_backbone(self):
        backbone_coordinates = get_coordinates(self.protein, atom_type="backbone")
        # Add assertions here to check the result
        # For example:
        self.assertIsNotNone(backbone_coordinates)
        # Add more assertions based on your expected results

    def test_get_coordinates_sidechain(self):
        sidechain_coordinates = get_coordinates(self.protein, atom_type="sidechain")
        # Add assertions here to check the result
        # For example:
        self.assertIsNotNone(sidechain_coordinates)
        # Add more assertions based on your expected results

if __name__ == '__main__':
    unittest.main()
