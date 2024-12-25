# tests/io/test_pdb.py

import pytest
from unittest.mock import mock_open, patch
from protein_design_tools.io.pdb import read_pdb
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.core.chain import Chain
from protein_design_tools.core.residue import Residue
from protein_design_tools.core.atom import Atom

# tests/io/test_pdb.py


@pytest.fixture
def sample_pdb_content():
    """from a previous AF2 test"""
    return """
ATOM  12920  N   GLU A 830      24.215 -73.330 -84.959  1.00 21.14           N
ATOM  12921  H   GLU A 830      24.534 -73.208 -85.909  1.00 21.14           H
ATOM  12922  CA  GLU A 830      25.271 -73.402 -83.948  1.00 21.14           C
ATOM  12923  HA  GLU A 830      25.219 -74.349 -83.410  1.00 21.14           H
ATOM  12924  C   GLU A 830      25.196 -72.274 -82.920  1.00 21.14           C
ATOM  12925  CB  GLU A 830      26.642 -73.259 -84.652  1.00 21.14           C
ATOM  12926  HB2 GLU A 830      26.593 -72.422 -85.349  1.00 21.14           H
ATOM  12927  HB3 GLU A 830      27.408 -73.025 -83.913  1.00 21.14           H
ATOM  12928  O   GLU A 830      26.017 -72.270 -82.006  1.00 21.14           O
ATOM  12929  CG  GLU A 830      27.136 -74.506 -85.392  1.00 21.14           C
ATOM  12930  HG2 GLU A 830      26.449 -74.735 -86.206  1.00 21.14           H
ATOM  12931  HG3 GLU A 830      27.129 -75.346 -84.696  1.00 21.14           H
ATOM  12932  CD  GLU A 830      28.559 -74.324 -85.964  1.00 21.14           C
ATOM  12933  OE1 GLU A 830      29.311 -75.329 -85.998  1.00 21.14           O
ATOM  12934  OE2 GLU A 830      28.903 -73.203 -86.407  1.00 21.14           O
ATOM  12935  N   ASP A 831      24.318 -71.289 -83.141  1.00 21.16           N
ATOM  12936  H   ASP A 831      23.643 -71.430 -83.879  1.00 21.16           H
ATOM  12937  CA  ASP A 831      24.553 -69.916 -82.707  1.00 21.16           C
ATOM  12938  HA  ASP A 831      25.383 -69.542 -83.306  1.00 21.16           H
ATOM  12939  C   ASP A 831      24.986 -69.816 -81.240  1.00 21.16           C
ATOM  12940  CB  ASP A 831      23.322 -69.023 -82.971  1.00 21.16           C
ATOM  12941  HB2 ASP A 831      22.443 -69.648 -83.123  1.00 21.16           H
ATOM  12942  HB3 ASP A 831      23.128 -68.420 -82.084  1.00 21.16           H
ATOM  12943  O   ASP A 831      24.343 -70.348 -80.338  1.00 21.16           O
ATOM  12944  CG  ASP A 831      23.450 -68.052 -84.157  1.00 21.16           C
ATOM  12945  OD1 ASP A 831      24.398 -68.178 -84.965  1.00 21.16           O
ATOM  12946  OD2 ASP A 831      22.571 -67.161 -84.245  1.00 21.16           O
ATOM  12947  N   ASP A 832      26.061 -69.049 -81.080  1.00 20.59           N
ATOM  12948  H   ASP A 832      26.446 -68.700 -81.946  1.00 20.59           H
ATOM  12949  CA  ASP A 832      26.772 -68.586 -79.893  1.00 20.59           C
ATOM  12950  HA  ASP A 832      27.783 -68.459 -80.279  1.00 20.59           H
ATOM  12951  C   ASP A 832      27.043 -69.526 -78.690  1.00 20.59           C
ATOM  12952  CB  ASP A 832      26.300 -67.170 -79.496  1.00 20.59           C
ATOM  12953  HB2 ASP A 832      25.222 -67.092 -79.641  1.00 20.59           H
ATOM  12954  HB3 ASP A 832      26.489 -67.020 -78.433  1.00 20.59           H
ATOM  12955  O   ASP A 832      26.347 -70.478 -78.352  1.00 20.59           O
ATOM  12956  CG  ASP A 832      27.007 -66.024 -80.251  1.00 20.59           C
ATOM  12957  OD1 ASP A 832      27.990 -66.276 -80.988  1.00 20.59           O
ATOM  12958  OD2 ASP A 832      26.623 -64.855 -80.002  1.00 20.59           O
ATOM  12959  N   LEU A 833      28.162 -69.163 -78.047  1.00 17.77           N
ATOM  12960  H   LEU A 833      28.557 -68.300 -78.392  1.00 17.77           H
ATOM  12961  CA  LEU A 833      28.981 -69.862 -77.042  1.00 17.77           C
ATOM  12962  HA  LEU A 833      29.378 -70.769 -77.497  1.00 17.77           H
ATOM  12963  C   LEU A 833      28.284 -70.350 -75.759  1.00 17.77           C
ATOM  12964  CB  LEU A 833      30.133 -68.894 -76.687  1.00 17.77           C
ATOM  12965  HB2 LEU A 833      30.552 -69.191 -75.726  1.00 17.77           H
ATOM  12966  HB3 LEU A 833      29.703 -67.902 -76.546  1.00 17.77           H
ATOM  12967  O   LEU A 833      27.559 -69.555 -75.116  1.00 17.77           O
ATOM  12968  CG  LEU A 833      31.257 -68.843 -77.730  1.00 17.77           C
ATOM  12969  HG  LEU A 833      30.857 -69.008 -78.731  1.00 17.77           H
ATOM  12970  CD1 LEU A 833      31.941 -67.476 -77.707  1.00 17.77           C
ATOM  12971 HD11 LEU A 833      32.322 -67.266 -76.708  1.00 17.77           H
ATOM  12972 HD12 LEU A 833      31.211 -66.716 -77.989  1.00 17.77           H
ATOM  12973 HD13 LEU A 833      32.753 -67.463 -78.434  1.00 17.77           H
ATOM  12974  CD2 LEU A 833      32.307 -69.910 -77.417  1.00 17.77           C
ATOM  12975 HD21 LEU A 833      33.074 -69.906 -78.192  1.00 17.77           H
ATOM  12976 HD22 LEU A 833      31.819 -70.884 -77.409  1.00 17.77           H
ATOM  12977 HD23 LEU A 833      32.755 -69.725 -76.441  1.00 17.77           H
ATOM  12978  OXT LEU A 833      28.699 -71.447 -75.317  1.00 17.77           O
TER   12979      LEU A 833
ATOM  12979  N   ASN B   1     -17.818  52.502 -35.651  1.00 21.86           N
ATOM  12980  H   ASN B   1     -18.510  52.279 -36.352  1.00 21.86           H
ATOM  12981  H2  ASN B   1     -18.065  53.382 -35.221  1.00 21.86           H
ATOM  12982  H3  ASN B   1     -17.827  51.766 -34.959  1.00 21.86           H
ATOM  12983  CA  ASN B   1     -16.496  52.599 -36.314  1.00 21.86           C
ATOM  12984  HA  ASN B   1     -16.283  51.648 -36.802  1.00 21.86           H
ATOM  12985  C   ASN B   1     -15.367  52.847 -35.322  1.00 21.86           C
ATOM  12986  CB  ASN B   1     -16.529  53.676 -37.424  1.00 21.86           C
ATOM  12987  HB2 ASN B   1     -16.930  54.615 -37.044  1.00 21.86           H
ATOM  12988  HB3 ASN B   1     -15.522  53.855 -37.802  1.00 21.86           H
ATOM  12989  O   ASN B   1     -14.769  53.917 -35.334  1.00 21.86           O
ATOM  12990  CG  ASN B   1     -17.363  53.189 -38.593  1.00 21.86           C
ATOM  12991  ND2 ASN B   1     -17.662  54.007 -39.576  1.00 21.86           N
ATOM  12992 HD21 ASN B   1     -17.326  54.958 -39.625  1.00 21.86           H
ATOM  12993 HD22 ASN B   1     -18.209  53.622 -40.333  1.00 21.86           H
ATOM  12994  OD1 ASN B   1     -17.774  52.045 -38.584  1.00 21.86           O
ATOM  12995  N   LYS B   2     -15.053  51.891 -34.436  1.00 28.22           N
ATOM  12996  H   LYS B   2     -15.457  50.968 -34.517  1.00 28.22           H
ATOM  12997  CA  LYS B   2     -13.839  52.015 -33.613  1.00 28.22           C
ATOM  12998  HA  LYS B   2     -13.754  53.030 -33.226  1.00 28.22           H
ATOM  12999  C   LYS B   2     -12.650  51.781 -34.539  1.00 28.22           C
ATOM  13000  CB  LYS B   2     -13.858  51.045 -32.418  1.00 28.22           C
ATOM  13001  HB2 LYS B   2     -14.251  50.076 -32.727  1.00 28.22           H
ATOM  13002  HB3 LYS B   2     -12.835  50.900 -32.071  1.00 28.22           H
ATOM  13003  O   LYS B   2     -12.550  50.704 -35.117  1.00 28.22           O
ATOM  13004  CG  LYS B   2     -14.686  51.613 -31.252  1.00 28.22           C
ATOM  13005  HG2 LYS B   2     -14.277  52.582 -30.966  1.00 28.22           H
ATOM  13006  HG3 LYS B   2     -15.717  51.754 -31.577  1.00 28.22           H
ATOM  13007  CD  LYS B   2     -14.666  50.685 -30.026  1.00 28.22           C
ATOM  13008  HD2 LYS B   2     -13.633  50.499 -29.732  1.00 28.22           H
ATOM  13009  HD3 LYS B   2     -15.130  49.736 -30.293  1.00 28.22           H
ATOM  13010  CE  LYS B   2     -15.421  51.322 -28.848  1.00 28.22           C
ATOM  13011  HE2 LYS B   2     -14.921  52.251 -28.575  1.00 28.22           H
ATOM  13012  HE3 LYS B   2     -16.429  51.573 -29.175  1.00 28.22           H
ATOM  13013  NZ  LYS B   2     -15.493  50.417 -27.668  1.00 28.22           N
ATOM  13014  HZ1 LYS B   2     -14.574  50.175 -27.327  1.00 28.22           H
ATOM  13015  HZ2 LYS B   2     -15.976  49.560 -27.899  1.00 28.22           H
ATOM  13016  HZ3 LYS B   2     -16.000  50.851 -26.911  1.00 28.22           H
END
"""


def test_read_pdb_success(sample_pdb_content):
    mock_file = mock_open(read_data=sample_pdb_content)

    with patch("builtins.open", mock_file):
        structure = read_pdb("dummy_path.pdb", chains=["A"], name="TestProtein")

    # Assertions
    assert isinstance(structure, ProteinStructure)
    assert structure.name == "TestProtein"
    assert len(structure.chains) == 1

    chainA = structure.chains[0]
    assert chainA.name == "A"
    assert len(chainA.residues) == 4  # 2 GLU, 1 ASP, 1 LEU

    chainB = structure.chains[1]
    assert chainB.name == "B"
    assert len(chainB.residues) == 2  # 1 ASN, 1 LYS

    # Check residues
    residue1 = chainA.residues[0]
    assert residue1.name == "GLU"
    assert residue1.res_seq == 830
    assert residue1.i_code == ""
    assert (
        len(residue1.atoms) == 13
    )  # N, H, CA, HA, C, CB, HB2, HB3, O, CG, HG2, HG3, CD

    residue2 = chainA.residues[1]
    assert residue2.name == "ASP"
    assert residue2.res_seq == 831
    assert residue2.i_code == ""
    assert len(residue2.atoms) == 10  # N, H, CA, HA, C, CB, HB2, HB3, O, CG

    residue3 = chainA.residues[2]
    assert residue3.name == "ASP"
    assert residue3.res_seq == 832
    assert residue3.i_code == ""
    assert len(residue3.atoms) == 10  # N, H, CA, HA, C, CB, HB2, HB3, O, CG

    residue4 = chainA.residues[3]
    assert residue4.name == "LEU"
    assert residue4.res_seq == 833
    assert residue4.i_code == ""
    assert (
        len(residue4.atoms) == 13
    )  # N, H, CA, HA, C, CB, HB2, HB3, O, CG, HG, CD1, HD11, HD12, HD13, CD2, HD21, HD22, HD23, OXT

    residue5 = chainB.residues[0]
    assert residue5.name == "ASN"
    assert residue5.res_seq == 1
    assert residue5.i_code == ""
    assert (
        len(residue5.atoms) == 10
    )  # N, H, H2, H3, CA, HA, C, CB, HB2, HB3, O, CG, ND2, HD21, HD22, OD1

    residue6 = chainB.residues[1]
    assert residue6.name == "LYS"
    assert residue6.res_seq == 2
    assert residue6.i_code == ""
    assert (
        len(residue6.atoms) == 16
    )  # N, H, CA, HA, C, CB, HB2, HB3, O, CG, HG2, HG3, CD, HD2, HD3, CE, HE2, HE3, NZ, HZ1, HZ2, HZ3

    # Check specific atoms
    atom_n = residue1.atoms[0]
    assert atom_n.name == "N"
    assert atom_n.element == "N"
    assert atom_n.x == 24.215
    assert atom_n.y == -73.330
    assert atom_n.z == -84.959
    assert atom_n.occupancy == 1.00
    assert atom_n.temp_factor == 21.14
    assert atom_n.charge == ""
    assert atom_n.atom_id == 12920


def test_read_pdb_missing_chains(sample_pdb_content):
    mock_file = mock_open(read_data=sample_pdb_content)

    # Specify chains that don't exist
    with patch("builtins.open", mock_file):
        structure = read_pdb("dummy_path.pdb", chains=["B"], name="TestProtein")

    # Assertions
    assert isinstance(structure, ProteinStructure)
    assert structure.name == "TestProtein"
    assert len(structure.chains) == 0  # No chains 'B' in the sample data


# tests/io/test_pdb.py


def test_read_pdb_malformed_lines():
    malformed_pdb_content = """
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      invalid_coordinates
ATOM      3  C   ALA A   1      13.104  14.207   3.100  1.00 20.00           C
END
"""
    mock_file = mock_open(read_data=malformed_pdb_content)

    with patch("builtins.open", mock_file):
        with pytest.raises(ValueError):
            read_pdb("dummy_path.pdb", chains=["A"], name="MalformedProtein")
