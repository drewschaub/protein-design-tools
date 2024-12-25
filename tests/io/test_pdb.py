# tests/io/test_pdb.py

import pytest
from pathlib import Path
from protein_design_tools.io.pdb import read_pdb, fetch_pdb


@pytest.fixture
def small_pdb_content():
    return """
ATOM      1  N   ASP A   1      18.175  39.116  -1.801  1.00 32.79           N  
ATOM      2  CA  ASP A   1      16.844  39.037  -1.185  1.00 31.89           C  
ATOM      3  C   ASP A   1      16.051  37.833  -1.643  1.00 31.52           C  
ATOM      4  O   ASP A   1      14.813  37.863  -1.663  1.00 32.78           O  
ATOM      5  N   TRP A   2      16.742  36.759  -1.988  1.00 30.27           N  
ATOM      6  CA  TRP A   2      16.080  35.525  -2.391  1.00 29.08           C  
ATOM      7  C   TRP A   2      15.402  34.954  -1.147  1.00 29.38           C  
ATOM      8  O   TRP A   2      16.021  34.882  -0.098  1.00 28.70           O  
ATOM      9  CB  TRP A   2      17.120  34.519  -2.886  1.00 28.06           C  
ATOM     10  CG  TRP A   2      17.719  34.859  -4.210  1.00 27.37           C  
ATOM     11  CD1 TRP A   2      19.004  35.245  -4.450  1.00 26.70           C  
ATOM     12  CD2 TRP A   2      17.058  34.860  -5.486  1.00 26.48           C  
ATOM     13  NE1 TRP A   2      19.177  35.500  -5.787  1.00 26.37           N  
ATOM     14  CE2 TRP A   2      17.999  35.265  -6.446  1.00 27.51           C  
ATOM     15  CE3 TRP A   2      15.762  34.553  -5.893  1.00 27.18           C  
ATOM     16  CZ2 TRP A   2      17.694  35.380  -7.808  1.00 26.12           C  
ATOM     17  CZ3 TRP A   2      15.451  34.663  -7.265  1.00 28.05           C  
ATOM     18  CH2 TRP A   2      16.422  35.067  -8.194  1.00 27.17           C  
ATOM     19  N   VAL A   3      14.133  34.576  -1.255  1.00 28.93           N  
ATOM     20  CA  VAL A   3      13.419  33.994  -0.115  1.00 29.14           C  
ATOM     21  C   VAL A   3      13.570  32.484  -0.211  1.00 28.02           C  
ATOM     22  O   VAL A   3      13.401  31.890  -1.294  1.00 28.27           O  
ATOM     23  CB  VAL A   3      11.898  34.400  -0.049  1.00 29.43           C  
ATOM     24  CG1 VAL A   3      11.566  35.501  -1.014  1.00 30.64           C  
ATOM     25  CG2 VAL A   3      10.985  33.224  -0.261  1.00 30.47           C  
ATOM     26  N   ILE A   4      13.925  31.851   0.901  1.00 26.19           N  
ATOM     27  CA  ILE A   4      14.102  30.411   0.891  1.00 24.98           C  
ATOM     28  C   ILE A   4      12.702  29.840   0.769  1.00 25.45           C  
ATOM     29  O   ILE A   4      11.753  30.392   1.342  1.00 25.02           O  
ATOM     30  CB  ILE A   4      14.799  29.940   2.209  1.00 25.48           C  
ATOM     31  CG1 ILE A   4      15.121  28.430   2.245  1.00 24.50           C  
ATOM     32  CG2 ILE A   4      13.937  30.287   3.365  1.00 27.32           C  
ATOM     33  CD1 ILE A   4      15.745  27.929   3.649  1.00 20.73           C  
ATOM     34  N   PRO A   5      12.523  28.812  -0.079  1.00 25.59           N  
ATOM     35  CA  PRO A   5      11.201  28.206  -0.240  1.00 25.45           C  
ATOM     36  C   PRO A   5      10.795  27.554   1.070  1.00 26.10           C  
ATOM     37  O   PRO A   5      11.616  26.909   1.726  1.00 24.94           O  
ATOM     38  CB  PRO A   5      11.457  27.102  -1.255  1.00 25.69           C  
ATOM     39  CG  PRO A   5      12.561  27.634  -2.076  1.00 25.83           C  
ATOM     40  CD  PRO A   5      13.482  28.218  -1.022  1.00 25.83           C  
ATOM     41  N   PRO A   6       9.494  27.587   1.383  1.00 26.63           N  
ATOM     42  CA  PRO A   6       8.933  27.001   2.607  1.00 26.50           C  
ATOM     43  C   PRO A   6       9.140  25.512   2.579  1.00 27.02           C  
ATOM     44  O   PRO A   6       9.157  24.905   1.491  1.00 27.51           O  
ATOM     45  CB  PRO A   6       7.445  27.317   2.489  1.00 26.44           C  
ATOM     46  CG  PRO A   6       7.427  28.599   1.681  1.00 27.76           C  
ATOM     47  CD  PRO A   6       8.443  28.256   0.600  1.00 27.62           C  
ATOM     48  N   ILE A   7       9.344  24.921   3.751  1.00 25.81           N  
ATOM     49  CA  ILE A   7       9.521  23.495   3.834  1.00 26.14           C  
ATOM     50  C   ILE A   7       8.110  22.999   4.115  1.00 25.68           C  
ATOM     51  O   ILE A   7       7.284  23.743   4.615  1.00 24.57           O  
ATOM     52  CB  ILE A   7      10.368  23.048   5.048  1.00 27.10           C  
ATOM     53  CG1 ILE A   7      11.770  23.663   5.098  1.00 28.29           C  
ATOM     54  CG2 ILE A   7      10.502  21.520   5.043  1.00 29.09           C  
ATOM     55  CD1 ILE A   7      12.588  23.132   6.349  1.00 24.14           C  
TER
END
"""


@pytest.fixture
def full_pdb_path():
    return Path("tests/io/1ncg.pdb")


def test_read_small_pdb(small_pdb_content):
    from io import StringIO
    pdb_stream = StringIO(small_pdb_content)
    structure = read_pdb(pdb_stream)
    assert len(structure.chains) == 1
    assert structure.chains[0].name == "A"
    assert len(structure.chains[0].residues) == 7


def test_read_full_pdb(full_pdb_path):
    structure = read_pdb(full_pdb_path)
    assert len(structure.chains) > 0
    assert all(len(chain.residues) > 0 for chain in structure.chains)


def test_fetch_pdb(mocker, small_pdb_content):
    mocker.patch("requests.get", return_value=mocker.Mock(status_code=200, text=small_pdb_content))
    structure = fetch_pdb("1NCG")
    assert len(structure.chains) == 1
    assert structure.chains[0].name == "A"
