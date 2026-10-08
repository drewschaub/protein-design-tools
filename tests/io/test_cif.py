# tests/io/test_cif.py

import sys
from io import StringIO

import pytest

from protein_design_tools.io.cif import _split_cif_line, fetch_cif, read_cif

# The same GTP O5' atom as RCSB writes it (quoted) and as ChimeraX re-saves it
# (bare).  The bare form is legal mmCIF but made shlex-based parsing fail with
# "No closing quotation" (GitHub issue from @shainato).
RCSB_ROW = (
    'HETATM 9337 O "O5\'" . GTP E 3 . ? 43.744 31.642 -15.533 1.00 65.09 ? '
    '401 GTP A "O5\'" 1'
)
CHIMERAX_ROW = (
    "HETATM 9337 O O5' . GTP E 3 . ? 43.744 31.642 -15.533 1.00 65.09 ? "
    "401 GTP A O5' 1"
)

ATOM_SITE_FIELDS = [
    "_atom_site.group_PDB",
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_alt_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_asym_id",
    "_atom_site.label_entity_id",
    "_atom_site.label_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
    "_atom_site.occupancy",
    "_atom_site.B_iso_or_equiv",
    "_atom_site.pdbx_formal_charge",
    "_atom_site.auth_seq_id",
    "_atom_site.auth_comp_id",
    "_atom_site.auth_asym_id",
    "_atom_site.auth_atom_id",
    "_atom_site.pdbx_PDB_model_num",
]


def make_cif(*rows):
    return "\n".join(["data_test", "#", "loop_", *ATOM_SITE_FIELDS, *rows, "#"]) + "\n"


def test_split_cif_line_prime_atom_names():
    assert _split_cif_line(RCSB_ROW)[3] == "O5'"
    assert _split_cif_line(CHIMERAX_ROW)[3] == "O5'"
    assert _split_cif_line(RCSB_ROW) == _split_cif_line(CHIMERAX_ROW)
    assert len(_split_cif_line(CHIMERAX_ROW)) == len(ATOM_SITE_FIELDS)


def test_split_cif_line_quoting_rules():
    # a quote closes only when followed by whitespace; '#' starts a comment
    # only at the start of a token
    line = "'it''s' \"a b\" bare#kept # comment"
    assert _split_cif_line(line) == ["it''s", "a b", "bare#kept"]
    assert _split_cif_line("   ") == []


@pytest.mark.parametrize(
    "row", [RCSB_ROW, CHIMERAX_ROW], ids=["rcsb-quoted", "chimerax-bare"]
)
def test_read_cif_prime_atom_names(row):
    structure = read_cif(StringIO(make_cif(row)))
    (chain,) = structure.chains
    assert chain.name == "A"
    (residue,) = chain.residues
    assert (residue.name, residue.res_seq) == ("GTP", 401)
    (atom,) = residue.atoms
    assert atom.name == "O5'"
    assert (atom.x, atom.y, atom.z) == (43.744, 31.642, -15.533)


def test_fetch_cif_without_requests_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)  # `import requests` fails
    with pytest.raises(ImportError, match=r"protein-design-tools\[fetch\]"):
        fetch_cif("1NCG")
