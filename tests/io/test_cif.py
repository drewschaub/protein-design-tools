# tests/io/test_cif.py

import sys

import pytest

from protein_design_tools.io.cif import fetch_cif


def test_fetch_cif_without_requests_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)  # `import requests` fails
    with pytest.raises(ImportError, match=r"protein-design-tools\[fetch\]"):
        fetch_cif("1NCG")
