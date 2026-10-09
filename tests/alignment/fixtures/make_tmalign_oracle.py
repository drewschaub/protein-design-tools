# tests/alignment/fixtures/make_tmalign_oracle.py
"""
Regenerate the TM-align oracle used by tests/alignment/test_tmalign.py.

Runs the reference TM-align C++ code (through the `tmtools` bindings, which
compile it unchanged) on the C-alpha fixtures in this directory and freezes
its answers in tmalign_oracle.json, so the test suite can check our NumPy
reimplementation against the reference without anything depending on tmtools.

Usage (only when the fixtures change):

    pip install tmtools
    python tests/alignment/fixtures/make_tmalign_oracle.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from protein_design_tools.io.pdb import read_pdb

HERE = Path(__file__).parent
PAIRS = [
    ("1mbn_A", "4hhb_A"),  # globins: homologous, low identity
    ("1ubq_A", "1ndd_A"),  # ubiquitin vs NEDD8: same fold, high identity
    ("1ncg_A", "1edh_A"),  # cadherin domains
    ("1ubq_A", "1ncg_A"),  # unrelated folds
    ("4hhb_A", "1mbn_A"),  # the globin pair the other way round
    ("1ncg_A", "1ncg_A"),  # self
]


def chain_ca(name):
    structure = read_pdb(HERE / f"{name}.pdb")
    chain = structure.chains[0]
    coords, seq = [], []
    for residue in chain.residues:
        ca = next(a for a in residue.atoms if a.name == "CA")
        coords.append((ca.x, ca.y, ca.z))
        seq.append(residue.one_letter_code or "X")
    return np.array(coords), "".join(seq)


def pairs_from_alignment(aln_x, aln_y):
    pairs, i, j = [], 0, 0
    for a, b in zip(aln_x, aln_y):
        if a != "-" and b != "-":
            pairs.append([i, j])
        i += a != "-"
        j += b != "-"
    return pairs


def main():
    import tmtools

    oracle = {"tmtools_version": tmtools.__version__, "cases": []}
    for name_x, name_y in PAIRS:
        x, seq_x = chain_ca(name_x)
        y, seq_y = chain_ca(name_y)
        res = tmtools.tm_align(x, y, seq_x, seq_y)
        oracle["cases"].append(
            {
                "x": name_x,
                "y": name_y,
                "len_x": len(x),
                "len_y": len(y),
                "tm_norm_x": res.tm_norm_chain1,
                "tm_norm_y": res.tm_norm_chain2,
                "rmsd": res.rmsd,
                "pairs": pairs_from_alignment(res.seqxA, res.seqyA),
                "aligned_x": res.seqxA,
                "aligned_y": res.seqyA,
                "aligned_mark": res.seqM,
                "rotation": np.asarray(res.u).tolist(),
                "translation": np.asarray(res.t).tolist(),
            }
        )
        print(
            f"{name_x} vs {name_y}: TM {res.tm_norm_chain1:.4f}/{res.tm_norm_chain2:.4f}"
        )
    (HERE / "tmalign_oracle.json").write_text(json.dumps(oracle, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
