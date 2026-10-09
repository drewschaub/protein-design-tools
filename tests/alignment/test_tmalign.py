# tests/alignment/test_tmalign.py
"""
The NumPy TM-align against answers frozen from the reference C++ code
(tests/alignment/fixtures/tmalign_oracle.json, made by make_tmalign_oracle.py).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from protein_design_tools.alignment.correspond import correspond
from protein_design_tools.alignment.tmalign import (
    secondary_structure,
    tm_align,
    tm_superpose,
)
from protein_design_tools.core.geometry import kabsch
from protein_design_tools.io.pdb import read_pdb
from protein_design_tools.metrics import compute_tmscore_numpy, tmscore

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE = json.loads((FIXTURES / "tmalign_oracle.json").read_text())


def chain_ca(name):
    chain = read_pdb(FIXTURES / f"{name}.pdb").chains[0]
    coords = [
        [a.x, a.y, a.z] for r in chain.residues for a in r.atoms if a.name == "CA"
    ]
    return np.array(coords), "".join(r.one_letter_code or "X" for r in chain.residues)


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda c: f"{c['x']}-vs-{c['y']}")
def test_matches_reference_tmalign(case):
    x, seq_x = chain_ca(case["x"])
    y, seq_y = chain_ca(case["y"])
    res = tm_align(x, y, seq_x, seq_y)
    assert res.tm_norm_x == pytest.approx(case["tm_norm_x"], abs=1e-4)
    assert res.tm_norm_y == pytest.approx(case["tm_norm_y"], abs=1e-4)
    assert res.rmsd == pytest.approx(case["rmsd"], abs=1e-3)
    assert res.pairs.tolist() == case["pairs"]
    assert (res.aligned_x, res.aligned_y) == (case["aligned_x"], case["aligned_y"])
    assert res.aligned_mark == case["aligned_mark"]
    # the reported superposition reproduces the reported score
    moved = x[res.pairs[:, 0]] @ res.rotation.T + res.translation
    tm = compute_tmscore_numpy(moved, y[res.pairs[:, 1]], L_ref=len(y))
    assert tm == pytest.approx(res.tm_norm_y, abs=1e-6)
    assert res.n_aligned == len(case["pairs"])


def test_fast_mode_runs_and_finds_the_fold():
    x, _ = chain_ca("1mbn_A")
    y, _ = chain_ca("4hhb_A")
    res = tm_align(x, y, fast=True)
    assert res.tm_norm_y > 0.8 and res.aligned_x is None


def test_tm_align_input_validation():
    x, seq_x = chain_ca("1ubq_A")
    with pytest.raises(ValueError, match="at least 3"):
        tm_align(x[:2], x)
    with pytest.raises(ValueError, match="seq_x"):
        tm_align(x, x, seq_x[:-1], seq_x)


def test_tm_superpose_is_at_least_the_kabsch_frame():
    rng = np.random.default_rng(0)
    P = rng.uniform(-20, 20, size=(80, 3))
    Q = P.copy()
    Q[::5] += rng.normal(scale=15.0, size=(16, 3))  # 20 % of the pairs thrown off
    tm_opt, R, t = tm_superpose(P, Q)
    Rk, tk = kabsch(P, Q)
    tm_kabsch = compute_tmscore_numpy(P @ Rk.T + tk, Q)
    assert tm_kabsch <= tm_opt <= 1.0
    assert tm_opt == pytest.approx(compute_tmscore_numpy(P @ R.T + t, Q), abs=1e-9)
    assert tm_superpose(P, P)[0] == pytest.approx(1.0)
    assert tm_superpose(P, Q, L_ref=160)[0] == pytest.approx(
        compute_tmscore_numpy(P @ R.T + t, Q, L_ref=160) * 1.0, rel=0.05
    )


def test_tmscore_optimize_matches_tm_superpose():
    rng = np.random.default_rng(1)
    P = rng.uniform(-20, 20, size=(60, 3))
    Q = P + rng.normal(scale=1.0, size=P.shape)
    assert tmscore(P, Q, optimize=True) == pytest.approx(tm_superpose(P, Q)[0])
    assert tmscore(P, Q, optimize=True) >= tmscore(P, Q)


def test_secondary_structure_from_ca_geometry():
    i = np.arange(30)
    helix = np.column_stack(
        [2.3 * np.cos(np.radians(100) * i), 2.3 * np.sin(np.radians(100) * i), 1.5 * i]
    )
    sec = secondary_structure(helix)
    assert "".join(sec[2:-2]) == "H" * 26 and "".join(sec[:2]) == "CC"
    line = np.column_stack([3.8 * i, np.zeros(30), np.zeros(30)])
    assert set(secondary_structure(line)) == {"C"}
    assert "".join(secondary_structure(helix[:4])) == "CCCC"


def test_correspond_by_structure():
    a = read_pdb(FIXTURES / "1mbn_A.pdb")
    b = read_pdb(FIXTURES / "4hhb_A.pdb")
    c = correspond(a, b, method="structure")
    case = next(k for k in ORACLE["cases"] if (k["x"], k["y"]) == ("1mbn_A", "4hhb_A"))
    assert c.method == "structure" and len(c) == len(case["pairs"])
    ra, rb = a.chains[0].residues, b.chains[0].residues
    expected = [
        (("A", ra[i].res_seq, ra[i].i_code), ("A", rb[j].res_seq, rb[j].i_code))
        for i, j in case["pairs"]
    ]
    assert c.pairs == expected
    assert c.score == pytest.approx(case["tm_norm_y"], abs=1e-4)
    assert c.alignments[("A", "A")] == (case["aligned_x"], case["aligned_y"])


def test_batched_superposition_search_matches_the_loop():
    from protein_design_tools.alignment.tmalign import (
        _parameters_for_search,
        _tmscore8_search,
        _tmscore8_search_loop,
    )

    rng = np.random.default_rng(3)
    for n in (4, 5, 23, 64, 97):
        P = rng.uniform(-20, 20, size=(n, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = P @ q.T + rng.normal(scale=1.5, size=P.shape)
        Q[::4] += rng.normal(scale=12.0, size=Q[::4].shape)  # some pairs far off
        _, Lnorm, score_d8, d0, d0_search, _ = _parameters_for_search(n + 7, n)
        for step in (1, 40):
            for method in (0, 8):
                args = (step, method, d0_search, Lnorm, score_d8, d0)
                score, R, t = _tmscore8_search(P, Q, *args)
                score_l, R_l, t_l = _tmscore8_search_loop(P, Q, *args)
                assert score == pytest.approx(score_l, abs=1e-12)
                np.testing.assert_allclose(R, R_l, atol=1e-9)
                np.testing.assert_allclose(t, t_l, atol=1e-9)


def test_compiled_dp_matches_the_python_loop():
    from protein_design_tools._optional import numba
    from protein_design_tools.alignment.tmalign import (
        _nwdp_arrays,
        _nwdp_jit,
        _nwdp_python,
    )

    rng = np.random.default_rng(5)
    cases = [rng.random((40, 37)), rng.random((7, 90))]
    cases.append(
        (rng.random((50, 50)) < 0.3).astype(float)
    )  # 0/1 scores: ties everywhere
    for score in cases:
        for gap in (-0.6, 0.0, -1.0):
            expected = _nwdp_python(score, gap)
            assert np.array_equal(_nwdp_arrays(score, gap), expected)
            if numba is not None:
                assert _nwdp_jit is not None
                assert np.array_equal(_nwdp_jit(score, gap), expected)
