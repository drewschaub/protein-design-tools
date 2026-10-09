# tests/metrics/test_structure_metrics.py
"""The structure-level rmsd/tmscore/gdt/lddt entry points and TM-score's L_ref."""

import numpy as np
import pytest

from protein_design_tools.alignment import correspond, paired_coordinates
from protein_design_tools.core.protein_structure import ProteinStructure
from protein_design_tools.metrics import (
    compute_gdt_numpy,
    compute_lddt_numpy,
    compute_rmsd_numpy,
    compute_tmscore_numpy,
    gdt,
    lddt,
    rmsd,
    tm_d0,
    tmscore,
)
from tests.helpers import (
    ca_coords,
    make_chain,
    make_structure,
    random_coords,
    shift_chain,
)

SEQ = "ACDEFGHIKLMNPQRSTVWY"


def test_tm_d0_matches_tmalign():
    # TMalign.cpp parameter_set4final: 0.5 for L <= 21, else max(0.5, 1.24 cbrt(L-15) - 1.8)
    assert tm_d0(10) == 0.5 and tm_d0(21) == 0.5
    assert tm_d0(22) == pytest.approx(1.24 * 7 ** (1 / 3) - 1.8)
    assert tm_d0(120) == pytest.approx(4.0499, abs=1e-4)


def test_tmscore_short_chains_no_longer_fail():
    P = np.random.default_rng(0).uniform(-10, 10, size=(10, 3))
    assert compute_tmscore_numpy(P, P) == 1.0
    assert compute_tmscore_numpy(P, P + 1.0) == pytest.approx(
        1 / (1 + (np.sqrt(3) / 0.5) ** 2)
    )


def test_tmscore_l_ref_normalises_by_the_reference_length():
    P = np.random.default_rng(1).uniform(-30, 30, size=(120, 3))
    Q = P + 2.0
    full = compute_tmscore_numpy(P, Q)
    half = compute_tmscore_numpy(P[:60], Q[:60], L_ref=120)
    assert half == pytest.approx(full / 2)  # same d0, half the pairs
    assert compute_tmscore_numpy(P[:60], Q[:60]) != pytest.approx(half)  # N = 60


@pytest.fixture
def placed_and_ref():
    rng = np.random.default_rng(20)
    ref = make_structure(
        make_chain("A", SEQ, random_coords(rng, 20)),
        make_chain("B", SEQ[:12], random_coords(rng, 12)),
    )
    return shift_chain(ref, "B", np.array([0.0, 0.0, 5.0])), ref


def test_structure_metrics_agree_with_the_array_functions(placed_and_ref):
    placed, ref = placed_and_ref
    P, Q, _ = paired_coordinates(placed, ref, over="B")
    assert rmsd(placed, ref, over="B") == compute_rmsd_numpy(P, Q) == pytest.approx(5.0)
    assert gdt(placed, ref, over="B") == compute_gdt_numpy(P, Q) == pytest.approx(25.0)
    assert (
        lddt(placed, ref, over="B") == compute_lddt_numpy(P, Q) == pytest.approx(100.0)
    )
    assert tmscore(placed, ref, over="B") == compute_tmscore_numpy(P, Q, L_ref=12)
    assert tmscore(placed, ref, over="B") == pytest.approx(1 / (1 + (5 / 0.5) ** 2))
    assert tmscore(placed, ref, over="B", L_ref=120) == compute_tmscore_numpy(
        P, Q, L_ref=120
    )
    assert rmsd(placed, ref, over="A") == pytest.approx(0.0)
    assert tmscore(placed, ref, over="A") == pytest.approx(1.0)


def test_tmscore_default_l_ref_is_the_reference_length(placed_and_ref):
    _, ref = placed_and_ref
    truncated = make_structure(make_chain("B", SEQ[:8], ca_coords(ref, "B")[:8]))
    assert tmscore(truncated, ref, over="B") == pytest.approx(8 / 12)
    pairs = correspond(truncated, ref, chain_a="B")
    assert tmscore(truncated, ref, pairs=pairs) == pytest.approx(
        8 / 12
    )  # chain B of ref
    assert tmscore(truncated, ref, pairs=pairs, L_ref=8) == pytest.approx(1.0)


def test_structure_metrics_reject_bad_inputs(placed_and_ref):
    placed, ref = placed_and_ref
    P = np.zeros((3, 3))
    with pytest.raises(TypeError):
        rmsd(P, P, over="A")
    with pytest.raises(TypeError):
        rmsd(P, P, pairs=[])
    with pytest.raises(TypeError):
        rmsd(P, ProteinStructure())
    with pytest.raises(ValueError, match="no paired residues"):
        rmsd(placed, ref, over={"B": [99]})
