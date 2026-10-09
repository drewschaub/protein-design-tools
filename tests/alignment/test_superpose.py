# tests/alignment/test_superpose.py
"""superpose(mobile, ref, on=...) -> Transform, and friends."""

import numpy as np
import pytest

from protein_design_tools.alignment import correspond, kabsch, superpose
from protein_design_tools.metrics import rmsd
from tests.helpers import (
    ca_coords,
    make_chain,
    make_structure,
    moved,
    random_coords,
    random_rigid,
    shift_chain,
)

SEQ = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def ref():
    rng = np.random.default_rng(10)
    return make_structure(
        make_chain("A", SEQ, random_coords(rng, 20)),
        make_chain("B", SEQ[:12], random_coords(rng, 12)),
        name="ref",
    )


def test_kabsch_recovers_a_rigid_motion():
    rng = np.random.default_rng(11)
    P = random_coords(rng, 30)
    R, t = random_rigid(rng)
    R2, t2 = kabsch(P, P @ R.T + t)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)


def test_kabsch_never_returns_a_reflection():
    rng = np.random.default_rng(12)
    P = random_coords(rng, 30)
    R, _ = kabsch(P, P * np.array([1.0, 1.0, -1.0]))  # mirror image
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_superpose_recovers_the_whole_structure(ref):
    rng = np.random.default_rng(13)
    R, t = random_rigid(rng)
    mobile = moved(ref, R, t)

    fit = superpose(mobile, ref)
    assert fit.n == 32 and fit.rmsd == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(fit.rotation @ R, np.eye(3), atol=1e-9)  # undoes R
    np.testing.assert_allclose(fit.matrix[:3, :3], fit.rotation)
    np.testing.assert_allclose(fit.matrix[:3, 3], fit.translation)

    back = fit.apply(mobile)
    np.testing.assert_allclose(ca_coords(back, "A"), ca_coords(ref, "A"), atol=1e-9)
    np.testing.assert_allclose(ca_coords(back, "B"), ca_coords(ref, "B"), atol=1e-9)
    # the input is untouched unless asked
    np.testing.assert_allclose(ca_coords(mobile, "A"), ca_coords(ref, "A") @ R.T + t)
    assert fit.apply(mobile, inplace=True) is mobile
    np.testing.assert_allclose(ca_coords(mobile, "B"), ca_coords(ref, "B"), atol=1e-9)
    np.testing.assert_allclose(fit.apply_to(np.zeros((1, 3)))[0], fit.translation)


def test_superpose_on_one_chain_and_measure_another(ref):
    rng = np.random.default_rng(14)
    model = shift_chain(ref, "B", np.array([0.0, 0.0, 5.0]))  # binder moved 5 A
    R, t = random_rigid(rng)
    model = moved(model, R, t)

    fit = superpose(model, ref, on="A")
    assert fit.n == 20 and fit.rmsd == pytest.approx(0.0, abs=1e-9)
    assert all(key[0] == "A" for _, key in fit.pairs)
    placed = fit.apply(model)
    assert rmsd(placed, ref, over="A") == pytest.approx(0.0, abs=1e-9)
    assert rmsd(placed, ref, over="B") == pytest.approx(5.0)
    assert rmsd(placed, ref) == pytest.approx(np.sqrt(12 * 25 / 32))
    # fitting on everything instead spreads the error over both chains
    assert 0.0 < superpose(model, ref).rmsd < 5.0


def test_superpose_with_different_numbering_needs_pairs(ref):
    renumbered = make_structure(
        make_chain("A", SEQ, ca_coords(ref, "A"), numbering=range(101, 121))
    )
    with pytest.raises(ValueError, match="correspond"):
        superpose(renumbered, ref)
    pairs = correspond(renumbered, ref)
    fit = superpose(renumbered, ref, pairs=pairs)
    assert fit.n == 20 and fit.rmsd == pytest.approx(0.0, abs=1e-9)
    assert rmsd(renumbered, ref, pairs=pairs) == pytest.approx(0.0, abs=1e-9)


def test_superpose_needs_three_pairs(ref):
    with pytest.raises(ValueError, match="at least 3"):
        superpose(ref, ref, on={"A": [1, 2]})
