# tests/alignment/test_correspond.py

import copy

import numpy as np
import pytest

from protein_design_tools.alignment._blosum62 import ALPHABET, BLOSUM62, INDEX
from protein_design_tools.alignment.correspond import (
    Correspondence,
    correspond,
    needleman_wunsch,
    paired_coordinates,
)
from tests.helpers import ca_coords, make_chain, make_structure, random_coords

SEQ = "ACDEFGHIKLMNPQRSTVWY"


def blosum(a, b):
    return BLOSUM62[INDEX[a]][INDEX[b]]


def self_score(seq):
    return sum(blosum(c, c) for c in seq)


def test_blosum62_table():
    assert len(ALPHABET) == len(BLOSUM62) == 24
    assert all(len(row) == 24 for row in BLOSUM62)
    assert all(BLOSUM62[i][j] == BLOSUM62[j][i] for i in range(24) for j in range(24))
    # values read off ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62
    assert (blosum("A", "A"), blosum("W", "W"), blosum("C", "C")) == (4, 11, 9)
    assert (blosum("A", "W"), blosum("A", "S"), blosum("X", "X")) == (-3, 1, -1)


def test_nw_identical_sequences():
    aln_a, aln_b, score = needleman_wunsch(SEQ, SEQ)
    assert aln_a == aln_b == SEQ
    assert score == self_score(SEQ)


def test_nw_point_mutation_stays_paired():
    mutant = SEQ[:9] + "V" + SEQ[10:]  # L -> V
    aln_a, aln_b, score = needleman_wunsch(SEQ, mutant)
    assert (aln_a, aln_b) == (SEQ, mutant)
    assert score == self_score(SEQ) - blosum("L", "L") + blosum("L", "V")


def test_nw_insertion_is_one_affine_gap():
    inserted = SEQ[:10] + "GGG" + SEQ[10:]
    aln_a, aln_b, score = needleman_wunsch(SEQ, inserted)
    assert aln_b == inserted
    assert aln_a == SEQ[:10] + "---" + SEQ[10:]
    assert score == pytest.approx(self_score(SEQ) - 10.0 - 2 * 0.5)


def test_nw_end_gaps_are_free_by_default():
    tagged = "M" + SEQ + "LEHHHHHH"
    aln_a, aln_b, score = needleman_wunsch(SEQ, tagged)
    assert aln_b == tagged
    assert aln_a == "-" + SEQ + "-" * 8
    assert score == self_score(SEQ)
    _, _, penalised = needleman_wunsch(SEQ, tagged, penalize_end_gaps=True)
    assert penalised == pytest.approx(score - 10.0 - (10.0 + 7 * 0.5))


def test_nw_letters_and_case():
    # unknown letters score as X; the original letters are kept in the output
    assert needleman_wunsch("A?A", "AXA")[2] == needleman_wunsch("AXA", "AXA")[2]
    assert needleman_wunsch("aca", "ACA") == ("aca", "ACA", 17.0)


def test_nw_empty_sequences():
    assert needleman_wunsch("", "ACD") == ("---", "ACD", 0.0)
    assert needleman_wunsch("", "ACD", penalize_end_gaps=True)[2] == -11.0
    assert needleman_wunsch("", "") == ("", "", 0.0)


def test_nw_is_symmetric():
    inserted = SEQ[:10] + "GGG" + SEQ[10:]
    aln_a, aln_b, score = needleman_wunsch(SEQ, inserted)
    aln_b2, aln_a2, score2 = needleman_wunsch(inserted, SEQ)
    assert (aln_a2, aln_b2, score2) == (aln_a, aln_b, score)


def test_correspond_sequence_skips_insertion_and_ignores_numbering():
    rng = np.random.default_rng(1)
    ref = make_structure(make_chain("A", SEQ, random_coords(rng, 20)))
    loop = SEQ[:10] + "GGG" + SEQ[10:]
    model = make_structure(
        make_chain("A", loop, random_coords(rng, 23), numbering=range(101, 124))
    )
    c = correspond(model, ref)
    assert c.method == "sequence" and len(c) == 20 and c.identity == 1.0
    assert c.b == [("A", i, "") for i in range(1, 21)]
    assert c.a == [("A", 101 + i, "") for i in range(10)] + [
        ("A", 114 + i, "") for i in range(10)
    ]
    assert c.alignments[("A", "A")] == (loop, SEQ[:10] + "---" + SEQ[10:])
    assert c.score == pytest.approx(self_score(SEQ) - 11.0)


def test_correspond_by_number_and_by_index():
    rng = np.random.default_rng(2)
    ref = make_structure(make_chain("A", SEQ, random_coords(rng, 20)))
    shifted = make_structure(
        make_chain("A", SEQ, random_coords(rng, 20), numbering=range(11, 31))
    )
    by_number = correspond(shifted, ref, method="number")
    assert by_number.pairs == [(("A", i, ""), ("A", i, "")) for i in range(11, 21)]
    assert by_number.identity == 0.0 and by_number.score is None
    by_index = correspond(shifted, ref, method="index")
    assert len(by_index) == 20 and by_index.identity == 1.0
    assert by_index.pairs[0] == (("A", 11, ""), ("A", 1, ""))


def test_correspond_number_respects_insertion_codes():
    rng = np.random.default_rng(3)
    numbering = [99, 100, (100, "A"), (100, "B")]
    a = make_structure(make_chain("H", "GHIK", random_coords(rng, 4), numbering))
    b = make_structure(
        make_chain("H", "GHK", random_coords(rng, 3), [99, 100, (100, "B")])
    )
    c = correspond(a, b, method="number")
    assert c.a == [("H", 99, ""), ("H", 100, ""), ("H", 100, "B")]


def test_correspond_chain_selection_and_defaults():
    rng = np.random.default_rng(4)
    a = make_structure(
        make_chain("A", "ACDEF", random_coords(rng, 5)),
        make_chain("B", "GHIKL", random_coords(rng, 5)),
    )
    b = make_structure(
        make_chain("B", "GHIKL", random_coords(rng, 5)),
        make_chain("C", "ACDEF", random_coords(rng, 5)),
    )
    default = correspond(a, b)  # only chain B is in both
    assert default.a == [("B", i, "") for i in range(1, 6)]
    assert correspond(a, b, chain_a="B").pairs == default.pairs
    assert correspond(a, b, chain_b="B").pairs == default.pairs
    explicit = correspond(a, b, chain_a="A", chain_b="C")
    assert len(explicit) == 5 and explicit.identity == 1.0
    assert explicit.b == [("C", i, "") for i in range(1, 6)]
    with pytest.raises(KeyError):
        correspond(a, b, chain_a="A")  # b has no chain A
    with pytest.raises(KeyError):
        correspond(a, b, chain_a="Q", chain_b="B")
    with pytest.raises(ValueError):
        correspond(a, b, method="magic")


def test_correspondence_container():
    c = Correspondence(
        pairs=[(("A", 1, ""), ("B", 2, ""))], method="number", identity=1.0
    )
    assert len(c) == 1 and list(c) == c.pairs
    assert c.a == [("A", 1, "")] and c.b == [("B", 2, "")]


def test_paired_coordinates_defaults_to_number_pairs_and_filters_with_over():
    rng = np.random.default_rng(5)
    ref = make_structure(
        make_chain("A", "ACDEF", random_coords(rng, 5)),
        make_chain("B", "GHIKL", random_coords(rng, 5)),
    )
    model = make_structure(
        make_chain("A", "ACDEF", random_coords(rng, 5)),
        make_chain("B", "GHI", random_coords(rng, 3)),
    )
    P, Q, used = paired_coordinates(model, ref)
    assert P.shape == Q.shape == (8, 3) and len(used) == 8
    np.testing.assert_allclose(P[:5], ca_coords(model, "A"))
    np.testing.assert_allclose(Q[5:], ca_coords(ref, "B")[:3])
    _, _, used = paired_coordinates(model, ref, over="B")
    assert used == [(("B", i, ""), ("B", i, "")) for i in range(1, 4)]
    _, _, used = paired_coordinates(model, ref, over={"A": [2, 4]})
    assert [key for _, key in used] == [("A", 2, ""), ("A", 4, "")]


def test_paired_coordinates_drops_pairs_missing_the_atom_and_checks_keys():
    rng = np.random.default_rng(6)
    ref = make_structure(make_chain("A", "ACDEF", random_coords(rng, 5)))
    model = copy.deepcopy(ref)
    model.chains[0].residues[2].atoms.clear()  # residue 3 has lost its CA
    _, _, used = paired_coordinates(model, ref)
    assert [key for key, _ in used] == [("A", i, "") for i in (1, 2, 4, 5)]
    P, Q, used = paired_coordinates(model, ref, atom="CB")
    assert used == [] and P.shape == Q.shape == (0, 3)
    with pytest.raises(KeyError):
        paired_coordinates(model, ref, pairs=[(("A", 1, ""), ("A", 99, ""))])
