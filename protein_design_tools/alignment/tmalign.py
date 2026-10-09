# protein_design_tools/alignment/tmalign.py
"""
TM-align: sequence-independent structural alignment, in NumPy.

A reimplementation of the TM-align algorithm

    Y. Zhang and J. Skolnick, "TM-align: a protein structure alignment
    algorithm based on the TM-score", Nucleic Acids Research 33, 2302-2309
    (2005), https://doi.org/10.1093/nar/gki524

and of the TM-score superposition search it is built on

    Y. Zhang and J. Skolnick, "Scoring function for automated assessment of
    protein structure template quality", Proteins 57, 702-710 (2004),
    https://doi.org/10.1002/prot.20264

written from the published method, with the parameters and tie-breaking
rules of the authors' reference implementation (TM-align 2019-08-22 as
distributed in US-align, Zhang lab, https://zhanggroup.org/TM-align/) so that
results match it.  Only NumPy is needed.

The algorithm searches jointly over residue correspondences and rigid
superpositions for the alignment that maximises TM-score:

1. several seed alignments — gapless threading, secondary-structure
   alignment (secondary structure is assigned from C-alpha geometry, so
   sequences are not needed), local fragment superposition, a
   superposition-plus-secondary-structure score, and fragment gapless
   threading;
2. each seed is refined by iterating dynamic programming on a
   ``1 / (1 + d^2 / d0^2)`` score matrix against the TM-score superposition
   search (:func:`tm_superpose`), with two gap penalties;
3. the best alignment is scored with the final d0 normalised by either
   chain's length.

:func:`tm_align` is the entry point for two C-alpha traces.  Residue-level
callers use ``correspond(a, b, method="structure")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometry import kabsch

#: Residue pairs closer than this (after superposition) are marked ':' in
#: the alignment, as in the reference program's output.
_D0_OUT = 5.0


# --------------------------------------------------------------------------- #
# Parameters (param_set.h)
# --------------------------------------------------------------------------- #


def _parameters_for_search(xlen: int, ylen: int):
    """d0 and cut-offs used while searching (normalised by the shorter chain)."""
    Lnorm = min(xlen, ylen)
    if Lnorm <= 19:
        d0 = 0.168
    else:
        d0 = 1.24 * (Lnorm - 15) ** (1.0 / 3.0) - 1.8
    D0_MIN = d0 + 0.8
    d0 = D0_MIN
    d0_search = min(8.0, max(4.5, d0))
    score_d8 = 1.5 * Lnorm**0.3 + 3.5
    dcu0 = 4.25
    return D0_MIN, Lnorm, score_d8, d0, d0_search, dcu0


def _parameters_for_final(length: float):
    """d0 of the reported TM-score for a given normalisation length."""
    D0_MIN = 0.5
    d0 = 0.5 if length <= 21 else 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, D0_MIN)
    d0_search = min(8.0, max(4.5, d0))
    return d0, d0_search


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _transform(x: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x @ R.T + t


def _sqdist_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Squared distances between every row of x and every row of y, via
    ``|x|^2 + |y|^2 - 2 x.y`` (BLAS) rather than explicit differences; the two
    agree to ~1e-12 A^2, far below any decision threshold in the search.
    """
    d = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(d, 0.0)


def _sqdist_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x - y
    return np.einsum("ij,ij->i", diff, diff)


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = kabsch(P, Q)
    return float(np.sqrt(np.mean(_sqdist_rows(_transform(P, R, t), Q))))


# --------------------------------------------------------------------------- #
# Secondary structure from C-alpha geometry (make_sec / sec_str)
# --------------------------------------------------------------------------- #


def secondary_structure(x: np.ndarray) -> np.ndarray:
    """
    Per-residue secondary structure from C-alpha distances, as TM-align
    assigns it: ``H`` helix, ``E`` strand, ``T`` turn, ``C`` coil.  The two
    residues at each end are always ``C``.
    """
    n = len(x)
    sec = np.full(n, "C")
    if n < 5:
        return sec
    d = lambda a, b: np.sqrt(_sqdist_rows(a, b))  # noqa: E731
    d13 = d(x[0 : n - 4], x[2 : n - 2])
    d14 = d(x[0 : n - 4], x[3 : n - 1])
    d15 = d(x[0 : n - 4], x[4:n])
    d24 = d(x[1 : n - 3], x[3 : n - 1])
    d25 = d(x[1 : n - 3], x[4:n])
    d35 = d(x[2 : n - 2], x[4:n])
    delta = 2.1
    helix = (
        (np.abs(d15 - 6.37) < delta)
        & (np.abs(d14 - 5.18) < delta)
        & (np.abs(d25 - 5.18) < delta)
        & (np.abs(d13 - 5.45) < delta)
        & (np.abs(d24 - 5.45) < delta)
        & (np.abs(d35 - 5.45) < delta)
    )
    delta = 1.42
    strand = (
        (np.abs(d15 - 13.0) < delta)
        & (np.abs(d14 - 10.4) < delta)
        & (np.abs(d25 - 10.4) < delta)
        & (np.abs(d13 - 6.1) < delta)
        & (np.abs(d24 - 6.1) < delta)
        & (np.abs(d35 - 6.1) < delta)
    )
    turn = d15 < 8.0
    sec[2 : n - 2] = np.where(
        helix, "H", np.where(strand, "E", np.where(turn, "T", "C"))
    )
    return sec


# --------------------------------------------------------------------------- #
# Dynamic programming (NW.h NWDP_TM)
# --------------------------------------------------------------------------- #


def _nwdp(score: np.ndarray, gap_open: float) -> np.ndarray:
    """
    TM-align's Needleman-Wunsch variant on a ``(xlen, ylen)`` score matrix.

    A gap costs ``gap_open`` only when it follows an aligned pair (the
    reference's ``path`` rule), there is no extension cost and end gaps are
    free.  Returns ``y2x``: for every residue j of y the index of its partner
    in x, or -1.  Pure Python inner loop; rows cannot be vectorised exactly
    because the gap charge depends on the previous cell's state.
    """
    len1, len2 = score.shape
    S = score.tolist()
    val = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
    path = [[False] * (len2 + 1) for _ in range(len1 + 1)]
    charged = gap_open != 0.0
    for i in range(1, len1 + 1):
        v_i, v_p, p_i, p_p, s_i = val[i], val[i - 1], path[i], path[i - 1], S[i - 1]
        for j in range(1, len2 + 1):
            d = v_p[j - 1] + s_i[j - 1]
            h = v_p[j]
            v = v_i[j - 1]
            if charged:
                if p_p[j]:
                    h += gap_open
                if p_i[j - 1]:
                    v += gap_open
            if d >= h and d >= v:
                p_i[j] = True
                v_i[j] = d
            else:
                v_i[j] = v if v >= h else h
    y2x = np.full(len2, -1, dtype=int)
    i, j = len1, len2
    while i > 0 and j > 0:
        if path[i][j]:
            y2x[j - 1] = i - 1
            i -= 1
            j -= 1
        else:
            h = val[i - 1][j] + (gap_open if path[i - 1][j] else 0.0)
            v = val[i][j - 1] + (gap_open if path[i][j - 1] else 0.0)
            if v >= h:
                j -= 1
            else:
                i -= 1
    return y2x


def _nwdp_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    d02: float,
    gap_open: float,
) -> np.ndarray:
    score = 1.0 / (1.0 + _sqdist_matrix(_transform(x, R, t), y) / d02)
    return _nwdp(score, gap_open)


# --------------------------------------------------------------------------- #
# TM-score superposition search (score_fun8 / TMscore8_search)
# --------------------------------------------------------------------------- #


def _score_fun8(xt, ytm, d, Lnorm, score_d8, d0, score_sum_method):
    """Pairs closer than ``d`` (relaxed until at least 3) and the TM-score."""
    di = _sqdist_rows(xt, ytm)
    n_ali = len(di)
    d_tmp = d * d
    inc = 0
    while True:
        i_ali = np.flatnonzero(di < d_tmp)
        if len(i_ali) < 3 and n_ali > 3:
            inc += 1
            d_tmp = (d + inc * 0.5) ** 2
        else:
            break
    d02 = d0 * d0
    if score_sum_method == 8:
        close = di[di <= score_d8 * score_d8]
        score_sum = np.sum(1.0 / (1.0 + close / d02))
    else:
        score_sum = np.sum(1.0 / (1.0 + di / d02))
    return i_ali, float(score_sum / Lnorm)


def _fragment_lengths(Lali: int) -> List[int]:
    """Fragment ladder L, L/2, L/4, ... down to 4 (at most six lengths)."""
    L_ini_min = min(4, Lali)
    L_ini: List[int] = []
    for k in range(5):
        L = Lali // 2**k
        if L <= L_ini_min:
            L_ini.append(L_ini_min)
            break
        L_ini.append(L)
    else:
        L_ini.append(L_ini_min)
    return L_ini


def _fragment_starts(Lali: int, L_frag: int, simplify_step: int) -> np.ndarray:
    """Start positions the reference visits: every ``simplify_step`` residues,
    plus the last possible start so that no fragment is missed."""
    iL_max = Lali - L_frag
    starts = np.arange(0, iL_max + 1, simplify_step)
    if starts[-1] != iL_max:
        starts = np.append(starts, iL_max)
    return starts


def _tmscore8_search_loop(
    xtm, ytm, simplify_step, score_sum_method, local_d0_search, Lnorm, score_d8, d0
):
    """
    One-fit-at-a-time form of :func:`_tmscore8_search`, kept because it reads
    like the algorithm and the batched version is tested against it: start
    from every fragment of length L, L/2, ..., 4 (stepping ``simplify_step``
    residues), fit it, keep the pairs within a cut-off, refit, and iterate
    (at most 20 times).  Returns the best score and its rotation and
    translation.
    """
    Lali = len(xtm)
    n_it = 20
    L_ini = _fragment_lengths(Lali)

    score_max = -1.0
    R0 = np.eye(3)
    t0 = np.zeros(3)
    for L_frag in L_ini:
        iL_max = Lali - L_frag
        i = 0
        while True:
            idx = np.arange(i, i + L_frag)
            R, t = kabsch(xtm[idx], ytm[idx])
            xt = _transform(xtm, R, t)
            i_ali, score = _score_fun8(
                xt, ytm, local_d0_search - 1, Lnorm, score_d8, d0, score_sum_method
            )
            if score > score_max:
                score_max, R0, t0 = score, R, t
            d = local_d0_search + 1
            for _ in range(n_it):
                if len(i_ali) == 0:
                    break
                k_ali = i_ali
                R, t = kabsch(xtm[k_ali], ytm[k_ali])
                xt = _transform(xtm, R, t)
                i_ali, score = _score_fun8(
                    xt, ytm, d, Lnorm, score_d8, d0, score_sum_method
                )
                if score > score_max:
                    score_max, R0, t0 = score, R, t
                if len(i_ali) == len(k_ali) and np.array_equal(i_ali, k_ali):
                    break
            if i < iL_max:
                i = min(i + simplify_step, iL_max)
            else:
                break
    return score_max, R0, t0


def _kabsch_batched(xw, yw, w):
    """
    Weighted Kabsch for a batch: rotations ``(W, 3, 3)`` and translations
    ``(W, 3)`` superposing ``xw[k]`` onto ``yw[k]`` over the points whose 0/1
    weight ``w[k]`` is one (the same arithmetic as :func:`kabsch`).  A row
    with no points gives some rotation and a zero translation; callers mask
    such rows out.
    """
    n = w.sum(axis=1)
    n = np.where(n > 0, n, 1.0)[:, None]
    wx = w[:, None, :]
    cx = (wx @ xw)[:, 0, :] / n
    cy = (wx @ yw)[:, 0, :] / n
    xc = (xw - cx[:, None, :]) * w[:, :, None]
    yc = yw - cy[:, None, :]
    H = np.transpose(xc, (0, 2, 1)) @ yc
    U, _, Vt = np.linalg.svd(H)
    VtT = np.transpose(Vt, (0, 2, 1))
    UT = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(VtT @ UT))
    d[d == 0] = 1.0
    VtT[:, :, 2] *= d[:, None]  # Vt.T @ diag(1, 1, d)
    R = VtT @ UT
    t = cy - (R @ cx[:, :, None])[:, :, 0]
    return R, t


def _sqdist_batched(xtm, ytm, R, t):
    """Squared distances ``(W, L)`` after moving ``xtm`` by each ``(R, t)``."""
    xt = xtm @ np.transpose(R, (0, 2, 1)) + t[:, None, :]
    diff = xt - ytm
    return (diff * diff).sum(axis=2)


def _score_fun8_batched(di, d, Lnorm, score_d8, d0, score_sum_method, live=None):
    """:func:`_score_fun8` for a batch of squared-distance rows ``(W, L)``;
    only rows flagged ``live`` have their cut-off relaxed."""
    W, n_ali = di.shape
    d_tmp = np.full(W, d * d)
    mask = di < d_tmp[:, None]
    if n_ali > 3:
        inc = np.zeros(W)
        short = mask.sum(axis=1) < 3
        if live is not None:
            short &= live
        while short.any():
            inc[short] += 1
            d_tmp[short] = (d + inc[short] * 0.5) ** 2
            mask[short] = di[short] < d_tmp[short, None]
            short = (mask.sum(axis=1) < 3) & short
    terms = 1.0 / (1.0 + di / (d0 * d0))
    if score_sum_method == 8:
        terms = np.where(di <= score_d8 * score_d8, terms, 0.0)
    return mask, terms.sum(axis=1) / Lnorm


def _tmscore8_search(
    xtm,
    ytm,
    simplify_step,
    score_sum_method,
    local_d0_search,
    Lnorm,
    score_d8,
    d0,
    chunk=256,
):
    """
    Superposition maximising the TM-score of pre-paired points ``xtm`` ->
    ``ytm`` (the TM-score program's search): start from every fragment of
    length L, L/2, ..., 4 (stepping ``simplify_step`` residues), fit it, keep
    the pairs within a cut-off, refit, and iterate (at most 20 times).
    Returns the best score and its rotation and translation.

    All fragments of one length are fitted together (one stacked SVD) and
    refined in lockstep with per-fragment 0/1 weights, which is where the
    time goes.  The winner is the first maximum in the order the
    one-at-a-time version visits, so the result matches
    :func:`_tmscore8_search_loop`.  ``chunk`` bounds how many fragments are
    in flight at once.
    """
    Lali = len(xtm)
    n_it = 20
    cols = np.arange(Lali)[None, :]
    score_max, R0, t0 = -1.0, np.eye(3), np.zeros(3)
    for L_frag in _fragment_lengths(Lali):
        starts = _fragment_starts(Lali, L_frag, simplify_step)
        for c0 in range(0, len(starts), chunk):
            s = starts[c0 : c0 + chunk]
            W = len(s)
            xw = np.broadcast_to(xtm, (W, Lali, 3))
            yw = np.broadcast_to(ytm, (W, Lali, 3))
            w = ((cols >= s[:, None]) & (cols < (s + L_frag)[:, None])).astype(float)
            scores = np.full((W, n_it + 1), -np.inf)
            Rs = np.empty((n_it + 1, W, 3, 3))
            ts = np.empty((n_it + 1, W, 3))
            R, t = _kabsch_batched(xw, yw, w)
            di = _sqdist_batched(xtm, ytm, R, t)
            mask, score = _score_fun8_batched(
                di, local_d0_search - 1, Lnorm, score_d8, d0, score_sum_method
            )
            scores[:, 0], Rs[0], ts[0] = score, R, t
            live = mask.any(axis=1)
            for it in range(1, n_it + 1):
                if not live.any():
                    break
                k_mask = mask
                R, t = _kabsch_batched(xw, yw, k_mask.astype(float))
                di = _sqdist_batched(xtm, ytm, R, t)
                new_mask, score = _score_fun8_batched(
                    di, local_d0_search + 1, Lnorm, score_d8, d0, score_sum_method, live
                )
                scores[live, it] = score[live]
                Rs[it], ts[it] = R, t
                converged = (new_mask == k_mask).all(axis=1)
                mask = np.where(live[:, None], new_mask, k_mask)
                live &= ~converged & new_mask.any(axis=1)
            best = int(np.argmax(scores))  # first maximum in visiting order
            wi, iti = divmod(best, n_it + 1)
            if scores[wi, iti] > score_max:
                score_max, R0, t0 = float(scores[wi, iti]), Rs[iti, wi], ts[iti, wi]
    return score_max, R0, t0


def _aligned(x, y, y2x):
    j = np.flatnonzero(y2x >= 0)
    return x[y2x[j]], y[j]


def _detailed_search(
    x, y, y2x, simplify_step, score_sum_method, local_d0_search, Lnorm, score_d8, d0
):
    xtm, ytm = _aligned(x, y, y2x)
    return _tmscore8_search(
        xtm, ytm, simplify_step, score_sum_method, local_d0_search, Lnorm, score_d8, d0
    )


def _get_score_fast(x, y, y2x, d0, d0_search) -> float:
    """Quick (three-iteration) estimate of how good an alignment is."""
    xtm, ytm = _aligned(x, y, y2x)
    n_ali = len(xtm)
    d02 = d0 * d0
    d002 = d0_search * d0_search

    def fit(sel):
        R, t = kabsch(xtm[sel], ytm[sel])
        dis = _sqdist_rows(_transform(xtm, R, t), ytm)
        return dis, float(np.sum(1.0 / (1.0 + dis / d02)))

    def select(dis, d002t):
        while True:
            sel = dis <= d002t
            if sel.sum() < 3 and n_ali > 3:
                d002t += 0.5
            else:
                return sel

    all_pairs = np.ones(n_ali, dtype=bool)
    dis, tmscore = fit(all_pairs)
    third = np.sort(dis)[min(2, n_ali - 1)]
    sel = select(dis, max(d002, third))
    if sel.sum() != n_ali:
        dis, tmscore1 = fit(sel)
        third = np.sort(dis)[min(2, n_ali - 1)]
        sel = select(dis, max(d002 + 1.0, third))
        _, tmscore2 = fit(sel)
    else:
        tmscore1 = tmscore2 = tmscore
    return max(tmscore, tmscore1, tmscore2)


# --------------------------------------------------------------------------- #
# Seed alignments (get_initial*)
# --------------------------------------------------------------------------- #


def _threading_map(ylen: int, xlen: int, k: int) -> np.ndarray:
    i = np.arange(ylen) + k
    return np.where((i >= 0) & (i < xlen), i, -1)


def _get_initial(x, y, d0, d0_search, fast):
    """Gapless threading: the best diagonal of the full chains."""
    xlen, ylen = len(x), len(y)
    min_ali = max(min(xlen, ylen) // 2, 5)
    n1, n2 = -ylen + min_ali, xlen - min_ali
    best, k_best = -1.0, n1
    for k in range(n1, n2 + 1, 5 if fast else 1):
        tm = _get_score_fast(x, y, _threading_map(ylen, xlen, k), d0, d0_search)
        if tm >= best:
            best, k_best = tm, k
    return _threading_map(ylen, xlen, k_best)


def _get_initial_ss(secx, secy):
    """Alignment of the secondary-structure strings (match 1, gap -1)."""
    return _nwdp((secx[:, None] == secy[None, :]).astype(float), -1.0)


def _get_initial5(x, y, d0, d0_search, fast, D0_MIN):
    """Local superposition of fragment pairs, each followed by DP."""
    xlen, ylen = len(x), len(y)
    d01 = max(d0 + 1.5, D0_MIN)
    d02 = d01 * d01
    aL = min(xlen, ylen)

    def jump(L):
        n = 45 if L > 250 else 35 if L > 200 else 25 if L > 150 else 15
        return max(1, min(n, L // 3))

    n_jump1, n_jump2 = jump(xlen), jump(ylen)
    if fast:
        n_jump1 *= 5
        n_jump2 *= 5
    n_frag = [max(1, min(20, aL // 3)), max(1, min(100, aL // 2))]
    GLmax, best = 0.0, None
    for nf in n_frag:
        for i in range(0, xlen - nf + 1, n_jump1):
            for j in range(0, ylen - nf + 1, n_jump2):
                R, t = kabsch(x[i : i + nf], y[j : j + nf])
                y2x = _nwdp_coordinates(x, y, R, t, d02, 0.0)
                GL = _get_score_fast(x, y, y2x, d0, d0_search)
                if GL > GLmax:
                    GLmax, best = GL, y2x
    return best


def _get_initial_ssplus(x, y, secx, secy, y2x0, D0_MIN, d0):
    """DP on the superposition of the best alignment so far, plus a bonus
    of 0.5 for matching secondary structure (gap -1)."""
    d01 = max(d0 + 1.5, D0_MIN)
    d02 = d01 * d01
    xtm, ytm = _aligned(x, y, y2x0)
    R, t = kabsch(xtm, ytm)
    score = 1.0 / (1.0 + _sqdist_matrix(_transform(x, R, t), y) / d02)
    score = score + 0.5 * (secx[:, None] == secy[None, :])
    return _nwdp(score, -1.0)


def _find_max_frag(x, dcu0, fast):
    """Longest run of consecutive C-alphas closer than dcu0 (relaxed if needed)."""
    n = len(x)
    fra_min = 8 if fast else 4
    r_min = min(n // 3, fra_min)
    d = _sqdist_rows(x[1:], x[:-1]) if n > 1 else np.empty(0)
    dcu_cut = dcu0 * dcu0
    inc = 0
    start_max = end_max = 0
    while True:
        Lfr_max, j, start = 0, 1, 0
        for i in range(1, n):
            if d[i - 1] < dcu_cut:
                j += 1
                if i == n - 1:
                    if j > Lfr_max:
                        Lfr_max, start_max, end_max = j, start, i
                    j = 1
            else:
                if j > Lfr_max:
                    Lfr_max, start_max, end_max = j, start, i - 1
                j, start = 1, i
        if Lfr_max < r_min:
            inc += 1
            dcu_cut = (1.1**inc * dcu0) ** 2
        else:
            break
    return start_max, end_max


def _get_initial_fgt(x, y, d0, d0_search, dcu0, fast):
    """Gapless threading of the longest continuous fragment of the shorter
    chain against the other chain."""
    xlen, ylen = len(x), len(y)
    fra_min1 = (8 if fast else 4) - 1
    step = 3 if fast else 1
    xstart, xend = _find_max_frag(x, dcu0, fast)
    ystart, yend = _find_max_frag(y, dcu0, fast)
    Lx, Ly = xend - xstart + 1, yend - ystart + 1
    L_fr = min(Lx, Ly)
    state = {"best": -1.0, "map": None}

    def consider(y2x):
        tm = _get_score_fast(x, y, y2x, d0, d0_search)
        if tm >= state["best"]:
            state["best"], state["map"] = tm, y2x

    def trim(ifr, L0):
        if len(ifr) == L0:
            return ifr[int(L0 * 0.1) : int(L0 * 0.89) + 1]
        return ifr

    def thread_x_fragment(ifr):  # fragment of x slid along y
        L1 = len(ifr)
        min_ali = max(int(min(L1, ylen) / 2.5), fra_min1)
        for k in range(-ylen + min_ali, L1 - min_ali + 1, step):
            i = np.arange(ylen) + k
            ok = (i >= 0) & (i < L1)
            y2x = np.full(ylen, -1, dtype=int)
            y2x[ok] = ifr[i[ok]]
            consider(y2x)

    def thread_y_fragment(ifr):  # fragment of y slid along x
        L2 = len(ifr)
        min_ali = max(int(min(xlen, L2) / 2.5), fra_min1)
        for k in range(-L2 + min_ali, xlen - min_ali + 1):
            j = np.arange(L2)
            i = j + k
            ok = (i >= 0) & (i < xlen)
            y2x = np.full(ylen, -1, dtype=int)
            y2x[ifr[j[ok]]] = i[ok]
            consider(y2x)

    if Lx < Ly or (Lx == Ly and xlen < ylen):
        thread_x_fragment(trim(xstart + np.arange(L_fr), min(xlen, ylen)))
    elif Lx > Ly or (Lx == Ly and xlen > ylen):
        thread_y_fragment(trim(ystart + np.arange(L_fr), min(xlen, ylen)))
    else:  # equal fragment and chain lengths: try both ways
        thread_x_fragment(trim(xstart + np.arange(L_fr), xlen))
        thread_y_fragment(trim(ystart + np.arange(Ly), xlen))
    return state["map"]


# --------------------------------------------------------------------------- #
# Iterative refinement (DP_iter) and the main search (TMalign_main)
# --------------------------------------------------------------------------- #


def _dp_iter(x, y, R, t, g1, g2, iteration_max, local_d0_search, Lnorm, d0, score_d8):
    """Alternate DP on the current superposition with the TM-score search."""
    gap_open = (-0.6, 0.0)
    d02 = d0 * d0
    best, best_map = -1.0, None
    tmscore_old = 0.0
    for g in range(g1, g2):
        for iteration in range(iteration_max):
            y2x = _nwdp_coordinates(x, y, R, t, d02, gap_open[g])
            tm, R, t = _detailed_search(
                x, y, y2x, 40, 8, local_d0_search, Lnorm, score_d8, d0
            )
            if tm > best:
                best, best_map = tm, y2x
            if iteration > 0 and abs(tmscore_old - tm) < 1e-6:
                break
            tmscore_old = tm
    return best, best_map


@dataclass
class TMAlignResult:
    """
    Result of :func:`tm_align` for C-alpha traces ``x`` and ``y``.

    ``pairs`` are index pairs ``(i in x, j in y)`` of the aligned residues
    that lie within the ``score_d8`` cut-off after superposition (what the
    reference program reports); ``tm_norm_x`` / ``tm_norm_y`` are the
    TM-scores normalised by the length of x / y; ``rotation`` and
    ``translation`` superpose x onto y (``x @ rotation.T + translation``) for
    the y-normalised score; ``rmsd`` is over the aligned pairs.  The gapped
    strings are filled in when sequences are given.
    """

    pairs: np.ndarray
    tm_norm_x: float
    tm_norm_y: float
    rmsd: float
    rotation: np.ndarray
    translation: np.ndarray
    d0_x: float
    d0_y: float
    aligned_x: Optional[str] = None
    aligned_y: Optional[str] = None
    aligned_mark: Optional[str] = None

    @property
    def n_aligned(self) -> int:
        return len(self.pairs)


def tm_align(
    x: np.ndarray,
    y: np.ndarray,
    seq_x: Optional[Sequence[str]] = None,
    seq_y: Optional[Sequence[str]] = None,
    fast: bool = False,
) -> TMAlignResult:
    """
    Align two C-alpha traces with the TM-align algorithm (no sequences
    needed) and return the residue correspondence, TM-scores and
    superposition.

    Parameters
    ----------
    x, y : (N, 3) and (M, 3) arrays
        C-alpha coordinates of the two chains, in chain order.  ``x`` is
        superposed onto ``y``.
    seq_x, seq_y : str, optional
        One-letter sequences, used only to render the gapped alignment.
    fast : bool
        The reference program's ``-fast`` mode: coarser seeds and two
        refinement rounds instead of thirty.  Faster, slightly less accurate.

    Raises
    ------
    ValueError
        If a chain has fewer than three residues or no alignment is found.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xlen, ylen = len(x), len(y)
    if min(xlen, ylen) < 3:
        raise ValueError("tm_align needs at least 3 residues in each chain")
    if seq_x is not None and len(seq_x) != xlen:
        raise ValueError("seq_x must have one letter per residue of x")
    if seq_y is not None and len(seq_y) != ylen:
        raise ValueError("seq_y must have one letter per residue of y")

    D0_MIN, Lnorm, score_d8, d0, d0_search, dcu0 = _parameters_for_search(xlen, ylen)
    local_d0_search = d0_search
    ddcc = 0.1 if Lnorm <= 40 else 0.4
    rounds = 2 if fast else 30
    secx, secy = secondary_structure(x), secondary_structure(y)

    def search(y2x):
        return _detailed_search(x, y, y2x, 40, 8, local_d0_search, Lnorm, score_d8, d0)

    def refine(R, t, g1, iteration_max):
        return _dp_iter(
            x, y, R, t, g1, 2, iteration_max, local_d0_search, Lnorm, d0, score_d8
        )

    TMmax = -1.0

    # 1. gapless threading
    y2x0 = _get_initial(x, y, d0, d0_search, fast)
    TM, R, t = search(y2x0)
    TMmax = max(TMmax, TM)
    TM, y2x = refine(R, t, 0, rounds)
    if TM > TMmax:
        TMmax, y2x0 = TM, y2x

    # 2. secondary structure
    y2x = _get_initial_ss(secx, secy)
    TM, R, t = search(y2x)
    if TM > TMmax:
        TMmax, y2x0 = TM, y2x
    if TM > TMmax * 0.2:
        TM, y2x = refine(R, t, 0, rounds)
        if TM > TMmax:
            TMmax, y2x0 = TM, y2x

    # 3. local fragment superposition
    y2x = _get_initial5(x, y, d0, d0_search, fast, D0_MIN)
    if y2x is not None:
        TM, R, t = search(y2x)
        if TM > TMmax:
            TMmax, y2x0 = TM, y2x
        if TM > TMmax * ddcc:
            TM, y2x = refine(R, t, 0, 2)
            if TM > TMmax:
                TMmax, y2x0 = TM, y2x

    # 4. superposition of the best alignment so far + secondary structure
    y2x = _get_initial_ssplus(x, y, secx, secy, y2x0, D0_MIN, d0)
    TM, R, t = search(y2x)
    if TM > TMmax:
        TMmax, y2x0 = TM, y2x
    if TM > TMmax * ddcc:
        TM, y2x = refine(R, t, 0, rounds)
        if TM > TMmax:
            TMmax, y2x0 = TM, y2x

    # 5. fragment gapless threading
    y2x = _get_initial_fgt(x, y, d0, d0_search, dcu0, fast)
    if y2x is not None:
        TM, R, t = search(y2x)
        if TM > TMmax:
            TMmax, y2x0 = TM, y2x
        if TM > TMmax * ddcc:
            TM, y2x = refine(R, t, 1, 2)
            if TM > TMmax:
                TMmax, y2x0 = TM, y2x

    if not np.any(y2x0 >= 0):
        raise ValueError("no alignment found between the two structures")

    # Final superposition of the best alignment, then keep pairs within score_d8.
    _, R, t = _detailed_search(
        x, y, y2x0, 40 if fast else 1, 8, local_d0_search, Lnorm, score_d8, d0
    )
    j = np.flatnonzero(y2x0 >= 0)
    i = y2x0[j]
    dist = np.sqrt(_sqdist_rows(_transform(x[i], R, t), y[j]))
    keep = dist <= score_d8
    m1, m2 = i[keep], j[keep]
    if len(m1) == 0:
        raise ValueError("no aligned residues within the distance cut-off")
    xtm, ytm = x[m1], y[m2]
    rmsd = _kabsch_rmsd(xtm, ytm)

    d0_y, d0_search_y = _parameters_for_final(ylen)
    tm_y, R0, t0 = _tmscore8_search(xtm, ytm, 1, 0, d0_search_y, ylen, score_d8, d0_y)
    d0_x, d0_search_x = _parameters_for_final(xlen)
    tm_x, _, _ = _tmscore8_search(xtm, ytm, 1, 0, d0_search_x, xlen, score_d8, d0_x)

    result = TMAlignResult(
        pairs=np.column_stack([m1, m2]),
        tm_norm_x=tm_x,
        tm_norm_y=tm_y,
        rmsd=rmsd,
        rotation=R0,
        translation=t0,
        d0_x=d0_x,
        d0_y=d0_y,
    )
    if seq_x is not None and seq_y is not None:
        result.aligned_x, result.aligned_y, result.aligned_mark = _render_alignment(
            seq_x, seq_y, m1, m2, np.sqrt(_sqdist_rows(_transform(xtm, R0, t0), ytm))
        )
    return result


def _render_alignment(seq_x, seq_y, m1, m2, dist) -> Tuple[str, str, str]:
    out_x: List[str] = []
    out_y: List[str] = []
    marks: List[str] = []
    i_old = j_old = 0
    for k in range(len(m1)):
        for i in range(i_old, m1[k]):
            out_x.append(seq_x[i])
            out_y.append("-")
            marks.append(" ")
        for j in range(j_old, m2[k]):
            out_x.append("-")
            out_y.append(seq_y[j])
            marks.append(" ")
        out_x.append(seq_x[m1[k]])
        out_y.append(seq_y[m2[k]])
        marks.append(":" if dist[k] < _D0_OUT else ".")
        i_old, j_old = m1[k] + 1, m2[k] + 1
    for i in range(i_old, len(seq_x)):
        out_x.append(seq_x[i])
        out_y.append("-")
        marks.append(" ")
    for j in range(j_old, len(seq_y)):
        out_x.append("-")
        out_y.append(seq_y[j])
        marks.append(" ")
    return "".join(out_x), "".join(out_y), "".join(marks)


def tm_superpose(
    P: np.ndarray, Q: np.ndarray, L_ref: Optional[int] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    The TM-score ``max`` over superpositions for points that are already in
    1:1 correspondence (the TM-score program's search, Zhang & Skolnick 2004):
    returns the maximal TM-score normalised by ``L_ref`` (default: the number
    of pairs) and the rotation and translation that achieve it, mapping ``P``
    onto ``Q``.  Unlike Kabsch, which minimises RMSD, this lets poorly fitting
    residues drift so that the well-fitting core scores as highly as possible.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if len(P) != len(Q):
        raise ValueError("P and Q must contain the same number of points")
    if len(P) == 0:
        raise ValueError("no points to superpose")
    L = len(P) if L_ref is None else L_ref
    d0, d0_search = _parameters_for_final(L)
    score, R, t = _tmscore8_search(P, Q, 1, 0, d0_search, L, 0.0, d0)
    return score, R, t
