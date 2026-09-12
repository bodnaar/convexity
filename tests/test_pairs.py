"""Equivalence and structure tests for the general (r, s) kernel.

The first test in this file is the gate on the whole general-pair line of work:
with `s` the perpendicular of `r`, `qsig.pairs` must agree with `qsig.fast`
BIT-FOR-BIT on every integer quantity. `qsig.fast` is in turn bit-identical to
the published `convexity.Convexity`, so passing here chains the new kernel back
to the reference implementation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qsig import directions, fast, pairs


def _blocky():
    a = np.zeros((14, 14), dtype=np.uint8)
    a[3:11, 3:11] = 1
    a[5, 5] = 0
    a[6:9, 7] = 0
    a[4, 9] = 0
    return a


def _disc_with_holes():
    a = np.zeros((16, 16), dtype=np.uint8)
    yy, xx = np.ogrid[:16, :16]
    a[((yy - 8) ** 2 + (xx - 8) ** 2) <= 36] = 1
    a[8, 8] = 0
    a[4:6, 10:12] = 0
    return a


def _two_blobs():
    a = np.zeros((13, 17), dtype=np.uint8)
    a[2:11, 2:8] = 1
    a[4:9, 9:15] = 1
    return a


IMAGES = [_blocky(), _disc_with_holes(), _two_blobs()]
POOL = directions.pool(max_norm2=130)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["points", "rows"])
@pytest.mark.parametrize("img_idx", range(len(IMAGES)))
def test_orthogonal_pair_matches_fast_bit_for_bit(img_idx, method):
    """s = r_perp must reproduce qsig.fast exactly, for every pool direction."""
    img = IMAGES[img_idx]
    for d in POOL:
        v = d.vec
        f = fast.compute(img, 1, 0, v, method)
        p = pairs.compute(img, 1, 0, v, pairs.perp(v), method)

        assert p["q0"] == f["q0"], f"phi sum differs for {v}"
        assert np.array_equal(p["_phi"], f["_phi"]), f"phi differs for {v}"
        assert np.array_equal(p["_norm_part"], f["_norm_part"]), f"norm differs for {v}"
        assert p["_card_f"] == f["_card_f"]
        assert p["_card_f_dash"] == f["_card_f_dash"]
        # The descriptor multiplies the four quadrants, so only the SET is
        # defined; pairs.py does not reproduce fast.py's labelling.
        assert (sorted(q.tobytes() for q in p["_quads"])
                == sorted(q.tobytes() for q in f["_quads"])), f"quadrants differ for {v}"
        assert p["q1"] == f["q1"], f"E differs for {v}"


def test_points_and_rows_agree_on_general_pairs():
    """The two kernels are two evaluations of one sum; they must not diverge."""
    img = _blocky()
    seen = 0
    for r, s in itertools.islice(
            ((r.vec, s.vec) for r in POOL[:14] for s in POOL[:14]
             if pairs.det(r.vec, s.vec) != 0), 0, 150):
        a = pairs.compute(img, 1, 0, r, s, "points")
        b = pairs.compute(img, 1, 0, r, s, "rows")
        assert np.array_equal(a["_phi"], b["_phi"]), f"phi differs for {r},{s}"
        assert a["q1"] == b["q1"], f"E differs for {r},{s}"
        seen += 1
    assert seen > 100


# --------------------------------------------------------------------------
# Lattice structure -- the cost law rests on these
# --------------------------------------------------------------------------

def test_det_generalises_the_norm2_cost_variable():
    """|det(r, r_perp)| = p^2 + q^2, the conference paper's cost variable."""
    for d in POOL:
        assert pairs.det(d.vec, pairs.perp(d.vec)) == d.norm2


def test_mask_has_exactly_det_minus_one_points():
    """Pick's theorem: a primitive pair gives B = 4, so I = |det| - 1.

    This is what makes the cost Theta(mn |det(r, s)|), so it is pinned rather
    than trusted.
    """
    checked = 0
    for r, s in itertools.islice(
            ((r.vec, s.vec) for r in POOL[:20] for s in POOL[:20]
             if pairs.det(r.vec, s.vec) != 0), 0, 250):
        n = len(pairs.mask_offsets(pairs.step(s), pairs.step(r)))
        assert n == pairs.det(r, s) - 1, f"mask size wrong for {r},{s}"
        checked += 1
    assert checked > 200


def test_mask_rows_are_contiguous_for_every_pair():
    """The mask is the interior of a convex parallelogram, so every horizontal
    slice is one run -- for a GENERAL pair, not only an orthogonal one. This is
    what lets the row-prefix kernel generalise."""
    for r, s in itertools.islice(
            ((r.vec, s.vec) for r in POOL[:20] for s in POOL[:20]
             if pairs.det(r.vec, s.vec) != 0), 0, 250):
        pairs.mask_rows(pairs.step(s), pairs.step(r))   # raises if not


def test_row_count_is_bounded_by_the_parallelogram_row_extent():
    """rows <= |r_0| + |s_0| + 1, i.e. the kernel is Theta(mn(|r_0|+|s_0|)).

    Reduces to Theta(mn(p+q)) when s = r_perp, which is the conference paper's
    row-prefix bound.
    """
    for r, s in itertools.islice(
            ((r.vec, s.vec) for r in POOL[:20] for s in POOL[:20]
             if pairs.det(r.vec, s.vec) != 0), 0, 250):
        n = len(pairs.mask_rows(pairs.step(s), pairs.step(r)))
        assert n <= abs(r[0]) + abs(s[0]) + 1, f"row bound violated for {r},{s}"


def test_separating_functional_is_strictly_negative_on_both_steps():
    for r, s in itertools.islice(
            ((r.vec, s.vec) for r in POOL[:20] for s in POOL[:20]
             if pairs.det(r.vec, s.vec) != 0), 0, 250):
        T1, T2 = pairs.step(s), pairs.step(r)
        f = pairs.separating_functional(T1, T2)
        assert f[0] * T1[0] + f[1] * T1[1] < 0
        assert f[0] * T2[0] + f[1] * T2[1] < 0


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_non_primitive_input_is_reduced_not_rejected():
    """k*r is the same direction at k^2 the cost; reducing loses nothing."""
    img = _blocky()
    a = pairs.compute(img, 1, 0, (2, -4), (2, 1), "points")
    b = pairs.compute(img, 1, 0, (1, -2), (2, 1), "points")
    assert a["q1"] == b["q1"]
    assert pairs.as_primitive((3, -6)) == (1, -2)


def test_parallel_directions_are_refused():
    with pytest.raises(ValueError):
        pairs.compute(_blocky(), 1, 0, (1, -2), (2, -4), "points")


def test_zero_vector_is_refused():
    with pytest.raises(ValueError):
        pairs.as_primitive((0, 0))


def test_empty_object_is_refused():
    with pytest.raises(ValueError):
        pairs.compute(np.zeros((8, 8), dtype=np.uint8), 1, 0, (1, 0), (0, 1))
