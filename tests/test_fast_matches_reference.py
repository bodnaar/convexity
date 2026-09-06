"""The single most important test in the project -- handoff sec 7.4 item 3.

The refactor changes the numeric type AND the execution path at once. Without an
equality test against the published implementation, a silent discrepancy would
not surface until the results look odd in week 4, by which point days of pool
computation would be wrong.

Every integer quantity must match BIT-FOR-BIT: the four quadrant tables, phi,
the normalisation denominator, |F| and |F-bar|.

The final scalar is a floating-point REDUCTION, and there bit-equality is the
wrong thing to demand. The elementwise quotients are bit-identical (scaling by
256 is exact in binary), but numpy sums pairwise where the reference's object
array sums sequentially, so the totals differ by about an ulp. The assertion
that actually means something is made against the EXACT RATIONAL value computed
with `fractions.Fraction`: both implementations must be within a few ulp of it,
and the fast path must be no worse than the reference. Pairwise summation is in
fact the more accurate of the two.

Coverage is EVERY direction in the pool, including the degenerate axis case
(1,0), on random binary images -- not a hand-picked few.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from convexity import Convexity
from qsig import fast
from qsig import directions as D

POOL = D.pool(max_norm2=26)          # (1,0) .. (5,1)/(1,5): 15 directions
EPS = np.finfo(np.float64).eps       # 2.22e-16


def random_image(rng, h, w, density=0.5):
    """A random binary image that is guaranteed to have both phases."""
    while True:
        img = (rng.random((h, w)) < density).astype(np.uint8)
        if 0 < img.sum() < img.size:
            return img


def reference_internals(img, vec, obj_a=1, obj_b=0):
    """Recompute the reference's intermediate arrays, for exact comparison."""
    f = np.asarray(img)
    a = f == obj_a
    pad = max(abs(vec[0]), abs(vec[1]))
    ap = np.pad(a, pad, "constant")
    sl = slice(pad, -pad) if pad else slice(None)
    c = Convexity(f, verbose=False)
    q1 = c.rotsat(ap, vec)[sl, sl]
    q2 = np.rot90(c.rotsat(np.rot90(ap, 1), vec), -1)[sl, sl]
    q3 = np.rot90(c.rotsat(np.rot90(ap, -1), vec), 1)[sl, sl]
    q4 = np.rot90(c.rotsat(np.rot90(ap, 2), vec), -2)[sl, sl]
    return [np.array(q.tolist(), dtype=object) for q in (q1, q2, q3, q4)]


@pytest.mark.parametrize("d", POOL, ids=lambda d: f"{d.p}x{d.q}")
def test_quadrant_tables_are_bit_identical(d):
    rng = np.random.default_rng(hash((d.p, d.q)) % (2**32))
    img = random_image(rng, 11, 13)
    ref = reference_internals(img, d.vec)
    got = fast.compute(img, 1, 0, d.vec)["_quads"]
    for k, (r, g) in enumerate(zip(ref, got)):
        assert np.array_equal(np.array(r.tolist(), dtype=np.int64), g), \
            f"quadrant {k + 1} differs for direction ({d.p},{d.q})"


@pytest.mark.parametrize("d", POOL, ids=lambda d: f"{d.p}x{d.q}")
def test_descriptor_matches_reference_on_random_images(d):
    rng = np.random.default_rng(1000 + d.norm2)
    for _ in range(3):
        img = random_image(rng, 12, 12)
        r = fast.compare_to_reference(img, d.vec)
        assert r["q0_equal"], f"phi sum differs for ({d.p},{d.q})"
        assert r["integers_exact_in_float64"], "test image unexpectedly exceeded 2^53"
        assert r["rel_diff"] <= 8 * EPS, (
            f"({d.p},{d.q}): ref={r['ref_q1']!r} fast={r['fast_q1']!r} "
            f"reldiff={r['rel_diff']:.3e} = {r['rel_diff']/EPS:.1f} eps"
        )


@pytest.mark.parametrize("shape", [(7, 7), (9, 15), (15, 9), (16, 16)])
def test_matches_across_image_shapes(shape):
    """Non-square images exercise the rot90 bookkeeping in both directions."""
    rng = np.random.default_rng(sum(shape))
    img = random_image(rng, *shape)
    for d in (D.Direction(1, 0), D.Direction(1, 1), D.Direction(3, 1), D.Direction(1, 3)):
        r = fast.compare_to_reference(img, d.vec)
        assert r["rel_diff"] <= 8 * EPS, f"{shape} {d}: {r}"


def test_matches_on_structured_shapes():
    """Random noise and real silhouettes stress different parts of the DP."""
    img = np.zeros((20, 20), np.uint8)
    img[4:16, 4:16] = 1                       # Q-convex: phi must be 0
    r = fast.compare_to_reference(img, (3, -1))
    assert r["fast_q1"] == r["ref_q1"] == 0.0      # exactly zero: no rounding involved

    img[8:12, 4:10] = 0                       # carve a notch: phi > 0
    r = fast.compare_to_reference(img, (3, -1))
    assert r["fast_q1"] > 0.0 and r["rel_diff"] <= 8 * EPS


def test_sparse_and_dense_images():
    rng = np.random.default_rng(7)
    for density in (0.05, 0.5, 0.95):
        img = random_image(rng, 14, 14, density)
        r = fast.compare_to_reference(img, (2, -1))
        assert r["rel_diff"] <= 8 * EPS, f"density {density}: {r}"


def test_mask_offset_count_equals_pick_bound():
    """|interior points| = |det(r, r_perp)| - 1 = p^2 + q^2 - 1 (Pick).

    This is the geometric fact the whole cost law rests on, checked against the
    mask the reference actually builds rather than against the derivation.
    """
    for d in D.pool(max_norm2=50):
        n = len(fast.mask_offsets(d.vec))
        assert n == d.norm2 - 1, f"({d.p},{d.q}): {n} interior points, expected {d.norm2 - 1}"


def test_protocol_guard_allows_128_and_rejects_256():
    fast.assert_protocol_safe(128)
    fast.assert_protocol_safe(181)
    with pytest.raises(OverflowError):
        fast.assert_protocol_safe(256)


def test_empty_phase_raises():
    with pytest.raises(ValueError):
        fast.compute(np.ones((8, 8), np.uint8), 1, 0, (1, 0))
    with pytest.raises(ValueError):
        fast.compute(np.zeros((8, 8), np.uint8), 1, 0, (1, 0))


def test_fast_is_actually_faster():
    """Guards against a 'refactor' that is merely a rewrite. Modest threshold so
    it does not flake on a loaded machine -- the real numbers come from
    scripts/fit_cost_law.py."""
    import time

    rng = np.random.default_rng(11)
    img = random_image(rng, 40, 40)
    vec = (5, -1)

    t0 = time.perf_counter()
    Convexity(img, verbose=False).compute(1, 0, vec)
    t_ref = time.perf_counter() - t0

    fast.compute(img, 1, 0, vec)                      # warm any jit
    t0 = time.perf_counter()
    fast.compute(img, 1, 0, vec)
    t_fast = time.perf_counter() - t0

    assert t_fast < t_ref / 3, f"speedup only {t_ref / t_fast:.1f}x (numba={fast.HAVE_NUMBA})"


@pytest.mark.parametrize("d", [D.Direction(1, 0), D.Direction(1, 1), D.Direction(3, 1),
                               D.Direction(1, 3), D.Direction(4, 3)],
                         ids=lambda d: f"{d.p}x{d.q}")
def test_both_implementations_round_the_same_exact_rational(d):
    """The assertion that actually establishes correctness.

    Computes the descriptor as an exact Fraction and measures each
    implementation's error against it in ulp. If the fast path were WRONG rather
    than merely rounded differently, its error would be enormous; if the
    reference were wrong, the same. Requiring the fast path to be no worse than
    the reference also stops a future 'optimisation' from quietly trading
    accuracy for speed.
    """
    rng = np.random.default_rng(4242 + d.norm2)
    img = random_image(rng, 10, 10)
    r = fast.compare_to_reference(img, d.vec, exact=True)
    assert r["ulp_ref"] <= 8, f"reference is {float(r['ulp_ref']):.1f} ulp from exact"
    assert r["ulp_fast"] <= 8, f"fast path is {float(r['ulp_fast']):.1f} ulp from exact"
    assert r["ulp_fast"] <= r["ulp_ref"] + 1, (
        f"fast path ({float(r['ulp_fast']):.1f} ulp) is less accurate than the "
        f"reference ({float(r['ulp_ref']):.1f} ulp)"
    )


def test_elementwise_quotients_are_bit_identical():
    """Locates the ulp difference precisely: it is in the SUM, not the terms.

    If this ever fails, the discrepancy has moved into the arithmetic and the
    tolerance in the tests above is hiding a real bug.
    """
    from convexity import Convexity

    rng = np.random.default_rng(99)
    img = random_image(rng, 11, 11)
    vec = (3, -1)
    f = fast.compute(img, 1, 0, vec)
    denom = np.power(f["_norm_part"], 4)
    mine = 256.0 * (f["_phi"].astype(np.float64) / denom.astype(np.float64))
    theirs = np.array(
        [[float(256 * int(f["_phi"][i, j])) / float(int(denom[i, j]))
          for j in range(denom.shape[1])] for i in range(denom.shape[0])]
    )
    assert np.array_equal(mine, theirs), "elementwise quotients differ -- not just summation order"
    assert Convexity(img, verbose=False).compute(1, 0, vec)["q1"] is not None
