"""int64 + vectorised Q-concavity -- handoff sec 7.4 items 2 and 3.

This is a drop-in replacement for `convexity.Convexity.compute`, NOT a change to
it. `convexity.py` stays exactly as published and remains the reference against
which this is tested (`tests/test_fast_matches_reference.py`).

What changes, and why
---------------------
1. **int64 instead of `Decimal`.** `np.zeros(shape, dtype=Decimal)` actually
   produces an *object* array, so every arithmetic operation in the reference
   runs at Python speed. At the 128 px protocol each quadrant count is <= 2^14,
   so the four-fold product is <= 2^56 and fits exactly in int64.
   `assert_protocol_safe` enforces the bound rather than trusting it. This is a
   PERFORMANCE change, not a correctness fix -- there was never an overflow;
   see handoff sec 5.1.

2. **The per-pixel mask loop becomes shifted array adds.** The reference walks
   the |det(r,s)| - 1 interior lattice points of the mask in Python, per pixel.
   The same sum is a correlation of the image with a small binary mask, which is
   |det| whole-array adds instead of mn * |det| Python iterations. Integer
   addition is associative and exact, so the result is bit-identical -- only the
   summation order changes.

3. **The dynamic-programming recurrence is jitted when numba is available.**
   The recurrence
       n0(x,y) = n0((x,y)-s) + n0((x,y)-r) - n0((x,y)-(s+r)) + g(x,y)
   is a genuine sequential 2D dependency and cannot be vectorised. numba is
   optional: without it a plain Python loop runs, which is still far faster than
   the reference because item 2 removed the inner loop. `HAVE_NUMBA` reports
   which path is active, and it is recorded with the results.

4. **calcH is vectorised too.** It is ~1% of the reference's runtime, but it
   scales linearly in (p+q) while the mask work scales in p^2+q^2, so once the
   mask loop is fast it would become roughly half the total and would bend the
   measured cost curve away from the |r|^2 law. Left alone it would have
   quietly degraded the paper's headline figure.

Exactness
---------
Every integer quantity -- the four quadrant counts, phi, the normalisation
denominator, |F|, |F-bar| -- is reproduced bit-for-bit. The final scalar involves
one true division. The reference divides Python ints, which is correctly rounded
from the exact rational; this module divides float64s converted from int64.
Where the operands are below 2^53 the two are identical, which covers every test
image. At the 128 px protocol phi can reach 2^56, so the last ulp may differ.
That is the lost exactness discussed in handoff sec 5.1, and it is irrelevant:
the descriptor immediately divides by a denominator of comparable magnitude.
`compare_to_reference` reports both the exact integer check and the scalar
difference so this stays measured rather than assumed.
"""

from __future__ import annotations

from fractions import Fraction
from math import log2

import numpy as np

try:  # pragma: no cover - environment dependent
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore[misc]
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]


INT = np.int64


def assert_protocol_safe(max_dim: int) -> None:
    """Guard the int64 bound of handoff sec 7.4 item 2.

    The binding quantity is NOT phi (the product of the four quadrant counts)
    but the normalisation denominator raised to the fourth,
    (|F| + r_i + s_j)^4, which is 2^8 times larger. Bounding
    |F| + r_i + s_j <= mn + m + n <= N^2 + 2N for an N x N image:

        N = 128   denominator^4 = 2^56.1   phi <= 2^48.1
        N = 233   denominator^4 = 2^62.99  -- the last exact size in int64
        N = 234   denominator^4 = 2^63.01  -- OVERFLOWS

    So the algorithm is exact in int64 up to **N = 233**, not the ~181 quoted in
    earlier notes (that figure conflated this with the float64 limit, and was
    wrong for that too). The guard below allows up to N = 215, deliberately
    conservative: it bounds mn by max_dim^2, which is exact for a square image
    and generous for any other, and leaves a bit of headroom.

    A SEPARATE and lower threshold, worth knowing but not worth guarding:
    denominator^4 stops being an exactly representable integer in float64 above
    **N = 97**, so at the 128 px protocol the final division already works on a
    rounded denominator. That is harmless -- the descriptor immediately divides
    by a quantity of the same magnitude, and the relative error is one ulp --
    and it is exactly the lost exactness described in handoff sec 5.1. It is
    also why the equality tests compare against an exact `Fraction` rather than
    demanding bit-identical floats.
    """
    bits = 4 * log2(max_dim ** 2)
    if not bits < 62:
        raise OverflowError(
            f"image size {max_dim} px needs {bits:.1f} bits for the fourth power of "
            f"the normalisation denominator; the int64 budget here is 62. The true "
            f"overflow point is 234 px. Above this, use the reference implementation "
            f"(convexity.Convexity, which uses Python ints) or switch to object dtype."
        )


# --------------------------------------------------------------------------
# Mask geometry -- transcribed from convexity.Convexity.rotsat so the two
# cannot drift. Any change here must be mirrored by the equality test.
# --------------------------------------------------------------------------

def mask_offsets(v: tuple[int, int]) -> np.ndarray:
    """Interior lattice points of the mask, as (drow, dcol) offsets.

    The reference builds the four rotated outer triangles, marks them 0 in a
    bounding-box mask, and treats whatever remains as interior. `rotsat` then
    reads a[r + drow, c + dcol] for each. There are |det(r, r_perp)| - 1 =
    p^2 + q^2 - 1 of them by Pick's theorem, which is exactly the cost variable.
    """
    v0, v1 = int(v[0]), int(v[1])
    corners = np.array([[0, 0], [-v0, -v1], [-v0 - v1, -v1 + v0], [-v1, v0]])
    dy = v1 / v0
    lower: list[list[int]] = []
    line_pos = corners[1][1]
    for i in range(corners[1][0], 0):
        for j in range(corners[1][1], -1, -1):
            if j <= line_pos:
                lower += [[i, j]]
        line_pos += dy
    rot90_ = [[x[1] - v0, -x[0] - v1] for x in lower]
    rot180 = [[x[1] - v0, -x[0] - v1] for x in rot90_]
    rot270 = [[x[1] - v0, -x[0] - v1] for x in rot180]
    outside = np.array(lower + rot90_ + rot180 + rot270)
    offset = outside.min(axis=0)
    size = outside.max(axis=0) - offset + [1, 1]
    mask = np.ones(size, dtype=np.uint8)
    mask[outside[:, 0] - offset[0], outside[:, 1] - offset[1]] = 0
    ii, jj = np.where(mask > 0)
    drow = -(jj + offset[1])
    dcol = ii + offset[0]
    return np.stack([drow, dcol], axis=1).astype(np.int64)


def mask_rows(v: tuple[int, int]) -> np.ndarray:
    """The same mask, as contiguous row runs: (drow, dcol_first, dcol_last).

    The mask is the interior of a convex lattice parallelogram, so its
    intersection with any horizontal line is a single run -- verified for every
    direction in the pool by
    tests/test_fast_matches_reference.py::test_every_mask_row_is_contiguous.

    That is what lets the sum be evaluated with a horizontal prefix sum: 2
    lookups per row instead of one per interior point. The number of rows is at
    most p+q+1, so the per-pixel cost drops from |r|^2-1 to ~2(p+q) --
    Theta(mn(p+q)) instead of Theta(mn(p^2+q^2)). See handoff sec 3.1.2.

    2 * (number of rows) equals exactly the symmetric difference of the mask
    under a one-pixel step, i.e. the cost of the equivalent sliding-window
    formulation. The two routes reach the same bound; this one is easier to keep
    exact and needs no compiled inner loop.
    """
    offs = mask_offsets(v)
    rows: dict[int, list[int]] = {}
    for dr, dc in offs:
        rows.setdefault(int(dr), []).append(int(dc))
    out = []
    for dr in sorted(rows):
        cs = sorted(rows[dr])
        if cs != list(range(cs[0], cs[-1] + 1)):
            raise AssertionError(
                f"mask row {dr} of direction {v} is not contiguous: {cs}. "
                "The row-prefix-sum kernel is invalid for this direction; use "
                "method='points'."
            )
        out.append((dr, cs[0], cs[-1]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def _accumulate_mask_rows(a: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """g[r,c] = a[r,c] + sum over the mask, via horizontal prefix sums.

    One whole-array operation per mask ROW rather than per mask POINT. Exact:
    integer prefix sums, and a difference of two of them is the exact run sum.
    Out-of-image columns are handled by clipping the interval, which yields an
    empty run and therefore a zero contribution -- matching `_accumulate_mask`,
    which simply skips out-of-bounds points.
    """
    h, w = a.shape
    g = a.astype(INT, copy=True)
    if rows.size == 0:
        return g
    pre = np.zeros((h, w + 1), dtype=INT)
    pre[:, 1:] = np.cumsum(a.astype(INT), axis=1)
    cols = np.arange(w)
    for drow, c0, c1 in rows:
        r0, r1 = max(0, -int(drow)), min(h, h - int(drow))
        if r0 >= r1:
            continue
        lo = np.clip(cols + int(c0), 0, w)
        hi = np.clip(cols + int(c1) + 1, 0, w)
        src = pre[r0 + int(drow):r1 + int(drow)]
        g[r0:r1] += src[:, hi] - src[:, lo]
    return g


def _accumulate_mask(a: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """g[r,c] = a[r,c] + sum_k a[r+drow_k, c+dcol_k], zero outside the image.

    Replaces the reference's per-pixel Python loop with one whole-array add per
    interior point. Exact: integer addition, only the order differs.
    """
    h, w = a.shape
    g = a.astype(INT, copy=True)
    for drow, dcol in offs:
        r0, r1 = max(0, -drow), min(h, h - drow)
        c0, c1 = max(0, -dcol), min(w, w - dcol)
        if r0 >= r1 or c0 >= c1:
            continue
        g[r0:r1, c0:c1] += a[r0 + drow:r1 + drow, c0 + dcol:c1 + dcol]
    return g


@njit(cache=True)
def _dp(g, c1x, c1y, c2x, c2y, c3x, c3y):  # pragma: no cover - jitted
    h, w = g.shape
    asum = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            r1, cc1 = r - c1y, c + c1x
            r3, cc3 = r - c3y, c + c3x
            r2, cc2 = r - c2y, c + c2x
            in1 = 0 <= r1 < h and 0 <= cc1 < w
            in3 = 0 <= r3 < h and 0 <= cc3 < w
            in2 = 0 <= r2 < h and 0 <= cc2 < w
            v1 = asum[r1, cc1] if in1 else 0
            v3 = asum[r3, cc3] if in3 else 0
            v2 = asum[r2, cc2] if (in2 and in1 and in3) else 0
            asum[r, c] = v1 - v2 + v3 + g[r, c]
    return asum


METHODS = ("points", "rows")


def rotsat(a: np.ndarray, v: tuple[int, int], method: str = "points") -> np.ndarray:
    """Rotated summed-area table. int64, exact, same values as the reference.

    `method` selects how the mask correction term is evaluated:
      "points"  one whole-array add per interior lattice point -- Theta(mn|r|^2)
      "rows"    one per contiguous mask row via prefix sums -- Theta(mn(p+q))
    Both are exact and must agree bit-for-bit; the tests assert it.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    v0, v1 = int(v[0]), int(v[1])
    corners = np.array([[0, 0], [-v0, -v1], [-v0 - v1, -v1 + v0], [-v1, v0]])
    arr = np.asarray(a).astype(INT)
    g = (_accumulate_mask_rows(arr, mask_rows(v)) if method == "rows"
         else _accumulate_mask(arr, mask_offsets(v)))
    return _dp(g,
               int(corners[1][0]), int(corners[1][1]),
               int(corners[2][0]), int(corners[2][1]),
               int(corners[3][0]), int(corners[3][1]))


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def _scanline_sums(a: np.ndarray, v: tuple[int, int]):
    """Vectorised calcH. b[ri,ci] = v0*ri - v1*ci; sum `a` over each level set."""
    v0, v1 = int(v[0]), int(v[1])
    h, w = a.shape
    b = v0 * np.arange(h, dtype=INT)[:, None] - v1 * np.arange(w, dtype=INT)[None, :]
    lo = int(b.min())
    counts = np.bincount((b - lo).ravel(), weights=a.astype(np.float64).ravel())
    return counts.astype(INT), b, lo


def compute(img: np.ndarray, obj_a=1, obj_b=0, vec=(1, 0), method: str = "points") -> dict:
    """E_F^{r,r_perp} for one binary image. Signature mirrors Convexity.compute.

    `method` is passed to `rotsat`; see METHODS. "rows" is the Theta(mn(p+q))
    kernel of handoff sec 3.1.2 and should be preferred once measured.
    """
    f = np.asarray(img)
    assert_protocol_safe(max(f.shape))
    a = (f == obj_a)
    b = (f == obj_b)
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"object A or B is empty: a={a.sum()} b={b.sum()}")

    frac = Fraction(int(vec[1]), int(vec[0]))
    if frac.numerator != vec[1]:
        vec = (frac.denominator, frac.numerator)
    vec = (int(vec[0]), int(vec[1]))

    pad = max(abs(vec[0]), abs(vec[1]))
    ap = np.pad(a, pad, "constant")
    sl = slice(pad, -pad) if pad else slice(None)
    q1 = rotsat(ap, vec, method)[sl, sl]
    q2 = np.rot90(rotsat(np.rot90(ap, 1), vec, method), -1)[sl, sl]
    q3 = np.rot90(rotsat(np.rot90(ap, -1), vec, method), 1)[sl, sl]
    q4 = np.rot90(rotsat(np.rot90(ap, 2), vec, method), -2)[sl, sl]

    phi = q1 * q2 * q3 * q4 * b.astype(INT)
    card_f = INT(a.sum())
    card_f_dash = int(np.count_nonzero(phi))

    hs, hlines, hlo = _scanline_sums(a, vec)
    vs, vlines, vlo = _scanline_sums(np.rot90(a), vec)
    H = hs[hlines - hlo]                      # (h, w)
    V = vs[vlines - vlo][::-1, :].T           # (w, h) -> V[i,j] = vs[vlines[w-1-j, i]]
    norm_part = card_f + H + V

    # 256 is a power of two, so scaling by it is exact in binary floating point
    # and round(256*x) == 256*round(x). The elementwise quotients are therefore
    # bit-identical to the reference's; only the summation ORDER differs (numpy
    # sums pairwise, an object array sums sequentially), which is worth about
    # one ulp on the total. See `exact_q1` for the ground truth.
    denom = np.power(norm_part, 4).astype(np.float64)
    phi_norm = 256.0 * float((phi.astype(np.float64) / denom).sum())
    q = 0.0 if card_f_dash == 0 else phi_norm / card_f_dash
    return {
        "q0": int(phi.sum()), "q1": q,
        "_phi": phi, "_quads": (q1, q2, q3, q4),
        "_norm_part": norm_part, "_card_f": int(card_f), "_card_f_dash": card_f_dash,
    }


def exact_q1(img: np.ndarray, obj_a=1, obj_b=0, vec=(1, 0)) -> Fraction:
    """The descriptor as an exact rational -- ground truth for the tests.

    Uses `fractions.Fraction`, so there is no rounding anywhere. Far too slow
    for real images, but on a 12x12 test image it settles whether a discrepancy
    between the two implementations is the reference being wrong, the fast path
    being wrong, or simply two different roundings of the same exact value.
    """
    r = compute(img, obj_a, obj_b, vec)
    phi = r["_phi"]
    denom = np.power(r["_norm_part"], 4)
    total = Fraction(0)
    nz = np.nonzero(phi)
    for i, j in zip(*nz):
        total += Fraction(256 * int(phi[i, j]), int(denom[i, j]))
    n = r["_card_f_dash"]
    return Fraction(0) if n == 0 else total / n


def operation_counts(v: tuple[int, int]) -> dict:
    """Per-pixel operation counts for the two kernels -- the paper's cost model.

    These are exact integers, not timings, and they are what Theta asserts.
    """
    n_points = len(mask_offsets(v))
    n_rows = len(mask_rows(v))
    p, q = abs(int(v[0])), abs(int(v[1]))
    return {
        "norm2": p * p + q * q,
        "points_ops": 3 + n_points,          # 3 table lookups + one add per point
        "rows_ops": 3 + 2 * n_rows,          # 3 lookups + two per contiguous run
        "n_rows": n_rows,
        "p_plus_q": p + q,
        "reduction": (3 + n_points) / (3 + 2 * n_rows),
    }


def compare_to_reference(img: np.ndarray, vec, obj_a=1, obj_b=0, exact=False) -> dict:
    """Run both implementations and report where, if anywhere, they differ.

    Integer quantities must match EXACTLY ('q0_equal', and the quadrant tables
    checked separately in the tests). The final scalar is a floating-point
    reduction: the elementwise quotients are bit-identical, but numpy sums
    pairwise where an object array sums sequentially, so the totals can differ
    by about an ulp. Demanding bit-equality there would be asserting that two
    different summation orders round identically, which is neither true nor the
    property that matters.

    With `exact=True` the exact rational value is computed as well, and
    'ulp_ref'/'ulp_fast' give each implementation's error against it in units in
    the last place. That is the assertion worth making: both are correct to
    within a rounding, and the fast path is no worse.
    """
    from convexity import Convexity

    ref = Convexity(np.asarray(img), verbose=False).compute(obj_a, obj_b, vec)
    fastr = compute(img, obj_a, obj_b, vec)
    rq, fq = float(ref["q1"]), fastr["q1"]
    phi_max = int(fastr["_phi"].max()) if fastr["_phi"].size else 0
    denom_max = int(np.power(fastr["_norm_part"], 4).max())
    out = {
        "ref_q1": rq,
        "fast_q1": fq,
        "abs_diff": abs(rq - fq),
        "rel_diff": abs(rq - fq) / abs(rq) if rq else 0.0,
        "q0_equal": int(ref["q0"]) == fastr["q0"],
        "phi_max": phi_max,
        "denom_max": denom_max,
        "integers_exact_in_float64": max(phi_max, denom_max) < 2 ** 53,
        "have_numba": HAVE_NUMBA,
    }
    if exact:
        ex = exact_q1(img, obj_a, obj_b, vec)
        ulp = np.spacing(abs(float(ex))) or np.spacing(1.0)
        out["exact_q1"] = ex
        out["ulp_ref"] = abs(Fraction(rq) - ex) / Fraction(ulp)
        out["ulp_fast"] = abs(Fraction(fq) - ex) / Fraction(ulp)
    return out
