"""Q-concavity for an ARBITRARY pair of lattice directions -- E_F^{r,s}.

Why this is a separate module
-----------------------------
`qsig.fast` computes `E_F^{r,r_perp}`: a single direction goes in, and the four
quadrants come out of ONE rotated summed-area table reused four times via
`np.rot90`. That trick is only available because the second direction is the
perpendicular of the first, so the four cones are related by 90 degree
rotations of the lattice.

For a general pair `(r, s)` the four cones are NOT related by rotations, so
each needs its own summed-area pass over its own mask. That is a different
algorithm sharing one recurrence, not a flag on the old one -- hence a new
module. `fast.compute` is left untouched so that the published numbers and the
bit-identity tests against `convexity.Convexity` keep their meaning.

Priority note. The descriptor is already DEFINED for a general pair: IWCIA 2025
writes it as `E^{r,s}` and states that the normalisation of its Proposition 1
"can be easily generalized to hold for any pair of directions r and s",
replacing the row and column sums by the number of points of F on the lines
parallel to r and to s through M. That generalisation had not been computed.
This module computes it.

The construction
----------------
Let `L = <r, s>` be the sublattice spanned by the pair. A summed-area recurrence
stepping by `r` and `s`,

    asum(P) = asum(P + T1) + asum(P + T2) - asum(P + T1 + T2) + g(P),

only ever visits points congruent to P modulo `L`, so it sums over a sublattice
of index `|det(r, s)|` rather than over every point of the cone. `g` repairs
that: `g(P)` pre-adds one complete set of coset representatives of `Z^2 / L`
positioned at P, so that the translates tile the cone exactly once.

Those representatives are the **strict interior of the fundamental
parallelogram spanned by the two steps, plus P itself**. When `r` and `s` are
both primitive, neither edge carries an interior lattice point, so the
parallelogram has exactly 4 boundary lattice points and, by Pick's theorem,

    I = area - B/2 + 1 = |det(r, s)| - 2 + 1 = |det(r, s)| - 1

interior points. Adding `P` gives `|det(r, s)|` representatives -- exactly the
index, as required. **So the cost of one evaluation is Theta(mn |det(r, s)|),**
which is the cost law of the conference paper with `|det(r, r_perp)| = p^2 + q^2`
as the orthogonal special case. Primitivity of BOTH directions is what makes
the count exact, and it is enforced below.

What is NOT yet generalised
---------------------------
The `Theta(mn(p+q))` row-prefix kernel. In the orthogonal case the mask's
intersection with any horizontal line is a single run, so the sum needs 2
lookups per mask ROW instead of one per mask POINT. Whether that survives for a
general pair is an open question -- `mask_rows` raises if a row is not
contiguous, and `method="rows"` is offered here only so the question can be
answered by measurement rather than argued. Until it is answered, treat
`method="points"` as the only kernel with a known bound.

Conventions, both inherited unchanged
-------------------------------------
- Quadrants are CLOSED: they include the lines through P parallel to r and s.
- This returns Q-CONCAVITY (`E`). The papers report Q-convexity.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import numpy as np

from .fast import (INT, _accumulate_mask, _accumulate_mask_rows, assert_protocol_safe,
                   njit)

METHODS = ("points", "rows")


# --------------------------------------------------------------------------
# Lattice bookkeeping
# --------------------------------------------------------------------------

def as_primitive(w) -> tuple[int, int]:
    """Reduce a lattice vector to its primitive representative.

    A non-primitive multiple `k*w` is the SAME direction at `k^2` the cost,
    since `|det(k*w, .)| = k * |det(w, .)|`, so nothing is lost by reducing --
    and the Pick count above is only valid for primitive vectors.
    """
    w0, w1 = int(w[0]), int(w[1])
    if w0 == 0 and w1 == 0:
        raise ValueError("the zero vector is not a direction")
    g = gcd(abs(w0), abs(w1))
    return (w0 // g, w1 // g)


def det(r, s) -> int:
    """|det(r, s)|, the index of <r, s> in Z^2 and the cost variable."""
    r, s = as_primitive(r), as_primitive(s)
    return abs(r[0] * s[1] - r[1] * s[0])


def step(w) -> tuple[int, int]:
    """The summed-area step, in (drow, dcol), for a direction vector.

    Matches `qsig.fast`: there the two steps are `-(v0, v1)` for the direction
    itself and `-(v_perp0, v_perp1) = (v1, -v0)` for its perpendicular. So the
    map from a direction to its step is `w -> -w`, applied to both members of
    the pair here.
    """
    w0, w1 = int(w[0]), int(w[1])
    return (-w0, -w1)


def _corner(T) -> tuple[int, int]:
    """A (drow, dcol) step as the (cix, ciy) pair `fast._dp` expects.

    `_dp` reads `asum[r - ciy, c + cix]`; we want `asum[P + T]`, so
    `ciy = -T_row` and `cix = T_col`.
    """
    return (int(T[1]), -int(T[0]))


def mask_offsets(T1, T2) -> np.ndarray:
    """Coset representatives other than P, as (drow, dcol) offsets.

    The strict interior of the lattice parallelogram spanned by `T1` and `T2`.
    Exact integer arithmetic: a point `p` is interior iff `p = a*T1 + b*T2` with
    `0 < a, b < 1`, which after clearing the determinant is a strict inequality
    between integers.

    Returns `|det| - 1` offsets for a primitive pair.
    """
    t1r, t1c = int(T1[0]), int(T1[1])
    t2r, t2c = int(T2[0]), int(T2[1])
    d = t1r * t2c - t1c * t2r
    if d == 0:
        raise ValueError(f"steps {T1} and {T2} are parallel; det is 0")

    rows = [0, t1r, t2r, t1r + t2r]
    cols = [0, t1c, t2c, t1c + t2c]
    out = []
    for pr in range(min(rows), max(rows) + 1):
        for pc in range(min(cols), max(cols) + 1):
            # a = (pr*t2c - pc*t2r)/d, b = (t1r*pc - t1c*pr)/d
            na = pr * t2c - pc * t2r
            nb = t1r * pc - t1c * pr
            if d > 0:
                inside = 0 < na < d and 0 < nb < d
            else:
                inside = d < na < 0 and d < nb < 0
            if inside:
                out.append((pr, pc))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def mask_rows(T1, T2) -> np.ndarray:
    """The same mask as contiguous row runs: (drow, dcol_first, dcol_last).

    Raises if any row is not contiguous -- see the module docstring: whether
    contiguity survives for a general pair is exactly the open question, so
    this fails loudly rather than silently computing something else.
    """
    offs = mask_offsets(T1, T2)
    per_row: dict[int, list[int]] = {}
    for dr, dc in offs:
        per_row.setdefault(int(dr), []).append(int(dc))
    out = []
    for dr in sorted(per_row):
        cs = sorted(per_row[dr])
        if cs != list(range(cs[0], cs[-1] + 1)):
            raise AssertionError(
                f"mask row {dr} of the pair with steps {tuple(T1)}, {tuple(T2)} "
                f"is not contiguous: {cs}. The row-prefix kernel is invalid "
                "here; use method='points'."
            )
        out.append((dr, cs[0], cs[-1]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def separating_functional(T1, T2) -> tuple[int, int]:
    """An integer `f` with `f.T1 < 0` and `f.T2 < 0`.

    Needed because the summed-area recurrence must visit `P + T1` and
    `P + T2` before `P`, so the traversal has to run in increasing `f.P` for
    some `f` strictly negative on both steps.

    `qsig.fast` never needs this: it forces every cone into the up-left
    quadrant with `np.rot90`, where plain row-major order works. That is only
    possible because an orthogonal pair spans exactly 90 degrees, so each cone
    fits in a quadrant. **For a general pair two of the four cones are WIDER
    than 90 degrees and no reflection or rotation of the array fixes them** --
    this is the one structural difference between the two kernels.

    Closed form rather than a search. With `perp(a, b) = (-b, a)`:
    `perp(T2).T2 = 0` and `perp(T2).T1 = -det`, and symmetrically for `T1`, so
    `f = sign(det) * (perp(T2) - perp(T1))` is negative on both. Two linearly
    independent steps span a proper convex cone, so such an `f` always exists.
    """
    t1r, t1c = int(T1[0]), int(T1[1])
    t2r, t2c = int(T2[0]), int(T2[1])
    d = t1r * t2c - t1c * t2r
    if d == 0:
        raise ValueError(f"steps {T1} and {T2} are parallel; det is 0")
    sign = 1 if d > 0 else -1
    f = (sign * (-t2c + t1c), sign * (t2r - t1r))
    if f[0] * t1r + f[1] * t1c >= 0 or f[0] * t2r + f[1] * t2c >= 0:
        raise AssertionError(
            f"separating functional {f} is not strictly negative on both "
            f"{T1} and {T2}; the traversal order would be invalid")
    return f


def visit_order(shape, T1, T2):
    """Row and column indices of every pixel, in a valid traversal order."""
    h, w = shape
    f0, f1 = separating_functional(T1, T2)
    key = f0 * np.arange(h, dtype=np.int64)[:, None] + \
        f1 * np.arange(w, dtype=np.int64)[None, :]
    flat = np.argsort(key.ravel(), kind="stable")
    return (flat // w).astype(np.int64), (flat % w).astype(np.int64)


@njit(cache=True)
def _dp_ordered(g, rows, cols, c1x, c1y, c2x, c2y, c3x, c3y):  # pragma: no cover
    """`fast._dp` with an explicit traversal order instead of a raster scan.

    The body is the reference recurrence verbatim, including the quirk that the
    `c2` term is dropped unless `c1` and `c3` are BOTH in bounds -- that is
    what `convexity.Convexity.rotsat` does, so it is kept for equivalence.
    """
    h, w = g.shape
    asum = np.zeros((h, w), dtype=np.int64)
    for k in range(rows.size):
        r = rows[k]
        c = cols[k]
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


def rotsat(a: np.ndarray, T1, T2, method: str = "points") -> np.ndarray:
    """Summed-area table over the cone spanned by `T1` and `T2`. int64, exact."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    arr = np.asarray(a).astype(INT)
    g = (_accumulate_mask_rows(arr, mask_rows(T1, T2)) if method == "rows"
         else _accumulate_mask(arr, mask_offsets(T1, T2)))
    c1, c3 = _corner(T1), _corner(T2)
    c2 = _corner((int(T1[0]) + int(T2[0]), int(T1[1]) + int(T2[1])))
    rows, cols = visit_order(arr.shape, T1, T2)
    return _dp_ordered(g, rows, cols, c1[0], c1[1], c2[0], c2[1], c3[0], c3[1])


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def line_sums(a: np.ndarray, w) -> np.ndarray:
    """For each pixel, the sum of `a` over the line through it parallel to `w`.

    This is the generalisation the IWCIA 2025 Proposition 1 remark calls for:
    the row and column sums become the counts on the lines parallel to r and to
    s. Lines parallel to `(w1, w0)` in (row, col) are the level sets of
    `w0*row - w1*col`, so one `bincount` over that invariant gives them all.

    `qsig.fast` obtains the second family by `np.rot90`, which only works when
    the second direction is the perpendicular of the first. This does not need
    the trick and reduces to it in that case.
    """
    w0, w1 = int(w[0]), int(w[1])
    h, wd = a.shape
    b = w0 * np.arange(h, dtype=INT)[:, None] - w1 * np.arange(wd, dtype=INT)[None, :]
    lo = int(b.min())
    counts = np.bincount((b - lo).ravel(), weights=a.astype(np.float64).ravel(),
                         minlength=int(b.max()) - lo + 1)
    return counts.astype(INT)[b - lo]


# --------------------------------------------------------------------------
# The descriptor
# --------------------------------------------------------------------------

def compute(img: np.ndarray, obj_a=1, obj_b=0, r=(1, 0), s=(0, 1),
            method: str = "points") -> dict:
    """E_F^{r,s} for one binary image, for an arbitrary primitive pair.

    With `s` the perpendicular of `r` this must agree with
    `qsig.fast.compute(img, obj_a, obj_b, r, method)` bit-for-bit on every
    integer quantity; `tests/test_pairs.py` is that assertion, and it is the
    gate on this whole line of work.

    Returns the same dict shape as `fast.compute`, plus `_det`.
    """
    f = np.asarray(img)
    assert_protocol_safe(max(f.shape))
    a = (f == obj_a)
    b = (f == obj_b)
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"object A or B is empty: a={a.sum()} b={b.sum()}")

    r = as_primitive(r)
    s = as_primitive(s)
    d = det(r, s)
    if d == 0:
        raise ValueError(f"directions {r} and {s} are parallel")

    A, B = step(s), step(r)
    negA, negB = (-A[0], -A[1]), (-B[0], -B[1])

    # Pad enough that no mask reaches around the array. The mask lives inside
    # the parallelogram spanned by the two steps, so its extent is bounded by
    # the sum of the absolute components. The descriptor is invariant to
    # background padding (pinned in tests/test_descriptor.py), so over-padding
    # relative to fast.compute cannot change a value.
    pad = max(abs(r[0]) + abs(s[0]), abs(r[1]) + abs(s[1]))
    ap = np.pad(a, pad, "constant")
    sl = slice(pad, -pad) if pad else slice(None)

    # The four cones at P, one summed-area pass each. The descriptor multiplies
    # all four, so their labelling is irrelevant -- only the SET matters.
    quads = tuple(rotsat(ap, t1, t2, method)[sl, sl] for t1, t2 in
                  ((A, B), (A, negB), (negA, B), (negA, negB)))

    phi = quads[0] * quads[1] * quads[2] * quads[3] * b.astype(INT)
    card_f = INT(a.sum())
    card_f_dash = int(np.count_nonzero(phi))

    norm_part = card_f + line_sums(a, r) + line_sums(a, s)

    # Same arithmetic as fast.compute: 256 is a power of two, so scaling by it
    # is exact and the elementwise quotients are bit-identical to the
    # reference's; only the summation order differs.
    denom = np.power(norm_part, 4).astype(np.float64)
    phi_norm = 256.0 * float((phi.astype(np.float64) / denom).sum())
    q = 0.0 if card_f_dash == 0 else phi_norm / card_f_dash

    # Disjunctive combination over the two OPPOSING cone pairs. The cones were
    # built in the order ((A,B), (A,-B), (-A,B), (-A,-B)), so the cone opposite
    # to index 0 is index 3 and the one opposite to index 1 is index 2.
    #   phi   = n0 n1 n2 n3 = (n0 n3)(n1 n2)   -- conjunctive, the product
    #   phi_D = n0 n3 + n1 n2                  -- disjunctive, the sum
    # Normalisation: with u = n0+n3, v = n1+n2 and u+v <= norm_part = N,
    # AM-GM gives n0 n3 <= u^2/4 and n1 n2 <= v^2/4, and u^2+v^2 <= (u+v)^2,
    # so phi_D <= N^2/4. The tight factor is therefore 4 (a 2-fold bound),
    # against 256 = 4^4 for the 4-fold product above.
    phi_d = (quads[0] * quads[3] + quads[1] * quads[2]) * b.astype(INT)
    card_f_dash_d = int(np.count_nonzero(phi_d))
    denom_d = np.power(norm_part, 2).astype(np.float64)
    phi_d_norm = 4.0 * float((phi_d.astype(np.float64) / denom_d).sum())
    q_d = 0.0 if card_f_dash_d == 0 else phi_d_norm / card_f_dash_d

    # The asymmetry between the two opposing pair-products; see qsig.fast.
    #   u = n0 n3, v = n1 n2;  phi = u v, phi_D = u + v, phi_A = |u - v|.
    # Same 2-fold bound as phi_D, hence the same factor 4 over norm_part^2.
    phi_a = np.abs(quads[0] * quads[3] - quads[1] * quads[2]) * b.astype(INT)
    card_f_dash_a = int(np.count_nonzero(phi_a))
    phi_a_norm = 4.0 * float((phi_a.astype(np.float64) / denom_d).sum())
    q_a = 0.0 if card_f_dash_a == 0 else phi_a_norm / card_f_dash_a

    return {
        "q0": int(phi.sum()), "q1": q, "_det": d,
        "q0_d": int(phi_d.sum()), "q1_d": q_d,
        "q0_a": int(phi_a.sum()), "q1_a": q_a,
        "_phi": phi, "_quads": quads, "_phi_d": phi_d, "_phi_a": phi_a,
        "_card_f_dash_a": card_f_dash_a,
        "_norm_part": norm_part, "_card_f": int(card_f), "_card_f_dash": card_f_dash,
        "_card_f_dash_d": card_f_dash_d,
    }


def perp(w) -> tuple[int, int]:
    """The perpendicular partner `qsig.fast` uses implicitly: (-w1, w0)."""
    w = as_primitive(w)
    return (-w[1], w[0])


def exact_q1(img: np.ndarray, obj_a=1, obj_b=0, r=(1, 0), s=(0, 1)) -> Fraction:
    """The descriptor as an exact rational -- ground truth for small tests."""
    out = compute(img, obj_a, obj_b, r, s, method="points")
    phi, norm_part = out["_phi"], out["_norm_part"]
    if out["_card_f_dash"] == 0:
        return Fraction(0)
    total = Fraction(0)
    nz = np.nonzero(phi)
    for i, j in zip(*nz):
        total += Fraction(int(phi[i, j]) * 256, int(norm_part[i, j]) ** 4)
    return total / out["_card_f_dash"]
