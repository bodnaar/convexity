"""The local normalisation, checked against the cones it is meant to bound.

Why this module exists
----------------------
The descriptor divides phi = n0 n1 n2 n3 at a background point P by
((|F| + h + v)/4)^4. That bound is AM-GM applied to IWCIA 2017 Property 3,

    n0 + n1 + n2 + n3 = |F| + h + v + f(P),

which holds when h and v count the object on the two lattice lines through P
that bound the cones -- the lines along the cone GENERATORS. The four cones
are closed, so exactly those points are counted twice.

`convexity.Convexity.calcH` (mirrored by `fast._scanline_sums` and
`pairs.line_sums`) counts on a different family. The cone code (`rotsat`)
reads its vector as Cartesian with y pointing UP ("v(x,y) is from the descartes
lower right quad"); calcH reads the same vector with y pointing DOWN ("v2 = y
(row) offset, positive is downwards"). In (row, col) the cone generators of
`fast.compute(vec=(v0, v1))` are (v0, v1) and (-v1, v0), while calcH's lines
run along (v1, v0) and (v0, -v1): the transpose. The two families coincide only
when the direction set is closed under transposition -- (1,0) and (1,+-1) in
the orthogonal case.

What this module does
---------------------
Computes the four cones ONCE (via `fast.compute` or `pairs.compute`, untouched)
and evaluates the three combination rules under both normalisations:

  "code"       the published line family. Reproduces fast/pairs exactly.
  "generator"  lines along the cone generators. Property 3 then holds exactly,
               so phi_hat <= 1 pointwise and every descriptor lies in [0, 1].

Nothing published is modified. `convexity.py`, `qsig.fast` and `qsig.pairs`
keep their meaning; this sits beside them.
"""

from __future__ import annotations

import numpy as np

from . import fast as _fast
from . import pairs as _pairs
from .fast import INT

NORMS = ("code", "generator")


def generator_line_sums(a: np.ndarray, w) -> np.ndarray:
    """For each pixel, the count of `a` on the lattice line through it along `w`.

    `w` is a (drow, dcol) vector -- the same coordinates `pairs.step` uses, in
    which the cone generators of `pairs.compute(r=w, ...)` are `w` itself.
    `pairs.line_sums(a, w)` runs along the transpose (w1, w0); swapping the
    components before the call is the whole correction.
    """
    return _pairs.line_sums(a, (int(w[1]), int(w[0])))


def rules(quads, norm_part: np.ndarray, bg: np.ndarray) -> dict:
    """q1, q1_d, q1_a from four cone tables and a normalisation term.

    The arithmetic is copied from `fast.compute` operation for operation, so
    with the code's `norm_part` the results are bit-identical to it (and to
    `pairs.compute`, which uses the same lines). Also returns the largest
    pointwise phi_hat = 256 phi / norm^4, the quantity the AM-GM bound caps at 1.
    """
    n0, n1, n2, n3 = quads
    b = bg.astype(INT)
    phi = n0 * n1 * n2 * n3 * b
    nz = int(np.count_nonzero(phi))
    denom = np.power(norm_part, 4).astype(np.float64)
    phi_hat = 256.0 * (phi.astype(np.float64) / denom)
    q = 0.0 if nz == 0 else 256.0 * float((phi.astype(np.float64) / denom).sum()) / nz

    denom_d = np.power(norm_part, 2).astype(np.float64)
    phi_d = (n0 * n3 + n1 * n2) * b
    nz_d = int(np.count_nonzero(phi_d))
    q_d = 0.0 if nz_d == 0 else 4.0 * float((phi_d.astype(np.float64) / denom_d).sum()) / nz_d
    phi_a = np.abs(n0 * n3 - n1 * n2) * b
    nz_a = int(np.count_nonzero(phi_a))
    q_a = 0.0 if nz_a == 0 else 4.0 * float((phi_a.astype(np.float64) / denom_d).sum()) / nz_a

    return {"q1": q, "q1_d": q_d, "q1_a": q_a, "q0": int(phi.sum()),
            "phi_hat_max": float(phi_hat.max()) if phi_hat.size else 0.0,
            "phi_hat_d_max": float((4.0 * phi_d / denom_d).max()) if phi_d.size else 0.0}


def _both(out: dict, a: np.ndarray, bg: np.ndarray, g1, g2) -> dict:
    gen = out["_card_f"] + generator_line_sums(a, g1) + generator_line_sums(a, g2)
    return {"code": rules(out["_quads"], out["_norm_part"], bg),
            "generator": rules(out["_quads"], gen, bg),
            "_quads": out["_quads"], "_norm_code": out["_norm_part"], "_norm_gen": gen}


def orthogonal(img: np.ndarray, vec, obj_a=1, obj_b=0, method: str = "rows") -> dict:
    """Both normalisations for `fast.compute(img, obj_a, obj_b, vec, method)`.

    Cone generators in (row, col): `vec` and its perpendicular (-v1, v0) -- the
    negated summed-area steps, exactly as `pairs.step` documents.
    """
    f = np.asarray(img)
    out = _fast.compute(f, obj_a, obj_b, vec, method)
    v0, v1 = int(vec[0]), int(vec[1])
    return _both(out, f == obj_a, f == obj_b, (v0, v1), (-v1, v0))


def pair(img: np.ndarray, r, s, obj_a=1, obj_b=0, method: str = "rows") -> dict:
    """Both normalisations for `pairs.compute(img, obj_a, obj_b, r, s, method)`."""
    f = np.asarray(img)
    out = _pairs.compute(f, obj_a, obj_b, r, s, method)
    return _both(out, f == obj_a, f == obj_b, _pairs.as_primitive(r), _pairs.as_primitive(s))


def property3_residual(res: dict, img: np.ndarray, obj_a=1) -> dict:
    """max |n0+n1+n2+n3 - (|F| + h + v + f)| under each line family. 0 = holds."""
    a = (np.asarray(img) == obj_a).astype(INT)
    total = sum(res["_quads"])
    return {"code": int(np.abs(total - (res["_norm_code"] + a)).max()),
            "generator": int(np.abs(total - (res["_norm_gen"] + a)).max())}
