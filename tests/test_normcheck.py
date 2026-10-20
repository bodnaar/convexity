"""The normalisation check: which line family makes Property 3 hold.

IWCIA 2017 Property 3: n0 + n1 + n2 + n3 = |F| + h + v + f(P), with h and v the
object counts on the two lines through P that bound the (closed) cones. These
tests pin that the cone generators are those lines, that the published line
family (calcH / fast._scanline_sums / pairs.line_sums) is a different family
except for transpose-closed direction sets, and what that does to the bound.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from convexity import Convexity
from qsig import directions, fast, normcheck, pairs

POOL = directions.by_angle(directions.pool(max_norm2=130))
WIDE = [(tuple(d["r"]), tuple(d["s"])) for d in json.load(open(os.path.join(
    os.path.dirname(__file__), "..", "experiments", "pairs_and_rules", "wide_pairs.json")))]


def _random_masks(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        h, w = rng.integers(12, 26, 2)
        a = (rng.random((h, w)) < rng.uniform(0.2, 0.8)).astype(np.uint8)
        if 0 < a.sum() < a.size:
            out.append(a)
    return out


MASKS = _random_masks()


def _x_shape(vec, k=4):
    """Object = the two generator lines through the centre, centre removed."""
    v0, v1 = vec
    size = 2 * k * max(abs(v0), abs(v1)) + 3
    a = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    for gr, gc in ((v0, v1), (-v1, v0)):
        for t in range(-k, k + 1):
            if t:
                a[c + t * gr, c + t * gc] = 1
    return a


def _transpose_closed(g1, g2):
    """True when {g1, g2} as lines equals {g1^T, g2^T}."""
    def line(w):
        w = pairs.as_primitive(w)
        return w if (w[0], w[1]) > (-w[0], -w[1]) else (-w[0], -w[1])
    s = {line(g1), line(g2)}
    return s == {line((g1[1], g1[0])), line((g2[1], g2[0]))}


@pytest.mark.parametrize("img_idx", range(len(MASKS)))
def test_generator_lines_satisfy_property3_orthogonal(img_idx):
    img = MASKS[img_idx]
    for d in POOL:
        res = normcheck.orthogonal(img, d.vec)
        r3 = normcheck.property3_residual(res, img)
        assert r3["generator"] == 0, d
        v0, v1 = d.vec
        assert (r3["code"] == 0) or not _transpose_closed((v0, v1), (-v1, v0)), d


@pytest.mark.parametrize("img_idx", range(len(MASKS)))
def test_generator_lines_satisfy_property3_pairs(img_idx):
    img = MASKS[img_idx]
    for r, s in WIDE + [((1, 0), (2, 1)), ((1, 0), (0, 1)), ((1, 1), (1, -1)), ((2, 1), (1, 2))]:
        res = normcheck.pair(img, r, s)
        r3 = normcheck.property3_residual(res, img)
        assert r3["generator"] == 0, (r, s)
        if _transpose_closed(r, s):
            assert r3["code"] == 0, (r, s)


def test_code_lines_fail_exactly_off_the_transpose_closed_set():
    """On a dense random mask the code's family breaks Property 3 for every
    orthogonal pool direction except (1,0) and (1,1)."""
    img = _random_masks(1, seed=3)[0]
    ok = [(d.p, d.q) for d in POOL
          if normcheck.property3_residual(normcheck.orthogonal(img, d.vec), img)["code"] == 0]
    assert ok == [(1, 0), (1, 1)]


def test_code_variant_reproduces_fast_and_pairs():
    img = MASKS[0]
    for d in POOL[::5]:
        f = fast.compute(img, 1, 0, d.vec, "rows")
        r = normcheck.orthogonal(img, d.vec)["code"]
        assert (r["q1"], r["q1_d"], r["q1_a"]) == (f["q1"], f["q1_d"], f["q1_a"])
    for r_, s_ in WIDE[:4]:
        o = pairs.compute(img, 1, 0, r_, s_, "rows")
        r = normcheck.pair(img, r_, s_)["code"]
        assert (r["q1"], r["q1_d"], r["q1_a"]) == (o["q1"], o["q1_d"], o["q1_a"])


def test_code_lines_are_the_published_calcH():
    """The line family under test is the reference's, not a qsig artefact."""
    img = MASKS[1]
    for vec in ((2, -1), (3, -2), (1, -1)):
        cv = Convexity(img)
        a = img == 1
        h, hl = cv.calcH(a, vec)
        v, vl = cv.calcH(np.rot90(a), vec)
        H, W = img.shape
        ref = np.array([[int(a.sum()) + int(h[hl[i, j]]) + int(v[vl[W - 1 - j, i]])
                         for j in range(W)] for i in range(H)])
        assert (normcheck.orthogonal(img, vec)["_norm_code"] == ref).all()


def test_generator_normalisation_bounds_every_rule_by_one():
    for img in MASKS:
        for d in POOL[::3]:
            g = normcheck.orthogonal(img, d.vec)["generator"]
            assert g["phi_hat_max"] <= 1.0 + 1e-12
            assert g["phi_hat_d_max"] <= 1.0 + 1e-12
            assert 0.0 <= g["q1"] <= 1.0 and 0.0 <= g["q1_d"] <= 1.0 and 0.0 <= g["q1_a"] <= 1.0


@pytest.mark.parametrize("vec", [(2, -1), (3, -1), (3, -2), (7, -4)])
def test_code_normalisation_counterexample(vec):
    """The X along the generators: only the centre has four non-empty cones,
    each holding half of each arm. Generator lines give exactly the bound (1);
    the code's lines miss both arms and give 4^4 / 4^2 = 16."""
    img = _x_shape(vec)
    res = normcheck.orthogonal(img, vec)
    assert res["generator"]["q1"] == pytest.approx(1.0)
    assert res["code"]["q1"] == pytest.approx(16.0)


@pytest.mark.parametrize("vec", [(1, 0), (1, -1)])
def test_axis_and_diagonal_unaffected(vec):
    for img in MASKS:
        res = normcheck.orthogonal(img, vec)
        assert (res["_norm_code"] == res["_norm_gen"]).all()


def test_zero_sets_do_not_depend_on_the_normalisation():
    for img in MASKS:
        for d in POOL[::4]:
            res = normcheck.orthogonal(img, d.vec)
            for k in ("q1", "q1_d", "q1_a"):
                assert (res["code"][k] == 0) == (res["generator"][k] == 0)
            assert res["code"]["q0"] == res["generator"]["q0"]
