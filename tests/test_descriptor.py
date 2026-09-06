"""Tests that touch the reference implementation `convexity.Convexity`.

These are the guardrails for the eventual Numba/int64 rewrite (handoff sec 7.4
item 3): whatever replaces the inner loop must reproduce these values exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsig import directions as D
from qsig.dataset import BACKGROUND, OBJECT
from qsig.descriptor import q_concavity


def _square(n=16, pad=4):
    img = np.full((n + 2 * pad, n + 2 * pad), BACKGROUND, dtype=np.uint8)
    img[pad:pad + n, pad:pad + n] = OBJECT
    return img


def _notched(n=16, pad=4):
    img = _square(n, pad)
    img[pad + n // 3: pad + 2 * n // 3, pad: pad + n // 2] = BACKGROUND
    return img


def test_every_pool_direction_is_computable():
    """The repo's rotsat raises for (p, +q) and for (0, +-1); Direction.vec is
    the only spelling that works. If this test fails the pool enumerator has
    drifted from the convention."""
    img = _notched(12, 3)
    for d in D.pool(max_norm2=26):
        value, secs = q_concavity(img, d)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
        assert secs > 0


def test_convex_shape_has_zero_q_concavity():
    """phi_F == 0 iff F is Q-convex (IWCIA 2025 sec 2). A filled axis-aligned
    square is Q-convex with respect to every direction pair."""
    img = _square()
    for d in [D.Direction(1, 0), D.Direction(1, 1), D.Direction(3, 1)]:
        value, _ = q_concavity(img, d)
        assert value == pytest.approx(0.0, abs=1e-12)


def test_notched_shape_is_not_q_convex_horizontally():
    value, _ = q_concavity(_notched(), D.Direction(1, 0))
    assert value > 0.0


def test_reflection_symmetry_of_cost_and_value():
    """(p,q) and (q,p) are mirror images about 45 degrees, so a shape that is
    symmetric about its own 45-degree axis must give the same value for both --
    and the two always cost the same, since |r|^2 is symmetric."""
    assert D.Direction(3, 1).norm2 == D.Direction(1, 3).norm2
    n = 20
    img = np.full((n, n), BACKGROUND, dtype=np.uint8)
    idx = np.arange(n)
    img[np.triu_indices(n, 0)] = OBJECT           # symmetric about the diagonal
    img[idx[5:9][:, None], idx[11:15][None, :]] = BACKGROUND
    img = np.maximum(img, img.T)
    v1, _ = q_concavity(img, D.Direction(3, 1))
    v2, _ = q_concavity(img, D.Direction(1, 3))
    assert v1 == pytest.approx(v2, rel=1e-9)


def test_cost_grows_with_norm2_not_with_the_component_sum():
    """A weak, fast version of the cost-law claim, run on a small image so it
    stays a unit test. The full regression is scripts/fit_cost_law.py."""
    img = _notched(24, 6)
    timings = {}
    for d in [D.Direction(1, 0), D.Direction(1, 1), D.Direction(3, 1), D.Direction(5, 1)]:
        _, secs = q_concavity(img, d)
        timings[d.norm2] = secs
    ordered = [timings[k] for k in sorted(timings)]
    assert ordered == sorted(ordered), f"cost not monotone in |r|^2: {timings}"


def test_values_are_reproducible():
    img = _notched()
    a, _ = q_concavity(img, D.Direction(3, 1))
    b, _ = q_concavity(img, D.Direction(3, 1))
    assert a == b


def test_E_is_invariant_to_background_padding():
    """The descriptor depends on the OBJECT, not on the canvas it sits in.

    Measured 2026-09-06: max relative change 3.2e-16 over 36 (shape, direction)
    pairs under asymmetric padding. This is what rules out the canvas SIZE as an
    explanation for anything: `phi` is zero wherever a point has an empty
    quadrant, so padded rows and columns contribute nothing to either the sum or
    to card(F'), and `norm_part` is unchanged. The only thing that separates
    `expand=True` from `expand=False` in qsig.rotational is therefore the object
    pixels that clipping DESTROYS -- not the array shape.

    An earlier revision of the handoff conjectured an angle-dependent
    normalisation denominator under canvas expansion. This test is why that
    conjecture was withdrawn.
    """
    import numpy as np

    from qsig import directions as D
    from qsig.descriptor import q_concavity

    rng = np.random.default_rng(0)
    img = np.zeros((48, 48), dtype=np.uint8)
    img[10:38, 10:38] = 1
    img[16:24, 10:26] = 0                  # a notch, so it is not Q-convex
    img[30:34, 28:36] = 0
    worst = 0.0
    for _ in range(3):
        pad = rng.integers(1, 20, size=4)
        big = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])))
        for d in list(D.pool(max_norm2=26))[:6]:
            a, _ = q_concavity(img, d, "rows")
            b, _ = q_concavity(big, d, "rows")
            worst = max(worst, abs(a - b) / max(a, 1e-12))
    assert worst < 1e-12, f"padding changed E by {worst:.2e}"
