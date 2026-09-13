"""qsig.classical tests -- synthetic shapes only, no dataset file needed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qsig.classical import FEATURE_NAMES, classical_features
from qsig.dataset import OBJECT


def _disc(n=101, r=40):
    img = np.zeros((n, n), dtype=np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    c = n // 2
    img[(yy - c) ** 2 + (xx - c) ** 2 <= r * r] = OBJECT
    return img


def _square(n=101, side=60):
    img = np.zeros((n, n), dtype=np.uint8)
    lo, hi = (n - side) // 2, (n + side) // 2
    img[lo:hi, lo:hi] = OBJECT
    return img


def _plus(n=101, arm=15, length=40):
    img = np.zeros((n, n), dtype=np.uint8)
    c = n // 2
    img[c - arm:c + arm, c - length:c + length] = OBJECT
    img[c - length:c + length, c - arm:c + arm] = OBJECT
    return img


def test_feature_vector_length_and_names_match():
    f = classical_features(_disc())
    assert len(f) == len(FEATURE_NAMES) == 9


def test_disc_is_convex_and_near_circular():
    f = classical_features(_disc())
    area_ratio, circularity = f[0], f[1]
    assert area_ratio > 0.98, "a discretised disc should be (near-)convex"
    assert circularity > 0.85, "a disc should score close to the circularity ceiling of 1.0"


def test_square_is_convex_but_not_circular():
    f = classical_features(_square())
    area_ratio, circularity = f[0], f[1]
    assert area_ratio > 0.99, "a square is exactly convex"
    assert circularity < 0.85, "a square is markedly less circular than a disc"


def test_plus_shape_is_neither_convex_nor_circular():
    f_plus = classical_features(_plus())
    f_disc = classical_features(_disc())
    assert f_plus[0] < 0.76, "a plus sign should have a low convex-hull area ratio"
    assert f_plus[1] < f_disc[1], "a plus sign should be less circular than a disc"


def test_hu_moments_are_translation_and_scale_stable_after_log_transform():
    """Hu invariants are translation/scale/rotation invariant by construction;
    this checks the wrapper doesn't break that (e.g. via absolute pixel coords
    leaking in)."""
    a = classical_features(_disc(n=101, r=30))
    b = classical_features(_disc(n=201, r=60))  # same shape, 2x scale, bigger canvas
    hu_a, hu_b = a[2:], b[2:]
    assert np.allclose(hu_a, hu_b, atol=0.3), (hu_a, hu_b)
