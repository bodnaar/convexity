"""The rotational signature R_F^D -- handoff sec 3.3, sec 7.4 item 12.

Without this the paper's decisive experiment (test 0: does cost-aware
rotation-free beat the rotational shortcut at equal cost?) cannot be run at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsig import directions as D
from qsig import rotational
from qsig.dataset import BACKGROUND, OBJECT

pytestmark = pytest.mark.skipif(not rotational.HAVE_CV2, reason="opencv not installed")


def _blob(n=40):
    img = np.full((n, n), BACKGROUND, dtype=np.uint8)
    img[8:32, 8:32] = OBJECT
    img[14:20, 8:20] = BACKGROUND          # a notch, so it is not Q-convex
    return img


def test_zero_rotation_is_the_identity():
    img = _blob()
    out = rotational.rotate_binary(img, 0.0)
    assert np.array_equal(out, (img == OBJECT).astype(np.uint8))


def test_rotation_is_binary_and_preserves_the_object_approximately():
    img = _blob()
    for a in (10.0, 45.0, 80.0):
        r = rotational.rotate_binary(img, a, expand=True)
        assert set(np.unique(r)) <= {0, 1}
        ratio = r.sum() / (img == OBJECT).sum()
        assert 0.9 < ratio < 1.1, f"{a} deg changed the object area by {ratio:.3f}"


def test_expand_prevents_clipping_and_no_expand_does_not():
    """The choice IWCIA left unstated. Their flat 2.54 s/component implies they
    clipped; clipping destroys object pixels of a shape that touches the frame."""
    n = 40
    img = np.full((n, n), BACKGROUND, dtype=np.uint8)
    img[:, 18:22] = OBJECT                 # a bar spanning the full height
    kept_expand = rotational.rotate_binary(img, 45.0, expand=True).sum()
    kept_clip = rotational.rotate_binary(img, 45.0, expand=False).sum()
    base = (img == OBJECT).sum()
    assert kept_clip < 0.95 * base, "clipping should lose pixels for this shape"
    assert kept_expand > kept_clip


def test_rotational_value_matches_the_axis_descriptor_at_zero_degrees():
    """R's component at 0 degrees is exactly S's (1,0) component."""
    from qsig.descriptor import q_concavity

    img = _blob()
    v_rot, secs = rotational.rotational_value(img, D.Direction(1, 0), "rows")
    v_axis, _ = q_concavity(img, D.Direction(1, 0), "rows")
    assert v_rot == pytest.approx(v_axis, rel=1e-12)
    assert secs > 0


def test_rotational_values_are_finite_across_the_pool():
    img = _blob()
    for d in D.pool(max_norm2=30):
        v, s = rotational.rotational_value(img, d, "rows")
        assert np.isfinite(v) and 0.0 <= v <= 1.0
        assert s > 0


def test_rotation_loss_is_measured_not_assumed():
    """Rotating by theta and back is not the identity on Z^2. Quantify it, so
    R's limitation can be stated with a number (handoff sec 3.5)."""
    loss = rotational.rotation_loss(_blob(), angles=(15.0, 30.0, 45.0))
    assert set(loss) == {15.0, 30.0, 45.0}
    assert all(v >= 0 for v in loss.values())
    assert max(loss.values()) > 0, "round-tripping should lose something"


def test_a_rotation_that_empties_the_image_raises():
    img = np.full((12, 12), BACKGROUND, dtype=np.uint8)
    img[0, 0] = OBJECT                     # single corner pixel: rotates away
    with pytest.raises(ValueError):
        rotational.rotational_value(img, D.Direction(1, 1), "rows", expand=False)
