"""qsig.skeview tests. The convention test needs no data file; the archive
tests are skipped automatically if the relevant skeview `-GT.zip` is absent."""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import scipy.io as sio

from qsig import skeview
from qsig.dataset import OBJECT

CANDIDATE_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"),
]


def _find(name):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


ANIMAL2000 = _find("Animal2000-GT.zip")
SWEDISHLEAVES = _find("SwedishLeaves-GT.zip")


def _fake_mat_bytes(mask: np.ndarray) -> bytes:
    """A minimal stand-in for a skeview GT .mat: a (1,3) object cell array
    whose element 0 is the mask, matching the real files' structure closely
    enough for `_load_mask` (elements 1-2 are never read)."""
    cell = np.empty((1, 3), dtype=object)
    cell[0, 0] = mask
    cell[0, 1] = np.zeros((1, 2), dtype=np.uint16)
    cell[0, 2] = np.zeros((0, 3), dtype=object)
    buf = io.BytesIO()
    sio.savemat(buf, {"mysaving_mat": cell})
    return buf.getvalue()


def test_mask_convention_is_0_is_object_1_is_background():
    """Regression test for the bug found 2026-09-13: a naive `> 0` read makes
    the WHOLE FRAME the object (since border pixels are 1), which sends
    convex-hull area ratio to 1.0 for every shape, silently."""
    raw = np.ones((20, 20), dtype=np.uint8)
    raw[5:15, 5:15] = 0  # a 10x10 "object" hole in an all-1 "background"

    class _FakeZip:
        def read(self, member):
            return _fake_mat_bytes(raw)

    mask = skeview._load_mask(_FakeZip(), "whatever.mat")
    assert mask.sum() == 100, "the 10x10 zero-region must become the object"
    assert set(np.unique(mask)) <= {0, OBJECT}
    # border of the raw array was all-1 (background); confirm it reads as background
    assert mask[0, 0] == 0


def test_class_extractors():
    assert skeview._class_animal2000("bird44") == "bird"
    assert skeview._class_animal2000("flyingbird12") == "flyingbird"
    assert skeview._class_swedishleaves("01_001bw") == "01"
    assert skeview._class_swedishleaves("15_075bw") == "15"


@pytest.mark.skipif(ANIMAL2000 is None, reason="Animal2000-GT.zip not found")
def test_animal2000_shape_and_class_counts():
    shapes = skeview.load_skeview(ANIMAL2000, "animal2000")
    assert len(shapes) == 2000
    classes = {s.cls for s in shapes}
    assert len(classes) == 20
    assert all(sum(1 for s in shapes if s.cls == c) == 100 for c in classes)


@pytest.mark.skipif(SWEDISHLEAVES is None, reason="SwedishLeaves-GT.zip not found")
def test_swedishleaves_shape_and_class_counts():
    shapes = skeview.load_skeview(SWEDISHLEAVES, "swedishleaves")
    assert len(shapes) == 1125
    classes = {s.cls for s in shapes}
    assert len(classes) == 15
    assert all(sum(1 for s in shapes if s.cls == c) == 75 for c in classes)


@pytest.mark.skipif(ANIMAL2000 is None, reason="Animal2000-GT.zip not found")
def test_animal2000_object_is_the_minority_class():
    """Sanity in the same spirit as test_dataset.py's MPEG-7 check: real
    silhouettes should not fill most of the frame after the convention fix."""
    shapes = skeview.load_skeview(ANIMAL2000, "animal2000", classes=("bird",))
    fractions = [s.n_object / s.img.size for s in shapes]
    assert sum(f < 0.5 for f in fractions) / len(fractions) > 0.8
