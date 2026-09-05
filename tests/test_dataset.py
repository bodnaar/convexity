"""Dataset-loader tests. Skipped automatically if MPEG7dataset.zip is absent."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsig import dataset

CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MPEG7dataset.zip"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "MPEG7dataset.zip"),
]
DATA = next((p for p in CANDIDATES if os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(DATA is None, reason="MPEG7dataset.zip not found")


def test_device_subset_is_200_shapes_in_10_classes():
    shapes = dataset.load_mpeg7(DATA, classes=dataset.DEVICE_CLASSES)
    assert len(shapes) == 200
    assert len({s.cls for s in shapes}) == 10
    assert all(sum(1 for t in shapes if t.cls == c) == 20 for c in dataset.DEVICE_CLASSES)


def test_full_dataset_is_1400_shapes_in_70_classes():
    shapes = dataset.load_mpeg7(DATA)
    assert len(shapes) == 1400
    assert len({s.cls for s in shapes}) == 70


def test_long_side_is_128_and_aspect_is_preserved():
    shapes = dataset.load_mpeg7(DATA, classes=("device0",))
    for s in shapes:
        assert max(s.img.shape) == 128
        assert min(s.img.shape) >= 1


def test_images_are_binary_and_non_degenerate():
    shapes = dataset.load_mpeg7(DATA, classes=("device0", "device5"))
    for s in shapes:
        assert set(np.unique(s.img)) <= {dataset.BACKGROUND, dataset.OBJECT}
        assert 0 < s.n_object < s.img.size


def test_object_is_the_minority_class():
    """Sanity on the binarisation: MPEG-7 silhouettes do not fill the frame."""
    shapes = dataset.load_mpeg7(DATA, classes=("device0",))
    assert all(s.n_object / s.img.size < 0.5 for s in shapes)


def test_order_is_stable_across_loads():
    a = [s.shape_id for s in dataset.load_mpeg7(DATA, classes=("device3",))]
    b = [s.shape_id for s in dataset.load_mpeg7(DATA, classes=("device3",))]
    assert a == b
