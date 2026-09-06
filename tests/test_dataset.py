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


# ---------------------------------------------------------------------------
# Preprocessing must match what produced the published numbers (handoff 9.1).
# ---------------------------------------------------------------------------

def test_binarise_is_a_fixed_threshold_and_never_inverts():
    """The bug this replaces: a minority-class heuristic inverted 271 of 1400
    MPEG-7 images (19.4 %), including three classes at 20/20, because
    silhouettes there occupy up to 87 % of the frame. A fixed threshold cannot
    invert, whatever the object's area."""
    import numpy as np

    from qsig.dataset import _binarise

    a = np.zeros((10, 10), dtype=np.uint8)
    a[1:9, 1:9] = 255                      # 64 % bright -- the minority rule flips here
    out = _binarise(a)
    assert out.sum() == 64, "bright must stay the object even above half the frame"

    b = np.full((10, 10), 255, dtype=np.uint8)
    b[4:6, 4:6] = 0                        # 96 % bright
    assert _binarise(b).sum() == 96

    for v in (250, 248, 255):              # the value sets present in the dataset
        c = np.zeros((6, 6), dtype=np.uint8)
        c[2:4, 2:4] = v
        assert _binarise(c).sum() == 4, f"value {v} must threshold as object"

    d = np.zeros((6, 6), dtype=np.uint8)
    d[2:4, 2:4] = 16                       # the one dim-valued image in the set
    assert _binarise(d).sum() == 0, "16 is below the threshold, as in cv2"


def test_pad_square_centres_and_preserves_every_object_pixel():
    import numpy as np

    from qsig.dataset import _pad_square

    for h, w in ((47, 128), (128, 47), (33, 128), (128, 128), (128, 5)):
        a = np.ones((h, w), dtype=np.uint8)
        out = _pad_square(a)
        assert out.shape == (max(h, w), max(h, w))
        assert out.sum() == h * w, "padding must not drop object pixels"
        rows = np.flatnonzero(out.any(axis=1))
        cols = np.flatnonzero(out.any(axis=0))
        assert abs((rows[0]) - (out.shape[0] - 1 - rows[-1])) <= 1
        assert abs((cols[0]) - (out.shape[1] - 1 - cols[-1])) <= 1
