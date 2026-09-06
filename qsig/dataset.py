"""MPEG-7 CE-Shape-1 loader, matching the IWCIA 2025 protocol.

Protocol (IWCIA 2025 sec 5.4, reproduced exactly so accuracies are comparable
to their Table 2):

  * 1400 shapes, 70 classes of 20.
  * Each image rescaled so the LONG side is 128 px, aspect ratio preserved.
  * `Device` subset = the 10 'device0'..'device9' classes, 200 shapes.

The distributed archive contains 1402 images: the 1400 shapes plus two stray
files. Classes are taken to be the filename stem before the final '-', and only
classes with exactly `PER_CLASS` members are kept, which drops the strays.

Resampling: nearest-neighbour. The image is binary and any interpolating
filter would produce grey values that then need re-thresholding, which changes
the concavity structure -- the same hazard as the pyramid re-binarisation of
handoff sec 6 test 1. Nearest keeps it a binary operation. This choice is
recorded in the results table metadata; see `qsig.store`.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

PER_CLASS = 20
LONG_SIDE = 128
DEVICE_CLASSES = tuple(f"device{i}" for i in range(10))

OBJECT = 1
BACKGROUND = 0


@dataclass(frozen=True)
class Shape:
    shape_id: str          # e.g. 'apple-1'
    cls: str               # e.g. 'apple'
    img: np.ndarray        # uint8, values in {OBJECT, BACKGROUND}

    @property
    def n_object(self) -> int:
        return int((self.img == OBJECT).sum())


def _binarise(arr: np.ndarray) -> np.ndarray:
    """Return a {0,1} uint8 array with the *shape* as 1: bright is the object.

    This MUST match the preprocessing that produced the published numbers
    (IWCIA 2025), which is `cv2.threshold(img, 128, 255, THRESH_BINARY)` --
    a fixed threshold, bright is object, no adaptation.

    An earlier version of this function took the MINORITY intensity class as the
    object, on the reasoning that silhouettes occupy well under half the frame.
    They do not: measured across all 1402 images, the bright fraction runs from
    0.019 to 0.870 and exceeds 0.5 in **271 of 1400** (19.4 %) -- including
    Misk, Heart and HCircle at 20/20. That heuristic therefore INVERTED a fifth
    of the dataset, and it is the leading explanation for the +3.35 point
    reproduction gap on the full set against only -1.5 on Device, where just
    2 of 200 images invert (handoff sec 9.1).

    The fixed threshold is safe here and was checked, not assumed: every image
    is dark background plus a bright object, with value sets {0,255} (1346),
    {0,250} (48) and four other all-bright-or-black variants.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 128).astype(np.uint8)


def _rescale(bin_img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bin_img.shape
    if max(h, w) == long_side:
        return bin_img
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    im = Image.fromarray((bin_img * 255).astype(np.uint8), mode="L")
    im = im.resize((nw, nh), resample=Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8)


def _pad_square(bin_img: np.ndarray) -> np.ndarray:
    """Centre the shape on a square canvas of side max(h, w).

    Also part of the published preprocessing. The descriptor itself is
    invariant to background padding (tests/test_descriptor.py), so this changes
    nothing for family S -- but family R rotates inside whatever canvas it is
    given, so the canvas decides how much of the object clipping destroys
    (handoff sec 3.3.3). Reproducing family R requires reproducing this.
    """
    h, w = bin_img.shape
    d = max(h, w)
    out = np.zeros((d, d), dtype=np.uint8)
    r0, c0 = int(d / 2 - h / 2), int(d / 2 - w / 2)
    out[r0:r0 + h, c0:c0 + w] = bin_img
    return out


def load_mpeg7(
    path: str,
    long_side: int = LONG_SIDE,
    classes: "tuple[str, ...] | None" = None,
    per_class: int = PER_CLASS,
) -> list[Shape]:
    """Load the dataset from a .zip or an extracted directory.

    `classes` restricts to a subset, e.g. `DEVICE_CLASSES`.
    Shapes are returned sorted by (class, numeric index) for a stable order.
    """
    if os.path.isdir(path):
        entries = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".gif", ".png", ".bmp", ".pgm", ".tif")):
                    entries.append(os.path.join(root, f))
        reader = lambda p: Image.open(p)  # noqa: E731
        names = entries
    elif zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith((".gif", ".png", ".bmp", ".pgm", ".tif"))]
        reader = lambda p: Image.open(io.BytesIO(zf.read(p)))  # noqa: E731
    else:
        raise FileNotFoundError(f"not a zip or directory: {path}")

    by_class: dict[str, list[tuple[str, str]]] = {}
    for n in names:
        stem = os.path.splitext(os.path.basename(n))[0]
        if "-" not in stem:
            continue
        cls = stem.rsplit("-", 1)[0]
        by_class.setdefault(cls, []).append((stem, n))

    keep = {c: v for c, v in by_class.items() if len(v) == per_class}
    if classes is not None:
        missing = set(classes) - set(keep)
        if missing:
            raise ValueError(f"requested classes not present with {per_class} members: {sorted(missing)}")
        keep = {c: keep[c] for c in classes}

    def idx(stem: str) -> int:
        tail = stem.rsplit("-", 1)[1]
        return int(tail) if tail.isdigit() else 0

    shapes: list[Shape] = []
    for cls in sorted(keep):
        for stem, member in sorted(keep[cls], key=lambda t: idx(t[0])):
            arr = np.asarray(reader(member).convert("L"))
            img = _pad_square(_rescale(_binarise(arr), long_side))
            if img.sum() == 0 or img.sum() == img.size:
                raise ValueError(f"{stem}: degenerate after binarisation/rescale")
            shapes.append(Shape(shape_id=stem, cls=cls, img=img))
    return shapes


def protocol_metadata(long_side: int = LONG_SIDE) -> dict:
    """Recorded alongside results so a cache can be attributed to a protocol
    (handoff sec 7.4 item 10)."""
    return {
        "dataset": "MPEG-7 CE-Shape-1",
        "long_side": long_side,
        "resample": "nearest",
        "binarise": "fixed threshold 128, bright is object (matches IWCIA 2025)",
        "canvas": "centred on a square of side max(h,w)",
        "classifier": "1NN",
        "validation": "leave-one-out",
        "distance": "d_C (orbit: cyclic shift + reversal, mean-centred, min L2)",
    }
