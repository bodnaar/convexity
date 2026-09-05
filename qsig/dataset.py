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
    """Return a {0,1} uint8 array with the *shape* as 1.

    MPEG-7 gifs are a white silhouette on black. Rather than trust that, take
    the minority intensity class as the object -- silhouettes occupy well under
    half the frame in this dataset -- and fall back to 'bright is object' if the
    split is near even.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    thr = (int(a.max()) + int(a.min())) / 2.0
    bright = a > thr
    frac = bright.mean()
    obj = bright if frac <= 0.5 else ~bright
    return obj.astype(np.uint8)


def _rescale(bin_img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = bin_img.shape
    if max(h, w) == long_side:
        return bin_img
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    im = Image.fromarray((bin_img * 255).astype(np.uint8), mode="L")
    im = im.resize((nw, nh), resample=Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8)


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
            img = _rescale(_binarise(arr), long_side)
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
        "binarise": "minority-intensity-class as object",
        "classifier": "1NN",
        "validation": "leave-one-out",
        "distance": "d_C (orbit: cyclic shift + reversal, mean-centred, min L2)",
    }
