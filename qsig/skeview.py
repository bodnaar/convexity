"""Loader for the skeview shape catalogue's ground-truth `.mat` masks.

Source: https://github.com/cong-yang/skeview -- "Kimia216 ... selected from
the MPEG7 dataset" is what disqualified Kimia (second_benchmark_2026-09-13.md).
Animal2000 and SwedishLeaves come from the same catalogue and were checked
separately: no shared images with MPEG-7 found (spot-checked on the two
classes -- bird, butterfly -- whose NAME also appears in MPEG-7; see
second_benchmark_results_2026-09-13.md sec 2).

Mask format, reverse-engineered and verified empirically (not documented
upstream): each `<stem>.mat` holds a MATLAB cell array under the key
`mysaving_mat`, shape (1, 3). Element 0 is the ground-truth mask, elements 1-2
are skeleton/branch data this loader ignores.

CONVENTION TRAP: the mask array uses **0 = object, 1 = background** -- the
OPPOSITE of a naive `arr > 0` read, and the opposite of `qsig.dataset`'s own
MPEG-7 loader. Verified two ways: every mask's border pixels are 1 in every
file checked (an object cannot touch all four edges of a padded frame), and a
visual render of `Animal2000-GT/mat/bird1.mat` confirms the silhouette is
where the array is 0. Getting this backwards is SILENT: convex-hull area
ratio comes out as 1.0 for every shape, because the "object" becomes the
whole frame minus one hole, which is already convex.

Both datasets reuse `qsig.dataset`'s preprocessing (`_rescale` + `_pad_square`,
long side 128 px, nearest, downscale only) so a `Shape` produced here is
interchangeable with one from `load_mpeg7` -- `qsig.classify.build_signatures`
and every downstream tool take a `list[Shape]` and don't care where it came
from.
"""

from __future__ import annotations

import io
import os
import re
import zipfile

import numpy as np
import scipy.io as sio

from .dataset import OBJECT, Shape, _pad_square, _rescale

LONG_SIDE = 128


def _class_animal2000(stem: str) -> str:
    m = re.match(r"^([a-zA-Z]+)", stem)
    if not m:
        raise ValueError(f"cannot extract class from Animal2000 stem {stem!r}")
    return m.group(1)


def _class_swedishleaves(stem: str) -> str:
    return stem.split("_", 1)[0]


CLASS_EXTRACTORS = {
    "animal2000": _class_animal2000,
    "swedishleaves": _class_swedishleaves,
}

# Expected (n_shapes, n_classes, per_class), for a loud failure if the archive
# doesn't match what second_benchmark_2026-09-13.md recorded.
EXPECTED_SHAPE = {
    "animal2000": (2000, 20, 100),
    "swedishleaves": (1125, 15, 75),
}


def _load_mask(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    data = sio.loadmat(io.BytesIO(zf.read(member)))
    raw = np.asarray(data["mysaving_mat"][0, 0])
    return (raw == 0).astype(np.uint8) * OBJECT  # see module docstring


def load_skeview(
    path: str,
    dataset: str,
    long_side: int = LONG_SIDE,
    classes: "tuple[str, ...] | None" = None,
) -> list[Shape]:
    """Load one skeview `<Dataset>-GT.zip` archive as `qsig.dataset.Shape`s.

    `dataset` selects the class-name convention: "animal2000" (alphabetic
    prefix, e.g. "bird1" -> "bird") or "swedishleaves" (prefix before the
    first "_", e.g. "01_001bw" -> "01"). Shapes are returned sorted by
    (class, shape_id) for a stable order, matching `load_mpeg7`.
    """
    if dataset not in CLASS_EXTRACTORS:
        raise ValueError(f"dataset must be one of {sorted(CLASS_EXTRACTORS)}, got {dataset!r}")
    class_of = CLASS_EXTRACTORS[dataset]

    zf = zipfile.ZipFile(path)
    members = sorted(n for n in zf.namelist() if n.endswith(".mat"))

    shapes: list[Shape] = []
    for member in members:
        stem = os.path.splitext(os.path.basename(member))[0]
        cls = class_of(stem)
        if classes is not None and cls not in classes:
            continue
        mask = _load_mask(zf, member)
        if mask.sum() == 0:
            raise ValueError(f"{stem}: empty mask")
        img = _pad_square(_rescale(mask, long_side))
        shapes.append(Shape(shape_id=stem, cls=cls, img=img))
    shapes.sort(key=lambda s: (s.cls, s.shape_id))

    if classes is None and dataset in EXPECTED_SHAPE:
        n, k, per = EXPECTED_SHAPE[dataset]
        classes_seen = {s.cls for s in shapes}
        if len(shapes) != n or len(classes_seen) != k:
            raise ValueError(
                f"{dataset}: expected {n} shapes in {k} classes, "
                f"got {len(shapes)} shapes in {len(classes_seen)} classes -- "
                "archive layout may have changed"
            )
    return shapes
