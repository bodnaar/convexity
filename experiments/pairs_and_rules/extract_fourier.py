"""Fourier descriptors (FD) as an external reference descriptor.

Centroid-distance signature of the largest external contour
(cv2.CHAIN_APPROX_NONE), resampled to N = 128 points equally spaced in arc
length; the descriptor is |F_m| / |F_0|, m = 1..32 (invariant to translation,
rotation, starting point and scale). Zhang & Lu, J. Vis. Commun. Image
Represent. 14(1):39-57, 2003. The paper uses M = 12, fixed in advance; 8, 16
and 32 are robustness checks.

Computed on exactly the preprocessed images of the Q-convexity features
(qsig.dataset.load_mpeg7; extract_features.load_mask + _rescale + _pad_square).
Writes <dataset>_fourier.csv next to this file.
"""
from __future__ import annotations
import csv, os, sys, zipfile
import paths  # noqa: F401  -- puts the repository root on sys.path
import numpy as np
import cv2
from qsig.dataset import load_mpeg7, _rescale, _pad_square
from extract_features import load_mask, class_of

N, MMAX, LONG_SIDE = 128, 32, 128


def fourier_descriptor(bin_img: np.ndarray) -> np.ndarray:
    img8 = (bin_img * 255).astype(np.uint8)
    cs, _ = cv2.findContours(img8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    c = np.vstack([c, c[:1]])
    s = np.r_[0.0, np.cumsum(np.sqrt((np.diff(c, axis=0) ** 2).sum(1)))]
    t = np.linspace(0.0, s[-1], N, endpoint=False)
    x, y = np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])
    m = cv2.moments(img8, binaryImage=True)
    r = np.hypot(x - m["m10"] / m["m00"], y - m["m01"] / m["m00"])
    F = np.abs(np.fft.fft(r))
    return F[1:MMAX + 1] / F[0]


def shapes():
    for sh in load_mpeg7(paths.data("MPEG7dataset.zip")):
        yield "mpeg7", sh.shape_id, sh.cls, sh.img
    for ds, z in (("animal2000", "Animal2000-GT.zip"), ("swedishleaves", "SwedishLeaves-GT.zip")):
        zf = zipfile.ZipFile(paths.data(z))
        for mem in sorted(n for n in zf.namelist() if n.endswith(".mat")):
            stem = os.path.splitext(os.path.basename(mem))[0]
            mask = load_mask(zf, mem)
            if mask.sum() == 0:
                continue
            yield ds, stem, class_of(ds, stem), _pad_square(_rescale(mask, LONG_SIDE))


def main():
    out = {}
    for ds, sid, cls, img in shapes():
        out.setdefault(ds, []).append([sid, cls] + [f"{v:.10g}" for v in fourier_descriptor(img)])
    for ds, rows in out.items():
        with open(os.path.join(paths.HERE, f"{ds}_fourier.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shape_id", "cls"] + [f"fd{m}" for m in range(1, MMAX + 1)])
            w.writerows(rows)
        print(f"{ds}: {len(rows)} shapes")


if __name__ == "__main__":
    main()
