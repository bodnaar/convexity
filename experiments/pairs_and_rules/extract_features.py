"""Second-benchmark feature extraction: classical descriptors + 64-orthogonal
Q-concavity signature, for Animal2000 and SwedishLeaves.

Classical baseline protocol:
  - preprocessing: qsig.dataset._rescale (long side 128, nearest, downscale only)
    + _pad_square, same as load_mpeg7.
  - classical descriptors: convex-hull area ratio, circularity (4*pi*A/P^2),
    log-transformed Hu moments 1-7.
  - Q-concavity signature: directions.by_angle(directions.pool(max_norm2=130)),
    64 directions, family S, impl='rows' (numba).

Usage: python3 extract_features.py <dataset_name> <gt_zip_path> <out_csv>
  dataset_name in {animal2000, swedishleaves}
"""
from __future__ import annotations
import io, os, re, sys, time, zipfile
import numpy as np
import scipy.io as sio
import cv2

import paths  # noqa: F401  -- puts the repository root on sys.path
from qsig import directions as D
from qsig.descriptor import q_concavity, warm_up
from qsig.dataset import OBJECT, BACKGROUND, _rescale, _pad_square

LONG_SIDE = 128
DIRS = D.by_angle(D.pool(max_norm2=130))
assert len(DIRS) == 64, len(DIRS)


def class_of(dataset: str, stem: str) -> str:
    if dataset == "animal2000":
        m = re.match(r"^([a-zA-Z]+)", stem)
        return m.group(1)
    elif dataset == "swedishleaves":
        return stem.split("_")[0]
    else:
        raise ValueError(dataset)


def load_mask(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Skeview GT .mat convention, verified empirically on bird1.mat and
    01_001bw.mat: border pixels are 1 in both, and a visual check of bird1
    shows the silhouette rendered where the raw array is 0. So 0 = object,
    1 = background -- the OPPOSITE of a naive >0 read. Object is returned as 1
    to match qsig.dataset's OBJECT=1 convention."""
    data = zf.read(member)
    d = sio.loadmat(io.BytesIO(data))
    v = d["mysaving_mat"]
    mask = np.asarray(v[0, 0])
    return (mask == 0).astype(np.uint8)


def classical_features(bin_img: np.ndarray) -> np.ndarray:
    img8 = (bin_img * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no contour found")
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        area = float(bin_img.sum())
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    area_ratio = area / hull_area if hull_area > 0 else np.nan
    perim = cv2.arcLength(c, True)
    circularity = (4.0 * np.pi * area / (perim ** 2)) if perim > 0 else np.nan
    m = cv2.moments(c)
    hu = cv2.HuMoments(m).flatten()
    eps = 1e-30
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return np.concatenate([[area_ratio, circularity], hu_log])


def main():
    dataset, gt_zip, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    warm_up("rows")
    zf = zipfile.ZipFile(gt_zip)
    members = [n for n in zf.namelist() if n.endswith(".mat")]
    members.sort()

    header = (
        ["shape_id", "cls", "area_ratio", "circularity"]
        + [f"hu{i+1}" for i in range(7)]
        + [f"sig{i}" for i in range(64)]
    )

    ncols = len(header)
    done = set()
    mode = "w"
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            lines = f.read().splitlines()
        if lines and lines[0] == ",".join(header):
            good_lines = [lines[0]]
            for ln in lines[1:]:
                if ln.count(",") == ncols - 1:
                    good_lines.append(ln)
                else:
                    print(f"dropping malformed/truncated line: {ln[:60]!r}", flush=True)
            if len(good_lines) != len(lines):
                with open(out_csv, "w") as fw:
                    fw.write("\n".join(good_lines) + "\n")
            done = {ln.split(",", 1)[0] for ln in good_lines[1:]}
            mode = "a"

    t_start = time.perf_counter()
    n_total = len(members)
    with open(out_csv, mode) as f:
        if mode == "w":
            f.write(",".join(header) + "\n")
        for i, member in enumerate(members):
            stem = os.path.splitext(os.path.basename(member))[0]
            if stem in done:
                continue
            cls = class_of(dataset, stem)
            mask = load_mask(zf, member)
            if mask.sum() == 0:
                print(f"SKIP {stem}: empty mask", flush=True)
                continue
            img = _pad_square(_rescale(mask, LONG_SIDE))
            cfeat = classical_features(img)
            sig = np.array([q_concavity(img, d, impl="rows")[0] for d in DIRS])
            row = [stem, cls] + list(cfeat) + list(sig)
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_total - i - 1) / rate if rate > 0 else float("inf")
                print(f"{dataset}: {i+1}/{n_total} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
    print(f"{dataset}: FINISHED", flush=True)


if __name__ == "__main__":
    main()
