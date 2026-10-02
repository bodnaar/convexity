"""phi, phi_D and phi_A over a set of GENERAL (non-orthogonal) direction pairs.

Exercises the general-pair kernel (qsig/pairs.py) with all three combination
rules; the 64-direction extractions measure them on orthogonal pairs only.

Pair set: general_pairs.spread(12) -- Farey neighbours (|det| = 1), aperture in
[10, 85] deg, spread over bisector orientation. Emits, per shape,
  gp{i}, gpd{i}, gpa{i}   for i = 0..11   (phi, phi_D, phi_A)
Resumable. Usage: python3 extract_generalpairs.py {mpeg7|animal2000|swedishleaves}
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

import qsig.threadguard  # noqa: F401  -- before numpy
import numpy as np
from qsig import pairs as _pairs
from qsig.descriptor import warm_up
from qsig.dataset import OBJECT, BACKGROUND
import general_pairs as GP

SET = GP.spread(12)
K = len(SET)
HEADER = (["shape_id"]
          + [f"gp{i}" for i in range(K)]
          + [f"gpd{i}" for i in range(K)]
          + [f"gpa{i}" for i in range(K)])


def _find(fname):
    return paths.data(fname)


def iter_shapes(dataset):
    if dataset == "mpeg7":
        from qsig.dataset import load_mpeg7
        for sh in load_mpeg7(_find("MPEG7dataset.zip")):
            yield sh.shape_id, sh.img
    else:
        import zipfile
        from qsig.dataset import _rescale, _pad_square
        from extract_features import load_mask
        zp = {"animal2000": "Animal2000-GT.zip",
              "swedishleaves": "SwedishLeaves-GT.zip"}[dataset]
        zf = zipfile.ZipFile(_find(zp))
        for m in sorted(n for n in zf.namelist() if n.endswith(".mat")):
            stem = os.path.splitext(os.path.basename(m))[0]
            mask = load_mask(zf, m)
            if mask.sum() == 0:
                print(f"SKIP {stem}: empty mask", flush=True)
                continue
            yield stem, _pad_square(_rescale(mask, 128))


def main():
    dataset = sys.argv[1]
    out = os.path.join(HERE, f"{dataset}_generalpairs.csv")
    done, mode = set(), "w"
    if os.path.exists(out):
        lines = open(out).read().splitlines()
        if lines and lines[0] == ",".join(HEADER):
            good = [lines[0]] + [l for l in lines[1:] if l.count(",") == len(HEADER) - 1]
            if len(good) != len(lines):
                open(out, "w").write("\n".join(good) + "\n")
            done = {l.split(",", 1)[0] for l in good[1:]}
            mode = "a"
    warm_up("rows")
    t0, i = time.perf_counter(), 0
    with open(out, mode) as f:
        if mode == "w":
            f.write(",".join(HEADER) + "\n")
        for sid, img in iter_shapes(dataset):
            i += 1
            if sid in done:
                continue
            a, b, c = [], [], []
            for d in SET:
                o = _pairs.compute(img, OBJECT, BACKGROUND, d["r"], d["s"], "rows")
                a.append(o["q1"]); b.append(o["q1_d"]); c.append(o["q1_a"])
            f.write(",".join([sid] + [repr(v) for v in a + b + c]) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{dataset}: {i} seen, {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"{dataset}: FINISHED ({i} seen, {time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
