"""phi_D = n0 n3 + n1 n2 signature (disjunctive combination over the two
OPPOSING quadrant pairs), same preprocessing and same 64-direction pool as
extract_features.py / extract_mpeg7_full.py.

Emits shape_id + sigd0..sigd63, to be joined to the existing *_features.csv
by shape_id. Resumable: re-running appends only the missing shape_ids.

Usage: python3 extract_disjunctive.py {mpeg7|animal2000|swedishleaves}
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

from qsig import directions as D
from qsig import fast as _fast
from qsig.descriptor import warm_up
from qsig.dataset import OBJECT, BACKGROUND

DIRS = D.by_angle(D.pool(max_norm2=130))
assert len(DIRS) == 64, len(DIRS)
HEADER = ["shape_id"] + [f"sigd{i}" for i in range(64)]


def iter_shapes(dataset: str):
    """Yield (shape_id, preprocessed binary image) in the SAME way the
    existing feature files were produced."""
    if dataset == "mpeg7":
        from qsig.dataset import load_mpeg7
        for sh in load_mpeg7(paths.data("MPEG7dataset.zip")):
            yield sh.shape_id, sh.img
    else:
        import zipfile
        from qsig.dataset import _rescale, _pad_square
        from extract_features import load_mask
        zp = {"animal2000": "Animal2000-GT.zip",
              "swedishleaves": "SwedishLeaves-GT.zip"}[dataset]
        zf = zipfile.ZipFile(paths.data(zp))
        members = sorted(n for n in zf.namelist() if n.endswith(".mat"))
        for m in members:
            stem = os.path.splitext(os.path.basename(m))[0]
            mask = load_mask(zf, m)
            if mask.sum() == 0:
                print(f"SKIP {stem}: empty mask", flush=True)
                continue
            yield stem, _pad_square(_rescale(mask, 128))


def main():
    dataset = sys.argv[1]
    out = os.path.join(HERE, f"{dataset}_disjunctive.csv")

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
    t0 = time.perf_counter()
    i = 0
    with open(out, mode) as f:
        if mode == "w":
            f.write(",".join(HEADER) + "\n")
        for sid, img in iter_shapes(dataset):
            i += 1
            if sid in done:
                continue
            vals = [_fast.compute(img, OBJECT, BACKGROUND, d.vec, "rows")["q1_d"]
                    for d in DIRS]
            f.write(",".join([sid] + [repr(v) for v in vals]) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{dataset}: {i} seen, {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"{dataset}: FINISHED ({i} seen, {time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
