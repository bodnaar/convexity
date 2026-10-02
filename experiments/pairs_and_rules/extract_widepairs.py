"""phi, phi_D, phi_A over WIDE-aperture non-orthogonal pairs.

Separates aperture from non-orthogonality: the Farey set runs 11-27 deg aperture against orthogonal 90 deg, so the
degeneracy of phi there could be narrowness rather than non-orthogonality.
Farey pairs cap at 45 deg, so widening REQUIRES |det| > 1 and a costlier
kernel. Pair set in wide_pairs.json (aperture 53-79 deg, 12 pairs, max
bisector gap 15 deg).

Emits wp{i}, wpd{i}, wpa{i}. Resumable.
Usage: python3 extract_widepairs.py {mpeg7|animal2000|swedishleaves}
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

import qsig.threadguard  # noqa: F401
import numpy as np
from qsig import pairs as _pairs
from qsig.descriptor import warm_up
from qsig.dataset import OBJECT, BACKGROUND
from extract_generalpairs import iter_shapes

SET = [{"r": tuple(d["r"]), "s": tuple(d["s"])} for d in
       json.load(open(os.path.join(HERE, "wide_pairs.json")))]
K = len(SET)
HEADER = (["shape_id"] + [f"wp{i}" for i in range(K)]
          + [f"wpd{i}" for i in range(K)] + [f"wpa{i}" for i in range(K)])


def main():
    dataset = sys.argv[1]
    out = os.path.join(HERE, f"{dataset}_widepairs.csv")
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
