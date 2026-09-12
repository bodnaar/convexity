#!/usr/bin/env python3
"""Does the descriptor need a 90-degree aperture? -- the aperture sweep.

    python scripts/aperture_sweep.py --data ../dgmm_article/MPEG7dataset.zip \
        --out ../journal_results/aperture_sweep_bis0_bis45_n1400.csv

An orthogonal pair `(r, r_perp)` has one free parameter, its orientation. A
general pair `(r, s)` has two: orientation and APERTURE, the angle between the
directions. This sweeps aperture with orientation held fixed, which is the
experiment that decides whether non-orthogonality is worth anything.

WHY THE BISECTORS ARE 0 AND 45 AND NOTHING ELSE
-----------------------------------------------
Holding the bisector fixed while varying aperture needs a pair symmetric about
that bisector. Reflection about a line maps the lattice to itself only on the
square lattice's symmetry axes, so EXACT bisectors exist only at 0, 45, 90 and
135 degrees:

    45 degrees:  reflecting (a, b) gives (b, a)    -- |det| = |a^2 - b^2|
     0 degrees:  reflecting (a, b) gives (a, -b)   -- |det| = 2|ab|

Any other bisector can only be approximated, which reintroduces exactly the
orientation confound this experiment exists to remove. A first attempt paired
each direction with `pairs.perp(r)`; that moves the bisector by 45 degrees, so
aperture and orientation varied together and the run was worthless.

COST UNITS -- READ THIS BEFORE QUOTING A NUMBER
-----------------------------------------------
`|det(r,s)|` is the cost variable of the POINT kernel. The row-prefix kernel,
which is what the papers actually use, costs
`min(|r_0|+|s_0|, |r_1|+|s_1|)` -- the parallelogram's extent along the cheaper
axis. The two disagree sharply: the pair (7,-6),(6,-5) has `|det| = 1` but row
cost 11. Cost claims about the kernel in use must be in row units.
"""

import argparse
import csv
import os
import sys
from math import atan2, degrees, gcd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

from qsig import dataset, descriptor, pairs  # noqa: E402

TARGET_APERTURES = [5, 10, 15, 20, 25, 30, 37, 45, 53, 60, 68, 75, 82, 90]


def line_angle(v) -> float:
    """A direction's angle as a LINE, in [0, 180). r and -r are the same line."""
    return degrees(atan2(-v[1], v[0])) % 180.0


def geometry(r, s) -> tuple[float, float]:
    """(bisector, aperture) of the narrow cone, both in degrees."""
    tr, ts = line_angle(r), line_angle(s)
    d = abs(tr - ts)
    aperture = min(d, 180.0 - d)
    bisector = (tr + ts) / 2 if d <= 90 else ((tr + ts) / 2 + 90) % 180
    return bisector, aperture


def row_cost(r, s) -> int:
    """Row-prefix kernel cost variable, taking the cheaper axis."""
    return min(abs(r[0]) + abs(s[0]), abs(r[1]) + abs(s[1]))


def exact_bisector_family(bisector: int, limit: int = 12):
    """All pairs symmetric about 0 or 45 degrees, as (aperture, det, r, s)."""
    if bisector not in (0, 45):
        raise ValueError("only 0 and 45 admit an exact lattice bisector")
    out = []
    for a in range(1, limit + 1):
        for b in range(-limit, 1):
            if gcd(abs(a), abs(b)) != 1:
                continue
            r = (a, b)
            s = (b, a) if bisector == 45 else (a, -b)
            if pairs.det(r, s) == 0:
                continue
            bis, ap = geometry(r, s)
            if abs(bis - bisector) > 1e-9:
                continue
            out.append((ap, pairs.det(r, s), r, s))
    return sorted(out)


def spread(family, targets=TARGET_APERTURES):
    """One pair per target aperture, cheapest on ties, no duplicates."""
    chosen, seen = [], set()
    for t in targets:
        ap, d, r, s = min(family, key=lambda x: (abs(x[0] - t), x[1]))
        if r in seen:
            continue
        seen.add(r)
        chosen.append((ap, d, r, s))
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=("device", "all"), default="all")
    ap.add_argument("--method", choices=("points", "rows"), default="rows")
    args = ap.parse_args()

    classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
    shapes = dataset.load_mpeg7(args.data, classes=classes)
    descriptor.warm_up("fast")
    print(f"{len(shapes)} shapes, {len({s.cls for s in shapes})} classes")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["bisector"]), row["r"], row["s"]))

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["bisector", "aperture", "det", "row_cost", "r", "s",
                         "shape_id", "cls", "E", "card_f_dash"])
        for bis in (45, 0):
            for aperture, det, r, s in spread(exact_bisector_family(bis)):
                if (bis, str(r), str(s)) in done:
                    continue
                for sh in shapes:
                    out = pairs.compute(sh.img, dataset.OBJECT, dataset.BACKGROUND,
                                        r, s, args.method)
                    wr.writerow([bis, f"{aperture:.4f}", det, row_cost(r, s),
                                 str(r), str(s), sh.shape_id, sh.cls,
                                 repr(out["q1"]), out["_card_f_dash"]])
                fh.flush()
                print(f"  bisector {bis:2d}  aperture {aperture:6.2f}  "
                      f"|det| {det:4d}  row cost {row_cost(r, s):3d}  {r} {s}")


if __name__ == "__main__":
    main()
