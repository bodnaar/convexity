#!/usr/bin/env python3
"""Timing calibration against IWCIA 2025 Table 1 -- handoff sec 1.3, sec 7.4 item 5.

The question this answers has changed since the first version, and the change
matters. Reproducing two of their published seconds is NOT the goal, because a
per-direction time in an interpreted implementation is a property of the
software stack as much as of the algorithm. The goal is:

    Does T(r) = A + B*|r|^2 still hold on THIS machine, and what are A and B?

The functional form is the paper's proposition (handoff sec 3.1) and it is
implementation-independent. The constants are not, and are not claimed to be.
So this script fits the law over several directions rather than checking two
seconds, and reports the published values alongside as context.

Protocol note, easy to get wrong: IWCIA Table 1 is the mean over the ENTIRE
1400-shape dataset, whose mean area is 13022 px (0.795 of a 128x128 square,
since only some classes are square). The Device shapes are all exactly 128x128,
i.e. the LARGEST in the set, so timing on Device and comparing to their table
overstates this machine's cost by about 26%. `--subset all` is therefore the
default.

Run it SINGLE-PROCESS -- this is a measurement run, not a bulk run -- under the
`performance` governor, pinned, and on a quiet machine:

    sudo cpupower frequency-set -g performance
    uptime                                    # confirm nobody else is on
    numactl --cpunodebind=0 --membind=0 \
        python scripts/calibrate.py --data ../MPEG7dataset.zip --shapes 8

Contention does not merely add noise: a longer job is descheduled more often, so
a loaded machine inflates the expensive directions MORE than the cheap ones and
flatters nothing -- it exaggerates the |r|^2 slope. Medians are reported
alongside means because one stray scheduling event moves a mean and not a median.
"""

import argparse
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401  -- MUST precede numpy

import numpy as np  # noqa: E402

from qsig import dataset, directions, store  # noqa: E402
from qsig.descriptor import q_concavity  # noqa: E402

# A spread of |r|^2 from 1 to 109, so the law can actually be fitted.
DEFAULT_DIRS = "1x0,2x1,3x1,5x1,1x7,10x3"


def turbo_state():
    """Best-effort read of the turbo/boost setting, to record with the result.

    Only one of these files exists: `no_turbo` under the intel_pstate driver,
    `boost` under acpi-cpufreq. An empty read of the other one is normal.
    """
    for path, on_value in (
        ("/sys/devices/system/cpu/intel_pstate/no_turbo", "0"),
        ("/sys/devices/system/cpu/cpufreq/boost", "1"),
    ):
        try:
            with open(path) as fh:
                v = fh.read().strip()
            return f"{os.path.basename(path)}={v} (turbo {'on' if v == on_value else 'off'})"
        except OSError:
            continue
    return "unknown"


def load_average():
    try:
        one, five, fifteen = os.getloadavg()
        return f"{one:.2f} {five:.2f} {fifteen:.2f}"
    except OSError:
        return "unknown"


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return a, b, r2, float(np.abs((pred - y) / y).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shapes", type=int, default=8, help="shapes to average over")
    ap.add_argument("--subset", choices=("all", "device"), default="all",
                    help="'all' matches IWCIA Table 1's protocol; 'device' is 128x128 only")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--dirs", default=DEFAULT_DIRS, help="comma list of pxq")
    ap.add_argument("--out", default="", help="optional CSV to append the raw timings to")
    args = ap.parse_args()

    dirs = []
    for token in args.dirs.split(","):
        p, q = token.lower().split("x")
        dirs.append(directions.Direction(int(p), int(q)))
    dirs.sort(key=lambda d: d.norm2)

    classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
    shapes = dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)
    # spread the sample across classes rather than taking the first N of one class
    step = max(1, len(shapes) // args.shapes)
    shapes = shapes[::step][: args.shapes]
    areas = np.array([s.img.size for s in shapes], float)

    gov = store.cpu_governor()
    print(f"host       {platform.node()}")
    print(f"python     {platform.python_version()}   numpy {np.__version__}")
    print(f"platform   {platform.platform()}")
    print(f"governor   {gov}")
    print(f"turbo      {turbo_state()}")
    print(f"loadavg    {load_average()}   (single-process run: ~1.0 means the machine is yours)")
    print(f"threads    {qsig.threadguard.report()}")
    print(f"protocol   {args.long_side} px long side, subset={args.subset}, "
          f"{len(shapes)} shapes, mean area {areas.mean():.0f} px")
    print()
    if gov != "performance":
        print("WARNING: governor is not 'performance'; timings will not be reproducible.\n")

    st = store.ResultStore(args.out) if args.out else None
    print(f"{'dir':>8s} {'|r|^2':>6s} {'mean':>8s} {'median':>8s} {'min':>8s} {'max':>8s} "
          f"{'IWCIA T1':>9s} {'ratio':>7s}")
    print("-" * 68)
    med = []
    for d in dirs:
        secs, rows = [], []
        for sh in shapes:
            value, dt = q_concavity(sh.img, d)
            secs.append(dt)
            rows.append({"shape_id": sh.shape_id, "cls": sh.cls, "p": d.p, "q": d.q,
                         "angle_deg": round(d.angle, 6), "norm2": d.norm2,
                         "resolution": args.long_side, "E": value, "seconds": dt})
        if st:
            st.append(rows)
        a = np.array(secs)
        med.append(float(np.median(a)))
        pub = directions.IWCIA_TABLE1.get((d.p, d.q))
        print(f"{f'({d.p},{d.q})':>8s} {d.norm2:6d} {a.mean():8.3f} {np.median(a):8.3f} "
              f"{a.min():8.3f} {a.max():8.3f} "
              f"{(f'{pub:.2f}' if pub else '--'):>9s} "
              f"{(f'{np.median(a) / pub:.3f}' if pub else '--'):>7s}")

    if len(dirs) < 3:
        print("\nGive at least 3 directions (--dirs) to fit the cost law; two points fit any line.")
        return

    x1 = [d.norm2 for d in dirs]
    x2 = [(d.p + d.q) ** 2 for d in dirs]
    a1, b1, r1, m1 = fit(x1, med)
    a2, b2, r2_, m2 = fit(x2, med)
    print(f"\ncost-law fit on THIS machine ({len(dirs)} directions, medians):")
    print(f"  |r|^2 = |det(r,r_perp)|      T = {a1:7.4f} + {b1:.5f}*x   R2={r1:.5f}  MAPE={m1:5.2f}%")
    print(f"  (r1+r2)^2  [published bound] T = {a2:7.4f} + {b2:.5f}*x   R2={r2_:.5f}  MAPE={m2:5.2f}%")
    print(f"  published (IWCIA Table 1)    T = {directions.COST_IWCIA[0]:7.4f} + "
          f"{directions.COST_IWCIA[1]:.5f}*|r|^2")

    cheap, dear = med[0], med[-1]
    spread = dear / cheap
    pub_spread = 65.21 / 2.54
    print(f"\ncost spread, dearest / cheapest direction:")
    print(f"  this machine  {spread:6.2f}x   ({dirs[-1].p},{dirs[-1].q}) vs ({dirs[0].p},{dirs[0].q})")
    print(f"  IWCIA Table 1 {pub_spread:6.2f}x   (10,3) vs (1,0)")
    print(f"  cost-aware selection is worth roughly the spread, so this is the number")
    print(f"  the paper's saving scales with -- not the published one.")

    print()
    if r1 > 0.98 and r1 > r2_:
        print("VERDICT: THE COST LAW HOLDS on this machine.")
        print(f"         T = {a1:.4f} + {b1:.5f}*|r|^2, R2 = {r1:.4f}, and it beats the")
        print(f"         published (r1+r2)^2 bound (R2 = {r2_:.4f}). The proposition is")
        print("         confirmed independently of the published constants.")
        print()
        print("         The CONSTANTS differ from IWCIA Table 1 and are not expected to")
        print("         match: in an interpreted implementation they are dominated by")
        print("         interpreter and numpy-scalar overheads, which have changed a lot")
        print("         since Ubuntu 16.04. Report this machine's own measurements, state")
        print("         the software stack printed above, and do NOT quote their seconds")
        print("         alongside your accuracies (handoff sec 3.1.1).")
    elif r1 > 0.9:
        print("VERDICT: the law roughly holds but the fit is loose (R2 = "
              f"{r1:.3f}). Most likely another process is on the machine -- check the")
        print("         loadavg above and re-run when it is near 1.0. Contention inflates")
        print("         long directions more than short ones.")
    else:
        print(f"VERDICT: the |r|^2 law does NOT fit here (R2 = {r1:.3f}). This is the one")
        print("         outcome that threatens the paper's proposition. Before concluding")
        print("         anything, re-run on a quiet machine; then check the mask geometry")
        print("         with tests/test_fast_matches_reference.py::"
              "test_mask_offset_count_equals_pick_bound.")


if __name__ == "__main__":
    main()
