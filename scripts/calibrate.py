#!/usr/bin/env python3
"""Timing calibration against IWCIA 2025 Table 1 -- handoff sec 1.3, sec 7.4 item 5.

The execution server uses the SAME CPU part as the IWCIA 2025 experiment
(Xeon E5-2670 v2). If two of their published per-direction timings reproduce
here, the paper can state that its timings are measured on hardware identical to
[1] and calibrated against its published figures, and every later comparison to
their Table 1 and Table 2 is free. If they diverge, the fallback is to report
all timings as ratios rather than absolutes -- and it is much better to learn
that in week 1 than in week 4.

Two directions are enough and they bracket the range:
    (1,0)   |r|^2 =   1   they report  2.54 s
    (10,3)  |r|^2 = 109   they report 65.21 s

Run it SINGLE-PROCESS (this is a measurement run, not a bulk run), under the
`performance` governor, ideally pinned:

    sudo cpupower frequency-set -g performance
    numactl --cpunodebind=0 --membind=0 \
        python scripts/calibrate.py --data MPEG7dataset.zip --shapes 12

Report medians as well as means (handoff sec 7.2): a single stray scheduling
event moves a mean and not a median.
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

TARGETS = {(1, 0): 2.54, (10, 3): 65.21}


def turbo_state():
    """Best-effort read of the turbo/boost setting, to record with the result."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shapes", type=int, default=12, help="shapes to average over")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--dirs", default="1x0,10x3", help="comma list of pxq, e.g. 1x0,10x3,1x1")
    ap.add_argument("--out", default="", help="optional CSV to append the raw timings to")
    args = ap.parse_args()

    dirs = []
    for token in args.dirs.split(","):
        p, q = token.lower().split("x")
        dirs.append(directions.Direction(int(p), int(q)))

    shapes = dataset.load_mpeg7(args.data, long_side=args.long_side,
                                classes=dataset.DEVICE_CLASSES)[: args.shapes]

    gov = store.cpu_governor()
    print(f"host      {platform.node()}")
    print(f"cpu       {platform.processor() or 'unknown'}")
    print(f"governor  {gov}")
    print(f"turbo     {turbo_state()}")
    print(f"threads   {qsig.threadguard.report()}")
    print(f"protocol  {args.long_side} px long side, {len(shapes)} shapes, single process\n")
    if gov != "performance":
        print("WARNING: governor is not 'performance'. Timings will not be reproducible.")
        print("         sudo cpupower frequency-set -g performance\n")

    st = store.ResultStore(args.out) if args.out else None
    print(f"{'dir':>8s} {'|r|^2':>6s} {'mean':>8s} {'median':>8s} {'min':>8s} {'max':>8s} "
          f"{'published':>10s} {'ratio':>7s}")
    print("-" * 68)
    ratios = []
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
        pub = TARGETS.get((d.p, d.q))
        r = float(np.median(a) / pub) if pub else float("nan")
        if pub:
            ratios.append(r)
        print(f"{f'({d.p},{d.q})':>8s} {d.norm2:6d} {a.mean():8.3f} {np.median(a):8.3f} "
              f"{a.min():8.3f} {a.max():8.3f} "
              f"{(f'{pub:.2f}' if pub else '--'):>10s} {(f'{r:.3f}' if pub else '--'):>7s}")

    if args.long_side != dataset.LONG_SIDE:
        print(f"\nNOTE: run at {args.long_side} px, not the protocol's {dataset.LONG_SIDE} px. "
              "The published\n      figures are 128 px only, so the ratios below are NOT "
              "comparable. Re-run\n      without --long-side for the real calibration.")
    elif len(ratios) >= 2:
        spread = max(ratios) / min(ratios)
        print(f"\nmedian/published ratios: {[round(r, 3) for r in ratios]}")
        print(f"ratio spread across directions: {spread:.3f}")
        print()
        if 0.85 <= min(ratios) and max(ratios) <= 1.15:
            print("VERDICT: within +-15% of the published figures. The paper may state that")
            print("         timings are measured on hardware identical to [1] and calibrated")
            print("         against its published values. Absolute seconds are comparable.")
        elif spread <= 1.15:
            print("VERDICT: offset from the published figures, but by a CONSISTENT factor")
            print(f"         (~{np.mean(ratios):.2f}x). The cost LAW is unaffected -- report")
            print("         timings as this machine's own measurements, and compare to [1]")
            print("         as ratios rather than absolutes (handoff sec 1.3 fallback).")
        else:
            print("VERDICT: the offset is NOT a constant factor across directions. Something")
            print("         differs structurally (image preprocessing, numpy version, an")
            print("         accidental thread pool). Investigate before running the pool --")
            print("         check qsig.threadguard.report() above and the binarisation in")
            print("         qsig.dataset against what their pipeline did.")


if __name__ == "__main__":
    main()
