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

from qsig import dataset, descriptor, directions, fast, store  # noqa: E402
from qsig.descriptor import q_concavity  # noqa: E402

# A spread of |r|^2 from 1 to 109 so the law can be fitted, PLUS the
# discriminating triple (10,1) (10,3) (7,8).
#
# Those three have nearly equal |r|^2 (101, 109, 113) but wildly different
# (p+q)^2 (121, 169, 225). If cost tracks |r|^2 their times differ by ~12%; if
# it tracks the published (p+q)^2 bound they differ by ~86%. A regression over
# directions that happen to rank the same way under both models cannot separate
# them however good its R^2 is -- this triple can, in three measurements.
DEFAULT_DIRS = "1x0,2x1,3x1,5x1,1x7,10x1,10x3,7x8"


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


def fit_area_normalised(area, xval, secs):
    """Fit t = A * (c0 + c1 * x) over every individual (shape, direction) run.

    `x` is the cost predictor of the kernel being timed -- |r|^2 for the point
    kernel, p+q for the rows kernel. Hard-coding |r|^2 here was a bug of the
    same kind as offering only two candidate models: it reported R^2 = 0.84 for
    the rows kernel and made a correct implementation look wrong.

    With --subset all the image areas span 2048..16384 px, an 8x range, and both
    the direction-independent and the direction-dependent work scale with area.
    Fitting raw seconds therefore charges the shape-size variation to the
    residuals and makes an excellent model look mediocre. Normalising by the
    PADDED area -- the array the algorithm actually walks, (h+2p)(w+2p) with
    p = max(|r1|,|r2|) -- removes that, and folds in the fact that a long
    direction pads the image more, which is a genuine part of its cost.

    Returns c0, c1 in seconds per pixel, plus R^2 over the individual runs.
    """
    A = np.asarray(area, float)
    xv = np.asarray(xval, float)
    y = np.asarray(secs, float)
    X = np.vstack([A, A * xv]).T
    (c0, c1), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ np.array([c0, c1])
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return float(c0), float(c1), float(r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shapes", type=int, default=8, help="shapes to average over")
    ap.add_argument("--subset", choices=("all", "device"), default="all",
                    help="'all' matches IWCIA Table 1's protocol; 'device' is 128x128 only")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--dirs", default=DEFAULT_DIRS, help="comma list of pxq")
    ap.add_argument("--impl", choices=descriptor.IMPLEMENTATIONS, default="reference",
                    help="'reference' = convexity.Convexity as published (the default, "
                         "and the only one comparable to IWCIA Table 1); 'fast' = qsig.fast")
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

    tag = descriptor.implementation_tag(args.impl)
    descriptor.warm_up(args.impl)   # JIT compile must not land inside a timed run
    gov = store.cpu_governor()
    print(f"host       {platform.node()}")
    print(f"impl       {tag}")
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

    st = store.ResultStore(args.out, impl=tag) if args.out else None
    print(f"{'dir':>8s} {'|r|^2':>6s} {'mean':>8s} {'median':>8s} {'min':>8s} {'max':>8s} "
          f"{'IWCIA T1':>9s} {'ratio':>7s}")
    if args.impl != "reference":
        print("  (IWCIA T1 column is for context only -- their seconds are from the "
              "reference\n   implementation on a 2016 stack, not comparable to this one)")
    print("-" * 68)
    med = []
    obs_area, obs_n2, obs_sec, obs_idx = [], [], [], []
    for di_, d in enumerate(dirs):
        pad = max(d.p, d.q)
        secs, rows = [], []
        for sh in shapes:
            value, dt = q_concavity(sh.img, d, args.impl)
            secs.append(dt)
            h, w = sh.img.shape
            obs_area.append((h + 2 * pad) * (w + 2 * pad))
            obs_n2.append(d.norm2)
            obs_idx.append(di_)
            obs_sec.append(dt)
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

    # ------------------------------------------------------------------
    # Candidate cost models.
    #
    # WHICH MODELS ARE OFFERED MATTERS. A fit can only pick the best of the
    # candidates it is given: with |r|^2 and (p+q)^2 as the only options, the
    # rows kernel -- which is Theta(mn(p+q)) by construction -- was reported as
    # "follows |r|^2", because |r|^2 was the closer of two wrong answers. So the
    # operation count of the kernel ACTUALLY RUNNING is always a candidate, and
    # every model is shown ranked rather than a single verdict being asserted.
    # ------------------------------------------------------------------
    oc = [fast.operation_counts(d.vec) for d in dirs]
    kernel_ops = ("rows_ops" if args.impl == "rows" else "points_ops")
    candidates = {
        "|r|^2 = |det(r,r_perp)|": [d.norm2 for d in dirs],
        "(p+q)^2  [published bound]": [(d.p + d.q) ** 2 for d in dirs],
        "p+q": [d.p + d.q for d in dirs],
        f"operation count ({kernel_ops})": [o[kernel_ops] for o in oc],
    }
    print(f"\ncost-model fit on THIS machine ({len(dirs)} directions, medians), ranked:")
    fits = {}
    for name, x in candidates.items():
        a_, b_, r_, m_ = fit(x, med)
        fits[name] = (a_, b_, r_, m_, x)
    for name in sorted(fits, key=lambda n: -fits[n][2]):
        a_, b_, r_, m_, _ = fits[name]
        print(f"  {name:30s} T = {a_:8.5f} + {b_:.6f}*x   R2={r_:.5f}  MAPE={m_:5.2f}%")
    best = max(fits, key=lambda n: fits[n][2])
    print(f"  {'published (IWCIA Table 1)':30s} T = {directions.COST_IWCIA[0]:8.5f} + "
          f"{directions.COST_IWCIA[1]:.6f}*|r|^2")

    bx = candidates[best]
    obs_x = [bx[k] for k in obs_idx]
    c0, c1, r_area = fit_area_normalised(obs_area, obs_x, obs_sec)
    ref = (args.long_side + 2) ** 2
    ref_dir = directions.Direction(10, 3)
    x109 = {"|r|^2 = |det(r,r_perp)|": ref_dir.norm2,
            "(p+q)^2  [published bound]": (ref_dir.p + ref_dir.q) ** 2,
            "p+q": ref_dir.p + ref_dir.q}.get(
        best, fast.operation_counts(ref_dir.vec)[kernel_ops])
    print(f"\narea-normalised fit over all {len(obs_sec)} individual runs, in the "
          f"best-fitting predictor\n({best}; padded area, so shape size is not charged "
          f"to the residuals):")
    print(f"  t = A * ({c0 * 1e6:.3f} + {c1 * 1e6:.5f} * x) microseconds per padded pixel"
          f"   R2={r_area:.5f}")
    print(f"  at a {args.long_side}x{args.long_side} shape that is "
          f"T = {c0 * ref:.5f} + {c1 * ref:.6f}*x s")
    print(f"  direction-dependent share at (10,3): "
          f"{c1 * x109 / (c0 + c1 * x109) * 100:.1f}% of the work")

    # ------------------------------------------------------------------
    # Model discrimination, WITH the padding correction.
    #
    # pad = max(p,q), so directions of equal |r|^2 can still walk arrays of
    # different size -- (7,8) pads by 8 where (10,3) pads by 10 and is therefore
    # ~6% cheaper for reasons that have nothing to do with the cost model.
    # Comparing raw seconds across the triple silently charges that to whichever
    # model is on trial.
    # ------------------------------------------------------------------
    areas = {}
    for d, o in zip(dirs, oc):
        pad = max(d.p, d.q)
        areas[(d.p, d.q)] = np.mean([(sh.img.shape[0] + 2 * pad) * (sh.img.shape[1] + 2 * pad)
                                     for sh in shapes])
    groups = []
    for i, di in enumerate(dirs):
        grp = [j for j, dj in enumerate(dirs) if abs(dj.norm2 - di.norm2) <= 0.15 * di.norm2]
        if len(grp) >= 2:
            box = [(dirs[j].p + dirs[j].q) ** 2 for j in grp]
            if max(box) > 1.25 * min(box):
                grp = tuple(sorted(grp))
                if grp not in groups:
                    groups.append(grp)
    rng = lambda v: (max(v) / min(v) - 1) * 100
    for grp in groups:
        base = areas[(dirs[grp[-1]].p, dirs[grp[-1]].q)]
        print("\nmodel discrimination -- near-equal |r|^2, very different (p+q)^2:")
        print(f"  {'dir':>8s} {'|r|^2':>6s} {'(p+q)^2':>8s} {'p+q':>4s} {'ops':>5s} "
              f"{'measured':>9s} {'per padded area':>16s}")
        corr = []
        for j in grp:
            d = dirs[j]
            c = med[j] / (areas[(d.p, d.q)] / base)
            corr.append(c)
            print(f"  {f'({d.p},{d.q})':>8s} {d.norm2:6d} {(d.p + d.q) ** 2:8d} "
                  f"{d.p + d.q:4d} {oc[j][kernel_ops]:5d} {med[j]:9.5f} {c:16.5f}")
        print(f"  spread across this group -- measured raw {rng([med[j] for j in grp]):5.1f}%, "
              f"area-corrected {rng(corr):5.1f}%")
        for name, xs in candidates.items():
            print(f"      {name:30s} predicts {rng([xs[j] for j in grp]):5.1f}%")
        winner = min(candidates,
                     key=lambda n: abs(rng([candidates[n][j] for j in grp]) - rng(corr)))
        print(f"  -> the area-corrected measurement follows: {winner}")

    cheap, dear = med[0], med[-1]
    spread = dear / cheap
    pub_spread = 65.21 / 2.54
    print(f"\ncost spread, dearest / cheapest direction:")
    print(f"  this machine  {spread:6.2f}x   ({dirs[-1].p},{dirs[-1].q}) vs ({dirs[0].p},{dirs[0].q})")
    print(f"  IWCIA Table 1 {pub_spread:6.2f}x   (10,3) vs (1,0)")
    print("  cost-aware selection is worth roughly the spread, so this is the number")
    print("  the paper's saving scales with -- not the published one.")

    print()
    print(f"VERDICT: best-fitting cost model on this machine for impl={tag}:")
    print(f"         {best}   (R2 = {fits[best][2]:.4f})")
    expected = "p+q" if args.impl == "rows" else "|r|^2 = |det(r,r_perp)|"
    if best.startswith(expected) or best.startswith("operation count"):
        print(f"         This is what the algorithm predicts for this kernel. Good.")
    else:
        print(f"         EXPECTED {expected} for impl={tag}. Investigate before")
        print("         trusting any cost number from this run.")
    print("         The CONSTANTS never transfer between implementations or software")
    print("         stacks; only the functional form does. Report this machine's own")
    print("         measurements with the stack printed above, and never quote IWCIA")
    print("         Table 1 seconds alongside accuracies measured here (handoff 3.1.1).")


if __name__ == "__main__":
    main()
