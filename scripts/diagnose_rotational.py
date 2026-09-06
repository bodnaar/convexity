#!/usr/bin/env python3
"""WHY the rotational signature R loses to the rotation-free signature S.

IWCIA 2025 Table 2 reports R below S at every set size and offers no mechanism.
This script supplies one, in five steps, from the pool tables:

    python scripts/diagnose_rotational.py --data ../MPEG7dataset.zip \
        --table results/device_n130_rows.csv results/device_n130_rotational.csv \
        --sets sint maxgap10 maxgap8

A. Deviation |E_R - E_S| / E_S per direction, MEAN AND MEDIAN side by side.
   The mean is 6-35% and the median a flat 1.7-3.0%. The gap is not an angular
   anomaly, it is a denominator effect: near-Q-convex shapes have E_S ~ 1e-6, so
   an absolute difference of 1e-4 reads as 20000%. Reporting the mean of a ratio
   here invents a direction-dependent effect that does not exist. (An earlier
   revision of this analysis did exactly that and "found" a 34.6% spike at 18.4
   degrees; step A exists so that mistake cannot be repeated silently.)

B. SIGNED difference by decile of E_S. R exceeds S in 84.5% of the smallest
   decile and falls below it in the largest: rotation adds spurious concavity
   where there is almost none and smooths away real concavity where there is a
   lot. A compression of the dynamic range, not symmetric noise.

C. Per-direction Fisher ratio. R keeps 99.7% of S's marginal class
   separability -- so the loss is NOT in the individual components. This kills
   the obvious explanation and forces step D.

D. Decomposition of the per-shape displacement d = R - S into a constant offset
   across directions plus an angle-dependent residual. The orbit distance d_C
   mean-centres, so the offset is invisible to it; measured on Device, the
   offset carries ~43% of the energy and costs exactly 0 accuracy points, while
   the residual carries ~57% and reproduces R's accuracy to the decimal. Two
   controls bound the claim: iid Gaussian noise of the same magnitude would cost
   ~36 points, so the residual is strongly structured, not random.

E. The noise floor, on synthetic Q-convex shapes where the true value is 0 for
   every direction. A digital square and a right triangle stay exactly 0 under
   both descriptors -- their rotated staircase edges are still monotone. A disc
   stays 0 under S and picks up 2e-5 to 9e-5 under R. Curved boundary plus
   re-binarisation manufactures concavity; that is the floor that swamps the
   near-Q-convex shapes in step B.

Steps A-D need the tables; step E is self-contained (--only E).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qsig.threadguard  # noqa: E402,F401

import numpy as np  # noqa: E402

from qsig import classify, dataset, directions, store  # noqa: E402
from qsig.descriptor import q_concavity, warm_up  # noqa: E402
from qsig.rotational import rotate_binary  # noqa: E402
from qsig.signature import euclidean  # noqa: E402
from scripts.run_pool import resolve_dirs  # noqa: E402


# -- synthetic Q-convex shapes (step E) -------------------------------------

def _square(n=128, s=90):
    a = np.zeros((n, n), np.uint8)
    o = (n - s) // 2
    a[o:o + s, o:o + s] = 1
    return a


def _disc(n=128, r=45):
    y, x = np.ogrid[:n, :n]
    return (((y - n / 2 + .5) ** 2 + (x - n / 2 + .5) ** 2) <= r * r).astype(np.uint8)


def _triangle(n=128):
    y, x = np.ogrid[:n, :n]
    return ((y >= 20) & (y <= 110) & (x >= 20) & (x - 20 <= (y - 20))).astype(np.uint8)


SYNTHETIC = {"square": _square, "disc": _disc, "triangle": _triangle}


def step_A(pairs):
    print("\n=== A. deviation per direction: the mean is a denominator artefact ===")
    print(f"{'p':>3s}{'q':>3s} {'ang':>6s} {'n':>5s} {'mean':>8s} {'median':>8s} "
          f"{'p90':>8s} {'min E_S':>10s}")
    for (p, q), v in sorted(pairs.items(), key=lambda kv: np.arctan2(kv[0][1], kv[0][0])):
        dev = np.array([abs(er - es) / es for es, er, _ in v if es > 0])
        es = np.array([es for es, _, _ in v])
        print(f"{p:3d}{q:3d} {np.degrees(np.arctan2(q, p)):6.2f} {len(dev):5d} "
              f"{100*dev.mean():7.2f}% {100*np.median(dev):7.2f}% "
              f"{100*np.percentile(dev, 90):7.2f}% {es.min():10.2e}")


def step_B(es, er):
    print("\n=== B. signed difference by decile of E_S: R inflates the small end ===")
    qs = np.quantile(es, np.linspace(0, 1, 11))
    print(f"{'E_S range':>19s} {'n':>6s} {'mean E_S':>10s} {'mean E_R':>10s} "
          f"{'mean diff':>10s} {'R>S':>7s}")
    for i in range(10):
        m = (es >= qs[i]) & ((es <= qs[i + 1]) if i == 9 else (es < qs[i + 1]))
        d = er[m] - es[m]
        print(f"{qs[i]:9.5f}-{qs[i+1]:9.5f} {m.sum():6d} {es[m].mean():10.5f} "
              f"{er[m].mean():10.5f} {d.mean():+10.5f} {100*(d>0).mean():6.1f}%")


def _fisher(v, lab):
    v, lab = np.asarray(v), np.asarray(lab)
    gm = v.mean()
    sb = sum((lab == c).sum() * (v[lab == c].mean() - gm) ** 2 for c in set(lab))
    sw = sum(((v[lab == c] - v[lab == c].mean()) ** 2).sum() for c in set(lab))
    return sb / max(sw, 1e-30)


def step_C(table, shapes, pool):
    print("\n=== C. per-direction Fisher ratio: the components are NOT the problem ===")
    fs, fr = [], []
    lab = [sh.cls for sh in shapes]
    res = max(shapes[0].img.shape)
    print(f"{'p':>3s}{'q':>3s} {'ang':>6s} {'sd S':>9s} {'sd R':>9s} "
          f"{'fisher S':>9s} {'fisher R':>9s} {'R/S':>6s}")
    for d in pool:
        try:
            vs = [table[(sh.shape_id, d.p, d.q, res, "S")]["E"] for sh in shapes]
            vr = [table[(sh.shape_id, d.p, d.q, res, "R")]["E"] for sh in shapes]
        except KeyError:
            continue
        a, b = _fisher(vs, lab), _fisher(vr, lab)
        fs.append(a)
        fr.append(b)
        print(f"{d.p:3d}{d.q:3d} {d.angle:6.2f} {np.std(vs):9.5f} {np.std(vr):9.5f} "
              f"{a:9.4f} {b:9.4f} {b/a:6.3f}")
    if fs:
        print(f"\n  mean Fisher  S={np.mean(fs):.4f}  R={np.mean(fr):.4f}  "
              f"-> R retains {100*np.mean(fr)/np.mean(fs):.1f}% of S's marginal separability")


def step_D(table, shapes, specs, seeds=5):
    print("\n=== D. what the displacement R-S is made of, and what each part costs ===")
    lab = np.asarray([sh.cls for sh in shapes])
    for spec in specs:
        dirs = directions.by_angle(resolve_dirs(spec, 0, 0))
        try:
            S, _, _ = classify.build_signatures(table, shapes, dirs, "S")
            R, _, _ = classify.build_signatures(table, shapes, dirs, "R")
        except KeyError as exc:
            print(f"  {spec}: SKIPPED -- {exc}")
            continue
        k = len(dirs)
        d = R - S
        off = d.mean(axis=1, keepdims=True)          # constant across directions
        rez = d - off                                 # angle-dependent remainder
        e_tot = (d ** 2).sum()
        D0 = euclidean(S)
        np.fill_diagonal(D0, np.inf)
        same = np.array([D0[i][lab == lab[i]].min() for i in range(len(lab))])
        other = np.array([D0[i][lab != lab[i]].min() for i in range(len(lab))])
        rng = np.random.default_rng(0)

        def noisy(mag):
            s = mag / np.sqrt(k)
            return np.mean([classify.loo_1nn(S + rng.normal(0, s, S.shape), lab, "dC")
                            for _ in range(seeds)])

        n_d = np.linalg.norm(d, axis=1).mean()
        n_r = np.linalg.norm(rez, axis=1).mean()
        print(f"\n  --- {spec}  (k={k}) ---")
        print(f"  ||R-S|| per shape                  : {n_d:.5f}")
        print(f"    constant offset part             : {100*(off**2).sum()*k/e_tot:4.0f}% of energy")
        print(f"    angle-dependent residual         : {100*(rez**2).sum()/e_tot:4.0f}% of energy"
              f"  (||.||={n_r:.5f})")
        print(f"  class margin (nearest other-same)  : {(other-same).mean():.5f}"
              f"   -> residual is {n_r/(other-same).mean():.2f}x the margin")
        print(f"  accuracy  S                        : {classify.loo_1nn(S, lab, 'dC'):6.2f}%")
        print(f"            S + the offset only      : {classify.loo_1nn(S+off, lab, 'dC'):6.2f}%"
              "   <- d_C mean-centres, so this is free")
        print(f"            S + the residual only    : {classify.loo_1nn(S+rez, lab, 'dC'):6.2f}%"
              "   <- the whole loss")
        print(f"            R                        : {classify.loo_1nn(R, lab, 'dC'):6.2f}%")
        print(f"  control: iid noise, |d| matched    : {noisy(n_d):6.2f}%")
        print(f"  control: iid noise, residual matched: {noisy(n_r):6.2f}%"
              "   <- so the residual is structured, not random")


def step_E(max_norm2=26, impl="rows"):
    print("\n=== E. the noise floor, on shapes whose true value is 0 everywhere ===")
    warm_up(impl)
    pool = directions.by_angle(directions.pool(max_norm2=max_norm2))
    axis = directions.Direction(1, 0)
    print(f"{'shape':>10s} {'px':>7s} {'max E_S':>10s} {'max E_R':>10s} {'nonzero E_R':>12s}")
    for name, make in SYNTHETIC.items():
        img = make()
        s_max = r_max = 0.0
        nz = 0
        for d in pool:
            es, _ = q_concavity(img, d, impl)
            er, _ = q_concavity(rotate_binary(img, d.angle), axis, impl)
            s_max = max(s_max, es)
            r_max = max(r_max, er)
            nz += er > 0
        print(f"{name:>10s} {int(img.sum()):7d} {s_max:10.2e} {r_max:10.2e} "
              f"{nz:6d}/{len(pool):<5d}")
    print("  A rotated square or triangle keeps a monotone staircase edge and stays")
    print("  Q-convex; a disc does not. Curvature plus re-binarisation is the source.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--table", nargs="*", default=[])
    ap.add_argument("--subset", choices=("device", "all"), default="device")
    ap.add_argument("--long-side", type=int, default=dataset.LONG_SIDE)
    ap.add_argument("--sets", nargs="+", default=["sint", "maxgap10"])
    ap.add_argument("--only", default="ABCDE", help="subset of the steps, e.g. --only E")
    args = ap.parse_args()

    if set(args.only) & set("ABCD"):
        if not (args.data and args.table):
            raise SystemExit("steps A-D need --data and --table (use --only E otherwise)")
        classes = dataset.DEVICE_CLASSES if args.subset == "device" else None
        shapes = dataset.load_mpeg7(args.data, long_side=args.long_side, classes=classes)
        table = {}
        for t in args.table:
            table.update(store.load_table(t))
        res = max(shapes[0].img.shape)
        pool, pairs, es, er = [], {}, [], []
        seen = {(k[1], k[2]) for k in table}
        for p, q in sorted(seen):
            d = directions.Direction(p, q)
            rows = []
            for sh in shapes:
                a = table.get((sh.shape_id, p, q, res, "S"))
                b = table.get((sh.shape_id, p, q, res, "R"))
                if a and b:
                    rows.append((a["E"], b["E"], sh.cls))
            if rows:
                pool.append(d)
                pairs[(p, q)] = rows
                es += [x[0] for x in rows]
                er += [x[1] for x in rows]
        pool = directions.by_angle(pool)
        print(f"{args.subset} subset, {len(shapes)} shapes, {len(pool)} directions "
              f"with both families, {len(es)} paired values")
        if not es:
            raise SystemExit("no direction has BOTH family S and family R rows; "
                             "run run_pool.py --family R over the same set first")
        es, er = np.array(es), np.array(er)
        if "A" in args.only:
            step_A(pairs)
        if "B" in args.only:
            step_B(es, er)
        if "C" in args.only:
            step_C(table, shapes, pool)
        if "D" in args.only:
            step_D(table, shapes, args.sets)
    if "E" in args.only:
        step_E()


if __name__ == "__main__":
    main()
