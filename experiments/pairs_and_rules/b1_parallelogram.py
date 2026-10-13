"""B1 control: is the phi_D gain just a rectangularity-type feature?

For each orthogonal slot-set pair (r, r_perp) the feature is
    par(r) = |F| / #(lattice points of the closed bounding parallelogram of F
                     with sides parallel to r and r_perp),
i.e. the filled fraction of the parallelogram whose unfilled part E_D is said
to measure. par = 1 exactly for the phi_D-convex sets (lattice parallelograms).

Step 1 (extract): writes {dataset}_parallelogram.csv (shape_id, par<i> for the
pool indices used by slot_set(6) and slot_set(12)), same preprocessing as
extract_disjunctive.py.
Step 2 (evaluate): 1NN leave-one-out on z-scored features, paired bootstrap
(4000 resamples, seed 0), LDA 5-fold as a second classifier, with and without
M=12 Fourier descriptors. Writes b1_results.json.

Usage: python3 b1_parallelogram.py extract {mpeg7|animal2000|swedishleaves}
       python3 b1_parallelogram.py evaluate
"""
from __future__ import annotations
import csv, json, os, pickle, sys, time
import paths  # noqa: F401
import numpy as np
from qsig import directions as D

HERE = paths.HERE
SLOT = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))
DIRS = D.by_angle(D.pool(max_norm2=130))
IDX = sorted(set(SLOT[6]) | set(SLOT[12]))


def par_ratio(img, r):
    ys, xs = np.nonzero(img)
    r1, r2 = r
    s1, s2 = -r2, r1                       # r_perp
    det = r1 * s2 - r2 * s1
    a = (xs * s2 - ys * s1) / det           # P = a r + b s
    b = (r1 * ys - r2 * xs) / det
    a0, a1, b0, b1 = a.min(), a.max(), b.min(), b.max()
    cx = [aa * r1 + bb * s1 for aa in (a0, a1) for bb in (b0, b1)]
    cy = [aa * r2 + bb * s2 for aa in (a0, a1) for bb in (b0, b1)]
    gx, gy = np.meshgrid(np.arange(int(np.floor(min(cx))), int(np.ceil(max(cx))) + 1),
                         np.arange(int(np.floor(min(cy))), int(np.ceil(max(cy))) + 1))
    ga = (gx * s2 - gy * s1) / det
    gb = (r1 * gy - r2 * gx) / det
    e = 1e-9
    n = np.count_nonzero((ga >= a0 - e) & (ga <= a1 + e) & (gb >= b0 - e) & (gb <= b1 + e))
    return len(xs) / n


def extract(dataset):
    from extract_disjunctive import iter_shapes
    from qsig.dataset import OBJECT
    out = os.path.join(HERE, f"{dataset}_parallelogram.csv")
    t0 = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape_id"] + [f"par{i}" for i in IDX])
        for k, (sid, img) in enumerate(iter_shapes(dataset), 1):
            m = (np.asarray(img) == OBJECT)
            w.writerow([sid] + [repr(par_ratio(m, DIRS[i].vec)) for i in IDX])
            if k % 500 == 0:
                print(f"{dataset}: {k} shapes, {time.time() - t0:.0f}s", flush=True)
    print(f"{dataset}: done, {k} shapes -> {out}")


def evaluate():
    from fourier_test import DATASETS, load, load_ids, correct, ci
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, stem, fn, dfn in DATASETS:
        y, CL, S, SD, FDall = load(stem, fn, dfn)
        ids = load_ids(stem, fn, dfn)
        pr = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, f"{stem}_parallelogram.csv")))}
        assert all(i in pr for i in ids)
        P6 = np.array([[float(pr[i][f"par{j}"]) for j in SLOT[6]] for i in ids])
        F = FDall[:, :12]
        p6, p12, d6 = S[:, SLOT[6]], S[:, SLOT[12]], SD[:, SLOT[6]]
        arms = {}
        for base, B in (("cl", CL), ("cl+FD", np.c_[CL, F])):
            arms[f"{base}"] = B
            arms[f"{base}+phi6"] = np.c_[B, p6]
            arms[f"{base}+phi12"] = np.c_[B, p12]
            arms[f"{base}+phi6+phiD6"] = np.c_[B, p6, d6]
            arms[f"{base}+phi6+par6"] = np.c_[B, p6, P6]
            arms[f"{base}+par6"] = np.c_[B, P6]
            arms[f"{base}+phi6+phiD6+par6"] = np.c_[B, p6, d6, P6]
        c = {k: correct(X, y) for k, X in arms.items()}
        lda = {k: round(cross_val_score(make_pipeline(StandardScaler(), LDA()), X, y, cv=cv).mean() * 100, 2)
               for k, X in arms.items()}
        acc = {k: {"1nn": round(v.mean() * 100, 2), "lda": lda[k]} for k, v in c.items()}
        tests = {}
        for base in ("cl", "cl+FD"):
            for a, b in ((f"{base}+phi6", f"{base}+phi6+phiD6"), (f"{base}+phi6", f"{base}+phi6+par6"),
                         (f"{base}+phi6+par6", f"{base}+phi6+phiD6"), (f"{base}+phi6+phiD6", f"{base}+phi6+phiD6+par6"),
                         (f"{base}", f"{base}+par6")):
                tests[f"{b} | {a}"] = [round(x, 2) for x in ci(c[a], c[b])]
        corr = [abs(np.corrcoef(P6[:, j], d6[:, j])[0, 1]) for j in range(6)]
        res[name] = {"n": int(len(y)), "acc": acc, "tests": tests,
                     "abs_corr_par_phiD_per_direction": [round(x, 3) for x in corr]}
        print(f"\n== {name} (n={len(y)})")
        for k, v in acc.items():
            print(f"  {k:24s} 1NN {v['1nn']:6.2f}  LDA {v['lda']:6.2f}")
        for k, v in tests.items():
            print(f"  diff {k:44s} {v[0]:+6.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")
        print("  |corr(par, phi_D)| per direction:", res[name]["abs_corr_par_phiD_per_direction"])
    json.dump(res, open(os.path.join(HERE, "b1_results.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        evaluate()
