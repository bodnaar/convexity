"""Class-disjoint selection protocol on the phi/phi_D result.

The phi+phi_D result fixed the DIRECTION SET before seeing labels, but two
decisions were still made with all labels in view -- the number of directions k,
and the choice of the `both` arm over `phi` alone. This run makes both decisions
on classes that are never scored.

  SELECT on a class-disjoint subset: choose (arm, k) by 1NN LOO there.
  EVALUATE the chosen (arm, k) by 1NN LOO on the held-out classes.

MPEG-7 uses the Device split the literature already treats as separate.
Animal2000 and SwedishLeaves have no such convention, so a random class split is
repeated over many seeds and the distribution reported.

Reported alongside: the classical baseline on the SAME evaluation shapes; the
ORACLE (best arm/k chosen on the evaluation set itself) so the selection gap is
visible; and phi at the same k, which is the comparison that matters.
"""
from __future__ import annotations
import sys, os, pickle, csv
import paths  # noqa: F401  -- puts the repository root on sys.path
import numpy as np  # BLAS multithreaded on purpose here: this is an accuracy
# run, not a timing run, so qsig.threadguard is deliberately NOT imported.
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT = pickle.load(open(os.path.join(HERE, "slotcols.pkl"), "rb"))
KS = sorted(SLOT)


def load(fn, dfn):
    rows = list(csv.DictReader(open(os.path.join(HERE, fn))))
    drows = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, dfn)))}
    rows = [r for r in rows if r["shape_id"] in drows]
    y = np.array([r["cls"] for r in rows])
    CL = np.array([[float(r[k]) for k in ("area_ratio", "circularity", "hu1", "hu2")] for r in rows])
    SIG = np.array([[float(r[f"sig{i}"]) for i in range(64)] for r in rows])
    SIGD = np.array([[float(drows[r["shape_id"]][f"sigd{i}"]) for i in range(64)] for r in rows])
    return y, CL, SIG, SIGD


def nn1_correct(X, y):
    """1NN LOO, z-scored, squared-L2 via the gram identity.

    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  -- one GEMM instead of an n x n x d
    broadcast, which at n=1333, d=68 would allocate ~480 MB per call. Identical
    nearest neighbours; the constant ||a||^2 per row does not move the argmin,
    and the -2 a.b term is what BLAS computes.
    """
    X = np.asarray(X, float)
    s = X.std(0, keepdims=True); s[s == 0] = 1
    # float64 throughout: in float32 the gram identity loses enough precision on
    # near-ties to flip individual nearest neighbours (0.09-0.18 points on
    # SwedishLeaves against the broadcast reference). At float64 it is exact
    # against that reference on every arm tested.
    Z = np.ascontiguousarray((X / s), dtype=np.float64)
    sq = (Z * Z).sum(1)
    d = sq[None, :] - 2.0 * (Z @ Z.T)      # + sq[:, None] omitted: constant per row
    np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def arms(CL, SIG, SIGD, k):
    c = SLOT[k]
    return {"phi": np.c_[CL, SIG[:, c]],
            "phi_D": np.c_[CL, SIGD[:, c]],
            "both": np.c_[CL, SIG[:, c], SIGD[:, c]]}


def grid(CL, SIG, SIGD, y, idx):
    out = {}
    for k in KS:
        for a, X in arms(CL, SIG, SIGD, k).items():
            out[(a, k)] = nn1_correct(X[idx], y[idx]).mean() * 100
    return out


def run_split(CL, SIG, SIGD, y, sel, ev):
    gs = grid(CL, SIG, SIGD, y, sel)
    ge = grid(CL, SIG, SIGD, y, ev)
    chosen = max(gs, key=gs.get)
    oracle = max(ge, key=ge.get)
    return dict(chosen=chosen, chosen_eval=ge[chosen], chosen_sel=gs[chosen],
                base=nn1_correct(CL[ev], y[ev]).mean() * 100,
                oracle=oracle, oracle_eval=ge[oracle],
                ref_phi64=nn1_correct(np.c_[CL, SIG][ev], y[ev]).mean() * 100,
                phi_same_k=ge[("phi", chosen[1])])


def main():
    print("=" * 78)
    print("CLASS-DISJOINT SELECTION  (arm and k chosen on classes that are never scored)")
    print("=" * 78)

    y, CL, SIG, SIGD = load("mpeg7_full_features.csv", "mpeg7_disjunctive.csv")
    dev = np.array([c.startswith("device") for c in y])
    print(f"\nMPEG-7  select = Device ({dev.sum()} shapes, {len(set(y[dev]))} classes)  "
          f"eval = rest ({(~dev).sum()} shapes, {len(set(y[~dev]))} classes)")
    r = run_split(CL, SIG, SIGD, y, np.where(dev)[0], np.where(~dev)[0])
    print(f"  selected on Device : arm={r['chosen'][0]:6s} k={r['chosen'][1]:2d}  (Device acc {r['chosen_sel']:.2f})")
    print(f"  EVAL, 60 classes   : {r['chosen_eval']:.2f}%")
    print(f"    classical 4d     : {r['base']:6.2f}%   gain {r['chosen_eval']-r['base']:+.2f}")
    print(f"    phi at same k    : {r['phi_same_k']:6.2f}%   gain over phi {r['chosen_eval']-r['phi_same_k']:+.2f}")
    print(f"    phi all 64 (ref) : {r['ref_phi64']:6.2f}%   gain over all-64 {r['chosen_eval']-r['ref_phi64']:+.2f}")
    print(f"    ORACLE on eval   : arm={r['oracle'][0]:6s} k={r['oracle'][1]:2d} -> {r['oracle_eval']:.2f}%   "
          f"selection gap {r['oracle_eval']-r['chosen_eval']:.2f}")

    for name, fn, dfn in [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
                          ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
                          ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]:
        y, CL, SIG, SIGD = load(fn, dfn)
        classes = np.array(sorted(set(y)))
        nsel = max(2, len(classes) // 3)
        rng = np.random.default_rng(20260926)
        picks, gc, gp, g6, gg = [], [], [], [], []
        NREP = 25
        for _ in range(NREP):
            selc = set(rng.choice(classes, nsel, replace=False))
            m = np.array([c in selc for c in y])
            r = run_split(CL, SIG, SIGD, y, np.where(m)[0], np.where(~m)[0])
            picks.append(r["chosen"])
            gc.append(r["chosen_eval"] - r["base"])
            gp.append(r["chosen_eval"] - r["phi_same_k"])
            g6.append(r["chosen_eval"] - r["ref_phi64"])
            gg.append(r["oracle_eval"] - r["chosen_eval"])
        gc, gp, g6, gg = map(np.array, (gc, gp, g6, gg))
        print(f"\n{name}: {NREP} random class splits, {nsel}/{len(classes)} classes select")
        print("  arm chosen      : " + ", ".join(f"{a}x{n}" for a, n in Counter(a for a, k in picks).most_common()))
        print("  k chosen        : " + ", ".join(f"{k}x{n}" for k, n in Counter(k for a, k in picks).most_common(5)))
        for lbl, g in [("over classical ", gc), ("over phi@same k", gp), ("over phi all 64", g6)]:
            print(f"  gain {lbl}: mean {g.mean():+6.2f}  median {np.median(g):+6.2f}  "
                  f"[{g.min():+.2f}, {g.max():+.2f}]  wins {(g>0).sum()}/{NREP}")
        print(f"  selection gap vs oracle: mean {gg.mean():.2f}  max {gg.max():.2f}")


if __name__ == "__main__":
    main()
