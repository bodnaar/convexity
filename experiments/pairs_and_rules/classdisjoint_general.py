"""Class-disjoint selection protocol on the GENERAL-pair bands.

classdisjoint.py and classdisjoint2.py cover the orthogonal set only; this
runs the same control on the general-pair aperture bands.

SELECT (arm, k) on a class-disjoint subset; EVALUATE on the held-out classes.
Arms: phi, phi_D, both -- all at a matched COLUMN BUDGET, so `both` at k = c/2
competes against phi/phi_D at k = c and no arm wins by carrying more features.
Bands: narrow general (11-27 deg), wide general (53-79 deg), orthogonal (90).
MPEG-7 additionally uses the Device split the literature already treats as
separate; all three datasets use 25 random class splits.
"""
from __future__ import annotations
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from collections import Counter

BUDGETS = [4, 6, 8, 12]          # both at 2,3,4,6 ; phi/phi_D at 4,6,8,12
NREP = 25


def load(feat, band, pre, k=12):
    fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, feat)))}
    if band is None:                                   # orthogonal reference
        dm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, pre)))}
        ids = [s for s in fm if s in dm]
        idx = list(range(0, 64, 64 // k))[:k]
        P = np.array([[float(fm[s][f"sig{i}"]) for i in idx] for s in ids])
        D = np.array([[float(dm[s][f"sigd{i}"]) for i in idx] for s in ids])
    else:
        bm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, band)))}
        ids = [s for s in fm if s in bm]
        P = np.array([[float(bm[s][f"{pre}{i}"]) for i in range(k)] for s in ids])
        D = np.array([[float(bm[s][f"{pre}d{i}"]) for i in range(k)] for s in ids])
    y = np.array([fm[s]["cls"] for s in ids])
    CL = np.array([[float(fm[s][c]) for c in ("area_ratio","circularity","hu1","hu2")] for s in ids])
    return y, CL, P, D


def correct(X, y):
    X = np.asarray(X, float)
    s = X.std(0, keepdims=True); s[s == 0] = 1
    Z = np.ascontiguousarray(X / s, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d = sq[None, :] - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y)


def arms(CL, P, D, c):
    """Every arm has exactly c signature columns."""
    h = c // 2
    return {f"phi@{c}": np.c_[CL, P[:, :c]],
            f"phi_D@{c}": np.c_[CL, D[:, :c]],
            f"both@{h}": np.c_[CL, P[:, :h], D[:, :h]]}


def run(y, CL, P, D, sel, ev):
    gs, ge = {}, {}
    for c in BUDGETS:
        for lbl, X in arms(CL, P, D, c).items():
            gs[lbl] = correct(X[sel], y[sel]).mean() * 100
            ge[lbl] = correct(X[ev], y[ev]).mean() * 100
    chosen = max(gs, key=gs.get)
    phi_only = {k: v for k, v in gs.items() if k.startswith("phi@")}
    chosen_phi = max(phi_only, key=phi_only.get)
    return chosen, ge[chosen], ge[chosen_phi], ge[max(ge, key=ge.get)]


SPECS = [("MPEG-7", "mpeg7_full_features.csv",
          [("narrow", "mpeg7_generalpairs.csv", "gp"),
           ("wide", "mpeg7_widepairs.csv", "wp"),
           ("orthog", None, "mpeg7_disjunctive.csv")]),
         ("Animal2000", "animal2000_features.csv",
          [("narrow", "animal2000_generalpairs.csv", "gp"),
           ("wide", "animal2000_widepairs.csv", "wp"),
           ("orthog", None, "animal2000_disjunctive.csv")]),
         ("SwedishLeaves", "swedishleaves_features.csv",
          [("narrow", "swedishleaves_generalpairs.csv", "gp"),
           ("wide", "swedishleaves_widepairs.csv", "wp"),
           ("orthog", None, "swedishleaves_disjunctive.csv")])]


def main():
    print("CLASS-DISJOINT SELECTION, matched column budget, general-pair bands")
    print("(arm and k chosen on classes that are never scored)\n")
    # MPEG-7 Device split
    print("--- MPEG-7, Device split (select 10 device* classes, score the other 60) ---")
    print(f"  {'band':>8s} {'selected':>12s} {'eval':>7s} {'vs sel. phi':>12s} {'oracle':>7s} {'gap':>6s}")
    for band, bf, pre in SPECS[0][2]:
        y, CL, P, D = load(SPECS[0][1], bf, pre)
        dev = np.array([c.startswith("device") for c in y])
        ch, ev_acc, phi_acc, orc = run(y, CL, P, D, np.where(dev)[0], np.where(~dev)[0])
        print(f"  {band:>8s} {ch:>12s} {ev_acc:7.2f} {ev_acc-phi_acc:+12.2f} {orc:7.2f} {orc-ev_acc:6.2f}")

    print(f"\n--- {NREP} random class splits, all datasets ---")
    print(f"  {'dataset':>14s} {'band':>8s} {'arm chosen':>24s} {'gain over sel. phi':>20s} {'wins':>7s}")
    for name, feat, bands in SPECS:
        for band, bf, pre in bands:
            y, CL, P, D = load(feat, bf, pre)
            classes = np.array(sorted(set(y)))
            nsel = max(2, len(classes) // 3)
            rng = np.random.default_rng(20260926)
            picks, gains = [], []
            for _ in range(NREP):
                selc = set(rng.choice(classes, nsel, replace=False))
                m = np.array([c in selc for c in y])
                ch, ev_acc, phi_acc, _ = run(y, CL, P, D, np.where(m)[0], np.where(~m)[0])
                picks.append(ch.split("@")[0]); gains.append(ev_acc - phi_acc)
            g = np.array(gains)
            cnt = ", ".join(f"{a}x{n}" for a, n in Counter(picks).most_common())
            print(f"  {name:>14s} {band:>8s} {cnt:>24s} "
                  f"{f'{g.mean():+.2f} [{g.min():+.2f},{g.max():+.2f}]':>20s} {(g>0).sum():4d}/{NREP}")


if __name__ == "__main__":
    main()
