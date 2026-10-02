"""Two follow-ups to classdisjoint.py.

A. MATCHED-DIMENSION under class-disjoint selection.
   classdisjoint.py's "gain over phi@same k" row is contaminated two ways: the
   `both` arm at k has 2k signature columns against phi's k, and when the
   selected arm IS phi the gain is 0 by construction. Here the selection is
   restricted to {phi, phi_D, both} at COLUMN BUDGET c, i.e. phi/phi_D at k=c
   against both at k=c/2, so every arm compared has exactly c signature columns
   and no arm can win by carrying more features.

B. THIRD CLASSIFIER -- a tree ensemble. 1NN and LDA agree on the phi_D result,
   but those two disagree on direction selection, so their agreement
   is not an independent third opinion. Random forest, 5-fold stratified CV,
   same arms as the LDA comparison in disjdim.py.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from classdisjoint import load, nn1_correct, SLOT

DATASETS = [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
            ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
            ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]

# column budgets reachable as both (k=c/2) and as phi/phi_D (k=c)
BUDGETS = [c for c in sorted(SLOT) if c % 2 == 0 and (c // 2) in SLOT]


def arms_at_budget(CL, SIG, SIGD, c):
    h = c // 2
    return {f"phi@{c}":   np.c_[CL, SIG[:, SLOT[c]]],
            f"phi_D@{c}": np.c_[CL, SIGD[:, SLOT[c]]],
            f"both@{h}":  np.c_[CL, SIG[:, SLOT[h]], SIGD[:, SLOT[h]]]}


print("=" * 78)
print("A. MATCHED COLUMN BUDGET, class-disjoint selection")
print(f"   budgets tested: {BUDGETS}")
print("=" * 78)

for name, fn, dfn in DATASETS:
    y, CL, SIG, SIGD = load(fn, dfn)
    classes = np.array(sorted(set(y)))
    nsel = max(2, len(classes) // 3)
    rng = np.random.default_rng(20260926)
    picks, gains, wins_vs_phi = [], [], []
    NREP = 25
    for _ in range(NREP):
        selc = set(rng.choice(classes, nsel, replace=False))
        m = np.array([cc in selc for cc in y])
        sel, ev = np.where(m)[0], np.where(~m)[0]
        gs, ge = {}, {}
        for c in BUDGETS:
            for lbl, X in arms_at_budget(CL, SIG, SIGD, c).items():
                gs[lbl] = nn1_correct(X[sel], y[sel]).mean() * 100
                ge[lbl] = nn1_correct(X[ev], y[ev]).mean() * 100
        chosen = max(gs, key=gs.get)
        picks.append(chosen.split("@")[0])
        # best phi-only arm the same selection procedure would have picked
        phi_only = {k: v for k, v in gs.items() if k.startswith("phi@")}
        chosen_phi = max(phi_only, key=phi_only.get)
        gains.append(ge[chosen] - ge[chosen_phi])
        wins_vs_phi.append(ge[chosen] > ge[chosen_phi])
    g = np.array(gains)
    print(f"\n{name}: {NREP} splits, {nsel}/{len(classes)} classes select")
    print("  arm chosen: " + ", ".join(f"{a}x{n}" for a, n in Counter(picks).most_common()))
    print(f"  eval gain of the selected arm over the selected PHI-ONLY arm, same budget:")
    print(f"    mean {g.mean():+6.2f}  median {np.median(g):+6.2f}  [{g.min():+.2f}, {g.max():+.2f}]  "
          f"wins {sum(wins_vs_phi)}/{NREP}  ties {(g==0).sum()}")

print("\n" + "=" * 78)
print("B. THIRD CLASSIFIER -- random forest, 5-fold stratified CV")
print("=" * 78)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def rf(X, y):
    clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    return cross_val_score(clf, np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100

print(f"\n{'dataset':>14s} {'class':>7s} {'phi12':>7s} {'phiD12':>7s} {'both6':>7s} "
      f"{'both12':>7s} {'phi24':>7s} {'phi64':>7s} {'phiD64':>7s}")
for name, fn, dfn in DATASETS:
    y, CL, SIG, SIGD = load(fn, dfn)
    cells = [rf(CL, y),
             rf(np.c_[CL, SIG[:, SLOT[12]]], y),
             rf(np.c_[CL, SIGD[:, SLOT[12]]], y),
             rf(np.c_[CL, SIG[:, SLOT[6]], SIGD[:, SLOT[6]]], y),
             rf(np.c_[CL, SIG[:, SLOT[12]], SIGD[:, SLOT[12]]], y),
             rf(np.c_[CL, SIG[:, SLOT[24]]], y),
             rf(np.c_[CL, SIG], y),
             rf(np.c_[CL, SIGD], y)]
    print(f"{name:>14s} " + " ".join(f"{v:7.2f}" for v in cells))
