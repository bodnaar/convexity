"""Matched-dimension control: is phi+phi_D at k=12 (24 cols) better than phi
alone at k=24, or is the gain just more columns? Plus the LDA row per dataset."""
import sys, os, pickle, csv
import paths  # noqa: F401  -- puts the repository root on sys.path
import qsig.threadguard  # noqa: F401
import numpy as np
from disjmeasure import load, correct, ci, lda, SLOT

for name, fn, dfn in [("MPEG-7","mpeg7_full_features.csv","mpeg7_disjunctive.csv"),
                      ("Animal2000","animal2000_features.csv","animal2000_disjunctive.csv"),
                      ("SwedishLeaves","swedishleaves_features.csv","swedishleaves_disjunctive.csv")]:
    y, CL, SIG, SIGD = load(fn, dfn)
    print(f"\n=== {name} ===")
    for k in (6, 12):
        both = np.c_[CL, SIG[:, SLOT[k]], SIGD[:, SLOT[k]]]
        phi2k = np.c_[CL, SIG[:, SLOT[2*k]]]
        phid2k = np.c_[CL, SIGD[:, SLOT[2*k]]]
        cb, cp, cd = correct(both, y), correct(phi2k, y), correct(phid2k, y)
        m1, l1, h1 = ci(cp, cb); m2, l2, h2 = ci(cd, cb)
        s1 = "  *" if not (l1 < 0 < h1) else ""
        s2 = "  *" if not (l2 < 0 < h2) else ""
        print(f"  1NN  both(k={k},{2*k}c) {cb.mean()*100:6.2f}  vs phi(k={2*k}) {cp.mean()*100:6.2f} "
              f"{m1:+6.2f} [{l1:+.2f},{h1:+.2f}]{s1}   vs phi_D(k={2*k}) {cd.mean()*100:6.2f} "
              f"{m2:+6.2f} [{l2:+.2f},{h2:+.2f}]{s2}")
        print(f"  LDA  both {lda(both,y):6.2f}  phi(k={2*k}) {lda(phi2k,y):6.2f}  "
              f"phi_D(k={2*k}) {lda(phid2k,y):6.2f}  classical {lda(CL,y):6.2f}")
