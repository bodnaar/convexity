import sys, os
import paths  # noqa: F401  -- puts the repository root on sys.path
import qsig.threadguard  # noqa: F401
import numpy as np
from disjmeasure import load, correct, ci, lda, SLOT
print("headline: classical + [phi,phi_D] at slot_set(6)  vs  classical + phi at all 64")
for name, fn, dfn in [("MPEG-7","mpeg7_full_features.csv","mpeg7_disjunctive.csv"),
                      ("Animal2000","animal2000_features.csv","animal2000_disjunctive.csv"),
                      ("SwedishLeaves","swedishleaves_features.csv","swedishleaves_disjunctive.csv")]:
    y, CL, SIG, SIGD = load(fn, dfn)
    A = np.c_[CL, SIG]                                   # 64 dirs, phi only
    B = np.c_[CL, SIG[:, SLOT[6]], SIGD[:, SLOT[6]]]     # 6 dirs, both
    ca, cb = correct(A, y), correct(B, y)
    m, lo, hi = ci(ca, cb)
    star = " *" if not (lo < 0 < hi) else ""
    print(f"  {name:>14s}  1NN  all64-phi {ca.mean()*100:6.2f} -> 6dir-both {cb.mean()*100:6.2f} "
          f"{m:+6.2f} [{lo:+.2f},{hi:+.2f}]{star}    LDA {lda(A,y):6.2f} -> {lda(B,y):6.2f}")
