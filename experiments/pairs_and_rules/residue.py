"""WHERE does the complementarity of phi and phi_D live?

phi and phi_D each decompose as a shared scalar (their own mean over
directions) plus a near-uncorrelated residue. The question is whether the two
descriptors are complementary because of (a) their two MEANS, (b) their two
RESIDUES, or (c) the cross terms.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from classdisjoint import load, nn1_correct, SLOT

K = 6
for name, fn, dfn in [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
                      ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
                      ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]:
    y, CL, SIG, SIGD = load(fn, dfn)
    c = SLOT[K]
    A, B = SIG[:, c], SIGD[:, c]
    mA, mB = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    rA, rB = A - mA, B - mB
    print(f"\n=== {name} (k={K}) ===")
    # per-direction correlation between the two residues
    per = np.array([np.corrcoef(rA[:, i], rB[:, i])[0, 1] for i in range(K)])
    print(f"  corr(residue_phi, residue_phi_D) per direction: "
          + " ".join(f"{v:+.3f}" for v in per) + f"   mean {per.mean():+.3f}")
    # canonical correlations between the two residue blocks
    def cca(X, Z):
        X = X - X.mean(0); Z = Z - Z.mean(0)
        qx, _ = np.linalg.qr(X); qz, _ = np.linalg.qr(Z)
        return np.linalg.svd(qx.T @ qz, compute_uv=False)
    cc = cca(rA, rB)
    print(f"  canonical correlations of the two residue blocks: "
          + " ".join(f"{v:.3f}" for v in cc))
    print(f"  corr(mu_phi, mu_phi_D) = {np.corrcoef(mA.ravel(), mB.ravel())[0,1]:+.3f}")
    # which piece carries the accuracy
    combos = [("classical", CL),
              ("+ mu_phi", np.c_[CL, mA]),
              ("+ mu_phi, mu_phiD", np.c_[CL, mA, mB]),
              ("+ mu_phi, res_phi   (= phi alone)", np.c_[CL, mA, rA]),
              ("+ mu_phi, res_phi, mu_phiD", np.c_[CL, mA, rA, mB]),
              ("+ mu_phi, res_phi, res_phiD", np.c_[CL, mA, rA, rB]),
              ("+ all four (= both)", np.c_[CL, mA, rA, mB, rB])]
    base = None
    for lbl, X in combos:
        a = nn1_correct(X, y).mean() * 100
        if base is None:
            base = a
        print(f"    {lbl:>36s}  cols={X.shape[1]-4:2d}  {a:6.2f}  ({a-base:+5.2f})")
