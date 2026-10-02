"""Is phi's degeneracy on general pairs caused by NON-ORTHOGONALITY or by the
NARROW APERTURE of the Farey set?

Three aperture bands, all else held as close as possible:
  narrow general  11-27 deg  (Farey, |det| = 1)        *_generalpairs.csv
  wide   general  53-79 deg  (|det| 4-51)              *_widepairs.csv
  orthogonal      90 deg     (the conventional case)   *_disjunctive.csv (phi_D)
                                                       + *_features.csv  (phi)

If the phi == 0 rate falls monotonically with aperture, the driver is aperture.
If it stays high at 53-79 deg, the driver is non-orthogonality itself.
"""
from __future__ import annotations
import os, sys, csv, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np


def cols(path, pre, k):
    m = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, path)))}
    return m, lambda sid: [float(m[sid][f"{pre}{i}"]) for i in range(k)]


def main():
    print("phi == 0 rate by aperture band (fraction of shape x pair cells)\n")
    print(f"{'dataset':>14s} {'narrow 11-27':>13s} {'wide 53-79':>12s} {'orthogonal 90':>14s}")
    rows = {}
    for name, feat, gen, wide in [
            ("MPEG-7", "mpeg7_full_features.csv", "mpeg7_generalpairs.csv", "mpeg7_widepairs.csv"),
            ("Animal2000", "animal2000_features.csv", "animal2000_generalpairs.csv", "animal2000_widepairs.csv"),
            ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_generalpairs.csv", "swedishleaves_widepairs.csv")]:
        gm, gf = cols(gen, "gp", 12)
        wm, wf = cols(wide, "wp", 12)
        fm = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(HERE, feat)))}
        ids = [s for s in fm if s in gm and s in wm]
        G = np.array([gf(s) for s in ids])
        W = np.array([wf(s) for s in ids])
        # orthogonal phi: the 64-direction signature, restricted to 12 columns
        # spread evenly so the comparison is at matched k
        idx = list(range(0, 64, 64 // 12))[:12]
        O = np.array([[float(fm[s][f"sig{i}"]) for i in idx] for s in ids])
        rows[name] = (G, W, O)
        print(f"{name:>14s} {(G==0).mean()*100:12.2f}% {(W==0).mean()*100:11.2f}% {(O==0).mean()*100:13.2f}%")

    print("\nphi_D == 0 rate, same bands (should be ~0 everywhere)")
    print(f"{'dataset':>14s} {'narrow':>13s} {'wide':>12s}")
    for name, gen, wide in [("MPEG-7","mpeg7_generalpairs.csv","mpeg7_widepairs.csv"),
                            ("Animal2000","animal2000_generalpairs.csv","animal2000_widepairs.csv"),
                            ("SwedishLeaves","swedishleaves_generalpairs.csv","swedishleaves_widepairs.csv")]:
        gm, gf = cols(gen, "gpd", 12); wm, wf = cols(wide, "wpd", 12)
        ids = [s for s in gm if s in wm]
        G = np.array([gf(s) for s in ids]); W = np.array([wf(s) for s in ids])
        print(f"{name:>14s} {(G==0).mean()*100:12.2f}% {(W==0).mean()*100:11.2f}%")

    print("\nper-pair phi==0 rate vs aperture, MPEG-7 wide set")
    SETW = json.load(open(os.path.join(HERE, "wide_pairs.json")))
    _, _, _ = rows["MPEG-7"]
    W = rows["MPEG-7"][1]
    print(f"  {'apert':>7s} {'|det|':>6s} {'maxc':>5s} {'phi==0':>8s}")
    order = sorted(range(12), key=lambda i: SETW[i]["aperture"])
    for i in order:
        d = SETW[i]
        print(f"  {d['aperture']:7.2f} {d['det']:6d} {d['maxcoord']:5d} {(W[:,i]==0).mean()*100:7.2f}%")


if __name__ == "__main__":
    main()
