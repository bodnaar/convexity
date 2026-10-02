"""Third classifier on the phi/phi_D arms: random forest, 5-fold stratified CV.
Resumable -- appends one dataset per run to rf_results.csv."""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from classdisjoint import load, SLOT
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_results.csv")
NTREE = 300
DATASETS = [("MPEG-7", "mpeg7_full_features.csv", "mpeg7_disjunctive.csv"),
            ("Animal2000", "animal2000_features.csv", "animal2000_disjunctive.csv"),
            ("SwedishLeaves", "swedishleaves_features.csv", "swedishleaves_disjunctive.csv")]

def rf(X, y):
    clf = RandomForestClassifier(n_estimators=NTREE, random_state=0, n_jobs=-1)
    return cross_val_score(clf, np.asarray(X, float), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() * 100

done = set()
if os.path.exists(OUT):
    done = {(r["dataset"], r["arm"]) for r in csv.DictReader(open(OUT))}
mode = "a" if done else "w"
with open(OUT, mode, newline="") as f:
    w = csv.writer(f)
    if mode == "w":
        w.writerow(["dataset", "arm", "cols", "acc"])
    for name, fn, dfn in DATASETS:
        y, CL, SIG, SIGD = load(fn, dfn)
        spec = [("classical", CL),
                ("phi@12", np.c_[CL, SIG[:, SLOT[12]]]),
                ("phi_D@12", np.c_[CL, SIGD[:, SLOT[12]]]),
                ("both@6", np.c_[CL, SIG[:, SLOT[6]], SIGD[:, SLOT[6]]]),
                ("both@12", np.c_[CL, SIG[:, SLOT[12]], SIGD[:, SLOT[12]]]),
                ("phi@24", np.c_[CL, SIG[:, SLOT[24]]]),
                ("phi@64all", np.c_[CL, SIG]),
                ("phi_D@64all", np.c_[CL, SIGD])]
        for arm, X in spec:
            if (name, arm) in done:
                continue
            t = time.perf_counter()
            a = rf(X, y)
            w.writerow([name, arm, X.shape[1] - 4, f"{a:.4f}"]); f.flush()
            print(f"{name:>14s} {arm:>12s} cols={X.shape[1]-4:3d} acc={a:6.2f}  ({time.perf_counter()-t:.0f}s)", flush=True)
print("DONE" if True else "")
