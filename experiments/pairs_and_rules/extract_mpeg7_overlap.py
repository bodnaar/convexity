"""Extract classical descriptors + mu for MPEG-7 shapes in classes whose NAME
also appears in Animal2000 (bird, butterfly, deer, dog, elephant, fish, horse,
rat), for the provenance spot-check against Animal2000. Uses load_mpeg7 (the
CE-Shape-1 zip, not the SkeView masks).
"""
import sys, os
import numpy as np
import paths  # noqa: F401  -- puts the repository root on sys.path
from qsig.dataset import load_mpeg7

OVERLAP_CLASSES = ("bird", "butterfly", "deer", "dog", "elephant", "fish", "horse", "rat")

shapes = load_mpeg7(paths.data("MPEG7dataset.zip"),
                     classes=OVERLAP_CLASSES)
print(f"loaded {len(shapes)} MPEG-7 shapes in overlap classes")

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import classical_features, DIRS
from qsig.descriptor import q_concavity, warm_up
warm_up("rows")

out = os.path.join(os.path.dirname(__file__), "mpeg7_overlap_features.csv")
header = ["shape_id", "cls", "area_ratio", "circularity"] + [f"hu{i+1}" for i in range(7)] + [f"sig{i}" for i in range(64)]
with open(out, "w") as f:
    f.write(",".join(header) + "\n")
    for sh in shapes:
        cfeat = classical_features(sh.img)
        sig = np.array([q_concavity(sh.img, d, impl="rows")[0] for d in DIRS])
        row = [sh.shape_id, sh.cls] + list(cfeat) + list(sig)
        f.write(",".join(str(x) for x in row) + "\n")
print("done", out)
