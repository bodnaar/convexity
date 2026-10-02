"""Full MPEG-7 (1400 shapes): classical descriptors + 64-direction signature,
written to mpeg7_full_features.csv. Resumable."""
import sys, os, time
import numpy as np
import paths  # noqa: F401  -- puts the repository root on sys.path
from qsig.dataset import load_mpeg7
sys.path.insert(0, os.path.dirname(__file__))
from extract_features import classical_features, DIRS
from qsig.descriptor import q_concavity, warm_up

out = os.path.join(os.path.dirname(__file__), "mpeg7_full_features.csv")
header = ["shape_id", "cls", "area_ratio", "circularity"] + [f"hu{i+1}" for i in range(7)] + [f"sig{i}" for i in range(64)]

shapes = load_mpeg7(paths.data("MPEG7dataset.zip"))
print(f"loaded {len(shapes)} MPEG-7 shapes", flush=True)

done = set()
mode = "w"
if os.path.exists(out):
    lines = open(out).read().splitlines()
    if lines and lines[0] == ",".join(header):
        good = [lines[0]]
        for ln in lines[1:]:
            if ln.count(",") == len(header) - 1:
                good.append(ln)
        if len(good) != len(lines):
            open(out, "w").write("\n".join(good) + "\n")
        done = {ln.split(",", 1)[0] for ln in good[1:]}
        mode = "a"

warm_up("rows")
t0 = time.perf_counter()
n = len(shapes)
with open(out, mode) as f:
    if mode == "w":
        f.write(",".join(header) + "\n")
    i = 0
    for sh in shapes:
        i += 1
        if sh.shape_id in done:
            continue
        cfeat = classical_features(sh.img)
        sig = np.array([q_concavity(sh.img, d, impl="rows")[0] for d in DIRS])
        row = [sh.shape_id, sh.cls] + list(cfeat) + list(sig)
        f.write(",".join(str(x) for x in row) + "\n")
        f.flush()
        if i % 50 == 0 or i == n:
            el = time.perf_counter() - t0
            print(f"{i}/{n} done, {el:.0f}s elapsed", flush=True)
print("FINISHED", flush=True)
