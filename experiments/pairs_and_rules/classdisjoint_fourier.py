"""Class-disjoint selection (classdisjoint_general.py, unchanged) with two
baselines: classical (4 columns), and classical + 12 Fourier descriptors.
The gain is the selected feature set's held-out 1NN accuracy minus that of the
phi-only feature set the same procedure selects. Writes
classdisjoint_fourier.json (optional arguments: dataset names, as in
fourier_test.py). Needs extract_fourier.py output.
"""
from __future__ import annotations
import csv, json, os, sys
import paths  # noqa: F401
import numpy as np
from collections import Counter
import classdisjoint_general as CD

M = 12
STEM = {"MPEG-7": "mpeg7", "Animal2000": "animal2000", "SwedishLeaves": "swedishleaves"}


def ids_of(feat, other):
    fm = [r["shape_id"] for r in csv.DictReader(open(os.path.join(CD.HERE, feat)))]
    keep = {r["shape_id"] for r in csv.DictReader(open(os.path.join(CD.HERE, other)))}
    return [s for s in fm if s in keep]


def main():
    path = os.path.join(CD.HERE, "classdisjoint_fourier.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    only = set(sys.argv[1:])
    for name, feat, bands in CD.SPECS:
        if only and name not in only:
            continue
        fr = {r["shape_id"]: r for r in csv.DictReader(open(os.path.join(CD.HERE, f"{STEM[name]}_fourier.csv")))}
        for band, bf, pre in bands:
            y, CL, P, D = CD.load(feat, bf, pre)
            ids = ids_of(feat, bf if bf else pre)
            assert len(ids) == len(y)
            F = np.array([[float(fr[s][f"fd{m}"]) for m in range(1, M + 1)] for s in ids])
            cell = {}
            for tag, base in (("classical", CL), ("classical+FD", np.c_[CL, F])):
                classes = np.array(sorted(set(y))); nsel = max(2, len(classes) // 3)
                rng = np.random.default_rng(20260926); picks, gains = [], []
                for _ in range(CD.NREP):
                    selc = set(rng.choice(classes, nsel, replace=False))
                    m = np.array([c in selc for c in y])
                    ch, ev, phi, _ = CD.run(y, base, P, D, np.where(m)[0], np.where(~m)[0])
                    picks.append(ch.split("@")[0]); gains.append(ev - phi)
                g = np.array(gains); mode = Counter(picks).most_common(1)[0]
                cell[tag] = dict(mean=round(float(g.mean()), 2), lo=round(float(g.min()), 2),
                                 hi=round(float(g.max()), 2), wins=int((g > 0).sum()), mode=[mode[0], mode[1]])
                if name == "MPEG-7":
                    dev = np.array([c.startswith("device") for c in y])
                    ch, ev, phi, orc = CD.run(y, base, P, D, np.where(dev)[0], np.where(~dev)[0])
                    cell[tag]["device_split"] = dict(chosen=ch, gain=round(ev - phi, 2), gap=round(orc - ev, 2))
            out[f"{name}|{band}"] = cell
            print(name, band, json.dumps(cell), flush=True)
            json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
