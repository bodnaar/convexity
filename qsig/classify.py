"""1NN classification with leave-one-out cross-validation.

Matches IWCIA 2025 sec 5.4 so accuracies sit next to their Table 2 without
qualification: 1-nearest-neighbour, leave-one-out, d_C distance.
"""

from __future__ import annotations

import numpy as np

from .signature import euclidean, pairwise_d_C


def loo_1nn_from_distances(dist: np.ndarray, labels) -> float:
    """Leave-one-out 1NN accuracy in percent, given a full distance matrix."""
    d = np.array(dist, dtype=float, copy=True)
    np.fill_diagonal(d, np.inf)          # leave-one-out
    labels = np.asarray(labels)
    nn = d.argmin(axis=1)
    return float((labels[nn] == labels).mean() * 100.0)


def loo_1nn(sigs: np.ndarray, labels, metric: str = "dC") -> float:
    """Accuracy in percent for a stack of signatures, shape (N, k)."""
    sigs = np.asarray(sigs, dtype=float)
    if metric == "dC":
        dist = pairwise_d_C(sigs)
    elif metric in ("l2", "euclidean"):
        dist = euclidean(sigs)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return loo_1nn_from_distances(dist, labels)


def build_signatures(table: dict, shapes, dirs):
    """Assemble an (N, k) signature matrix from a results table.

    `table` is `qsig.store.load_table` output; `shapes` a list of
    `qsig.dataset.Shape`; `dirs` a direction set already in signature order.
    Raises KeyError naming the first missing (shape, direction) pair, so a
    partial pool fails loudly rather than silently producing NaNs.
    """
    res = shapes[0].img.shape
    resolution = max(res)
    rows, labels, ids = [], [], []
    for sh in shapes:
        vals = []
        for d in dirs:
            key = (sh.shape_id, d.p, d.q, resolution)
            if key not in table:
                raise KeyError(f"missing result for {key}")
            vals.append(table[key]["E"])
        rows.append(vals)
        labels.append(sh.cls)
        ids.append(sh.shape_id)
    return np.array(rows, dtype=float), labels, ids


def measured_cost(table: dict, shapes, dirs, statistic: str = "median") -> float:
    """Mean/median seconds to compute one full signature, from the table.

    This is the x-axis of the paper's headline figure. Medians are more robust
    to a stray scheduling event over a long run; report both (handoff sec 7.2).
    """
    resolution = max(shapes[0].img.shape)
    total = 0.0
    for d in dirs:
        secs = [
            table[(sh.shape_id, d.p, d.q, resolution)]["seconds"]
            for sh in shapes
            if (sh.shape_id, d.p, d.q, resolution) in table
        ]
        if not secs:
            raise KeyError(f"no timings for direction ({d.p},{d.q})")
        total += float(np.median(secs) if statistic == "median" else np.mean(secs))
    return total
