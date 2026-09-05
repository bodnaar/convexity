"""One Q-concavity value for one (shape, direction) pair.

Thin wrapper over the existing `convexity.Convexity`, which is the REFERENCE
implementation for the whole project: every optimised version must reproduce it
exactly (handoff sec 7.4 item 3). Nothing here changes its numerics.

`Convexity.compute(obj_a, obj_b, vec)` returns `q1`, which is the descriptor
E_F^{r,s} of [IWCIA 2025, Def 2]: phi summed over background points with a
non-zero contribution, normalised by 4^4 / (|F| + r_i + s_j)^4 and divided by
the number of such points. The perpendicular s = r_perp is implicit in `rotsat`.
"""

from __future__ import annotations

import time

import numpy as np

from convexity import Convexity

from .dataset import BACKGROUND, OBJECT
from .directions import Direction


def q_concavity(img: np.ndarray, direction: Direction) -> tuple[float, float]:
    """Return (E_F^{r,r_perp}, elapsed_seconds) for one binary image.

    `img` must be the {OBJECT, BACKGROUND} array produced by `qsig.dataset`.
    """
    t0 = time.perf_counter()
    res = Convexity(np.asarray(img), verbose=False).compute(OBJECT, BACKGROUND, direction.vec)
    dt = time.perf_counter() - t0
    return float(res["q1"]), dt


def signature_values(img: np.ndarray, dirs) -> tuple[list[float], list[float]]:
    """Descriptor values and per-direction timings for a whole direction set.

    `dirs` must already be in signature order (`directions.by_angle`).
    """
    vals, secs = [], []
    for d in dirs:
        v, s = q_concavity(img, d)
        vals.append(v)
        secs.append(s)
    return vals, secs
