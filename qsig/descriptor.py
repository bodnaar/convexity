"""One Q-concavity value for one (shape, direction) pair.

Two implementations are selectable, and which one produced a number is part of
the number (handoff sec 3.1.1):

  "reference"  convexity.Convexity, exactly as published. Object dtype, the
               per-pixel interior-point loop in Python. This is the ground
               truth for correctness and the baseline for every timing claim.

  "fast"       qsig.fast. int64, the interior-point loop replaced by whole-array
               adds, the dynamic-programming recurrence jitted when numba is
               available. Bit-identical on every integer quantity; the final
               scalar agrees to within an ulp (see qsig.fast for why exact
               equality is the wrong assertion there).

Both are wired through the same entry point so an experiment can be run twice
and compared, rather than the choice being buried in an import.

TIMING WARNING. Under "fast", numba compiles `_dp` on its first call, which
costs about a second. A timed run must call `warm_up()` first, and every worker
process in a Pool must do the same before its first timed job -- otherwise the
compile lands inside one measurement and corrupts it.
"""

from __future__ import annotations

import time

import numpy as np

from convexity import Convexity

from . import fast as _fast
from .dataset import BACKGROUND, OBJECT
from .directions import Direction

IMPLEMENTATIONS = ("reference", "fast")


def warm_up(impl: str = "fast") -> None:
    """Trigger JIT compilation so it cannot land inside a timed run."""
    if impl != "fast":
        return
    img = np.zeros((8, 8), dtype=np.uint8)
    img[2:6, 2:6] = OBJECT
    img[3, 3] = BACKGROUND
    for vec in ((1, 0), (3, -1)):
        _fast.compute(img, OBJECT, BACKGROUND, vec)


def q_concavity(img: np.ndarray, direction: Direction,
                impl: str = "reference") -> tuple[float, float]:
    """Return (E_F^{r,r_perp}, elapsed_seconds) for one binary image."""
    if impl not in IMPLEMENTATIONS:
        raise ValueError(f"impl must be one of {IMPLEMENTATIONS}, got {impl!r}")
    arr = np.asarray(img)
    t0 = time.perf_counter()
    if impl == "reference":
        value = Convexity(arr, verbose=False).compute(OBJECT, BACKGROUND, direction.vec)["q1"]
    else:
        value = _fast.compute(arr, OBJECT, BACKGROUND, direction.vec)["q1"]
    dt = time.perf_counter() - t0
    return float(value), dt


def signature_values(img: np.ndarray, dirs, impl: str = "reference"):
    """Descriptor values and per-direction timings for a whole direction set.

    `dirs` must already be in signature order (`directions.by_angle`).
    """
    vals, secs = [], []
    for d in dirs:
        v, s = q_concavity(img, d, impl)
        vals.append(v)
        secs.append(s)
    return vals, secs


def implementation_tag(impl: str) -> str:
    """Short label to record with results, e.g. 'fast+numba'."""
    if impl == "reference":
        return "reference"
    return "fast+numba" if _fast.HAVE_NUMBA else "fast-nonumba"
