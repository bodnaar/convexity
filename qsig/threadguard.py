"""Pin every numeric library to one thread.

MUST be imported before numpy, anywhere in the codebase. Setting these
variables after numpy is imported has no effect.

Rationale (handoff sec. 7.4 item 1): the experiment runs `Pool(20)` on a
20-physical-core machine with SMT disabled. If each worker also spawns its own
BLAS thread pool the machine oversubscribes ~20x, which is slower than serial
*and* makes every per-job timing meaningless. Timing is the result in this
paper, so this is a correctness issue, not a tuning one.

Usage:
    import qsig.threadguard  # noqa: F401  -- first import in any entry point
    import numpy as np
"""

import os
import sys

_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _v in _VARS:
    os.environ.setdefault(_v, "1")

if "numpy" in sys.modules:  # pragma: no cover - defensive
    raise RuntimeError(
        "qsig.threadguard was imported after numpy. The thread limits have no "
        "effect now. Move `import qsig.threadguard` above the numpy import."
    )


def report():
    """Return the guard state, for recording alongside results."""
    return {v: os.environ.get(v) for v in _VARS}
