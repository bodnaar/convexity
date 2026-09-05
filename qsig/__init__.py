"""qsig -- cost-aware lattice direction sets for Q-concavity signatures.

Experiment layer for the DGMM 2027 paper, built on top of the existing
`convexity.Convexity` reference implementation (DGCI 2019 / IWCIA 2025).

Entry points MUST import `qsig.threadguard` before numpy. See its docstring.
"""

__all__ = [
    "threadguard",
    "directions",
    "dataset",
    "descriptor",
    "signature",
    "store",
    "classify",
]
