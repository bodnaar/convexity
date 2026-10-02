"""Locations used by the experiment scripts in this directory.

Datasets are not vendored. Put the archives in ./data/ (next to this file), or
point QCONV_DATA at the directory that holds them:

  MPEG7dataset.zip       https://dabi.temple.edu/external/shape/MPEG7/dataset.html
  Animal2000-GT.zip      https://github.com/cong-yang/skeview
  SwedishLeaves-GT.zip   https://github.com/cong-yang/skeview

Feature CSVs are written to and read from this directory. Generated LaTeX
tables go to ./tab and pgfplots data files to ./fig, unless QCONV_TAB or
QCONV_FIG point elsewhere.

Importing this module also puts the repository root (for `qsig`) and this
directory on sys.path. It imports nothing but os and sys, so it is safe to
import before `qsig.threadguard`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = os.environ.get("QCONV_DATA", os.path.join(HERE, "data"))
TAB_DIR = os.environ.get("QCONV_TAB", os.path.join(HERE, "tab"))
FIG_DIR = os.environ.get("QCONV_FIG", os.path.join(HERE, "fig"))


def data(fname: str) -> str:
    """Full path of a dataset archive, or a clear error if it is missing."""
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{fname} not found in {DATA_DIR}; download it (see paths.py) "
            f"or set QCONV_DATA")
    return p
