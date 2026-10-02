# Direction pairs and combination rules — experiment scripts

Scripts behind the experiments on general (non-orthogonal) direction pairs and
the conjunctive / disjunctive combination rules (`phi`, `phi_D`, `phi_A`). They
use the `qsig` package at the repository root; `paths.py` puts it on `sys.path`.

## Data

Not vendored. Download the archives into `./data/`, or set `QCONV_DATA` to the
directory that holds them:

| file | source |
|---|---|
| `MPEG7dataset.zip` | MPEG-7 CE Shape-1 Part B, https://dabi.temple.edu/external/shape/MPEG7/dataset.html |
| `Animal2000-GT.zip` | SkeView ground-truth collection, https://github.com/cong-yang/skeview |
| `SwedishLeaves-GT.zip` | SkeView ground-truth collection, https://github.com/cong-yang/skeview |

SkeView `.mat` masks use 0 = object, 1 = background; `extract_features.load_mask`
inverts them.

## Order of runs

All scripts are run from any directory as `python3 <script>`; CSVs are written
next to the scripts. Extraction is resumable.

1. Features (classical descriptors and the 64-direction orthogonal signature):
   `extract_mpeg7_full.py`;
   `extract_features.py animal2000 data/Animal2000-GT.zip animal2000_features.csv`;
   `extract_features.py swedishleaves data/SwedishLeaves-GT.zip swedishleaves_features.csv`.
2. Combination rules on the same 64 orthogonal directions, per dataset
   (`mpeg7`, `animal2000`, `swedishleaves`): `extract_disjunctive.py`,
   `extract_asymmetry.py`.
3. General pairs, per dataset: `extract_generalpairs.py` (narrow band, Farey
   pairs from `general_pairs.py`), `extract_widepairs.py` (wide band, pair list
   in `wide_pairs.json`).
4. Analyses: `analyze.py`, `disjmeasure.py`, `disjdim.py`, `disjhead.py`,
   `classdisjoint.py`, `classdisjoint2.py`, `classdisjoint_general.py`,
   `rfcheck.py` (writes `rf_results.csv`), `residue.py`, `asymmetry_test.py`,
   `generalpair_test.py`, `aperture_confound.py`.
5. Tables and figure data: `rfcheck.py` first, then `maketables.py`,
   `make_tab_replacement.py`, `make_tab_headline.py` (LaTeX into `./tab`) and
   `makefigdata.py` (pgfplots `.dat` into `./fig`).

`extract_mpeg7_overlap.py` produces the MPEG-7 subset used for the
Animal2000 provenance spot-check. `greedy_fast.py` is a helper for greedy
forward selection.

## Fixed inputs

- `slotcols.pkl` — for each `k`, the column indices (into the angle-sorted
  64-direction pool) of the `slot_set(k)` direction set. For `k` up to 24 it
  equals `qsig.directions.slot_set(k, 45/k, candidates=pool)`; the `k = 32`
  entry is stored as used.
- `wide_pairs.json` — the 12 wide-aperture (53–79°) pairs, with aperture,
  bisector, `|det|` and largest coordinate.

## Notes

- 1NN is leave-one-out on z-scored features; confidence intervals are paired
  bootstrap over shapes, 4000 resamples. LDA and random forest use 5-fold
  stratified CV.
- `maketables.py` computes every table from the CSVs except the 1NN and LDA
  columns of `tab_matched`, which are transcribed from the output of
  `disjmeasure.py`, `disjdim.py` and `disjhead.py`.
- `import qsig.threadguard` (timing runs only) must come before `numpy`.
