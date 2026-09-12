# Q-convexity shape descriptors — QSIG-FAST working branch

**Branch `qsig-harness`.** The experiment harness built for the DGMM 2027
paper *Minimum-Cost Direction Sets for Multidirectional Q-Concavity
Descriptors* (single-author, submitted 2026-09-12). Tagged at
`dgmm2027-submission`.

For the repository layout and the other branches, see the README on `main`.

## What this branch adds

`convexity.py` at the repository root is the published reference
implementation and **is not modified here** — it stays the correctness ground
truth. Everything new lives in `qsig/` and `scripts/`.

Three contributions, in the order they depend on each other:

1. **A measured cost law.** The cost of one (shape, direction) evaluation is
   `Theta(mn * |det(r, r_perp)|) = Theta(mn(p^2 + q^2))` for the direct
   evaluation, via Pick's theorem. Directions therefore differ in price by
   more than an order of magnitude, which is what makes choosing them worth
   doing.
2. **A cheaper kernel.** A row-prefix-sum formulation reduces this to
   `Theta(mn(p+q))`, cutting the cheap-to-dear spread from ~18x to ~5x.
3. **Minimum-cost direction selection.** Given a maximum angular gap `G`,
   choose the cheapest feasible direction set: an exact dynamic program /
   shortest path on an angle-sorted DAG, `O(n^2)` in the pool size. Anchoring
   the path at `(1,0)` is without loss for the reported sets — the `O(n^3)`
   anchor-free variant returns identical sets at `G = 10, 8, 6`.

## Layout

    convexity.py            published reference implementation (do not edit)
    examples.py             its usage examples
    input/                  its sample images

    qsig/
      descriptor.py         one (shape, direction) value; selects the impl
      fast.py               int64 + vectorised kernels: "points" and "rows"
      directions.py         the pool, cost models, and the selection DP
      signature.py          signatures and the orbit distance d_C
      rotational.py         family R (rotate the image, evaluate at (1,0))
      classify.py           1NN leave-one-out accuracy
      dataset.py            MPEG-7 loading and the published preprocessing
      store.py              append-only, resumable results table (CSV)
      threadguard.py        pins BLAS threads so timings mean something

    scripts/
      run_pool.py           produce a results table
      evaluate.py           accuracy against modelled cost — the main table
      fit_cost_law.py       fit and rank candidate cost models
      calibrate.py          timing calibration
      diagnose_rotational.py  why family R loses
      fig_rotloss.py        regenerates the rotation-loss figure and its counts

    tests/                  pytest; the reference-equivalence tests are the
                            ones that matter

## Three implementations, and why the choice is recorded

`IMPLEMENTATIONS = ("reference", "fast", "rows")`.

- `reference` — `convexity.Convexity`, exactly as published. Object dtype and a
  per-pixel Python loop. Ground truth, and the baseline for every timing claim.
- `fast` — int64, the interior-point loop replaced by whole-array adds, the DP
  recurrence jitted with numba. **Bit-identical on every integer quantity.**
- `rows` — `fast` with the `Theta(mn(p+q))` row-prefix kernel. Same numbers,
  different cost model — which is the point, so it gets its own tag.

The **cost constants depend on the implementation**, so a results table mixing
implementations would fit a meaningless cost law. Every row records its impl
tag, and `store.py` refuses to resume a table under a different protocol.

## Two descriptor families — never mix them in one signature

- **S**, rotation-free: the shape stays fixed and the descriptor is evaluated
  along a pair of slanted lattice directions.
- **R**, rotational: the image is rotated and the descriptor evaluated at
  `(1,0)`.

These are *different descriptors*, not two ways of computing one. Family R also
has two protocols: `iwcia2025` reproduces the published baseline including its
rounded rotation angle and its rotation centre, and `exact` is the repaired
descriptor. Comparisons in the paper use `exact`, so the paper beats a fixed
baseline rather than a broken one.

## Running it

    pip install -r requirements-qsig.txt
    python scripts/run_pool.py  --data ../MPEG7dataset.zip --subset all \
                                --impl rows --table results/all_n130_rows.csv
    python scripts/evaluate.py  --data ../MPEG7dataset.zip --subset all \
                                --table results/all_n130_rows.csv \
                                --sets sint maxgap10 maxgap8 maxgap6
    pytest -q

Results tables and the MPEG-7 archive are **not vendored** (see
`.gitignore`) — a full pool run is large and machine-specific.

## Things that will bite you

- **Quadrants are CLOSED.** They include the row and the column through `P`.
  A single boundary pixel on the far side of that row is enough to occupy a
  quadrant. This is the mechanism behind family R's resampling noise floor.
- **The code computes and stores Q-CONCAVITY (`E`); the papers report
  Q-convexity.** Check the convention before quoting any stored number.
- **Never report a mean relative error on this descriptor.** Its informative
  range approaches zero, so near-Q-convex shapes (`E ~ 1e-6`) produce
  meaningless ratios. Report the median. A phantom "34.6 % spike at 18.4
  degrees" came from exactly this and does not exist.
- **Costs are modelled, not measured from the table.** A pool run has many
  workers contending for memory bandwidth, which inflates per-job times
  unevenly across directions. Price from `directions.cost_of`; the table's own
  seconds are for diagnosis only.
- **numba is a hard requirement, not a nice-to-have.** Without it the
  direction-independent overhead dominates and the measured cost curve bends
  away from the law.
- **Do not form a hypothesis on the Device subset.** 200 shapes means half a
  point per shape; several conclusions drawn there reversed on the full 1400.
