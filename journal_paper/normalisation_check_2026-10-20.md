# Normalisation check: line family of the local Q-concavity normalisation

2026-10-20. Branch `norm-check` (from `qsig-pairs` at 6e0db73 = tag
`qconv-nonortho-pairs-disj`). `convexity.py`, `qsig/fast.py`, `qsig/pairs.py`
and all tags are untouched. New code: `qsig/normcheck.py`,
`tests/test_normcheck.py`, `experiments/normalisation_check/`.

Note on the brief: it gave the head as 019af02. That object does not exist in
this repository. The tag `qconv-nonortho-pairs-disj` points at 6e0db73, which
is the head of `qsig-pairs`, and this check used that commit. There is no
`.venv-g8`; `.venv` was used.

## Verdict

The suspected inconsistency is real. **The normalisation's line family is the
side that is wrong.** The cones are correct for the direction the signature
labels, provided angles are read in image coordinates (y = row index,
pointing down).

- The cone code (`rotsat`; `fast.rotsat`; `pairs.rotsat`) reads `vec` as
  Cartesian with y pointing up, as its comment says ("v(x,y) is from the
  descartes lower right quad").
- `calcH` reads the same `vec` with y pointing down, as its comment says
  ("v2 = y (row) offset, positive is downwards").

Each function does what its own comment says, but the two comments
contradict each other. In (row, col), the cone generators of
`compute(vec=(v0, v1))` are `(v0, v1)` and `(-v1, v0)`. The lines that
calcH counts run along their transposes, `(v1, v0)` and `(v0, -v1)`. The
same holds for `pairs.line_sums(a, w)` against `pairs.step(w) = -w`.

The two line families coincide only when the direction set is closed under
transposition. Among orthogonal pairs that means (1,0) and (1,±1). Among
general pairs it means pairs like {(3,1),(1,3)}, {(1,-3),(3,-1)}, and every
pair symmetric about the 45° bisector.

The fix belongs in the lines, not the cones. The bound being normalised by,
IWCIA 2017 Property 3 (n0+n1+n2+n3 = |F| + h + v + f), holds only when h and v
are counted on the lines that bound the closed cones. No relabelling makes the
code's lines and cones parallel to the same pair. The IWCIA 2025 remark
("lines parallel to r and to s") states the correct rule; the code does not
implement it.

### Worked example: Direction(2,1), label 26.57°, vec = (2,-1)

The figure is generated from the code itself
(`fast.compute` on single-pixel images, cross-checked against
`convexity.Convexity.calcH`). P is the centre. On the left, each digit is the
number of cones at P that contain Q; a 2 marks a generator line, because the
cones are closed. On the right, `#` marks a pixel that the code's
normalisation counts.

    cones containing Q                code normalisation counts Q
    1 1 1 1 1 1 2 1 1                 . . # . . . . . .
    1 1 1 1 1 1 1 1 1                 . . . . . . . . .
    2 1 1 1 1 2 1 1 1                 . . . # . . . . #
    1 1 2 1 1 1 1 1 1                 . . . . . . # . .
    1 1 1 1 P 1 1 1 1                 . . . . P . . . .
    1 1 1 1 1 1 2 1 1                 . . # . . . . . .
    1 1 1 2 1 1 1 1 2                 # . . . . # . . .
    1 1 1 1 1 1 1 1 1                 . . . . . . . . .
    1 1 2 1 1 1 1 1 1                 . . . . . . # . .

- **Cone boundaries.** In (row, col) they run along (2,-1) and (1,2). On
  screen (x right, y up) those are the lines at -26.57° and 63.43°: Cartesian
  (2,-1) and its perpendicular. In image coordinates (x = col, y = row, down)
  they are (2,1) and (-1,2), i.e. the labelled Direction(2,1) and its
  perpendicular.
- **Normalisation lines.** In (row, col) they run along (1,-2) and (2,1). On
  screen that is +26.57° and 116.57°, the vertical mirror image of the cones.
- For vec = (1,0) both panels show the same row and column through P.

What "direction (p,q)" means in the papers, stated precisely:

- **As computed**, E^(p,q) is Q-concavity for the pair {(p,q), (-q,p)}
  measured with y pointing down (image row axis). Equivalently, it is the pair
  {(p,-q), (q,p)} in y-up Cartesian.
- **If a paper measures angles counter-clockwise on the displayed image**
  (y up), the computed descriptor for label (p,q) belongs to the mirrored
  direction (q,p) at 90° - θ.

This relabelling changes no reported number. S_int and the 64-pool are
symmetric under (p,q) to (q,p), d_C includes reversal, and the cost depends
only on p, q. It only affects how a direction set is described, so the papers
should state the convention.

Uncertainty: I had no paper sources. My reading of IWCIA 2017 / IWCIA 2025
comes from the code comments that quote them (`qsig/pairs.py` docstring) and
from the brief. Check the convention sentence against the actual text.

## Evidence

All of it is reproducible from the new branch.

1. **Reproduction of the brief's snippet.** (1,0) and (1,-1) give True/True;
   (2,-1) and (3,-1) give False/True, as expected.
2. **Property 3, exact integers.** I checked 2,310 random mask × direction
   cases (64 pool directions plus the 12 Farey and 12 wide JMIV pairs).
   - With generator lines the residual is 0 everywhere.
   - With the code's lines, Property 3 fails for 62 of 64 orthogonal
     directions (all except (1,0) and (1,1)) and for 22 of 24 JMIV pairs (all
     except {(3,1),(1,3)} and {(1,-3),(3,-1)}).
   - Pinned in `tests/test_normcheck.py`. The new tests and the existing
     bit-identity suites (`test_pairs.py`, `test_fast_matches_reference.py`)
     pass: 136 tests.
3. **Bound.** With generator lines, AM-GM gives phi_hat <= 1 pointwise and
   phi_D, phi_A <= 1. This is tested, and the maximum over all MPEG-7 shapes
   is exactly 1.0000. **With the code's lines the bound is not guaranteed.**
   - Counterexample: the object is the two generator lines through P, with P
     removed (an "X"). Only P has four non-empty cones. The code's lines miss
     both arms, so the descriptor is **q1 = 16** for every non-axis direction
     tested: (2,-1), (3,-1), (3,-2), (7,-4). With generator lines it is
     exactly 1, so the bound is tight. This is pinned as a test.
   - Random masks reach a pointwise phi_hat of 1.37.
   - On MPEG-7, 150 of 1,400 shapes have some point with phi_hat > 1 (max
     1.2534). The averaged descriptor never exceeded 1 in practice: max phi
     0.728, phi_D 0.423, Farey-pair phi 0.431.
4. **Zero sets and exact Q-convexity flags** do not depend on the
   normalisation. phi, phi_D, phi_A and q0 are untouched and only the
   denominators change. Confirmed on MPEG-7: the zero pattern is identical in
   all nine feature families, for example 4,449 zeros out of 89,600 phi
   values in both variants. The degeneracy rates (phi = 0 per aperture band)
   are therefore unchanged.
5. **Size of the change on MPEG-7** (128 px).
   - Columns (1,0) and (1,1) are bit-identical between the variants.
   - Elsewhere, the median relative change per column is 0.1% (worst column
     0.6%, 95th percentile 2.8%), and the generator/code ratio stays within
     [0.910, 1.056].
   - Spearman rank correlation per column is at least 0.9996.
   - The Farey (narrow) pairs move most: median 1.2%, 95th percentile 10%.

## Recompute: cost and protocol

- **Cost.** Both normalisations come from one cone pass, at about 1.0 s per
  shape on one core (64 orthogonal directions plus 24 pairs). All 1,400
  MPEG-7 shapes took 1 min 41 s on 16 workers. The analysis scripts took
  about 20 minutes.
- **Validation.** The "code" variant reproduces the stored DGMM tables
  `results/{all,device}_n130_rows.csv` exactly: 102,400 values, 0 differ. It
  also reproduces every JMIV 1NN/LDA number transcribed in
  `maketables.tab_matched` for MPEG-7 (e.g. 79.21, 82.57, 82.07, 78.36;
  LDA 56.43 / 64.07 / 65.93 / 69.79 / 69.57 / 64.86).
- **Protocol.** The analysis scripts were rerun unchanged
  (`experiments/normalisation_check/run_jmiv.sh`), using scikit-learn 1.4.2
  and numpy 1.26.4 from `/usr/local/fsl/bin/python3`, since `.venv` has no
  scikit-learn.

## Impact per paper

### IWCIA 2025 (published, co-authored)

- **Rotational family R:** unaffected, since it evaluates only at (1,0).
- **Rotation-free family S:** affected in principle; the published numbers
  came from `convexity.py`.
  - On our reproduction, S_int 1NN d_C moves from 47.86 to 48.00 (all,
    +0.14 [-0.29, +0.64]) and from 73.50 to 73.50 (Device).
  - S_18 / S_9 / S_6 / S_3 were interpolated from S_int values, so they are
    affected at the same order.
- **The text's definition is right.** The Proposition 1 normalisation and
  the remark about lines parallel to r and s are correct. The published code
  deviates from them.
- **Bound claim.** Any claim that the computed descriptor lies in [0, 1]
  holds empirically on MPEG-7 but is not guaranteed by the code.

### DGMM 2027 (submitted 2026-09-12, tag `dgmm2027-submission`)

- **Cost results:** unaffected. That covers the cost law, both kernels, the
  max-gap DP and the selected sets. The corrected line sums cost the same
  O(mn).
- **Accuracies.** I covered 1NN LOO for 19 sets (sint, maxgap12/10/8/6,
  slot2 to slot32, pool64) × {all, Device} × {d_C, L2}, 76 comparisons in
  all.
  - Every difference is between -1.5 and +1.0 points (with d_C, -0.57 to
    +0.57 on all and -1.5 to +1.0 on Device).
  - One 95% interval excludes zero: L2 slot8 (all), -0.57 [-1.14, -0.07].
    That is in line with chance across 76 comparisons.
  - The maxgap sets show no systematic direction. With d_C they read
    maxgap10 -0.29 / -0.50, maxgap8 -0.57 / +1.00, maxgap6 -0.07 / -0.50
    (all / Device).
- **Conclusions:** no ordering or conclusion I could identify changes. That
  includes d_C vs L2 on all vs Device and rotational vs rotation-free.
- **Uncertainty:** I inferred the reported sets from the scripts
  (`evaluate.py`, `diagnose_rotational.py`), not from the paper. Full table
  in `experiments/normalisation_check/analyse_2026-10-20.log`.

### JMIV (submitted 2026-10-16), MPEG-7 only

Format: code → generator; MPEG-7 rows only.

| table / result | code | generator |
|---|---|---|
| tab_headline: 1NN, phi all-64 → phi+phi_D @6 | 75.79 → 82.57, +6.79* [+4.64, +8.93] | 75.57 → 82.71, +7.14* [+4.93, +9.29] |
| tab_headline: LDA | 62.79 → 69.79 | 62.50 → 69.86 |
| tab_replacement: phi_D - phi, slot12 / all64 | -0.07 / -2.14 [-4.64, +0.43] | -0.07 / -1.93 [-4.43, +0.64] |
| tab_degeneracy Δ: narrow / wide / orth | +5.21* / +1.79 / -1.14 | +4.86* / +1.79 / -0.93 |
| tab_degeneracy zero rates | unchanged | unchanged |
| tab_matched 1NN: phi12 / phiD12 / both6 / phi24 / both12 | 79.21 / 79.14 / 82.57 / 78.36 / 82.07 | 79.21 / 79.14 / 82.71 / 78.36 / 82.07 |
| tab_matched LDA: phi12 / phiD12 / both6 / phi24 / both12 | 64.07 / 65.93 / 69.79 / 64.86 / 69.57 | 63.86 / 65.79 / 69.86 / 64.36 / 69.36 |
| tab_matched RF: phi12 / phiD12 / both6 / phi24 / both12 | 80.00 / 80.71 / 84.14 / 78.79 / 83.57 | 79.71 / 80.43 / 83.79 / 78.93 / 83.50 |
| tab_classdisjoint narrow | phi_D (12/25), +3.44, 22/25 wins | both (12/25), +3.28, 21/25 wins |
| tab_classdisjoint wide | +2.17, 20/25 | +1.98, 20/25 |
| tab_classdisjoint orthogonal | +0.66, 13/25 | +0.69, 14/25 |
| tab_fourier: +phi,phi_D @6 over cl+FD | +2.29* [+0.93, +3.64] | +2.07* [+0.71, +3.43] |
| tab_count: spread / best small set / all 64 | 0.41 / 79.79 / 75.79 | 0.27 / 79.64 / 75.57 |
| generalpair_test: phi+phi_D (6+6) minus phi (all 12) | +4.93* | +4.21* |
| asymmetry_test: (phi+phi_A) - (phi+phi_D) | -0.50, indistinguishable | -0.57, indistinguishable |

Changes that could touch wording:

1. **disjmeasure, classical + phi(all 64) over classical:** +2.71
   [+0.00, +5.43]* becomes +2.50 [-0.21, +5.14]. The interval was borderline
   and now includes zero. Check whether the paper states this gain as
   significant.
2. **Modal arm in the class-disjoint narrow band:** it flips from phi_D
   (12/25) to both (12/25). It was a near-tie in both variants: 12 vs 11.
3. **Class-disjoint narrow Device split** (single split): gain +5.83 becomes
   +2.92 (`classdisjoint_general`, `classdisjoint_fourier`).

Every starred headline result stays starred, and every reported direction of
effect is preserved.

**Not recomputed.** Animal2000 and SwedishLeaves: the archives are not in
this checkout and the brief scoped the recompute to MPEG-7. The
`tab_fourier_variants` table was also not rerun: `fourier_variants.py` needs
the raw Animal2000/SwedishLeaves archives. The `aperture_sweep.py` 0° bisector
arm is affected; the 45° arm is transpose-closed and unaffected. The RF
column could not be checked against the original `rf_results.csv`, which is
not in the checkout.

## Options

1. **Nothing.** This is defensible for the numbers: every MPEG-7 change is
   within resampling noise, and no conclusion changes. It is not defensible
   for the bound. The code can produce values above 1 (q1 = 16 on a valid
   input), so any "in [0, 1]" statement is false for the code as published.
2. **Revision notes.** Recommended for the two submitted papers.
   - **JMIV:** switch to generator lines before the first revision,
     recompute all three datasets (minutes of compute), update the tables
     above, and state the coordinate convention. Recheck item 1 above.
   - **DGMM 2027:** at camera-ready, a one-sentence correction plus updated
     accuracies. No conclusion moves, and the cost results are unaffected.
3. **Erratum for IWCIA 2025.** Not warranted for the published numbers: the
   R family is untouched and S_int moves by +0.14 / 0.00. Because the paper's
   definition is correct and only the code deviates, a code-level correction
   is enough. That means a note in the repository README and a corrected
   kernel released beside the unchanged reference. Agree it with the
   co-authors. A formal erratum would be justified only if the paper claims
   the computed values are bounded by 1 for non-axis directions. I could not
   check that without the text.

Whichever option is chosen, the papers should name the coordinate convention
for "direction (p, q)", since the two conventions differ by a mirror image.

## Files

- `qsig/normcheck.py`: both normalisations from one cone pass;
  `generator_line_sums`; Property 3 residual.
- `tests/test_normcheck.py`: Property 3, the transpose-closure rule, the
  q1 = 16 counterexample, identical zero sets, and the code variant equal to
  fast/pairs.
- `experiments/normalisation_check/extract.py`: MPEG-7 features for both
  variants, in the CSV layout of `pairs_and_rules`.
- `experiments/normalisation_check/analyse.py`: validation, bound, zero sets,
  DGMM / IWCIA accuracies. Output: `analyse_2026-10-20.log`.
- `experiments/normalisation_check/run_jmiv.sh`: reruns the
  `pairs_and_rules` analysis scripts unchanged on each variant.
