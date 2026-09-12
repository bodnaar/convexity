# Q-convexity and Q-concavity shape descriptors

Reference implementation and research code for shape descriptors built on
**Quadrant-convexity** (Q-convexity), together with the experiment harnesses
used in the papers listed below.

This branch, `main`, holds the **published reference implementation** and
nothing else. It is deliberately small and deliberately unchanged: every later
kernel in this repository is validated bit-for-bit against it. The working code
lives on the branches.

## This branch

    convexity.py      the reference implementation: Q-concavity, enlacement
                      and interlacement, exactly as published
    examples.py       usage on the sample images
    input/            sample images (apple, discs, crosses, bullseye, ...)

From *A Spatial Convexity Descriptor for Object Enlacement* — S. Brunetti,
P. Balázs, P. Bodnár, J. Szűcs, DGCI 2019.
<https://doi.org/10.1007/978-3-030-14085-4_26>

Tagged **`published-reference`**. Treat it as read-only: later work adds
alongside it rather than editing it, so that the correctness of a faster kernel
is always decidable against something that has not moved.

## Branches

| branch | what it is |
|---|---|
| `main` | the published reference implementation above |
| `qsig-harness` | the DGMM 2027 work: fast kernels, a measured cost law, and minimum-cost direction-set selection. Has its own README |
| `qsig-pairs` | active. Journal-paper work: generalising the descriptor from an orthogonal pair `(r, r_perp)` to an arbitrary lattice pair `(r, s)`, plus the normalisation study |

## Tags

| tag | what it marks |
|---|---|
| `published-reference` | the reference implementation as published |
| `dgmm2027-submission` | the code behind the DGMM 2027 paper as submitted, 2026-09-12 |

## Papers

- S. Brunetti, P. Balázs, P. Bodnár, J. Szűcs. *A Spatial Convexity Descriptor
  for Object Enlacement.* DGCI 2019.
  <https://doi.org/10.1007/978-3-030-14085-4_26> — the reference
  implementation on this branch.
- P. Bodnár. *Minimum-Cost Direction Sets for Multidirectional Q-Concavity
  Descriptors.* DGMM 2027, submitted 2026-09-12. — branch `qsig-harness`,
  tag `dgmm2027-submission`.

## Orientation

Q-convexity generalises hv-convexity to any two — or more — lattice
directions. A point `P` and a pair of directions split the grid into four
quadrants; a set is Q-convex when every point whose four quadrants all meet
the set belongs to it. Counting the points that violate this gives a
continuous concavity measure, and evaluating it along several direction pairs
gives a vector descriptor (a *signature*) usable for shape classification.

Two conventions worth knowing before reading any number produced here:

- the quadrants at `P` are **closed** — they include the row and the column
  through `P`;
- the code computes and stores **Q-concavity** (`E`), while the papers report
  **Q-convexity**.
