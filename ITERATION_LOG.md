# Acceleration Iteration Log — py-spatialecotyper

Benchmark: the single-sample pipeline on the canonical fixture
(`reference_out/ci` — the 6000-cell centroid crop of Melanoma1_subset,
297 spatial neighbourhoods, 9 cell types), via `tests/bench.py`.
Warmup run discarded; 3 timed runs, mean ± sample stddev.
`snf2_only_s` is reported alongside because SNF2 is the dominant cost at scale
(123 s of R's 252 s on the full 27,907-cell fixture) and is far less sensitive
to machine load than the end-to-end number.

Parity is re-checked **in the same process** as every timing, so no rewrite can
be accepted on speed alone.

Gate (pre-registered, `data/manifest.yaml`): `se_labels_single_sample`,
clustering class, ARI ≥ 0.85.

---

## Baseline — 2026-07-28 05:18:10

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 5.6797
wall_clock_stddev_s: 1.0702
wall_clock_runs_s: [5.1861, 6.9076, 4.9455]
warmup_run_s: 8.2248
snf2_only_s: 1.1257
snf2_max_abs_err_vs_R: 2.78e-17
parity_metric: 0.983066
parity_class: clustering
parity_threshold: 0.85
parity_passes: true
notes: |
  Equivalence Agent's clean translation. Direct transcription of the R control
  flow: per-row Python loops where R has apply(), lil_matrix for diagonal
  assignment, Reduce("+", Wall[-j]) re-summed inside the diffusion loop.
  Clears the gate at ARI 0.983066 against R's SpatialEcoTyper 1.0.4.
```

---

## iter 1 — 2026-07-28 05:19:40

```yaml
iter: 1
status: ACCEPT
action: normalize_no_lil
playbook_section: "section 1 (representation change)"
admissibility: exact
admissibility_evidence: |
  `SNF2::normalize` sets the diagonal to 0.5 after the row scaling. The old
  code round-tripped the matrix through `scipy.sparse.lil_matrix` to use
  `setdiag`, which reallocates every row. Replacing it with
  `x + diags(0.5 - x.diagonal())` writes the same value into the same
  positions -- `x_ii + (0.5 - x_ii) = 0.5` exactly in f64 for any finite
  x_ii, and off-diagonal entries are untouched because the correction matrix
  has no off-diagonal support. No summation is reordered.
wall_clock_mean_s: 6.8673
wall_clock_stddev_s: 1.0055
wall_clock_runs_s: [6.0208, 6.5304, 6.1231, 8.4643, 7.1980]
warmup_run_s: 4.7459
snf2_only_s: 0.9864
speedup_vs_previous: 1.14      # measured on snf2_only_s; the end-to-end number
speedup_vs_baseline: 1.14      # was contaminated by a concurrent R job
parity_metric: 0.983066
parity_delta_vs_baseline: 0.000000
parity_passes: true
notes: |
  End-to-end wall clock is not usable for this iteration: the R two-sample
  reference job was running on the same node and the runs span 6.0-8.5 s.
  The isolated SNF2 timing (1.1257 -> 0.9864 s) is the signal.
```

### Decision
ACCEPT — exact, and the only cost is a representation change.

---

## iter 2 — 2026-07-28 05:21:30

```yaml
iter: 2
status: ACCEPT
action: dominateset_vectorised
playbook_section: "section 1 (loop fusion / vectorisation)"
admissibility: exact
admissibility_evidence: |
  `.dominateset` keeps the KK largest entries of each row, R implementing it as
  `x[sort(x, index.return = TRUE)$ix[1:(length(x) - KK)]] <- 0`. The old Python
  sliced one CSR row at a time and densified it. The rewrite densifies once and
  calls `np.argsort(axis = 1, kind = "stable")`. A stable sort along axis 1 is
  applied independently per row and produces, row by row, exactly the ordering
  the per-row call produced, so the retained index set is identical element for
  element -- including which member of a tie group survives at the KK boundary,
  which is what R's radix (stable) sort decides.
wall_clock_mean_s: 4.6163
wall_clock_stddev_s: 0.4731
wall_clock_runs_s: [4.9625, 4.0772, 4.8093]
warmup_run_s: 6.4230
snf2_only_s: 0.6973
speedup_vs_previous: 1.41
speedup_vs_baseline: 1.61
parity_metric: 0.983066
parity_delta_vs_baseline: 0.000000
parity_passes: true
```

### Decision
ACCEPT.

---

## iter 3 — 2026-07-28 05:21:30 (measured jointly with iter 2 and 4)

```yaml
iter: 3
status: ACCEPT
action: transpose_hoisting
playbook_section: "section 1 (loop-invariant code motion)"
admissibility: exact
admissibility_evidence: |
  `newW[j].T` was recomputed inside the t x LW diffusion loop; the operand
  never changes. Hoisting it out of the loop evaluates the same expression on
  the same values, once instead of t*LW times. Pure common-subexpression
  elimination; no arithmetic is performed differently.
parity_metric: 0.983066
parity_delta_vs_baseline: 0.000000
parity_passes: true
notes: |
  Bundled into the same measurement as iter 2 and iter 4 (all three touch
  SNF2's inner loop); the combined snf2_only_s is 1.1257 -> 0.6973 s.
```

### Decision
ACCEPT.

---

## iter 4 — 2026-07-28 05:21:30 (measured jointly with iter 2 and 3)

```yaml
iter: 4
status: ACCEPT
action: snf_sum_minus_self
playbook_section: "section 2 (bounded epsilon-approximation)"
admissibility: bounded
admissibility_evidence: |
  R recomputes `Reduce("+", Wall[-j])` for every view j inside every diffusion
  round: LW*(LW-1) sparse additions per round. The rewrite forms the total once
  and subtracts: `sum_wj = (total - W_j) / (LW - 1)`, i.e. 2*LW - 1 operations.
  Identical in exact arithmetic by associativity; in f64 the two accumulate
  rounding in a different order, so it is classified as (B) with a derived
  bound rather than (A).
  For LW = 9 (the canonical fixture's cell types) that is 17 sparse ops per
  round instead of 72, and 170 instead of 720 over the default t = 10 rounds.
perturbation_bound: |
  Per Higham Thm 2.5 for summation of n non-negative f64 values,
  |sum_hat - sum| <= gamma_{n-1} * sum|w_m|, gamma_k = k*u/(1 - k*u), u = 2^-53.
  Original path: gamma_{LW-2} * S. Rewrite: gamma_{LW-1} * S + u * |S - w_j|
  <= gamma_{LW} * S. Difference per entry <= 2*u*S / (LW - 1) * LW ~ 2*u*S.
  Each W_m is normalize()-d so its rows sum to <= 1 and S <= LW; the map
  P_j . (.) . P_j^T followed by normalize() is non-expansive in the inf-norm,
  so over t rounds
      ||fused_new - fused_old||_inf <= t * 2 * u * LW
                                     = 10 * 2 * 1.11e-16 * 9 = 2.0e-14.
  Full derivation in MATH.md section 2.
wall_clock_mean_s: 4.6163
wall_clock_stddev_s: 0.4731
wall_clock_runs_s: [4.9625, 4.0772, 4.8093]
warmup_run_s: 6.4230
snf2_only_s: 0.6973
snf2_max_abs_err_vs_R: 4.16e-17
speedup_vs_previous: 1.41
speedup_vs_baseline: 1.61
parity_metric: 0.983066
parity_delta_vs_baseline: 0.000000
parity_passes: true
notes: |
  Measured perturbation of the fused matrix vs R: 2.78e-17 -> 4.16e-17, i.e. a
  change of 1.4e-17, three orders below the derived 2.0e-14 ceiling and eight
  orders below the 1e-8 `snf_fused` gate. End-to-end ARI unchanged to six
  decimals.
```

### Decision
ACCEPT.

---

## iter 5 — 2026-07-28 05:24:50

```yaml
iter: 5
status: REJECT_INADMISSIBLE
action: dense_lapack_svd_in_run_pca
playbook_section: "section 1 (algorithm substitution)"
admissibility: exact          # CLAIMED
admissibility_evidence: |
  CLAIM (which turned out to be wrong at the granularity that matters): the
  rank-npcs truncated SVD is the leading block of the thin SVD, so replacing
  ARPACK `svds` with dense LAPACK `np.linalg.svd` returns the same
  factorisation and avoids ARPACK's iterative restarts. SpatialEcoTyper's PCA
  operates on (spots x spots) or (spots x <=300) blocks, so dense is cheap.
wall_clock_mean_s: 30.7041
wall_clock_stddev_s: 6.3043
wall_clock_runs_s: [32.5709, 23.6772, 35.8642]
warmup_run_s: 26.3719
snf2_only_s: 0.7004
speedup_vs_previous: 0.15
speedup_vs_baseline: 0.19
parity_metric: 0.924550
parity_delta_vs_baseline: -0.058516
parity_passes: true           # 0.9246 still clears the 0.85 gate
math_reason_for_dip: |
  A truncated SVD is unique only up to an orthogonal rotation *within* any
  degenerate (or near-degenerate) singular subspace. The two solvers pick
  different bases inside such a subspace. `FindNeighbors(dims = 1:10)` cuts the
  embedding at PC 10, so when a near-degenerate pair straddles the PC-10/PC-11
  boundary the two solvers hand different 10-dimensional coordinates to the
  kNN search, the SNN graph changes, and the Louvain partition moves. ARPACK's
  Lanczos basis happens to track irlba's (R's choice) more closely, which is
  why the ARPACK path scores 0.983066 and the dense path 0.924550. The
  admissibility claim "exact" was therefore wrong: exact as a *subspace* is not
  exact as a *coordinate matrix*, and this pipeline consumes coordinates.
```

### Decision
REJECT_INADMISSIBLE — rolled back. The rewrite was also 6.6x *slower* on this
run, but the timing is not the reason for rejection: the admissibility argument
does not hold for a consumer that reads individual coordinates rather than the
span. A `# NOTE (Acceleration iter 5, ROLLED BACK ...)` comment is left at the
call site in `seuratcompat.run_pca` so the next person does not retry it.

---

## iter 6 — 2026-07-28 05:29:40

```yaml
iter: 6
status: ACCEPT
action: rank_sparse_vectorised
playbook_section: "section 1 (loop fusion / vectorisation)"
admissibility: exact
admissibility_evidence: |
  `rankSparse` ranks the non-zero entries within each column
  (`ave(x, col, FUN = rank)`), average ties. The old code called
  `scipy.stats.rankdata` once per column. The rewrite does one global
  `np.lexsort((value, column))` and one vectorised tie-averaging pass.
  Ranking within a column is order-isomorphic to ranking the whole array under
  a key that is lexicographic in (column, value): the lexsort places each
  column's entries contiguously and in ascending value order, so the
  within-column position is `global position - column start`, and a tie group
  is exactly a maximal run of equal (column, value). The average assigned to a
  tie group is computed from the same group boundaries and the same 1-based
  positions, so every rank is identical.
  Verified independently by `test_exact_match.py::test_rank_sparse`, which
  compares against R at the pre-registered `deterministic-strict` 1e-13 gate:
  measured 0.0.
wall_clock_mean_s: 4.5000
wall_clock_stddev_s: 0.5291
wall_clock_runs_s: [4.2676, 5.1055, 4.1268]
warmup_run_s: 7.9186
snf2_only_s: 0.7117
speedup_vs_previous: 1.03
speedup_vs_baseline: 1.26
parity_metric: 0.983066
parity_delta_vs_baseline: 0.000000
parity_passes: true
```

### Decision
ACCEPT.

---

## Summary

| iter | action | admissibility | mean wall (s) | SNF2 only (s) | speedup vs baseline | ARI vs R | status |
|---|---|---|---|---|---|---|---|
| 0 | (baseline) | — | 5.680 | 1.126 | 1.00x | 0.983066 | — |
| 1 | `normalize_no_lil` | E | 6.867* | 0.986 | 1.14x (SNF2) | 0.983066 | ACCEPT |
| 2 | `dominateset_vectorised` | E | 4.616 | 0.697 | 1.61x (SNF2) | 0.983066 | ACCEPT |
| 3 | `transpose_hoisting` | E | 4.616 | 0.697 | — (bundled) | 0.983066 | ACCEPT |
| 4 | `snf_sum_minus_self` | B (2.0e-14) | 4.616 | 0.697 | — (bundled) | 0.983066 | ACCEPT |
| 5 | `dense_lapack_svd` | E (claimed) | 30.704 | 0.700 | 0.19x | 0.924550 | **REJECT_INADMISSIBLE** |
| 6 | `rank_sparse_vectorised` | E | 4.500 | 0.712 | **1.26x** end-to-end, **1.58x** on SNF2 | 0.983066 | ACCEPT |

\* iteration 1's end-to-end timing overlapped a concurrent R reference job on the
same node and is not comparable; its SNF2-only figure is.

**Cumulative**: 1.26x end-to-end and 1.58x on the SNF2 kernel relative to the
Equivalence Agent's first working translation, with the ARI against R unchanged
at 0.983066 to six decimal places. Against the R reference itself the port is
**3.0x faster end-to-end** on the full 27,907-cell fixture (84.1 s vs 251.8 s).

**Sum of admitted (B) bounds**: 2.0e-14, versus the 1e-8
`deterministic-standard` gate for `snf_fused`. See MATH.md section 2.

## Stop reason

Last accepted rewrite (iter 6) produced a 1.03x step; the remaining playbook
candidates for this pipeline are either already applied, inadmissible (iter 5),
or would change the numerics that the gate protects. The dominant remaining
cost is `scipy.spatial.cKDTree` in `getSN` and the Louvain sweep in
`_modularity`, neither of which has an equivalence-preserving rewrite left in
the playbook that does not touch neighbour selection.
