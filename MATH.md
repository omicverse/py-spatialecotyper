# MATH.md — derivations behind the port

Two kinds of claim live here:

1. **Convexity arguments** that explain why parts of this port are reproducible
   even though the R original draws from an unseeded RNG (section 1).
2. **Perturbation bounds** for every Acceleration rewrite that is not an exact
   identity, as required by `ACCELERATION_PLAYBOOK.md` (section 2).

---

## 1. Why `NMFGenerateW` and `NMFpredict` are reproducible without a seed

`NMFGenerateW` (`R/NMFGenerateW.R`) installs a custom `NMF::NMFStrategy` whose
`Update` function calls only `std.divergence.update.w`. `H` is seeded with the
binary SE-membership matrix `FracsF + eps` and never updated. Symmetrically,
`.nmf.predict` (`R/NMFpredict.R`) updates only `H` and holds `W` fixed.

**Claim.** With `H` fixed, the generalised KL objective

$$D(V \,\|\, WH) \;=\; \sum_{ij}\Big( V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij}\Big)$$

is convex in `W` on the non-negative orthant, so the Lee–Seung multiplicative
updates converge to a global minimiser and the random initialisation `W_0`
affects only how far along that path a finite number of iterations gets.

**Proof sketch.** `(WH)_{ij} = \sum_a W_{ia}H_{aj}` is *linear* in `W`. The
scalar map `t \mapsto v\log(v/t) - v + t` is convex on `t > 0` (second
derivative `v/t^2 > 0`). A convex function composed with an affine map is
convex, and a non-negative sum of convex functions is convex. The feasible set
`W \ge 0` is convex. Hence the problem is convex. The same argument with the
roles of `W` and `H` exchanged covers `.nmf.predict`. ∎

**Empirical confirmation.** `tests/r_nmf_driver.R` runs `NMFGenerateW` twice
from deliberately different RNG states (`set.seed(11)` and `set.seed(999)`);
the R package sets no seed of its own before `NMF::rnmf`, so `W_0` genuinely
differs between the two runs. Measured on the canonical fixture
(`reference_out/nmf/nmf_W_R_vs_R.json`):

| quantity | value |
|---|---|
| selected-feature Jaccard between the two R runs | 1.0000 |
| `max abs(W_run1 - W_run2)` on common features | 8.88e-16 |
| Pearson correlation of the two `W` matrices | 1.000000 |

So the fixed point is a property of the data, not of the seed. This is what
makes the `nmf_W` gate meaningful at all — without it, the only honest gate
would be distributional.

**Caveat, measured.** The *feature-selection* step downstream of the
factorisation (`delta = W + apply(W, 1, function(x) sort(-x)[2])`, keep
`delta > 0`) is a threshold on a quantity that can sit arbitrarily close to
zero. On an earlier, smaller fixture the two R runs kept 212 vs 218 features
(Jaccard 0.92) despite `W` itself agreeing to 1e-15. The selected feature set
is therefore **not** a stable output of the R code, and the port's gate is
applied to `W` on the common features, not to the feature set.

---

## 2. Perturbation bound for the SNF diffusion rewrite (Acceleration iter 4)

### The rewrite

`SNF2` (`R/SNF2.R`) computes, for each view `j` and each diffusion round,

```r
sumWJ <- Reduce("+", Wall[-j]) / (LW - 1)
```

i.e. it re-sums the other `LW - 1` views from scratch, `LW` times per round —
`LW(LW-1)` sparse additions. The port instead computes the total once and
subtracts:

```python
total  = sum_m W_m                       # LW - 1 additions
sum_wj = (total - W_j) / (LW - 1)        # 1 subtraction per view
```

`2LW - 1` operations instead of `LW(LW-1)`. For the 9 cell types in the
canonical melanoma fixture that is 17 versus 72 sparse matrix operations per
round, and 170 versus 720 over the default 10 rounds.

### Admissibility

In exact arithmetic `\sum_{m \ne j} W_m = \big(\sum_m W_m\big) - W_j`, so the
rewrite is an **exact algebraic identity**. In f64 it is a **bounded
ε-approximation**, because floating-point addition is not associative and the
two expressions accumulate rounding in a different order.

### Bound

Fix an entry `(p,q)` and write `w_m = (W_m)_{pq} \ge 0`, `S = \sum_m w_m`.
Summation of `n` non-negative f64 values in any fixed order satisfies the
standard bound (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 2.5)

$$\big|\widehat{\textstyle\sum} - \textstyle\sum\big| \;\le\; \gamma_{n-1}\sum_m |w_m|, \qquad \gamma_k = \frac{k u}{1-ku},\; u = 2^{-53}.$$

*Original*: sums `LW - 1` terms, error `\le \gamma_{LW-2}\, S`.
*Rewrite*: sums `LW` terms then performs one subtraction, error
`\le \gamma_{LW-1} S + u\,|S - w_j| \le \gamma_{LW} S`.

Therefore

$$\big|\text{sum\_wj}^{\text{new}}_{pq} - \text{sum\_wj}^{\text{old}}_{pq}\big| \;\le\; \frac{\gamma_{LW-2} + \gamma_{LW}}{LW-1}\, S_{pq} \;\le\; \frac{2\,LW\,u}{LW-1}\,S_{pq} \;\approx\; 2u\,S_{pq}.$$

Each `W_m` is the output of `normalize()`, whose rows sum to at most 1, so
`S_{pq} \le LW` and the per-entry perturbation entering one diffusion round is
at most `2 u LW ≈ 2.0e-15` for `LW = 9`.

Each round then applies `x = P_j · sum_wj · P_j^T` followed by `normalize()`.
`P_j` is row-substochastic after `dominateset` + `normalize`, so
`\|P_j\|_\infty \le 1` and the map is non-expansive in the `\infty`-norm; the
perturbation cannot grow across the `t = 10` rounds beyond a factor `t`:

$$\big\|\text{fused}^{\text{new}} - \text{fused}^{\text{old}}\big\|_\infty \;\le\; t \cdot 2u\,LW \;=\; 10 \cdot 2 \cdot 1.11\text{e-}16 \cdot 9 \;=\; 2.0\text{e-}14.$$

### Measured

Against the R reference on the canonical fixture (`tests/bench.py`):

| version | `max abs(fused_py - fused_R)` |
|---|---|
| before the rewrite | 2.78e-17 |
| after the rewrite | 4.16e-17 |

The observed change (1.4e-17) is three orders of magnitude below the derived
ceiling of 2.0e-14, and both are eight orders below the pre-registered
`snf_fused` gate of 1e-8. The end-to-end `se_labels_single_sample` ARI is
unchanged at 0.983066 to six decimals.

### Sum of all bounded rewrites

Only iteration 4 is a (B) bounded rewrite; iterations 2, 3, 5 and 6 are exact
(vectorisation, representation change, and truncated-vs-thin SVD, all of which
apply the same operation to the same values). So

$$\Sigma\ \text{bound} \;=\; 2.0\text{e-}14 \;\ll\; 1\text{e-}8 = \text{the } \texttt{deterministic-standard} \text{ gate}.$$

---

## 3. Why `matrixMultiply`'s blocking is bit-exact

`matrixMultiply(mat1, mat2, minibatch)` splits `mat2` **by column** and
concatenates the products. Entry `(i, q)` of the result is
`\sum_k (\text{mat1})_{ik} (\text{mat2})_{kq}` — a sum over `k`, the *shared*
index, which the blocking never touches. Splitting the `q` axis therefore
partitions the output entries without reordering any summation, and the result
is bit-identical for every `minibatch`. Verified in
`tests/test_exact_match.py::test_matrix_multiply` at `minibatch` 17, 64 and
5000: `max abs` difference 0.0.

---

## 4. Moran's I under row-standardised weights

`ComputeNormalizedMoranI` calls `spdep::moran(x, listw, n, S0)` with
`listw <- nb2listw(nb, style = "W")`. Under `style = "W"` every row of the
weight matrix sums to 1, so `S0 = \sum_{ij} w_{ij} = n` and Moran's I collapses
to a quadratic form:

$$I = \frac{n}{S_0}\cdot\frac{z^\top W z}{z^\top z} = \frac{z^\top W z}{z^\top z}, \qquad z = x - \bar{x}.$$

That closed form is what `pyspatialecotyper.stats._moran` evaluates, which is
why the port needs no `sf`/`GEOS`/`PROJ` stack. Validated against real
`spdep` 1.4.2 (`max abs` error 6.66e-16 on the observed statistic;
`spdep::Szero` returned exactly 4000 for the 4000-cell graph, confirming
`S0 = n`).
