# Discovery — `SpatialEcoTyper` → `py-spatialecotyper`

> Omicverse-RebuildR **Step 0**. Committed before any algorithmic Python code.
> Generated with `python -m engine.discover_omicverse_deps` (kit v7) on 2026-07-28,
> against the 95-repo `github.com/omicverse` listing.

## 1. Is the target already ported?

```
$ python -m engine.discover_omicverse_deps --check SpatialEcoTyper
## Discovery — `SpatialEcoTyper`
**No existing omicverse port found.** Safe to start a new port.
```

No sister port exists on `github.com/omicverse`, on PyPI (`pyspatialecotyper`,
`py-spatialecotyper`, `spatialecotyper` all return HTTP 404), or anywhere in the
Python spatial-omics ecosystem. Spatial EcoTyper (Zhang *et al.*, *Nature* 2026,
doi:10.1038/s41586-026-10452-4) is R + Seurat only. **Proceed with the port.**

Motivating request: omicverse issue #760 (spatial-ecotype analysis).

## 2. Upstream R dependency audit

Upstream `DESCRIPTION` (SpatialEcoTyper 1.0.4):

* **Depends**: R (>= 4.2.0), Seurat (>= 4.2.0), Matrix (>= 1.5), RANN (>= 2.6), parallel, NMF (>= 0.25), ggplot2, pals
* **Imports**: tidyr, dplyr, circlize (>= 0.4), grDevices, ComplexHeatmap (>= 2.12), spdep (>= 1.4.2), RColorBrewer, data.table (>= 1.14)

| R dep | omicverse mirror | Decision | Rationale |
|---|---|---|---|
| `NMF` (>= 0.25) | [`omicverse/rust-NMF`](https://github.com/omicverse/rust-NMF) → PyPI **`nmf-rs` 0.1.0** | **hard dep** | `nmf-rs` is bit-equivalent to R `NMF` brunet/lee multiplicative updates within 1e-12 and exposes exactly the two primitives SpatialEcoTyper's custom `NMFStrategy` objects need: `update_w_brunet` (W-only update, used by `NMFGenerateW`) and `update_h_brunet` (H-only update, used by `.nmf.predict`). Reusing it removes the need to re-derive and re-validate the KL multiplicative update kernels. A pure-NumPy fallback (`pyspatialecotyper.nmf._np_update_{w,h}_brunet`) is retained so the package still imports and passes its gate if the Rust wheel is unavailable on an exotic platform. |
| `Seurat` (>= 4.2.0) | [`omicverse/py-cca`](https://github.com/omicverse/py-cca) | **not reused** | py-cca ports `RunCCA` only. SpatialEcoTyper uses Seurat for `NormalizeData`, `ScaleData`, `FindVariableFeatures(selection.method="dispersion")`, `RunPCA`, `FindNeighbors`, `FindClusters`, `AverageExpression` — a disjoint surface. Ported natively in `pyspatialecotyper.seuratcompat` (see §3). |
| `Matrix` (>= 1.5) | [`omicverse/anndata-oom`](https://github.com/omicverse/anndata-oom) | **not reused** | anndata-oom is an out-of-memory AnnData backend, not a sparse-linear-algebra library. `scipy.sparse` is the direct equivalent (`dgCMatrix` ⇄ CSC). |
| `RANN` (>= 2.6) | — | native | `RANN::nn2` is an ANN kd-tree; `scipy.spatial.cKDTree.query` is exact and returns the same neighbour sets/distances for the Euclidean metric used here. |
| `parallel` | — | native | `joblib` / `concurrent.futures`; parallelism is orthogonal to numerics. |
| `NMF` runtime helpers (`posneg`, `rnmf`, `cophcor`, `NMFfitX`, connectivity-stop) | — | native | Not exposed by `nmf-rs`; ported in `pyspatialecotyper.nmf` and gated. |
| `dplyr`, `tidyr`, `data.table` | — | native | `pandas`. |
| `spdep` (>= 1.4.2) | — | native | Only `knearneigh`/`knn2nb`/`nb2listw`/`moran`/`Szero`, used by `ComputeNormalizedMoranI`. Ported directly (Moran's I under `style="W"` is a closed-form quadratic); validated against real `spdep` 1.4.2 in a dedicated conda env. |
| `ggplot2`, `ComplexHeatmap`, `circlize`, `RColorBrewer`, `grDevices` | — | native | `matplotlib` + `seaborn`; plotting is not parity-gated numerically. |
| `pals` | — | native | Palettes are static hex vectors; `kelly`, `cols25`, `polychrome`, `glasbey`, `alphabet`, `alphabet2`, `brewer.*` and the continuous ramps transcribed from `pals` 1.10 and byte-compared against it. |

### Action items resolved

* **1** hard dependency added from the omicverse ecosystem: `nmf-rs>=0.1.0`.
* **0** other omicverse mirrors were applicable.
* Everything else is either a pandas/scipy/matplotlib equivalent or ported natively.

### Ecosystem-reuse accounting

| Item | LOC avoided | Evidence |
|---|---|---|
| `nmf-rs` KL multiplicative update kernels (`std.divergence.update.w` / `.h`, including the `nbterms`/`ncterms` fixed-term handling and the `eps` floor) + their R bit-equivalence test suite | ~350 LOC + a full parity suite | `rust-NMF/tests/`, README "✅ 1e-12" rows |

### New alias registered

`SpatialEcoTyper → py-spatialecotyper` follows the `py-<X>` rule, so no
`engine/discover_omicverse_deps.py::ALIAS_MAP` entry is needed.

## 3. What must be ported natively (no faithful Python equivalent exists)

`pyspatialecotyper.rrandom` and `pyspatialecotyper.seuratcompat` are the two
"substrate" modules with no existing Python equivalent that is *numerically*
faithful to R:

* `rrandom` — R's Mersenne-Twister `unif_rand`, the post-R-3.6 `R_unif_index`
  rejection sampler behind `sample()`, and the inversion `norm_rand`. Required
  because `NMF::rnmf`, `nmfClustering`'s `sample(1:6280, ...)`, the
  down-sampling in `NMFGenerateWList`, and every permutation test in
  `Colocalization` / `CoassociationTest` / `ComputeNormalizedMoranI` draw from
  R's RNG stream. Porting the stream converts otherwise-stochastic outputs into
  deterministic ones.
* `seuratcompat` — `NormalizeData`, `ScaleData`, dispersion-based
  `FindVariableFeatures`, `RunPCA`, `FindNeighbors` (exact Seurat SNN with
  Jaccard pruning) and `FindClusters` (a port of Seurat's
  `ModularityOptimizer.cpp`, including its `JavaRandom` LCG, so Louvain is
  reproducible rather than merely similar).

## 4. Raw generator output (for the record)

<details><summary>engine.discover_omicverse_deps --description SpatialEcoTyper-ref/DESCRIPTION</summary>

| R dep | omicverse match |
|---|---|
| `tidyr` | — |
| `dplyr` | — |
| `circlize` | — |
| `grDevices` | — |
| `ComplexHeatmap` | — |
| `spdep` | — |
| `RColorBrewer` | — |
| `data.table` | — |
| `Seurat` | [`py-cca`](https://github.com/omicverse/py-cca) |
| `Matrix` | [`anndata-oom`](https://github.com/omicverse/anndata-oom) |
| `RANN` | — |
| `parallel` | — |
| `NMF` | [`rust-NMF`](https://github.com/omicverse/rust-NMF) |
| `ggplot2` | — |
| `pals` | — |

`3` of `15` R Imports/Depends have an omicverse-org Python mirror.
</details>
