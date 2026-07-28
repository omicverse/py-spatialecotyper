# py-spatialecotyper

A pure-Python port of the R package **[SpatialEcoTyper](https://github.com/digitalcytometry/spatialecotyper)**
(Zhang *et al.*, *Nature* 2026, [doi:10.1038/s41586-026-10452-4](https://www.nature.com/articles/s41586-026-10452-4)).

Spatial EcoTyper is an unsupervised framework for discovering **spatial
ecotypes** — recurrent multicellular communities — from single-cell spatial
transcriptomics. It builds one similarity network over spatial neighbourhoods
*per cell type*, fuses them with similarity network fusion, clusters the fused
graph, and then integrates spatial clusters across samples with consensus NMF to
recover the ecotypes that recur.

Upstream is R + Seurat only. This is a complete Python re-implementation,
validated function-by-function against R 4.4.3 / Seurat 5.4.0 / NMF 0.28 /
SpatialEcoTyper 1.0.4, with the parity gate committed **before** any algorithmic
Python was written (`data/manifest.yaml`).

## Parity with R, in one table

Every number below is measured, not asserted. Reproduce with
`pytest tests/test_exact_match.py`.

| Output | R function | Parity class | Gate | Measured |
|---|---|---|---|---|
| Filtered matrix | `PreprocessST` | deterministic-strict | < 1e-13 | **0** (bit-exact) |
| Spatial metacells | `GetSpatialMetacells` | deterministic | < 1e-8 | **3.6e-15** |
| Similarity network | `getSN` / `GetSNList` | deterministic | < 1e-8 | **1.0e-15** (4/9 cell types); see *Known divergences* |
| Fused network | `SNF2` | deterministic | < 1e-8 | **4.2e-17** |
| Rank transform | `rankSparse` | deterministic-strict | < 1e-13 | **0** (bit-exact) |
| Blocked matmul | `matrixMultiply` | deterministic | < 1e-8 | **0** (bit-exact) |
| Weighted z-score | `Znorm` | deterministic | < 1e-8 | **1.4e-14** |
| Cell-type PCs | `GetPCList` | embedding (Procrustes) | ≥ 0.95 | **1.000** |
| SE labels, 1 sample | `SpatialEcoTyper` | clustering (ARI) | ≥ 0.85 | **0.983** (6k cells) / **0.980** (27.9k cells) |
| Cell-level SE labels | `AnnotateCells` | classification | ≥ 0.99 | **1.000** |
| NMF basis `W` | `NMFGenerateW` | factorization (matched Pearson) | ≥ 0.95 | **1.000**, max abs diff **7.5e-15** |
| NMF loadings `H` | `NMFpredict` | factorization | ≥ 0.95 | **1.000**, max abs diff **8.9e-16** |
| Integrated matrix | `Integrate` | deterministic | < 1e-8 | **5.6e-16** (real 2-sample data) |
| Conserved SE labels | `nmfClustering` | clustering (ARI) | ≥ 0.85 | **1.000**; consensus matrix bit-identical |
| Coassociation index | `Coassociation` | deterministic | < 1e-8 | **7.8e-16** |
| Coassociation p-values | `CoassociationTest` | inference | ρ ≥ 0.90 | **ρ = 1.000**, Z max abs diff 4.4e-15 |
| Colocalization Z | `Colocalization` | ordinal (Pearson) | ≥ 0.99 | **1.000**, max abs diff 1.8e-13 |
| Colocalization meta | `ColocalizationMetaAnalysis` | ordinal | ≥ 0.99 | **1.000** |
| Normalised Moran's I | `ComputeNormalizedMoranI` | ordinal | ≥ 0.95 | **1.000**, max abs diff 1.7e-13 |
| Recovery metrics | `ComputeMetrics` | deterministic | < 1e-8 | **5.6e-16** |
| SE abundance by SN | `ComputeSEAbundanceBySN` | deterministic | < 1e-8 | **5.5e-16** (payload columns) |
| Palettes | `getColors` | byte equality | 100% | **323/323 cases identical to `pals` 1.10** |

**R-function coverage: 35/35 exported functions (100%), 9/10 reachable internal helpers.**
Full audit in [`AUDIT.md`](AUDIT.md); full evidence in
[`RECONSTRUCTION_REPORT.md`](RECONSTRUCTION_REPORT.md).

## Install

```bash
pip install pyspatialecotyper
```

From source:

```bash
git clone https://github.com/omicverse/py-spatialecotyper
cd py-spatialecotyper
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, and
[`nmf-rs`](https://github.com/omicverse/rust-NMF) (the omicverse Rust port of
R's `NMF`, bit-equivalent to the brunet multiplicative updates within 1e-12).
A pure-NumPy fallback for the NMF kernels is built in, so the package still
works if the Rust wheel is unavailable for your platform.

## Quick start — class API

```python
import pandas as pd, scipy.sparse as sp
from pyspatialecotyper import SpatialEcoTyper, normalize_data

counts = pd.read_csv("Melanoma1_subset_counts.tsv.gz", sep="\t", index_col=0)
meta   = pd.read_csv("Melanoma1_subset_scmeta.tsv", sep="\t", index_col=0)
meta   = meta.loc[counts.columns]          # X, Y, CellType columns required

normdata = normalize_data(sp.csc_matrix(counts.to_numpy(float)))

se = SpatialEcoTyper(normdata, meta, gene_names=list(counts.index),
                     radius=50, resolution=0.5, nfeatures=300).run()

print(se)                    # SpatialEcoTyper(1400 spatial neighborhoods, 8 SEs)
se.result.spot_metadata      # neighbourhood-level metadata with an 'SE' column
se.result.metadata           # single-cell metadata with an 'SE' column
se.result.fused              # the fused, rank-transformed similarity graph
```

## Low-level functional API — a one-to-one mirror of R

Function names map from R to snake_case; **argument names and defaults are
verbatim**. `pyspatialecotyper.R_FUNCTION_MAP` is the machine-readable
dictionary, and the mechanical transliterations (`spatial_eco_typer`,
`nm_fpredict`, …) are bound as aliases so code translated line-by-line from R
keeps working.

```python
import pyspatialecotyper as se

expdat, meta, genes, cells = se.preprocess_st(normdata, meta, min_cells=5, min_features=10)
mc, mc_cols = se.get_spatial_metacells(expdat, meta, k=20, radius=50, gene_names=genes)
emb  = se.get_pc_list(mc, mc_cols, genes, nfeatures=300)
nets, spots = se.get_sn_list(emb, npcs=20, k=50, min_cts_per_region=2)
fused = se.snf2([w for w, _ in nets.values()], t=10)
fused = se.rank_sparse(fused)
```

Multi-sample integration (upstream Tutorial 2):

```python
res = se.multi_spatial_ecotyper(
    data_list={"SKCM": norm1, "CRC": norm2},
    metadata_list={"SKCM": meta1, "CRC": meta2},
    gene_names=genes, nmf_ranks=range(4, 13), nrun_per_rank=10, seed=1)
res["cluster_SE"]   # conserved SE per spatial cluster
res["metadata"]     # single-cell metadata with the conserved SE label
```

SE recovery in a new dataset, and bulk deconvolution:

```python
Ws = se.nmf_generate_w_list(scdata, scmeta, gene_names=genes, Sample="Sample")
calls = se.recover_se(newdata, genes, cells, celltypes, Ws, min_score=0.6)
fracs = se.deconvolute_se(bulk, genes, samples, W, w_names, se_names)
```

## What's included

| R function | Python | Notes |
|---|---|---|
| `SpatialEcoTyper` | `spatial_ecotyper` / `SpatialEcoTyper` class | single-sample discovery |
| `MultiSpatialEcoTyper` | `multi_spatial_ecotyper` | end-to-end multi-sample |
| `IntegrateSpatialEcoTyper` | `integrate_spatial_ecotyper` | integration from existing results |
| `PreprocessST`, `Znorm` | `preprocess_st`, `znorm` | |
| `GetSpatialMetacells` | `get_spatial_metacells` | |
| `SNF2`, `matrixMultiply`, `rankSparse` | `snf2`, `matrix_multiply`, `rank_sparse` | |
| `NMFGenerateW`, `NMFGenerateWList`, `NMFpredict`, `nmfClustering` | `nmf_generate_w`, `nmf_generate_w_list`, `nmf_predict`, `nmf_clustering` | |
| `RecoverSE`, `DeconvoluteSE`, `AggregateRecoverModels`, `LoocvPredict` | `recover_se`, `deconvolute_se`, `aggregate_recover_models`, `loocv_predict` | |
| `Coassociation`, `CoassociationTest` | `coassociation`, `coassociation_test` | |
| `Colocalization`, `ColocalizationMetaAnalysis` | `colocalization`, `colocalization_meta_analysis` | |
| `ComputeMetrics`, `ComputeNormalizedMoranI` | `compute_metrics`, `compute_normalized_moran_i` | |
| `ComputeSEAbundanceBySN`, `SmoothSEAbundances` | `compute_se_abundance_by_sn`, `smooth_se_abundances` | |
| `CreatePseudobulks`, `InferNCells`, `AnnotateCells` | `create_pseudobulks`, `infer_ncells`, `annotate_cells` | |
| `AverageMarkerExpression` | `average_marker_expression` | gene sets must be supplied (upstream's are in the R package's `inst/extdata`) |
| `SpatialView`, `HeatmapView`, `CooccurrenceHeatmapView`, `drawRectangleAnnotation` | `spatial_view`, `heatmap_view`, `cooccurrence_heatmap_view`, `draw_rectangle_annotation` | matplotlib |
| `getColors`, `mostFrequent` | `get_colors`, `most_frequent` | |

Two substrate modules have no upstream Python equivalent and were written from
the R/C++ sources:

* **`pyspatialecotyper.rrandom`** — R's Mersenne-Twister `unif_rand` (bit-identical),
  the post-R-3.6 rejection sampler behind `sample()` (identical), and the
  inversion `rnorm` (2.2e-16). This is what makes the permutation tests,
  the NMF restart seeds and the down-sampling *reproducible* rather than merely
  similar.
* **`pyspatialecotyper._modularity`** — Seurat's `ComputeSNN` and a port of
  `ModularityOptimizer.cpp` including its `JavaRandom` LCG, so `FindClusters`
  reproduces Seurat's Louvain **exactly** (element-wise identical labels given
  the same graph, verified on 31/31 parameter combinations).

## Reproducing the R results yourself

```bash
# 1. reference (needs R 4.4.3 + Seurat 5.4.0 + NMF 0.28 + SpatialEcoTyper 1.0.4)
Rscript tests/r_reference_driver.R data/Melanoma1_subset_counts.tsv.gz \
                                   data/Melanoma1_subset_scmeta.tsv \
                                   reference_out/ci 6000
Rscript tests/r_nmf_driver.R      reference_out/nmf
Rscript tests/r_nmfclust_driver.R reference_out/nmf
Rscript tests/r_stats_driver.R    reference_out/stats

# 2. the pre-registered gate
pytest tests/test_exact_match.py -v
```

Every threshold is read out of `data/manifest.yaml`; no test hard-codes a
number, so the gate cannot be quietly widened.

## Known divergences from R (measured, not hand-waved)

1. **`getSN` on tied nearest neighbours.** 5 of 9 cell types in the canonical
   fixture contain spatial neighbourhoods whose cell-type-specific metacell
   profiles are *exactly identical* (adjacent grid spots drawing the same
   `k = 20` nearest cells of a rare type). Their distances tie at the k-th
   neighbour boundary and `RANN`'s ANN kd-tree and `scipy.spatial.cKDTree`
   break the tie differently. Effect: 20 of 5100 stored entries differ for DC
   (99.6% identical); `max abs` 0.11 on a 0–0.5 scale. Both answers are correct
   k-NN; the tie is inherent. This is the one pre-registered gate the port does
   **not** clear, and it is reported as a failure rather than papered over.
2. **`FindNeighbors` uses Annoy in Seurat 5, exact k-NN here.** Annoy is
   approximate; measured on the full fixture, 95.5% of cells get an identical
   neighbour set and 99.7% of (cell, neighbour) pairs are shared, and the exact
   search's summed neighbour distance is ≤ Annoy's for 2133/2133 cells. This is
   the sole source of the end-to-end ARI being 0.98 rather than 1.00 — given
   R's own SNN graph, the Louvain labels are element-wise identical.
3. **`NMFGenerateW`'s feature selection is not stable in R either.** R sets no
   seed before `NMF::rnmf`, so `W_0` differs run to run. The fitted `W` is
   unaffected (KL is convex in `W` when `H` is fixed — see `MATH.md`; measured
   R-vs-R `max abs` 8.9e-16, Pearson 1.000), but the downstream
   `delta > 0` feature filter can flip near-threshold genes: two R runs on a
   smaller fixture kept 212 vs 218 features (Jaccard 0.92).
4. **`Colocalization(ncores > 1)` is not reproducible in R.** `mclapply` forks
   and each child reseeds from its PID. Two R runs with the same `set.seed(1)`
   differ by `max abs` 11.29 (Pearson 0.991). At `ncores = 1` R is
   bit-reproducible, and that is what the port matches; `colocalization()`
   accepts `ncores` for signature parity and consumes the stream sequentially.
5. **`normalization.method = "SCT"`** raises `NotImplementedError` — it would
   require a port of `SCTransform`, which is out of scope.
6. **The bundled MERSCOPE recovery models** (`inst/extdata/*.rds`) are not
   redistributed. `recover_se` and `deconvolute_se` require explicit `Ws` / `W`;
   the default-model branch, including the published `SE01..SE11 → SE1..SE9`
   relabelling, is not reachable without those files.
7. **`nmfClustering` cophenetic coefficient** differs from R in the 5th decimal
   (0.960832 vs 0.960812) on identical consensus matrices — `hclust` and
   `scipy.cluster.hierarchy.average` order equal merges differently. Cluster
   assignments are unaffected (ARI 1.000).

## Performance

Full melanoma fixture (500 genes x 27,907 cells, 2133 spatial neighbourhoods),
single node, 17 cores:

| | R 4.4.3 | py-spatialecotyper | speed-up |
|---|---|---|---|
| `SpatialEcoTyper` end-to-end | 251.8 s | 84.1 s | **3.0x** |
| `SNF2` alone | 122.9 s | (see `ITERATION_LOG.md`) | |
| `Colocalization` (same seed, same nperm) | 18.4 s | 4.5 s | **4.1x** |

The Acceleration loop is logged in [`ITERATION_LOG.md`](ITERATION_LOG.md) with
per-iteration admissibility evidence; one rewrite was rejected and rolled back.
Perturbation bounds for the single bounded-ε rewrite are derived in
[`MATH.md`](MATH.md).

## Notebooks

* [`examples/compare_R_vs_Python.ipynb`](examples/compare_R_vs_Python.ipynb) — pipeline-level parity, one visualisation per manifest output.
* [`examples/tutorial_melanoma.ipynb`](examples/tutorial_melanoma.ipynb) — Python-only walkthrough, one subsection per public function.
* [`examples/function_by_function_R_parity.ipynb`](examples/function_by_function_R_parity.ipynb) — R⇄Python dictionary with a parameter table and a numerical comparison per function.
* [`examples/evolution.ipynb`](examples/evolution.ipynb) — the Acceleration iterations, one panel each.

## Relationship to omicverse

Built with the [omicverse-rebuildr](https://github.com/omicverse) protocol and
depends on [`omicverse/rust-NMF`](https://github.com/omicverse/rust-NMF) for the
KL multiplicative-update kernels. Motivated by omicverse issue #760.

## Citation

If you use this package, cite the original method:

> Zhang, W. *et al.* Spatial ecotypes of the tumour microenvironment.
> *Nature* (2026). doi:10.1038/s41586-026-10452-4

and, optionally, this port:

> py-spatialecotyper: a pure-Python port of SpatialEcoTyper.
> https://github.com/omicverse/py-spatialecotyper

## License

**Stanford Non-Commercial Software License Agreement** — mirrored verbatim from
upstream (see [`LICENSE`](LICENSE)). Spatial EcoTyper is provided free of charge
for **non-commercial use only**; use by any commercial entity for any purpose,
including research, is prohibited. Commercial entities should contact Stanford
University's Office of Technology Licensing and reference docket **S24-045**.

This port is a derivative work and inherits that restriction. It is *not*
MIT-licensed.
