## R function coverage audit

### Coverage summary

| Category | Ported | Total | % |
|---|---|---|---|
| Exported R functions | 35 | 35 | 100.0% |
| Internal helpers (reachable) | 9 | 10 | 90.0% |

_Python package exposes 196 unique names._

### Exported R functions

| R function | Python equivalent | Status |
|---|---|---|
| `AggregateRecoverModels` | `aggregate_recover_models` | ✅ ported |
| `AnnotateCells` | `annotate_cells` | ✅ ported |
| `AverageMarkerExpression` | `average_marker_expression` | ✅ ported |
| `Coassociation` | `coassociation` | ✅ ported |
| `CoassociationTest` | `coassociation_test` | ✅ ported |
| `Colocalization` | `colocalization` | ✅ ported |
| `ColocalizationMetaAnalysis` | `colocalization_meta_analysis` | ✅ ported |
| `ComputeMetrics` | `compute_metrics` | ✅ ported |
| `ComputeNormalizedMoranI` | `compute_normalized_moran_i` | ✅ ported |
| `ComputeSEAbundanceBySN` | `compute_se_abundance_by_sn` | ✅ ported |
| `CooccurrenceHeatmapView` | `cooccurrence_heatmap_view` | ✅ ported |
| `CreatePseudobulks` | `create_pseudobulks` | ✅ ported |
| `DeconvoluteSE` | `deconvolute_se` | ✅ ported |
| `GetSpatialMetacells` | `get_spatial_metacells` | ✅ ported |
| `HeatmapView` | `heatmap_view` | ✅ ported |
| `InferNCells` | `infer_n_cells` | ✅ ported |
| `IntegrateSpatialEcoTyper` | `integrate_spatial_eco_typer` | ✅ ported |
| `LoocvPredict` | `loocv_predict` | ✅ ported |
| `MultiSpatialEcoTyper` | `multi_spatial_eco_typer` | ✅ ported |
| `NMFGenerateW` | `nmf_generate_w` | ✅ ported |
| `NMFGenerateWList` | `nmf_generate_w_list` | ✅ ported |
| `NMFpredict` | `nm_fpredict` | ✅ ported |
| `PreprocessST` | `preprocess_st` | ✅ ported |
| `RecoverSE` | `recover_se` | ✅ ported |
| `SNF2` | `snf2` | ✅ ported |
| `SmoothSEAbundances` | `smooth_se_abundances` | ✅ ported |
| `SpatialEcoTyper` | `spatial_eco_typer` | ✅ ported |
| `SpatialView` | `spatial_view` | ✅ ported |
| `Znorm` | `znorm` | ✅ ported |
| `drawRectangleAnnotation` | `draw_rectangle_annotation` | ✅ ported |
| `getColors` | `get_colors` | ✅ ported |
| `matrixMultiply` | `matrix_multiply` | ✅ ported |
| `mostFrequent` | `most_frequent` | ✅ ported |
| `nmfClustering` | `nmf_clustering` | ✅ ported |
| `rankSparse` | `rank_sparse` | ✅ ported |

### Internal helpers reachable from exports

| R helper | File | Python equivalent | Status |
|---|---|---|---|
| `ComputeFCs` | `ComputeFCs.R` | `compute_f_cs` | ✅ ported |
| `GetKnnWeights` | `GetSpatialMetacells.R` | `get_knn_weights` | ✅ ported |
| `GetPCList` | `GetPCList.R` | `get_pc_list` | ✅ ported |
| `GetSNList` | `GetSNList.R` | `get_sn_list` | ✅ ported |
| `Integrate` | `Integrate.R` | `integrate` | ✅ ported |
| `aggregateByWeights` | `ComputeSEAbundanceBySN.R` | `aggregate_by_weights` | ✅ ported |
| `buildKNNWeights` | `ComputeSEAbundanceBySN.R` | `build_knn_weights` | ✅ ported |
| `fillspots` | `GetSNList.R` | `fillspots` | ✅ ported |
| `getSN` | `GetSNList.R` | `get_sn` | ✅ ported |
| `heatmap_annotation` | `HeatmapView.R` | `—` | 🔸 missing-or-inlined |
