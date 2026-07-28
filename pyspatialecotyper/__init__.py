"""py-spatialecotyper -- a pure-Python port of the R package SpatialEcoTyper.

Spatial EcoTyper discovers recurrent multicellular communities ("spatial
ecotypes") in single-cell spatial transcriptomics by building cell-type-specific
similarity networks over spatial neighbourhoods, fusing them with similarity
network fusion, and integrating across samples with consensus NMF.

Upstream : https://github.com/digitalcytometry/spatialecotyper (v1.0.4)
Paper    : Zhang et al., Nature (2026). doi:10.1038/s41586-026-10452-4
License  : Stanford Non-Commercial Software License Agreement (docket S24-045),
           mirrored from upstream.

Two APIs are exposed side by side:

* a class API -- ``SpatialEcoTyper(normdata, metadata).run()``;
* a functional API mirroring the R exports one-to-one, with names mapped to
  snake_case (``NMFGenerateW`` -> ``nmf_generate_w``) and argument names and
  defaults kept verbatim.
"""

__version__ = "0.1.0"

# --- single-sample pipeline ------------------------------------------------
from .core import (SpatialEcoTyper, SpatialEcoTyperResult, annotate_cells,
                   spatial_ecotyper)
from .preprocessing import preprocess_st, znorm
from .metacells import get_knn_weights, get_spatial_metacells
from .network import (dominateset, fillspots, get_pc_list, get_sn, get_sn_list,
                      snf2)
from .utils import matrix_multiply, most_frequent, rank_sparse

# --- Seurat / clustering substrate ----------------------------------------
from .seuratcompat import (find_variable_features, normalize_data, run_pca,
                           scale_data)
from ._modularity import compute_snn, find_clusters, find_neighbors
from .rrandom import RRandom

# --- multi-sample integration ---------------------------------------------
from .integration import (compute_fcs, integrate, integrate_spatial_ecotyper,
                        multi_spatial_ecotyper)

# --- NMF layer -------------------------------------------------------------
from .nmf import (aggregate_recover_models, cophcor, deconvolute_se,
                  loocv_predict, nmf_clustering, nmf_generate_w,
                  nmf_generate_w_list, nmf_predict, posneg, recover_se)
from .markers import average_expression, average_marker_expression

# --- downstream statistics -------------------------------------------------
from .stats import (aggregate_by_weights, build_knn_weights, coassociation,
                    coassociation_test, colocalization,
                    colocalization_meta_analysis, compute_metrics,
                    compute_normalized_moran_i, compute_se_abundance_by_sn,
                    create_pseudobulks, infer_ncells, partition_tissue,
                    smooth_se_abundances)

# --- visualisation ---------------------------------------------------------
from .palettes import get_colors
from .plotting import (cooccurrence_heatmap_view, draw_rectangle_annotation,
                       heatmap_view, spatial_view)

# --- literal R-name aliases -------------------------------------------------
# Five R exports contain acronym runs or the proper noun "EcoTyper", where a
# mechanical CamelCase -> snake_case mapping and a readable Python name
# disagree (`SpatialEcoTyper` -> `spatial_eco_typer` vs `spatial_ecotyper`).
# Both spellings are bound so that code translated mechanically from R keeps
# working, and so the R-function coverage audit resolves every export.
spatial_eco_typer = spatial_ecotyper
integrate_spatial_eco_typer = integrate_spatial_ecotyper
multi_spatial_eco_typer = multi_spatial_ecotyper
infer_n_cells = infer_ncells
nm_fpredict = nmf_predict            # NMFpredict, mechanical transliteration
compute_f_cs = compute_fcs           # ComputeFCs

R_FUNCTION_MAP = {
    "AggregateRecoverModels": "aggregate_recover_models",
    "AnnotateCells": "annotate_cells",
    "AverageMarkerExpression": "average_marker_expression",
    "Coassociation": "coassociation",
    "CoassociationTest": "coassociation_test",
    "Colocalization": "colocalization",
    "ColocalizationMetaAnalysis": "colocalization_meta_analysis",
    "ComputeMetrics": "compute_metrics",
    "ComputeNormalizedMoranI": "compute_normalized_moran_i",
    "ComputeSEAbundanceBySN": "compute_se_abundance_by_sn",
    "CooccurrenceHeatmapView": "cooccurrence_heatmap_view",
    "CreatePseudobulks": "create_pseudobulks",
    "DeconvoluteSE": "deconvolute_se",
    "GetSpatialMetacells": "get_spatial_metacells",
    "HeatmapView": "heatmap_view",
    "InferNCells": "infer_ncells",
    "IntegrateSpatialEcoTyper": "integrate_spatial_ecotyper",
    "LoocvPredict": "loocv_predict",
    "MultiSpatialEcoTyper": "multi_spatial_ecotyper",
    "NMFGenerateW": "nmf_generate_w",
    "NMFGenerateWList": "nmf_generate_w_list",
    "NMFpredict": "nmf_predict",
    "PreprocessST": "preprocess_st",
    "RecoverSE": "recover_se",
    "SNF2": "snf2",
    "SmoothSEAbundances": "smooth_se_abundances",
    "SpatialEcoTyper": "spatial_ecotyper",
    "SpatialView": "spatial_view",
    "Znorm": "znorm",
    "drawRectangleAnnotation": "draw_rectangle_annotation",
    "getColors": "get_colors",
    "matrixMultiply": "matrix_multiply",
    "mostFrequent": "most_frequent",
    "nmfClustering": "nmf_clustering",
    "rankSparse": "rank_sparse",
    # internal helpers reachable from the exports
    "ComputeFCs": "compute_fcs",
    "GetKnnWeights": "get_knn_weights",
    "GetPCList": "get_pc_list",
    "GetSNList": "get_sn_list",
    "Integrate": "integrate",
    "aggregateByWeights": "aggregate_by_weights",
    "buildKNNWeights": "build_knn_weights",
    "fillspots": "fillspots",
    "getSN": "get_sn",
    ".dominateset": "dominateset",
    ".colocalization": "stats._colocalization",
    ".moran": "stats._moran",
    ".nmf.predict": "nmf._nmf_predict",
    "PartitionTissue": "partition_tissue",
}

__all__ = [
    "__version__", "R_FUNCTION_MAP",
    # class API
    "SpatialEcoTyper", "SpatialEcoTyperResult",
    # single sample
    "spatial_ecotyper", "annotate_cells", "preprocess_st", "znorm",
    "get_spatial_metacells", "get_knn_weights", "get_pc_list", "get_sn",
    "get_sn_list", "fillspots", "snf2", "dominateset",
    "rank_sparse", "matrix_multiply", "most_frequent",
    # substrate
    "normalize_data", "scale_data", "find_variable_features", "run_pca",
    "find_neighbors", "find_clusters", "compute_snn", "RRandom",
    # integration
    "integrate_spatial_ecotyper", "multi_spatial_ecotyper", "integrate",
    "compute_fcs",
    # NMF
    "nmf_generate_w", "nmf_generate_w_list", "nmf_predict", "nmf_clustering",
    "recover_se", "deconvolute_se", "aggregate_recover_models",
    "loocv_predict", "posneg", "cophcor",
    "average_marker_expression", "average_expression",
    # statistics
    "coassociation", "coassociation_test", "colocalization",
    "colocalization_meta_analysis", "compute_metrics",
    "compute_normalized_moran_i", "compute_se_abundance_by_sn",
    "smooth_se_abundances", "build_knn_weights", "aggregate_by_weights",
    "create_pseudobulks", "infer_ncells", "partition_tissue",
    # visualisation
    "get_colors", "spatial_view", "heatmap_view",
    # literal R-name aliases
    "spatial_eco_typer", "integrate_spatial_eco_typer",
    "multi_spatial_eco_typer", "infer_n_cells", "nm_fpredict", "compute_f_cs",
    "cooccurrence_heatmap_view", "draw_rectangle_annotation",
]
