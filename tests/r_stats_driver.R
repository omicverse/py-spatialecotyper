#!/usr/bin/env Rscript
# =====================================================================
# py-spatialecotyper — R reference runner for the downstream-statistics
# surface (Coassociation / Colocalization / ComputeMetrics /
# ComputeNormalizedMoranI / ComputeSEAbundanceBySN / CreatePseudobulks /
# InferNCells / PartitionTissue).
#
# Usage:
#   Rscript r_stats_driver.R <counts.tsv.gz> <scmeta.tsv> <outdir> [part] [ncells]
#
#   part = "all" (default) | "main" | "moran"
#     main  — everything except ComputeNormalizedMoranI (needs Seurat +
#             SpatialEcoTyper; run under $R_TEST_ENV / the CMAP env).
#     moran — ComputeNormalizedMoranI only (needs spdep + dplyr).  spdep is not
#             in the CMAP env, so put the setref library on R_LIBS:
#               R_LIBS=...:/scratch/users/steorra/env/setref/lib/R/library
#             If SpatialEcoTyper is unavailable in that env the upstream source
#             file SpatialEcoTyper-ref/R/ComputeNormalizedMoranI.R is sourced
#             directly (it only needs spdep + dplyr + parallel).
#
# The fixture is built with NO RNG whatsoever: the `ncells` cells nearest the
# tissue centroid, sample IDs from a 4x4 spatial grid, SE labels from a
# diagonal spatial stripe, and a deterministic corruption of SE to play the
# role of a cross-validation prediction.  It is dumped as `00_fixture` so the
# Python side never has to re-derive it.
#
# Dump conventions follow tests/r_reference_driver.R:
#   sparse -> writeMM + .rows.txt/.cols.txt
#   dense  -> fwrite gz TSV + .rows.txt/.cols.txt
#   df     -> TSV with a `.rowname` first column
#   list/vector -> jsonlite::write_json(digits = NA)
# =====================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(dplyr)
  library(jsonlite)
})

args     <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]
meta_f   <- args[2]
outdir   <- args[3]
part     <- if (length(args) >= 4) args[4] else "all"
ncells   <- if (length(args) >= 5) as.integer(args[5]) else 4000L
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------
# Dump helpers (identical to tests/r_reference_driver.R)
# --------------------------------------------------------------------
dump_sparse <- function(x, name) {
  x <- as(as(as(x, "dMatrix"), "generalMatrix"), "CsparseMatrix")
  Matrix::writeMM(x, file.path(outdir, paste0(name, ".mtx")))
  writeLines(as.character(rownames(x)), file.path(outdir, paste0(name, ".rows.txt")))
  writeLines(as.character(colnames(x)), file.path(outdir, paste0(name, ".cols.txt")))
}
dump_dense <- function(x, name) {
  x <- as.matrix(x)
  data.table::fwrite(data.table::data.table(as.data.frame(x)),
                     file.path(outdir, paste0(name, ".tsv.gz")), sep = "\t")
  writeLines(as.character(rownames(x)), file.path(outdir, paste0(name, ".rows.txt")))
  writeLines(as.character(colnames(x)), file.path(outdir, paste0(name, ".cols.txt")))
}
dump_df <- function(x, name) {
  x <- as.data.frame(x)
  x <- cbind(.rowname = rownames(x), x)
  data.table::fwrite(x, file.path(outdir, paste0(name, ".tsv")), sep = "\t")
}
dump_json <- function(x, name) {
  write_json(x, file.path(outdir, paste0(name, ".json")),
             auto_unbox = TRUE, digits = NA, na = "null")
}
dump_named <- function(v, name) {
  dump_json(list(name = names(v), value = as.numeric(v)), name)
}

# --------------------------------------------------------------------
# Fixture — 100% deterministic, no RNG.
# --------------------------------------------------------------------
build_fixture <- function(meta_f, ncells) {
  meta <- read.table(meta_f, sep = "\t", header = TRUE, row.names = 1)
  cx <- median(meta$X); cy <- median(meta$Y)
  d  <- (meta$X - cx)^2 + (meta$Y - cy)^2
  keep <- sort(order(d)[seq_len(min(ncells, nrow(meta)))])
  m <- meta[keep, ]

  # 4 x 4 spatial grid of pseudo-samples (same arithmetic as PartitionTissue).
  gx <- pmin(3L, as.integer(floor((m$X - min(m$X)) /
                                  ((max(m$X) - min(m$X)) * 1.000001 / 4))))
  gy <- pmin(3L, as.integer(floor((m$Y - min(m$Y)) /
                                  ((max(m$Y) - min(m$Y)) * 1.000001 / 4))))
  m$Sample <- paste0("Sample", gx * 4L + gy + 1L)

  # Diagonal spatial stripes -> 4 spatially autocorrelated SEs.  Kept coarse so
  # that every (SE, CellType) cell state stays large enough to survive
  # Colocalization's `min.cell` filter in every permutation (an emptied state
  # makes R's own `PermuteSD[PermuteSD == 0] <- 1` abort on an NA subscript).
  m$SE <- paste0("SE", 1L + ((round(m$X / 250) + round(m$Y / 250)) %% 4L))
  # A finer, 8-level stripe pattern for Coassociation, which has no min.cell
  # filter.  More SE levels => CoassociationTest returns more P-values, so the
  # pre-registered `inference` gate is measured on 9 values rather than 4.
  m$SEf <- paste0("SE", 1L + ((round(m$X / 150) + 3L * round(m$Y / 150)) %% 8L))
  # A NonSE-carrying variant for Coassociation (every 11th cell).
  m$SEn <- m$SEf
  m$SEn[seq_len(nrow(m)) %% 11L == 0L] <- "NonSE"
  # A deterministic "cross-validation prediction": every 5th cell's SE is
  # rotated by one, everything else is called correctly.
  se_idx <- as.integer(sub("SE", "", m$SE))
  pred_idx <- se_idx
  flip <- seq_len(nrow(m)) %% 5L == 0L
  pred_idx[flip] <- (se_idx[flip] %% 4L) + 1L
  m$cvPred <- paste0("SE", pred_idx)
  m
}

m <- build_fixture(meta_f, ncells)
message("[R] fixture: ", nrow(m), " cells, ", length(unique(m$Sample)),
        " samples, ", length(unique(m$SE)), " SEs, ",
        length(unique(m$CellType)), " cell types")

# Guard: the port sorts with Python's code-point order (== R's C locale).  R's
# own sort() collates in LC_COLLATE.  Assert the two agree on every label the
# fixture puts through a sort(), so the parity numbers are not silently a
# collation artefact.
labels <- c(unique(m$SE), unique(m$SEf), unique(m$SEn), unique(m$CellType),
            unique(paste0(m$SEn, "_", m$CellType)),
            unique(paste0(m$SE, "_", m$CellType)))
stopifnot(identical(sort(labels), sort(labels, method = "radix")))

# ====================================================================
# PART: moran (needs spdep)
# ====================================================================
if (part %in% c("all", "moran")) {
  ok <- requireNamespace("spdep", quietly = TRUE)
  if (!ok) {
    message("[R] spdep not available — skipping ComputeNormalizedMoranI. ",
            "Re-run with part='moran' and setref's library on R_LIBS.")
  } else {
    suppressPackageStartupMessages(library(spdep))
    fn <- NULL
    if (requireNamespace("SpatialEcoTyper", quietly = TRUE)) {
      ns <- asNamespace("SpatialEcoTyper")
      if (exists("ComputeNormalizedMoranI", envir = ns, inherits = FALSE)) {
        fn <- get("ComputeNormalizedMoranI", envir = ns)
        moran_fn <- get(".moran", envir = ns)
      }
    }
    if (is.null(fn)) {
      src <- file.path(dirname(dirname(normalizePath(
        sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))),
        "SpatialEcoTyper-ref", "R", "ComputeNormalizedMoranI.R")
      suppressPackageStartupMessages(library(parallel))
      source(src)
      fn <- ComputeNormalizedMoranI
      moran_fn <- .moran
    }

    # The raw spdep pieces, so the closed-form port can be checked without the
    # permutation null in the way.
    co <- as.matrix(m[, c("X", "Y")])
    knn <- spdep::knearneigh(co, k = 3)
    nb  <- spdep::knn2nb(knn, row.names = rownames(m))
    lw  <- spdep::nb2listw(nb, style = "W", zero.policy = TRUE)
    dump_json(list(n = nrow(m), S0 = spdep::Szero(lw), k = 3), "moran_listw_meta")
    dump_named(moran_fn(m$SE, lw, ncores = 1), "moran_observed")

    set.seed(1)
    z1 <- fn(m, coords = c("X", "Y"), SE = "SE", CellType = "CellType",
             nperm = 200, k = 3, ncores = 1)
    dump_named(z1, "normalized_moran_i")
    # R-vs-R reproducibility of the (plain lapply) permutation loop.
    set.seed(1)
    z2 <- fn(m, coords = c("X", "Y"), SE = "SE", CellType = "CellType",
             nperm = 200, k = 3, ncores = 1)
    dump_json(list(reproducible = isTRUE(all.equal(as.numeric(z1),
                                                   as.numeric(z2), tolerance = 0)),
                   max_abs_diff = max(abs(as.numeric(z1) - as.numeric(z2)))),
              "normalized_moran_i_rvr")
    message("[R] ComputeNormalizedMoranI done")
  }
}

if (part == "moran") quit(save = "no")

# ====================================================================
# PART: main (needs SpatialEcoTyper + Seurat)
# ====================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(SpatialEcoTyper)
})
ns <- asNamespace("SpatialEcoTyper")
.colocalization    <- get(".colocalization",    envir = ns)
buildKNNWeights    <- get("buildKNNWeights",    envir = ns)
aggregateByWeights <- get("aggregateByWeights", envir = ns)
PartitionTissue    <- get("PartitionTissue",    envir = ns)

dump_df(m, "00_fixture")
dump_json(list(ncells = nrow(m),
               radius = 50, k_coloc = 100, min_cell = 10,
               nperm_coloc = 200, nperm_meta = 50, nperm_coassoc = 1000,
               grid_size = 50, min_cells_sn = 5,
               nperm_moran = 200, k_moran = 3,
               n_mixtures = 20, seed = 1), "params")

# --------------------------------------------------------------------
# 1. PartitionTissue  (exact)
# --------------------------------------------------------------------
pt <- PartitionTissue(m, nrow = 3, ncol = 5, X = "X", Y = "Y")
dump_json(list(cell = rownames(pt), partition = pt$Partition), "partition_tissue")

# --------------------------------------------------------------------
# 2. InferNCells  (exact) — real counts, subset to the fixture's cells
# --------------------------------------------------------------------
scdata <- fread(counts_f, sep = "\t", header = TRUE, data.table = FALSE)
rownames(scdata) <- scdata[, 1]
scdata <- as.matrix(scdata[, -1])
scdata <- scdata[, rownames(m), drop = FALSE]
dump_dense(scdata, "00_counts")
for (av in c(5, 1.5, 12)) {
  dump_json(list(avg.number = av, ncells = as.integer(InferNCells(scdata, av))),
            paste0("infer_ncells_", gsub("[.]", "p", as.character(av))))
}

# --------------------------------------------------------------------
# 3. ComputeMetrics  (deterministic, 1e-8)
# --------------------------------------------------------------------
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = "Sample", metric = "F1"), "metrics_f1")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = "Sample", metric = "F2"), "metrics_f2")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = "Sample", metric = "recall"), "metrics_recall")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = "Sample", metric = "precision"),
           "metrics_precision")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = "CellType",
                          Sample = "Sample", metric = "F1"), "metrics_f1_ct")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = NULL, metric = "F1"), "metrics_f1_nosample")
message("[R] ComputeMetrics done")

# --------------------------------------------------------------------
# 4. buildKNNWeights / aggregateByWeights / ComputeSEAbundanceBySN /
#    SmoothSEAbundances   (deterministic, 1e-8)
# --------------------------------------------------------------------
sc_coords <- m[, c("X", "Y")]
w_self <- buildKNNWeights(ref_coords = sc_coords, k = 20, radius = 40,
                          include.self = TRUE)
dump_sparse(w_self, "knn_weights_self")
w_noself <- buildKNNWeights(ref_coords = sc_coords, k = 20, radius = 40,
                            include.self = FALSE)
dump_sparse(w_noself, "knn_weights_noself")

se_levels <- sort(unique(m$SE))
cell2se <- Matrix(0, nrow = nrow(m), ncol = length(se_levels), sparse = TRUE)
rownames(cell2se) <- rownames(m); colnames(cell2se) <- se_levels
cell2se[cbind(seq_len(nrow(m)), match(m$SE, se_levels))] <- 1
dump_dense(aggregateByWeights(cell2se, w_self, sum2one = TRUE, min.cells = 5),
           "aggregate_weights")

seab <- ComputeSEAbundanceBySN(m, spot_coords = NULL, radius = 50,
                               grid.size = 50, X = "X", Y = "Y", SE = "SE",
                               min.cells = 5)
dump_df(seab, "se_abundance_by_sn")

sm <- SmoothSEAbundances(as.matrix(seab[, -c(1, 2)]), seab[, c("X", "Y")],
                         k = 7, X = "X", Y = "Y", include.self = TRUE,
                         min.neighbors = 3)
dump_df(sm, "smooth_se_abundances")
message("[R] ComputeSEAbundanceBySN done: ", nrow(seab), " SNs")

# --------------------------------------------------------------------
# 5. Coassociation (test = FALSE)  (deterministic, 1e-8)
#    + CoassociationTest on that matrix   (inference gate)
# --------------------------------------------------------------------
coassoc <- Coassociation(m, Sample = "Sample", SE = "SEn", CellType = "CellType",
                         NonSE = "NonSE", test = FALSE)
dump_dense(coassoc, "coassociation_index")
dump_json(list(diag_all_one = all(diag(coassoc) == 1),
               diag_min = min(diag(coassoc)), diag_max = max(diag(coassoc))),
          "coassociation_index_diag")

set.seed(1)
cp1 <- CoassociationTest(coassoc, nperm = 1000)
dump_named(cp1, "coassociation_pvals")
dump_named(attr(cp1, "Zscore"), "coassociation_zscore")
set.seed(1)
cp2 <- CoassociationTest(coassoc, nperm = 1000)
dump_json(list(reproducible = identical(as.numeric(cp1), as.numeric(cp2))),
          "coassociation_pvals_rvr")
message("[R] Coassociation done: ", nrow(coassoc), " cell states")

# --------------------------------------------------------------------
# 6. Colocalization   (ordinal gate, |Pearson| >= 0.99)
# --------------------------------------------------------------------
mm <- m
mm$CellState <- paste0(mm$SE, "_", mm$CellType)
dump_dense(.colocalization(mm, coords = c("X", "Y"), CellState = "CellState",
                           radius = 50, k = 100, min.cell = 10),
           "colocalization_observed")

set.seed(1)
t0 <- Sys.time()
coloc1 <- Colocalization(m, coords = c("X", "Y"), SE = "SE",
                         CellType = "CellType", radius = 50, k = 100,
                         min.cell = 10, nperm = 200, test = TRUE, ncores = 1)
message("[R] Colocalization(ncores=1, nperm=200) ",
        round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s")
dump_dense(coloc1$ColocIndex, "colocalization_index")
dump_named(coloc1$Pval, "colocalization_pvals")
dump_named(attr(coloc1$Pval, "Zscore"), "colocalization_zscore")

# R-vs-R reproducibility: ncores = 1 (mclapply short-circuits to lapply) versus
# ncores > 1 (mclapply forks and each child reseeds from its PID).
set.seed(1)
coloc1b <- Colocalization(m, coords = c("X", "Y"), SE = "SE",
                          CellType = "CellType", radius = 50, k = 100,
                          min.cell = 10, nperm = 200, test = FALSE, ncores = 1)
set.seed(1)
cA <- Colocalization(m, coords = c("X", "Y"), SE = "SE", CellType = "CellType",
                     radius = 50, k = 100, min.cell = 10, nperm = 60,
                     test = FALSE, ncores = 4)
set.seed(1)
cB <- Colocalization(m, coords = c("X", "Y"), SE = "SE", CellType = "CellType",
                     radius = 50, k = 100, min.cell = 10, nperm = 60,
                     test = FALSE, ncores = 4)
dump_json(list(
  ncores1_max_abs_diff = max(abs(as.numeric(coloc1$ColocIndex) -
                                 as.numeric(coloc1b))),
  ncores1_reproducible = identical(as.numeric(coloc1$ColocIndex),
                                   as.numeric(coloc1b)),
  ncores4_max_abs_diff = max(abs(as.numeric(cA) - as.numeric(cB))),
  ncores4_reproducible = identical(as.numeric(cA), as.numeric(cB)),
  ncores4_pearson = cor(as.numeric(cA), as.numeric(cB))
), "colocalization_rvr")

# --------------------------------------------------------------------
# 7. ColocalizationMetaAnalysis   (ordinal gate, |Pearson| >= 0.99)
#    Three spatial thirds play the role of three samples.
# --------------------------------------------------------------------
third <- cut(m$X, breaks = quantile(m$X, c(0, 1/3, 2/3, 1)),
             include.lowest = TRUE, labels = FALSE)
coloc_list <- list()
for (i in 1:3) {
  sub <- m[third == i, ]
  set.seed(100 + i)
  coloc_list[[paste0("S", i)]] <-
    Colocalization(sub, coords = c("X", "Y"), SE = "SE", CellType = "CellType",
                   radius = 50, k = 100, min.cell = 10, nperm = 50,
                   test = TRUE, ncores = 1)
  dump_dense(coloc_list[[i]]$ColocIndex, paste0("meta_input_index_S", i))
  dump_named(coloc_list[[i]]$Pval, paste0("meta_input_pval_S", i))
  dump_named(attr(coloc_list[[i]]$Pval, "Zscore"),
             paste0("meta_input_zscore_S", i))
}
meta <- ColocalizationMetaAnalysis(coloc_list, cap = 5, min.samples = 1)
dump_dense(meta$MetaColocIndex, "colocalization_meta_index")
dump_named(meta$MetaPval, "colocalization_meta_pvals")

meta2 <- ColocalizationMetaAnalysis(coloc_list, cap = 2, min.samples = 2)
dump_dense(meta2$MetaColocIndex, "colocalization_meta_index_cap2")
dump_named(meta2$MetaPval, "colocalization_meta_pvals_cap2")
message("[R] ColocalizationMetaAnalysis done")

# --------------------------------------------------------------------
# 8. CreatePseudobulks  (ungated diagnostic; exercises rnorm + sample)
# --------------------------------------------------------------------
groups <- m$SE
names(groups) <- rownames(m)
set.seed(1)
pb <- CreatePseudobulks(counts = scdata, groups = groups, n_mixtures = 20)
dump_dense(pb$Fracs, "pseudobulk_fracs")
dump_dense(pb$Mixtures, "pseudobulk_mixtures")
set.seed(1)
pb2 <- CreatePseudobulks(counts = scdata, groups = groups, n_mixtures = 20)
dump_json(list(reproducible = identical(as.numeric(pb$Mixtures),
                                        as.numeric(pb2$Mixtures))),
          "pseudobulk_rvr")
message("[R] CreatePseudobulks done")

message("[R] all outputs written to ", outdir)
