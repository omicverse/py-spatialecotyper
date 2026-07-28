#!/usr/bin/env Rscript
# =====================================================================
# py-spatialecotyper -- per-function R reference dump for
# examples/function_by_function_R_parity.ipynb (Notebook 3).
#
# Unlike tests/r_reference_driver.R (which runs the single-sample pipeline
# stage by stage) this driver calls each PUBLIC R function *in isolation* on
# one small, fully deterministic fixture and dumps its return value, so the
# notebook can compare R and Python function by function.
#
# Usage:
#   Rscript r_per_function_dump.R <outdir> [counts.tsv.gz] [scmeta.tsv]
#                                 [part] [ncells]
#
#   part = "main" (default) | "nmf" | "nmfclust" | "all"
#     main     -- everything except the NMF layer.
#     nmf      -- NMFGenerateW / NMFpredict / Integrate (the input matrix that
#                 nmfClustering consumes).
#     nmfclust -- nmfClustering ONLY, and it MUST be a separate Rscript
#                 invocation:  NMF:::nmf.stop.connectivity stores `.consold`
#                 in a local({}) closure, i.e. process-global state, so
#                 factorising matrices of different shapes in one R session
#                 aborts with "cons != .consold : non-conformable arrays".
#                 Same reason tests/r_nmfclust_driver.R is a separate file.
#     all      -- main + nmf in one process (still NOT nmfclust).
#
# Environment (Sherlock):
#   R_LIBS=/scratch/users/steorra/Rlibs_set:/scratch/users/steorra/env/CMAP/lib/R/library
#   TMPDIR=/scratch/users/steorra/tmp
#   /scratch/users/steorra/env/CMAP/bin/Rscript
#
# NOT covered here, on purpose:
#   * getColors  -- needs `pals`, which is not installed in the CMAP env that
#     carries Seurat + SpatialEcoTyper.  The 323-case byte-comparison against
#     `pals` 1.10 was already dumped to reference_out/palettes.json by the
#     palette driver; Notebook 3 reads that file instead of re-running R.
#   * ComputeNormalizedMoranI -- needs `spdep`, same situation; its reference
#     lives in reference_out/stats/normalized_moran_i.json (produced by
#     tests/r_stats_driver.R part="moran" under the setref env).
#
# Dump conventions are identical to tests/r_reference_driver.R:
#   sparse      -> Matrix::writeMM + <name>.rows.txt / <name>.cols.txt
#   dense       -> data.table::fwrite gz TSV + .rows.txt / .cols.txt
#   data.frame  -> TSV with a `.rowname` first column
#   list/vector -> jsonlite::write_json(..., digits = NA)
# =====================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
  library(data.table)
})

args     <- commandArgs(trailingOnly = TRUE)
outdir   <- args[1]
here     <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE),
                                            value = TRUE)[1]))
repo     <- dirname(here)
counts_f <- if (length(args) >= 2 && nzchar(args[2])) args[2] else
  file.path(repo, "data", "Melanoma1_subset_counts.tsv.gz")
meta_f   <- if (length(args) >= 3 && nzchar(args[3])) args[3] else
  file.path(repo, "data", "Melanoma1_subset_scmeta.tsv")
part     <- if (length(args) >= 4 && nzchar(args[4])) args[4] else "main"
ncells   <- if (length(args) >= 5) as.integer(args[5]) else 3000L
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------
# Dump helpers
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

timings <- list()
tick <- function(label, expr) {
  s <- Sys.time(); v <- force(expr); e <- Sys.time()
  timings[[label]] <<- as.numeric(difftime(e, s, units = "secs"))
  message(sprintf("[R] %-26s %8.2fs", label, timings[[label]]))
  v
}

# ====================================================================
# PART: nmfclust -- must be its own process (see the header).
# ====================================================================
if (part == "nmfclust") {
  suppressPackageStartupMessages({ library(NMF); library(SpatialEcoTyper) })
  S <- as.matrix(data.table::fread(file.path(outdir, "integrate_matrix.tsv.gz")))
  rownames(S) <- readLines(file.path(outdir, "integrate_matrix.rows.txt"))
  colnames(S) <- readLines(file.path(outdir, "integrate_matrix.cols.txt"))
  message("[R] nmfClustering input: ", nrow(S), " x ", ncol(S))

  fit <- nmfClustering(S, ranks = 4, nrun.per.rank = 10, seed = 2024, ncores = 1)
  lab <- predict(fit)
  dump_json(list(sample = names(lab), label = as.integer(lab),
                 cophenetic = as.numeric(cophcor(fit)),
                 ranks = 4, nrun.per.rank = 10, seed = 2024),
            "nmfclustering_k4")
  dump_dense(consensus(fit), "nmfclustering_consensus_k4")
  message("[R] nmfClustering done")
  quit(save = "no")
}

# ====================================================================
# Deterministic fixture -- NO RNG anywhere in its construction.
#
#  * the `ncells` cells closest to the tissue centroid (a contiguous
#    spatial crop, the same rule tests/r_reference_driver.R uses);
#  * pseudo-samples from a 4 x 4 spatial grid;
#  * SE labels from coarse diagonal spatial stripes (so they are spatially
#    autocorrelated and every (SE, CellType) state survives Colocalization's
#    min.cell filter under permutation);
#  * a finer 8-stripe variant plus a NonSE-carrying variant for Coassociation;
#  * a deterministic "cross-validation prediction" (every 5th cell rotated).
# ====================================================================
build_fixture <- function(meta_f, ncells) {
  meta <- read.table(meta_f, sep = "\t", header = TRUE, row.names = 1)
  cx <- median(meta$X); cy <- median(meta$Y)
  d  <- (meta$X - cx)^2 + (meta$Y - cy)^2
  keep <- sort(order(d)[seq_len(min(ncells, nrow(meta)))])
  m <- meta[keep, ]
  gx <- pmin(3L, as.integer(floor((m$X - min(m$X)) /
                                  ((max(m$X) - min(m$X)) * 1.000001 / 4))))
  gy <- pmin(3L, as.integer(floor((m$Y - min(m$Y)) /
                                  ((max(m$Y) - min(m$Y)) * 1.000001 / 4))))
  m$Sample <- paste0("Sample", gx * 4L + gy + 1L)
  m$SE  <- paste0("SE", 1L + ((round(m$X / 250) + round(m$Y / 250)) %% 4L))
  m$SEf <- paste0("SE", 1L + ((round(m$X / 150) + 3L * round(m$Y / 150)) %% 8L))
  m$SEn <- m$SEf
  m$SEn[seq_len(nrow(m)) %% 11L == 0L] <- "NonSE"
  se_idx <- as.integer(sub("SE", "", m$SE))
  pred_idx <- se_idx
  flip <- seq_len(nrow(m)) %% 5L == 0L
  pred_idx[flip] <- (se_idx[flip] %% 4L) + 1L
  m$cvPred <- paste0("SE", pred_idx)
  m
}

m <- build_fixture(meta_f, ncells)
message("[R] fixture: ", nrow(m), " cells, ",
        length(unique(m$Sample)), " samples, ",
        length(unique(m$SE)), " SEs, ",
        length(unique(m$CellType)), " cell types")

# The port sorts with Python's code-point order (== R's C locale); assert that
# R's own LC_COLLATE sort agrees, so no parity number is a collation artefact.
labels <- c(unique(m$SE), unique(m$SEf), unique(m$SEn), unique(m$CellType),
            unique(paste0(m$SEn, "_", m$CellType)),
            unique(paste0(m$SE, "_", m$CellType)))
stopifnot(identical(sort(labels), sort(labels, method = "radix")))

scdata <- fread(counts_f, sep = "\t", header = TRUE, data.table = FALSE)
rownames(scdata) <- scdata[, 1]
scdata <- as.matrix(scdata[, -1])
scdata <- scdata[, rownames(m), drop = FALSE]

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(SpatialEcoTyper)
})
ns <- asNamespace("SpatialEcoTyper")
PartitionTissue <- get("PartitionTissue", envir = ns)

normdata <- NormalizeData(scdata, verbose = FALSE)

dump_df(m, "00_fixture")
dump_dense(scdata, "00_counts")
dump_sparse(as(normdata, "CsparseMatrix"), "00_normdata")
dump_json(list(ncells = nrow(m), ngenes = nrow(scdata),
               min.cells = 5, min.features = 10,
               radius = 50, grid.size = 70, k = 20, k.sn = 50,
               nfeatures = 300, npcs = 20, resolution = 0.5,
               iterations = 10, minibatch = 5000,
               nperm_coassoc = 1000, nperm_coloc = 200,
               k_coloc = 100, min.cell_coloc = 10,
               seed = 1, nmf_seed = 2024),
          "params")

# ====================================================================
# PART: nmf  (NMFGenerateW / NMFpredict / Integrate)
# ====================================================================
if (part %in% c("nmf", "all")) {
  suppressPackageStartupMessages({ library(NMF) })
  source(file.path(repo, "tests", "nmf_fixture.R"))
  fx <- make_fixture()
  V <- fx$V; cell_se <- fx$cell_se; celltype <- fx$celltype
  dump_dense(V, "nmf_V")
  dump_json(list(cell = colnames(V), SE = cell_se, CellType = celltype),
            "nmf_cellse")

  ses   <- sort(unique(cell_se))
  Fracs <- matrix(0, ncol(V), length(ses), dimnames = list(colnames(V), ses))
  Fracs[cbind(seq_len(ncol(V)), match(cell_se, ses))] <- 1
  dump_dense(Fracs, "nmf_Fracs")

  # ---- NMFGenerateW -------------------------------------------------
  # NMFGenerateW does NOT set a seed before NMF::rnmf, so W_0 is drawn from
  # whatever RNG state the session is in.  set.seed here only pins THIS dump;
  # the fitted W is independent of W_0 (KL is convex in W when H is fixed).
  set.seed(11)
  W <- tick("NMFGenerateW",
            NMFGenerateW(Fracs, V, scale = TRUE, nfeature = 300,
                         nfeature.per.se = 50, method = "brunet"))
  dump_dense(W, "nmfgeneratew_W")

  # The exact pre-NMF training matrix, so Python can be fed R's own input.
  to_predict <- local({
    E <- Seurat::ScaleData(V, verbose = FALSE)
    E[is.na(E)] <- 0
    tp <- NMF::posneg(E)
    rn <- rownames(tp)
    idx <- duplicated(rn)
    rn[!idx] <- paste0(rn[!idx], "__pos"); rn[idx] <- paste0(rn[idx], "__neg")
    rownames(tp) <- rn
    tp[apply(tp, 1, function(x) var(x) > 0), , drop = FALSE]
  })
  dump_dense(to_predict, "nmf_to_predict")

  # ---- NMFpredict ---------------------------------------------------
  H <- tick("NMFpredict",
            NMFpredict(W, V, scale = FALSE, ncell.per.run = 5000,
                       sum2one = TRUE, ncores = 1))
  dump_dense(H, "nmfpredict_H")
  H_chunked <- NMFpredict(W, V, scale = FALSE, ncell.per.run = 25,
                          sum2one = TRUE, ncores = 1)
  dump_dense(H_chunked, "nmfpredict_H_chunked")

  # ---- Integrate ----------------------------------------------------
  # Two pseudo-samples so the cross-sample rank step engages; this is the
  # matrix nmfClustering consumes inside IntegrateSpatialEcoTyper.
  scmeta_nmf <- data.frame(CellType = celltype, SE = cell_se,
                           row.names = colnames(V))
  mkfc <- function(tag, shift) {
    meta2 <- scmeta_nmf
    meta2$SE <- paste0(tag, "..", meta2$SE)
    SpatialEcoTyper:::ComputeFCs(V + shift, meta2, cluster = "SE",
                                 scale = TRUE, ncores = 1)
  }
  f1 <- mkfc("S1", 0); f2 <- mkfc("S2", 0.25)
  genes <- intersect(rownames(f1), rownames(f2))
  avg <- cbind(f1[genes, ], f2[genes, ])
  dump_dense(avg, "integrate_avgexprs")
  integ <- tick("Integrate",
                SpatialEcoTyper:::Integrate(avg, nfeatures = 200,
                                            min.features = 5, minibatch = 5000,
                                            ncores = 1, seed = 1))
  dump_dense(as.matrix(integ), "integrate_matrix")
  message("[R] nmf part done")
}

if (part == "nmf") { dump_json(timings, "timings_nmf"); quit(save = "no") }

# ====================================================================
# PART: main
# ====================================================================

# --------------------------------------------------------------------
# 1. PreprocessST
# --------------------------------------------------------------------
pp <- tick("PreprocessST",
           PreprocessST(normdata, metadata = m, min.cells = 5,
                        min.features = 10, X = "X", Y = "Y"))
dump_sparse(as(pp$expdat, "CsparseMatrix"), "preprocessst_expdat")
dump_df(pp$metadata, "preprocessst_metadata")

expdat   <- pp$expdat
metadata <- pp$metadata

# --------------------------------------------------------------------
# 2. Znorm -- on a 60-gene slice so the dense dump stays small.
# --------------------------------------------------------------------
sub <- expdat[1:60, , drop = FALSE]
dump_dense(tick("Znorm", Znorm(sub)), "znorm_nogroup")
dump_dense(Znorm(sub, groups = metadata$Region), "znorm_region")
dump_json(list(genes = rownames(sub), groups = metadata$Region), "znorm_input")

# --------------------------------------------------------------------
# 3. mostFrequent
# --------------------------------------------------------------------
mf_inputs <- list(
  celltype = metadata$CellType,
  region   = metadata$Region,
  # a deliberate tie: table() sorts levels, arrange(desc(Freq)) is stable, so
  # the lexicographically smallest of the tied levels wins.
  tie      = c("b", "b", "a", "a", "c"),
  numeric  = as.character(c(3, 3, 1, 1, 2, 2, 2))
)
dump_json(lapply(mf_inputs, function(x) mostFrequent(x)), "mostfrequent_out")
dump_json(mf_inputs, "mostfrequent_in")

# --------------------------------------------------------------------
# 4. GetSpatialMetacells (spotCoord = NULL -> internal grid)
# --------------------------------------------------------------------
lognorm <- expdat
if (max(lognorm) > 50) lognorm <- log1p(lognorm)
ncem <- tick("GetSpatialMetacells",
             GetSpatialMetacells(lognorm, metadata, X = "X", Y = "Y",
                                 CellType = "CellType", spotCoord = NULL,
                                 k = 20, radius = 50,
                                 min.cells.per.region = 1, ncores = 1))
dump_sparse(as(ncem, "CsparseMatrix"), "getspatialmetacells")

# --------------------------------------------------------------------
# 5. rankSparse -- closed-form sparse input with deliberate ties.
# --------------------------------------------------------------------
nr <- 40L; nc <- 12L
ii <- integer(0); jj <- integer(0); xx <- numeric(0)
for (j in seq_len(nc)) {
  rows <- seq(from = ((j * 3L) %% 7L) + 1L, to = nr, by = 3L)
  ii <- c(ii, rows); jj <- c(jj, rep(j, length(rows)))
  # `%% 5` guarantees repeated values inside a column -> average-tie ranks.
  xx <- c(xx, ((rows * 7L + j * 11L) %% 5L) + 0.5)
}
rs_in <- sparseMatrix(i = ii, j = jj, x = xx, dims = c(nr, nc),
                      dimnames = list(paste0("r", seq_len(nr)),
                                      paste0("c", seq_len(nc))))
dump_sparse(rs_in, "ranksparse_in")
dump_sparse(tick("rankSparse", rankSparse(rs_in)), "ranksparse_out")

# --------------------------------------------------------------------
# 6. matrixMultiply -- closed-form dense inputs.
# --------------------------------------------------------------------
mkmat <- function(nr, nc, a, b) {
  outer(seq_len(nr), seq_len(nc),
        function(i, j) sin(i * a + j * b) + (i * j) %% 7 / 7)
}
mm1 <- mkmat(30, 50, 0.3011, 0.7391)
mm2 <- mkmat(50, 220, 0.5171, 0.2311)
dimnames(mm1) <- list(paste0("a", 1:30), paste0("k", 1:50))
dimnames(mm2) <- list(paste0("k", 1:50), paste0("b", 1:220))
dump_dense(mm1, "matrixmultiply_in1")
dump_dense(mm2, "matrixmultiply_in2")
dump_dense(tick("matrixMultiply",
                matrixMultiply(mm1, mm2, minibatch = 100, ncores = 1)),
           "matrixmultiply_out")

# --------------------------------------------------------------------
# 7. SNF2 -- three deterministic similarity views over the SN grid.
#    Built from the fixture's own spatial-neighbourhood centres with a
#    Gaussian kernel at three bandwidths, then top-15 sparsified and
#    symmetrised.  No RNG, and the inputs are dumped so the Python side can
#    be fed exactly the same three matrices.
# --------------------------------------------------------------------
grid.size <- 70
gm <- metadata
gm$SpotID <- paste0("X", round(gm$X / grid.size), "_Y", round(gm$Y / grid.size))
spotco <- gm %>% group_by(SpotID) %>%
  summarise(X = median(X), Y = median(Y), .groups = "drop") %>% as.data.frame
rownames(spotco) <- spotco$SpotID
spotco <- spotco[sort(rownames(spotco)), ]
snf_ids <- rownames(spotco)[seq_len(min(150L, nrow(spotco)))]
co <- as.matrix(spotco[snf_ids, c("X", "Y")])
D <- as.matrix(dist(co))
Wall <- list()
for (jj2 in seq_len(3)) {
  sigma <- c(60, 95, 140)[jj2]
  Wj <- exp(-(D^2) / (2 * sigma^2))
  thr <- apply(Wj, 1, function(r) sort(r, decreasing = TRUE)[16])
  Wj[Wj < thr] <- 0
  Wj <- (Wj + t(Wj)) / 2
  dimnames(Wj) <- list(snf_ids, snf_ids)
  Wj <- as(as(as(Matrix(Wj, sparse = TRUE), "dMatrix"),
              "generalMatrix"), "CsparseMatrix")
  Wall[[jj2]] <- Wj
  dump_sparse(Wj, paste0("snf2_in", jj2))
}
fused <- tick("SNF2", SNF2(Wall, K = 10, t = 10, minibatch = 5000,
                           ncores = 1, verbose = FALSE))
dump_sparse(fused, "snf2_out")

# --------------------------------------------------------------------
# 8. SpatialEcoTyper (end to end, one sample) + AnnotateCells.
#    RunUMAP inside SpatialEcoTyper occasionally fails on this cluster
#    (uwot/annoy temp-file handling), so the call is guarded; the staged
#    reconstruction below is byte-for-byte what SpatialEcoTyper() does after
#    FindClusters and is always dumped.
# --------------------------------------------------------------------
res <- tryCatch(
  tick("SpatialEcoTyper",
       SpatialEcoTyper(normdata, m, outprefix = NULL, radius = 50,
                       resolution = 0.5, nfeatures = 300,
                       min.cts.per.region = 2, npcs = 20, min.cells = 5,
                       min.features = 10, iterations = 10, minibatch = 5000,
                       ncores = 1, k = 20, k.sn = 50, dropcell = TRUE)),
  error = function(e) {
    message("[R] SpatialEcoTyper() failed: ", conditionMessage(e)); NULL })

if (!is.null(res)) {
  dump_json(list(spot = colnames(res$obj), SE = as.character(res$obj$SE)),
            "spatialecotyper_spot")
  dump_df(res$metadata, "spatialecotyper_cells")
  # ---- AnnotateCells, on the object SpatialEcoTyper just returned ----
  ac <- tick("AnnotateCells",
             AnnotateCells(scmeta = metadata, obj = res$obj, col = "SE",
                           dropcell = TRUE))
  dump_df(ac, "annotatecells_drop")
  ac2 <- AnnotateCells(scmeta = metadata, obj = res$obj, col = "SE",
                       dropcell = FALSE)
  dump_df(ac2, "annotatecells_keep")
  dump_json(list(project.name = res$obj@project.name), "spatialecotyper_project")
} else {
  dump_json(list(failed = TRUE), "spatialecotyper_failed")
}

# --------------------------------------------------------------------
# 9. PartitionTissue (internal, exported via :::)
# --------------------------------------------------------------------
pt <- tick("PartitionTissue",
           PartitionTissue(m, nrow = 3, ncol = 5, X = "X", Y = "Y"))
dump_json(list(cell = rownames(pt), partition = pt$Partition),
          "partitiontissue")

# --------------------------------------------------------------------
# 10. InferNCells
# --------------------------------------------------------------------
for (av in c(5, 1.5, 12)) {
  dump_json(list(avg.number = av, ncells = as.integer(InferNCells(scdata, av))),
            paste0("inferncells_", gsub("[.]", "p", as.character(av))))
}

# --------------------------------------------------------------------
# 11. ComputeMetrics
# --------------------------------------------------------------------
for (met in c("F1", "F2", "precision", "recall")) {
  dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                            Sample = "Sample", metric = met),
             paste0("computemetrics_", tolower(met)))
}
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = "CellType",
                          Sample = "Sample", metric = "F1"),
           "computemetrics_f1_ct")
dump_dense(ComputeMetrics(m, SE = "SE", Pred = "cvPred", CellType = NULL,
                          Sample = NULL, metric = "F1"),
           "computemetrics_f1_nosample")

# --------------------------------------------------------------------
# 12. ComputeSEAbundanceBySN + SmoothSEAbundances
# --------------------------------------------------------------------
seab <- tick("ComputeSEAbundanceBySN",
             ComputeSEAbundanceBySN(m, spot_coords = NULL, radius = 50,
                                    grid.size = 50, X = "X", Y = "Y",
                                    SE = "SE", min.cells = 5))
dump_df(seab, "computeseabundancebysn")
sm <- tick("SmoothSEAbundances",
           SmoothSEAbundances(as.matrix(seab[, -c(1, 2)]), seab[, c("X", "Y")],
                              k = 7, X = "X", Y = "Y", include.self = TRUE,
                              min.neighbors = 3))
dump_df(sm, "smoothseabundances")

# --------------------------------------------------------------------
# 13. Coassociation (test = FALSE) + CoassociationTest
# --------------------------------------------------------------------
coassoc <- tick("Coassociation",
                Coassociation(m, Sample = "Sample", SE = "SEn",
                              CellType = "CellType", NonSE = "NonSE",
                              test = FALSE))
dump_dense(coassoc, "coassociation_index")
set.seed(1)
cp <- tick("CoassociationTest", CoassociationTest(coassoc, nperm = 1000))
dump_named(cp, "coassociationtest_pvals")
dump_named(attr(cp, "Zscore"), "coassociationtest_zscore")

# --------------------------------------------------------------------
# 14. Colocalization (ncores = 1 -- mclapply short-circuits to lapply, which
#     is the only bit-reproducible branch; see README "Known divergences" 4)
# --------------------------------------------------------------------
set.seed(1)
coloc <- tick("Colocalization",
              Colocalization(m, coords = c("X", "Y"), SE = "SE",
                             CellType = "CellType", radius = 50, k = 100,
                             min.cell = 10, nperm = 200, test = TRUE,
                             ncores = 1))
dump_dense(coloc$ColocIndex, "colocalization_index")
dump_named(coloc$Pval, "colocalization_pvals")
dump_named(attr(coloc$Pval, "Zscore"), "colocalization_zscore")

dump_json(timings, "timings_main")
message("[R] all per-function outputs written to ", outdir)
