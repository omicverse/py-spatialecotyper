#!/usr/bin/env Rscript
# =====================================================================
# py-spatialecotyper — R reference runner (the executable spec).
#
# Runs upstream SpatialEcoTyper 1.0.4 on the canonical fixture and dumps
# EVERY intermediate of the single-sample pipeline so the Python port can
# be diffed stage-by-stage, not just end-to-end.
#
# Usage:
#   Rscript r_reference_driver.R <counts.tsv.gz> <scmeta.tsv> <outdir> [ncells]
#
# Invoked under $R_TEST_ENV (R 4.4.3 + Seurat 5.4.0 + NMF 0.28).
# =====================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(dplyr)
  library(data.table)
  library(SpatialEcoTyper)
  library(jsonlite)
})

args     <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]
meta_f   <- args[2]
outdir   <- args[3]
ncells   <- if (length(args) >= 4) as.integer(args[4]) else NA_integer_
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------
# Portable dump helpers.  Sparse -> MatrixMarket + dimnames; dense -> tsv.
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

t0 <- Sys.time()
timings <- list()
tick <- function(label, expr) {
  s <- Sys.time(); v <- force(expr); e <- Sys.time()
  timings[[label]] <<- as.numeric(difftime(e, s, units = "secs"))
  message(sprintf("[R] %-28s %8.2fs", label, timings[[label]]))
  v
}

# --------------------------------------------------------------------
# 0. Load fixture
# --------------------------------------------------------------------
scdata <- fread(counts_f, sep = "\t", header = TRUE, data.table = FALSE)
rownames(scdata) <- scdata[, 1]
scdata <- as.matrix(scdata[, -1])
scmeta <- read.table(meta_f, sep = "\t", header = TRUE, row.names = 1)
scmeta <- scmeta[colnames(scdata), ]

# Deterministic spatial crop for the CI fixture: keep the `ncells` cells
# closest to the tissue centroid (no RNG involved).
if (!is.na(ncells) && ncells < ncol(scdata)) {
  cx <- median(scmeta$X); cy <- median(scmeta$Y)
  d  <- (scmeta$X - cx)^2 + (scmeta$Y - cy)^2
  keep <- order(d)[seq_len(ncells)]
  keep <- sort(keep)
  scdata <- scdata[, keep, drop = FALSE]
  scmeta <- scmeta[keep, , drop = FALSE]
}
message("[R] fixture: ", nrow(scdata), " genes x ", ncol(scdata), " cells")

normdata <- NormalizeData(scdata, verbose = FALSE)
dump_sparse(as(normdata, "CsparseMatrix"), "00_normdata")
dump_df(scmeta, "00_scmeta")

# --------------------------------------------------------------------
# Pipeline parameters (upstream defaults, pinned)
# --------------------------------------------------------------------
radius <- 50; resolution <- 0.5; nfeatures <- 300
min.cts.per.region <- 2; npcs <- 20; min.cells <- 5; min.features <- 10
iterations <- 10; minibatch <- 5000; ncores <- 4
grid.size <- round(radius * 1.4); k <- 20; k.sn <- 50
dump_json(list(radius = radius, resolution = resolution, nfeatures = nfeatures,
               min.cts.per.region = min.cts.per.region, npcs = npcs,
               min.cells = min.cells, min.features = min.features,
               iterations = iterations, minibatch = minibatch,
               grid.size = grid.size, k = k, k.sn = k.sn), "params")

# --------------------------------------------------------------------
# 1. PreprocessST
# --------------------------------------------------------------------
tmp <- tick("PreprocessST", PreprocessST(normdata, metadata = scmeta,
                                         min.cells = min.cells,
                                         min.features = min.features))
normdata <- tmp$expdat
metadata <- tmp$metadata
dump_sparse(as(normdata, "CsparseMatrix"), "01_preprocess_expdat")
dump_df(metadata, "01_preprocess_metadata")

# --------------------------------------------------------------------
# 2. Spatial-neighbourhood metadata (the SpotID grid)
# --------------------------------------------------------------------
ncmeta <- metadata
ncmeta$SpotID <- paste0("X", round(ncmeta$X / grid.size),
                        "_Y", round(ncmeta$Y / grid.size))
ncmeta <- ncmeta %>% group_by(SpotID) %>%
  mutate(X = median(X), Y = median(Y)) %>% as.data.frame
ncmeta <- ncmeta[, colnames(ncmeta) != "CID"]
tmpmedian <- ncmeta %>% group_by(SpotID, X, Y) %>%
  summarise(across(where(is.numeric), median), .groups = "drop")
tmp_freq <- ncmeta %>% group_by(SpotID, X, Y) %>%
  summarise(across(where(is.character) | where(is.factor) | where(is.logical),
                   mostFrequent), .groups = "drop")
ncmeta <- merge(tmpmedian, tmp_freq, by = c("SpotID", "X", "Y"), all = TRUE) %>%
  as.data.frame
rownames(ncmeta) <- ncmeta$SpotID
dump_df(ncmeta, "02_ncmeta")

if (max(normdata) > 50) normdata <- log1p(normdata)

# --------------------------------------------------------------------
# 3. GetSpatialMetacells
# --------------------------------------------------------------------
ncem <- tick("GetSpatialMetacells",
             GetSpatialMetacells(normdata, metadata, spotCoord = ncmeta,
                                 k = k, radius = radius, ncores = ncores))
dump_sparse(as(ncem, "CsparseMatrix"), "03_metacells")

# --------------------------------------------------------------------
# 4. GetPCList
# --------------------------------------------------------------------
emb_list <- tick("GetPCList",
                 SpatialEcoTyper:::GetPCList(ncem, nfeatures = nfeatures,
                                             min.cells = min.cells,
                                             min.features = min.features,
                                             ncores = ncores))
writeLines(names(emb_list), file.path(outdir, "04_emb_celltypes.txt"))
for (ct in names(emb_list)) dump_dense(emb_list[[ct]], paste0("04_emb_", ct))

# --------------------------------------------------------------------
# 5. GetSNList
# --------------------------------------------------------------------
snlist <- tick("GetSNList",
               SpatialEcoTyper:::GetSNList(emb_list, npcs = npcs,
                                           min.cts.per.region = min.cts.per.region,
                                           k = k.sn, ncores = ncores))
writeLines(names(snlist), file.path(outdir, "05_sn_celltypes.txt"))
for (ct in names(snlist)) dump_sparse(snlist[[ct]], paste0("05_sn_", ct))

# --------------------------------------------------------------------
# 6. SNF2 + rankSparse
# --------------------------------------------------------------------
fused <- tick("SNF2", SNF2(snlist, t = iterations, minibatch = minibatch,
                           ncores = ncores, verbose = FALSE))
dump_sparse(fused, "06_snf_fused")
ranked <- tick("rankSparse", rankSparse(fused))
rownames(ranked) <- gsub("_", "-", rownames(ranked))
dump_sparse(ranked, "07_rank_sparse")

# --------------------------------------------------------------------
# 7. Seurat block: ScaleData -> PCA -> SNN -> Louvain
# --------------------------------------------------------------------
obj <- CreateSeuratObject(ranked, meta.data = ncmeta[colnames(ranked), ])
obj[["RNA"]]$data <- obj[["RNA"]]$counts
VariableFeatures(obj) <- rownames(obj)
obj <- ScaleData(obj, do.scale = FALSE, verbose = FALSE)
dump_dense(GetAssayData(obj, layer = "scale.data"), "08_scaled")
obj <- RunPCA(obj, verbose = FALSE)
dump_dense(Embeddings(obj, "pca"), "09_pca")
obj <- FindNeighbors(obj, dims = 1:10, verbose = FALSE)
dump_sparse(obj@graphs$RNA_snn, "10_snn")
obj <- FindClusters(obj, resolution = resolution, verbose = FALSE)
dump_json(list(spot = colnames(obj),
               cluster = as.character(obj$seurat_clusters)), "11_clusters")

# --------------------------------------------------------------------
# 8. SE labels + AnnotateCells, reconstructed from the staged objects.
#
# This is byte-for-byte what SpatialEcoTyper() does after FindClusters --
# `obj$SE <- paste0("NewSE", cluster + 1)`, a project.name that encodes the
# radius, then AnnotateCells.  Reconstructing it (rather than re-running the
# whole public entry point a second time) keeps the driver to one pass over
# the data and skips RunUMAP, whose uwot/annoy temp-file handling is fragile
# on this cluster and whose output no downstream step consumes.
# --------------------------------------------------------------------
obj$SE <- paste0("NewSE", as.numeric(as.character(obj$seurat_clusters)) + 1)
obj@project.name <- paste0("ref_radius", radius, "_nfeatures", nfeatures,
                           "_npcs", npcs, "_k.sn", k.sn, "_k", k)
dump_json(list(spot = colnames(obj), SE = as.character(obj$SE)), "12_se_spot")
cellmeta <- AnnotateCells(metadata, obj, dropcell = TRUE)
dump_df(cellmeta, "12_se_cells")

# --------------------------------------------------------------------
# 9. The public entry point itself, as an end-to-end cross-check.
#    Guarded: RunUMAP inside it can fail on this cluster (uwot/annoy tempfile).
# --------------------------------------------------------------------
res <- tryCatch(
  tick("SpatialEcoTyper", SpatialEcoTyper(normdata, metadata,
                                          outprefix = NULL, radius = radius,
                                          resolution = resolution,
                                          nfeatures = nfeatures,
                                          min.cts.per.region = min.cts.per.region,
                                          npcs = npcs, min.cells = min.cells,
                                          min.features = min.features,
                                          iterations = iterations,
                                          minibatch = minibatch, ncores = 1,
                                          k = k, k.sn = k.sn, dropcell = TRUE)),
  error = function(e) { message("[R] SpatialEcoTyper() failed: ", conditionMessage(e)); NULL })
if (!is.null(res)) {
  saveRDS(res, file.path(outdir, "13_spatialecotyper_result.rds"))
  dump_json(list(spot = colnames(res$obj), SE = as.character(res$obj$SE)), "13_se_spot")
  dump_df(res$metadata, "13_se_cells")
}

dump_json(timings, "timings")
message("[R] total ", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s")
