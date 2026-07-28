#!/usr/bin/env Rscript
# =====================================================================
# R reference for the REAL two-sample integration (upstream Tutorial 2):
# melanoma (SKCM) + colorectal (CRC) MERSCOPE subsets.
#
#   Rscript r_integration_driver.R <datadir> <outdir>
#
# Reproduces IntegrateSpatialEcoTyper()'s numerical path verbatim, skipping
# only its plotting tail -- `getColors` needs `pals`, which cannot be built
# in the reference R env, and no numeric output depends on it.
# =====================================================================

suppressPackageStartupMessages({
  library(Matrix); library(Seurat); library(dplyr); library(tidyr)
  library(data.table); library(NMF); library(SpatialEcoTyper); library(jsonlite)
})

args    <- commandArgs(trailingOnly = TRUE)
datadir <- args[1]
outdir  <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

dump_dense <- function(x, name) {
  x <- as.matrix(x)
  fwrite(data.table(as.data.frame(x)), file.path(outdir, paste0(name, ".tsv.gz")), sep = "\t")
  writeLines(as.character(rownames(x)), file.path(outdir, paste0(name, ".rows.txt")))
  writeLines(as.character(colnames(x)), file.path(outdir, paste0(name, ".cols.txt")))
}
dump_json <- function(x, name) {
  write_json(x, file.path(outdir, paste0(name, ".json")), auto_unbox = TRUE,
             digits = NA, na = "null")
}
dump_df <- function(x, name) {
  x <- as.data.frame(x); x <- cbind(.rowname = rownames(x), x)
  fwrite(x, file.path(outdir, paste0(name, ".tsv")), sep = "\t")
}

load_sample <- function(tag) {
  scdata <- fread(file.path(datadir, paste0(tag, "_subset_counts.tsv.gz")),
                  sep = "\t", header = TRUE, data.table = FALSE)
  rownames(scdata) <- scdata[, 1]
  scdata <- as.matrix(scdata[, -1])
  scmeta <- read.table(file.path(datadir, paste0(tag, "_subset_scmeta.tsv")),
                       sep = "\t", header = TRUE, row.names = 1)
  scmeta <- scmeta[colnames(scdata), ]
  list(data = NormalizeData(scdata, verbose = FALSE), meta = scmeta)
}

message(Sys.time(), " loading samples")
skcm <- load_sample("Melanoma1")
crc  <- load_sample("CRC2")
data_list <- list(SKCM = skcm$data, CRC = crc$data)
meta_list <- list(SKCM = skcm$meta, CRC = crc$meta)

set_list <- list()
for (s in names(data_list)) {
  message(Sys.time(), " SpatialEcoTyper: ", s)
  set_list[[s]] <- SpatialEcoTyper(data_list[[s]], meta_list[[s]],
                                   outprefix = NULL, radius = 50,
                                   nfeatures = 300, min.cts.per.region = 1,
                                   minibatch = 5000, ncores = 1)
  dump_json(list(spot = colnames(set_list[[s]]$obj),
                 SE = as.character(set_list[[s]]$obj$SE)),
            paste0("single_", s, "_se_spot"))
  dump_df(set_list[[s]]$metadata, paste0("single_", s, "_se_cells"))
  # the fused rank-transformed graph + SNN, so Python can be fed R's own graph
  gg <- set_list[[s]]$obj@graphs$RNA_snn
  Matrix::writeMM(as(gg, "CsparseMatrix"), file.path(outdir, paste0("single_", s, "_snn.mtx")))
  writeLines(colnames(gg), file.path(outdir, paste0("single_", s, "_snn.cols.txt")))
}
saveRDS(set_list, file.path(outdir, "SpatialEcoTyper_list.rds"))

# --------------------------------------------------------------------
# IntegrateSpatialEcoTyper(), numerical path only
# --------------------------------------------------------------------
subresolution <- 30; nfeatures <- 300; min.features <- 10
nmf_ranks <- 4:12; nrun.per.rank <- 10; min.coph <- 0.95; seed <- 1
sample_names <- names(data_list)

metadata_list <- lapply(set_list, function(x) {
  obj <- FindClusters(x$obj, resolution = subresolution, verbose = FALSE)
  obj$SE <- paste0("InitSE", obj$seurat_clusters)
  md <- AnnotateCells(x$metadata, obj)
  md <- md[!is.na(md$SE), ]
  colnames(md)[colnames(md) == "SE"] <- "InitSE"
  md
})
names(metadata_list) <- sample_names

data_list <- lapply(sample_names, function(s) {
  idx1 <- colnames(data_list[[s]]) %in% rownames(metadata_list[[s]])
  idx2 <- colSums(data_list[[s]] > 0) >= min.features
  data_list[[s]][, idx1 & idx2]
})
names(data_list) <- sample_names

metadata_list <- lapply(sample_names, function(s) {
  scmeta <- metadata_list[[s]]
  if (!"CID" %in% colnames(scmeta)) scmeta <- cbind(CID = rownames(scmeta), scmeta)
  else scmeta$CID <- rownames(scmeta)
  scmeta$Sample <- s
  scmeta$InitSE <- paste0(scmeta$Sample, "..", scmeta$InitSE)
  scmeta[match(colnames(data_list[[s]]), rownames(scmeta)), ]
})
names(metadata_list) <- sample_names

commoncols <- table(unlist(lapply(metadata_list, colnames)))
commoncols <- names(commoncols)[commoncols == max(commoncols)]
commoncols <- unique(c("CID", "Sample", "InitSE", "CellType", commoncols))
commoncols <- intersect(colnames(metadata_list[[1]]), commoncols)
commoncols <- setdiff(commoncols, c("SpotID", "Spot.X", "Spot.Y"))
metadatas <- do.call(rbind, lapply(metadata_list, function(x) x[, commoncols]))
metadatas <- metadatas %>% filter(!is.na(InitSE)) %>% as.data.frame
dump_df(metadatas, "integ_metadatas")

message(Sys.time(), " ComputeFCs")
avgexpr_list <- lapply(sample_names, function(ss)
  SpatialEcoTyper:::ComputeFCs(data_list[[ss]], metadata_list[[ss]],
                               cluster = "InitSE", Region = NULL,
                               scale = TRUE, ncores = 1))
genes <- table(unlist(lapply(avgexpr_list, rownames)))
genes <- na.omit(names(genes)[genes == length(avgexpr_list)])
avgexpr_list <- lapply(avgexpr_list, function(x) x[genes, ])
avgexprs <- do.call(cbind, avgexpr_list)
dump_dense(avgexprs, "integ_avgexprs")

message(Sys.time(), " Integrate")
integrated <- SpatialEcoTyper:::Integrate(avgexprs, nfeatures = nfeatures,
                                          min.features = min.features,
                                          minibatch = 5000, ncores = 1, seed = seed)
dump_dense(as.matrix(integrated), "integ_integrated")
message(Sys.time(), " done (nmfClustering runs in a separate process)")
