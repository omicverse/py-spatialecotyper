#!/usr/bin/env Rscript
# =====================================================================
# R reference for the NMF + integration layer.
#
#   Rscript r_nmf_driver.R <outdir>
#
# The fixture is fully deterministic (tests/nmf_fixture.R), so no RNG state
# leaks into it.  Each block also records whether R reproduces *itself*
# run-to-run -- that is what decides whether the Python port can be gated
# deterministically at all.
# =====================================================================

suppressPackageStartupMessages({
  library(Matrix); library(Seurat); library(dplyr)
  library(NMF); library(SpatialEcoTyper); library(jsonlite); library(data.table)
})

args   <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]
here   <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
source(file.path(here, "nmf_fixture.R"))

dump_dense <- function(x, name) {
  x <- as.matrix(x)
  data.table::fwrite(data.table::data.table(as.data.frame(x)),
                     file.path(outdir, paste0(name, ".tsv.gz")), sep = "\t")
  writeLines(as.character(rownames(x)), file.path(outdir, paste0(name, ".rows.txt")))
  writeLines(as.character(colnames(x)), file.path(outdir, paste0(name, ".cols.txt")))
}
dump_json <- function(x, name) {
  write_json(x, file.path(outdir, paste0(name, ".json")),
             auto_unbox = TRUE, digits = NA, na = "null")
}

fx <- make_fixture()
V <- fx$V; cell_se <- fx$cell_se; celltype <- fx$celltype
dump_dense(V, "nmf_V")
dump_json(list(cell = colnames(V), SE = cell_se, CellType = celltype), "nmf_cellse")

ses   <- sort(unique(cell_se))
Fracs <- matrix(0, ncol(V), length(ses), dimnames = list(colnames(V), ses))
Fracs[cbind(seq_len(ncol(V)), match(cell_se, ses))] <- 1
dump_dense(Fracs, "nmf_Fracs")

# --------------------------------------------------------------------
# 1. NMFGenerateW  (H fixed -> W-only KL updates)
#    Run twice from different RNG states: NMFGenerateW does NOT set a seed
#    before NMF::rnmf, so the initial W differs between the two runs.
# --------------------------------------------------------------------
set.seed(11)
W1 <- NMFGenerateW(Fracs, V, scale = TRUE, nfeature = 300, nfeature.per.se = 50)
set.seed(999)
W2 <- NMFGenerateW(Fracs, V, scale = TRUE, nfeature = 300, nfeature.per.se = 50)
dump_dense(W1, "nmf_W_seed11")
dump_dense(W2, "nmf_W_seed999")
common <- intersect(rownames(W1), rownames(W2))
A <- W1[common, , drop = FALSE]; B <- W2[common, , drop = FALSE]
dump_json(list(same_rownames = identical(rownames(W1), rownames(W2)),
               n_features_run1 = nrow(W1), n_features_run2 = nrow(W2),
               n_common = length(common),
               jaccard = length(common) / length(union(rownames(W1), rownames(W2))),
               max_abs_diff_common = max(abs(A - B)),
               cor_common = cor(as.vector(A), as.vector(B))),
          "nmf_W_R_vs_R")

# The pre-NMF `to_predict` matrix, so the Python side can be fed R's exact input.
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

# --------------------------------------------------------------------
# 2. NMFpredict  (W fixed -> H-only KL updates; set.seed(39) internally)
# --------------------------------------------------------------------
H1 <- NMFpredict(W1, V, scale = FALSE, ncell.per.run = 5000, sum2one = TRUE)
H2 <- NMFpredict(W1, V, scale = FALSE, ncell.per.run = 5000, sum2one = TRUE)
dump_dense(H1, "nmf_H")
dump_json(list(max_abs_diff = max(abs(H1 - H2))), "nmf_H_R_vs_R")
H3 <- NMFpredict(W1, V, scale = FALSE, ncell.per.run = 25, sum2one = TRUE)
dump_dense(H3, "nmf_H_chunked")

# --------------------------------------------------------------------
# 3. ComputeFCs + Znorm
# --------------------------------------------------------------------
scmeta <- data.frame(CellType = celltype, SE = cell_se, row.names = colnames(V))
dump_dense(SpatialEcoTyper:::ComputeFCs(V, scmeta, cluster = "SE",
                                        scale = TRUE, ncores = 1), "nmf_computefcs")
dump_dense(SpatialEcoTyper:::ComputeFCs(V, scmeta, cluster = "SE",
                                        scale = FALSE, ncores = 1), "nmf_computefcs_noscale")
dump_dense(Znorm(V, groups = rep(c("R1", "R2"), length.out = ncol(V))), "nmf_znorm")
dump_dense(Znorm(V), "nmf_znorm_nogroup")

# --------------------------------------------------------------------
# 4. Integrate (two pseudo-samples so the cross-sample rank step engages)
# --------------------------------------------------------------------
mkfc <- function(tag, shift) {
  meta2 <- scmeta
  meta2$SE <- paste0(tag, "..", meta2$SE)
  SpatialEcoTyper:::ComputeFCs(V + shift, meta2, cluster = "SE",
                               scale = TRUE, ncores = 1)
}
f1 <- mkfc("S1", 0); f2 <- mkfc("S2", 0.25)
genes <- intersect(rownames(f1), rownames(f2))
avg <- cbind(f1[genes, ], f2[genes, ])
dump_dense(avg, "nmf_avgexprs")
integ <- SpatialEcoTyper:::Integrate(avg, nfeatures = 200, min.features = 5,
                                     minibatch = 5000, ncores = 1, seed = 1)
dump_dense(as.matrix(integ), "nmf_integrated")

message("[R] nmf driver done")
