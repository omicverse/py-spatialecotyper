#!/usr/bin/env Rscript
# =====================================================================
# R reference for nmfClustering -- deliberately a SEPARATE script.
#
# `NMF:::nmf.stop.connectivity` keeps `.consold` inside a `local({})` closure,
# i.e. as process-global state.  Factorising matrices of different shapes in
# one session leaves a stale `.consold` behind and R errors out with
# "cons != .consold : non-conformable arrays".  A fresh R process per matrix
# shape is the only reliable way to obtain a reference.
#
#   Rscript r_nmfclust_driver.R <outdir>     (run AFTER r_nmf_driver.R)
# =====================================================================

suppressPackageStartupMessages({
  library(NMF); library(SpatialEcoTyper); library(jsonlite); library(data.table)
})
args   <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

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

# The clustering input is the *integrated* similarity matrix, i.e. the object
# nmfClustering actually receives inside IntegrateSpatialEcoTyper.
S <- as.matrix(data.table::fread(file.path(outdir, "nmf_integrated.tsv.gz")))
rownames(S) <- readLines(file.path(outdir, "nmf_integrated.rows.txt"))
colnames(S) <- readLines(file.path(outdir, "nmf_integrated.cols.txt"))
cat("[R] nmfClustering input:", nrow(S), "x", ncol(S), "\n")

fit <- nmfClustering(S, ranks = 4, nrun.per.rank = 10, seed = 2024, ncores = 1)
lab <- predict(fit)
dump_json(list(sample = names(lab), label = as.integer(lab),
               cophenetic = as.numeric(cophcor(fit))), "nmf_clustering_k4")
dump_dense(consensus(fit), "nmf_consensus_k4")

fit2 <- nmfClustering(S, ranks = 4, nrun.per.rank = 10, seed = 2024, ncores = 1)
dump_json(list(identical_labels = all(predict(fit2) == lab),
               max_abs_consensus_diff = max(abs(consensus(fit2) - consensus(fit)))),
          "nmf_clustering_R_vs_R")

message("[R] nmfClustering driver done")
