#!/usr/bin/env Rscript
# Dumps Integrate()'s per-cell-type rank-correlation networks and the fused
# matrix *before* rankSparse, so the Python port can be diffed at the exact
# point where the two diverge instead of only at the end.
#
#   Rscript r_integrate_internals.R <outdir>      (run AFTER r_nmf_driver.R)

suppressPackageStartupMessages({
  library(Matrix); library(SpatialEcoTyper); library(data.table); library(dplyr)
})
outdir <- commandArgs(trailingOnly = TRUE)[1]

dump <- function(x, name) {
  x <- as.matrix(x)
  fwrite(data.table(as.data.frame(x)), file.path(outdir, paste0(name, ".tsv.gz")), sep = "\t")
  writeLines(rownames(x), file.path(outdir, paste0(name, ".rows.txt")))
  writeLines(colnames(x), file.path(outdir, paste0(name, ".cols.txt")))
}

avg <- as.matrix(fread(file.path(outdir, "nmf_avgexprs.tsv.gz")))
rownames(avg) <- readLines(file.path(outdir, "nmf_avgexprs.rows.txt"))
colnames(avg) <- readLines(file.path(outdir, "nmf_avgexprs.cols.txt"))

nfeatures <- 200; min.features <- 5
celltypes <- gsub("\\.\\..*", "", rownames(avg))

cors_rank <- lapply(unique(celltypes), function(ct) {
  tmpdat <- avg[celltypes == ct, , drop = FALSE]
  if (nrow(tmpdat) <= min.features) return(NULL)
  tmpdat <- tmpdat[, colSums(!is.na(tmpdat)) > min.features, drop = FALSE]
  tmpdat <- tmpdat[, colSums(tmpdat != 0, na.rm = TRUE) > min.features, drop = FALSE]
  samples <- gsub("\\.\\..*", "", colnames(tmpdat))
  if (length(table(samples)) < 2 || min(table(samples)) < 3) return(NULL)
  vars <- lapply(unique(samples), function(ss)
    apply(tmpdat[, samples == ss, drop = FALSE], 1, var, na.rm = TRUE))
  vars <- do.call(cbind, vars)
  vars <- vars[apply(vars, 1, min) > 0, , drop = FALSE]
  var.ranks <- apply(-vars, 2, rank)
  vs <- rownames(var.ranks)[order(rowMeans(log(var.ranks)))]
  vs <- vs[1:min(length(vs), nfeatures)]
  tmpdat <- tmpdat[match(vs, rownames(tmpdat)), , drop = FALSE]
  tmpcor <- suppressWarnings(cor(as.matrix(tmpdat), method = "pearson"))
  tmpcor[is.na(tmpcor)] <- 0
  samples <- gsub("\\.\\..*", "", rownames(tmpcor))
  tmpcor <- lapply(unique(samples), function(ss) {
    s <- apply(tmpcor[samples == ss, , drop = FALSE], 2, rank)
    t(t(s) / colSums(s))
  })
  tmpcor <- do.call(rbind, tmpcor)
  tmpcor <- tmpcor[, rownames(tmpcor), drop = FALSE]
  tmpcor + t(tmpcor)
})
names(cors_rank) <- unique(celltypes)
cors_rank <- cors_rank[lengths(cors_rank) > 0]
cat("[R] networks:", paste(names(cors_rank), collapse = ", "), "\n")
writeLines(names(cors_rank), file.path(outdir, "integ_nets.txt"))
for (ct in names(cors_rank)) dump(cors_rank[[ct]], paste0("integ_net_", ct))

filled <- SpatialEcoTyper:::fillspots(cors_rank)
fused <- SNF2(filled, ncores = 1, minibatch = 5000, verbose = FALSE)
dump(as.matrix(fused), "integ_fused")
cat("[R] fused stored entries:", length(fused@x), "of", prod(dim(fused)), "\n")
cat("[R] fused exact zeros stored:", sum(fused@x == 0), "\n")
