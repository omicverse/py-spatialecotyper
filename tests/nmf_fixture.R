# Deterministic fixture shared by the NMF reference drivers.
#
# Closed-form arithmetic only -- no RNG, so the fixture itself never depends
# on which R session built it.  The "noise" term is a fixed irrational-phase
# sine so that no two columns are identical: duplicated or constant columns
# make cor() return NA, and an NA column makes NMF's connectivity stopping
# criterion produce a ragged `apply(h, 2, which.max)` and error out with
# "non-conformable arrays".

make_fixture <- function(ngene = 150, ncell = 240, nse = 12, nct = 4) {
  gset <- lapply(seq_len(nse), function(s) {
    lo <- ((s - 1) * 10) %% ngene + 1
    lo:min(lo + 9, ngene)
  })
  cell_se <- rep(paste0("SE", sprintf("%02d", seq_len(nse))),
                 each = ncell / nse)
  celltype <- rep(LETTERS[seq_len(nct)], length.out = ncell)

  V <- matrix(0, ngene, ncell)
  for (i in seq_len(ngene)) {
    for (j in seq_len(ncell)) {
      V[i, j] <- 1 + ((i * 7 + j * 13) %% 11) / 10 +
        0.35 * sin(i * 0.7391 + j * 0.3011)
    }
  }
  for (s in seq_len(nse)) {
    m <- cell_se == paste0("SE", sprintf("%02d", s))
    V[gset[[s]], m] <- V[gset[[s]], m] + 3
  }
  for (k in seq_len(nct)) {
    m <- celltype == LETTERS[k]
    rows <- seq(k, ngene, by = nct)
    V[rows, m] <- V[rows, m] + 1.2
  }
  V <- pmax(V, 0)
  rownames(V) <- paste0("G", seq_len(ngene))
  colnames(V) <- paste0("C", seq_len(ncell))
  list(V = V, cell_se = cell_se, celltype = celltype)
}
