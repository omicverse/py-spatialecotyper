"""``PreprocessST`` and ``Znorm`` -- the two data-conditioning steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .seuratcompat import scale_data

__all__ = ["preprocess_st", "znorm"]


def _nnz_per_row(x):
    if sp.issparse(x):
        return np.asarray((x > 0).sum(axis=1)).ravel()
    return (np.asarray(x) > 0).sum(axis=1)


def _nnz_per_col(x):
    if sp.issparse(x):
        return np.asarray((x > 0).sum(axis=0)).ravel()
    return (np.asarray(x) > 0).sum(axis=0)


def preprocess_st(expdat, metadata: pd.DataFrame,
                  min_cells: int = 3, min_features: int = 5,
                  X: str = "X", Y: str = "Y",
                  genes: list[str] | None = None,
                  cells: list[str] | None = None,
                  verbose: bool = True):
    """``PreprocessST(expdat, metadata, min.cells, min.features, X, Y)``.

    Drops genes detected in fewer than ``min_cells`` cells, then cells with
    fewer than ``min_features`` detected genes -- **in that order**, so the
    cell filter sees the already-gene-filtered matrix (``R/PreprocessST.R``).

    Returns ``(expdat, metadata, genes, cells)`` where ``genes``/``cells`` are
    the surviving names; R returns a list of two because its matrices carry
    their own dimnames.
    """
    metadata = pd.DataFrame(metadata)
    if genes is None:
        genes = list(range(expdat.shape[0]))
    if cells is None:
        cells = list(metadata.index)
    genes = np.asarray(genes, dtype=object)
    cells = np.asarray(cells, dtype=object)

    # metadata <- metadata[match(colnames(expdat), rownames(metadata)), ]
    metadata = metadata.loc[[c for c in cells]]

    keep_g = _nnz_per_row(expdat) >= min_cells
    n_drop = int((~keep_g).sum())
    if verbose and n_drop:
        print(f"Remove {n_drop} genes expressed in fewer than {min_cells} cells")
    expdat = expdat[keep_g, :] if sp.issparse(expdat) else np.asarray(expdat)[keep_g, :]
    genes = genes[keep_g]

    keep_c = _nnz_per_col(expdat) >= min_features
    n_drop = int((~keep_c).sum())
    if verbose and n_drop:
        print(f"Remove {n_drop} cells with fewer than {min_features} features")
    expdat = expdat[:, keep_c] if sp.issparse(expdat) else expdat[:, keep_c]
    cells = cells[keep_c]
    metadata = metadata.iloc[keep_c]

    # if(ncol(expdat) > 5000) expdat <- as(expdat, "sparseMatrix")
    if expdat.shape[1] > 5000 and not sp.issparse(expdat):
        expdat = sp.csc_matrix(expdat)

    metadata = metadata.copy()
    metadata["X"] = metadata[X].to_numpy()
    metadata["Y"] = metadata[Y].to_numpy()
    return expdat, metadata, list(genes), list(cells)


def znorm(mat, groups=None):
    """``Znorm(mat, groups)`` (``R/Znorm.R``).

    With ``groups = None`` this is plain ``Seurat::ScaleData``.  With groups it
    is a *weighted* z-score in which every group contributes equally: each cell
    gets weight ``1 / n_group``, so a large group cannot dominate the mean.

    The R code::

        wt        <- 1 / rgs[groups]                    # per-cell weight
        wtd.mean  <- rowSums(wt0 * mat) / sum(wt)
        wtd.var   <- rowSums(wt0 * (mat - wtd.mean)^2) / (sum(wt) - 1)
        wtd.var[wtd.var <= 0] <- 1
        (mat - wtd.mean) / sqrt(wtd.var)

    Note ``sum(wt)`` equals the number of distinct groups (each group's weights
    sum to 1), so the ``- 1`` denominator is ``n_groups - 1``, not ``n_cells - 1``.
    There is no ``scale.max`` clamp on this path.
    """
    if groups is None:
        return scale_data(mat)

    x = np.asarray(mat.todense(), dtype=np.float64) if sp.issparse(mat) \
        else np.asarray(mat, dtype=np.float64)
    groups = np.asarray([str(g) for g in np.asarray(groups).ravel()])
    if np.any(pd.isna(groups)):
        raise ValueError("The group labels can't be NAs.")
    if len(groups) != x.shape[1]:
        raise ValueError("Length of group labels does not match the number of "
                         "columns in the mat.")

    levels, counts = np.unique(groups, return_counts=True)
    per_group = dict(zip(levels, counts))
    wt = np.array([1.0 / per_group[g] for g in groups])
    sum_wt = wt.sum()

    wtd_mean = np.nansum(x * wt[None, :], axis=1) / sum_wt
    wtd_var = np.nansum(wt[None, :] * (x - wtd_mean[:, None]) ** 2, axis=1) / (sum_wt - 1)
    wtd_var = np.where(wtd_var <= 0, 1.0, wtd_var)   # Prevent 0 variance.
    return (x - wtd_mean[:, None]) / np.sqrt(wtd_var)[:, None]
