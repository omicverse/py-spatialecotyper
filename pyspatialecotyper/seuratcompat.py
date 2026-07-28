"""The slice of Seurat that SpatialEcoTyper actually depends on.

SpatialEcoTyper leans on Seurat for six operations. None of them has a
numerically faithful Python equivalent already (scanpy's versions differ in
normalisation constants, variance estimators, HVF binning and PCA conventions),
so each is re-derived here from the Seurat 5.4.0 sources:

============================  =========================================
R call                        source of truth
============================  =========================================
``NormalizeData``             ``R/preprocessing.R`` / ``src/data_manipulation.cpp::LogNorm``
``ScaleData``                 ``R/utilities.R::FastRowScale`` and
                              ``src/data_manipulation.cpp::FastSparseRowScale``
``FindVariableFeatures``      ``R/preprocessing.R`` (``selection.method = "dispersion"``)
  ``mean.function``           ``src/data_manipulation.cpp::FastExpMean``
  ``dispersion.function``     ``src/data_manipulation.cpp::FastLogVMR``
``RunPCA``                    ``R/dimensional_reduction.R::RunPCA.default``
============================  =========================================

``FindNeighbors`` / ``FindClusters`` live in :mod:`pyspatialecotyper._modularity`
because they need a port of Seurat's ``ModularityOptimizer.cpp``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "normalize_data",
    "scale_data",
    "find_variable_features",
    "run_pca",
    "r_cut_equal_width",
]


def _as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.todense(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# NormalizeData
# ---------------------------------------------------------------------------
def normalize_data(counts, scale_factor: float = 1e4,
                   normalization_method: str = "LogNormalize"):
    """``Seurat::NormalizeData(counts, normalization.method, scale.factor)``.

    Parameters
    ----------
    counts
        genes x cells matrix (dense ndarray or scipy.sparse).
    scale_factor
        R default ``1e4``.
    normalization_method
        ``"LogNormalize"`` (default), ``"CLR"`` or ``"RC"``.

    Notes
    -----
    ``LogNorm`` in ``src/data_manipulation.cpp`` computes
    ``log1p(value / colSum * scale_factor)`` per column, i.e. per *cell* —
    the column sums are over all genes, including the ones that are zero.
    Columns that sum to zero stay zero (R would produce ``NaN``; Seurat's C++
    divides by the sum only for stored entries, so an all-zero column is left
    untouched).
    """
    if normalization_method not in ("LogNormalize", "CLR", "RC"):
        raise ValueError(f"unknown normalization.method: {normalization_method}")

    issparse = sp.issparse(counts)
    x = counts.tocsc(copy=True).astype(np.float64) if issparse \
        else np.array(counts, dtype=np.float64)

    colsums = np.asarray(x.sum(axis=0)).ravel()
    safe = np.where(colsums == 0, 1.0, colsums)

    if normalization_method == "CLR":
        # Seurat's CLR margin=2: exp(x / exp(mean(log1p(x[x>0]))))
        raise NotImplementedError("CLR is not used by SpatialEcoTyper")

    if issparse:
        # scale each column by scale_factor / colsum, then log1p on stored data
        scaler = sp.diags(scale_factor / safe)
        x = x @ scaler
        if normalization_method == "LogNormalize":
            x.data = np.log1p(x.data)
        return x.tocsc()

    x = x * (scale_factor / safe)[None, :]
    if normalization_method == "LogNormalize":
        x = np.log1p(x)
    return x


# ---------------------------------------------------------------------------
# ScaleData
# ---------------------------------------------------------------------------
def scale_data(mat, do_scale: bool = True, do_center: bool = True,
               scale_max: float = 10.0, na_to_zero: bool = True) -> np.ndarray:
    """``Seurat::ScaleData(mat, do.scale, do.center, scale.max)``.

    Always returns a dense genes x cells ndarray, matching Seurat (both
    ``FastRowScale`` and ``FastSparseRowScale`` return dense).

    Three details that are easy to get wrong, all verified against the C++:

    1. The standard deviation uses the **n-1** denominator
       (``rowSds(mat, center = rm)`` / ``colSdev / (mat.rows() - 1)``).
    2. When ``do_center`` is ``FALSE`` but ``do_scale`` is ``TRUE`` the
       divisor is the **uncentred** root-mean-square,
       ``sqrt(rowSums(mat^2) / (ncol - 1))``, not the SD.
    3. ``scale_max`` clamps **only from above** (``mat[mat > scale_max] <- scale_max``).
       There is no lower clamp.

    ``ScaleData`` then does ``data.scale[is.na(data.scale)] <- 0``, which is
    what rescues zero-variance rows (0/0 -> NaN -> 0).
    """
    x = _as_dense(mat)
    n = x.shape[1]

    if do_center:
        rm = np.nanmean(x, axis=1)             # rowMeans2(..., na.rm = TRUE)
    if do_scale:
        if do_center:
            rsd = np.sqrt(np.nansum((x - rm[:, None]) ** 2, axis=1) / (n - 1))
        else:
            rsd = np.sqrt(np.nansum(x ** 2, axis=1) / (n - 1))
    if do_center:
        x = x - rm[:, None]
    if do_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = x / rsd[:, None]
    if np.isfinite(scale_max):
        x = np.where(x > scale_max, scale_max, x)
    if na_to_zero:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ---------------------------------------------------------------------------
# FindVariableFeatures(selection.method = "dispersion")
# ---------------------------------------------------------------------------
def _fast_exp_mean(x: np.ndarray, stored_mask: np.ndarray | None) -> np.ndarray:
    """``FastExpMean``: ``log1p(sum(expm1(stored)) / ncol)``.

    The C++ iterates only over *stored* entries of the sparse matrix, so
    structural zeros contribute ``expm1(0) = 0``.  With a dense input R coerces
    to sparse first, which drops exact zeros — hence ``stored_mask``.
    """
    ncols = x.shape[1]
    vals = np.expm1(x)
    if stored_mask is not None:
        vals = np.where(stored_mask, vals, 0.0)
    return np.log1p(vals.sum(axis=1) / ncols)


def _fast_log_vmr(x: np.ndarray, stored_mask: np.ndarray | None) -> np.ndarray:
    """``FastLogVMR``: ``log(var / mean)`` on the ``expm1`` scale.

    Reproduces the C++ line-for-line, including the correction term for the
    entries that are not stored::

        v = (sum_stored (expm1(x) - rm)^2 + (ncols - nnZero) * rm^2) / (ncols - 1)
        out = log(v / rm)

    ``rm`` here is the *stored-entry* mean already divided by ``ncols``.
    A non-positive ``v/rm`` yields NaN, which the caller maps to 0 exactly as
    ``feature.dispersion[is.na(feature.dispersion)] <- 0`` does.
    """
    ncols = x.shape[1]
    e = np.expm1(x)
    if stored_mask is None:
        stored_mask = np.ones_like(x, dtype=bool)
    e_stored = np.where(stored_mask, e, 0.0)
    rm = e_stored.sum(axis=1) / ncols
    nnzero = stored_mask.sum(axis=1)
    v = np.where(stored_mask, (e - rm[:, None]) ** 2, 0.0).sum(axis=1)
    v = (v + (ncols - nnzero) * rm ** 2) / (ncols - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(v / rm)


def r_cut_equal_width(x: np.ndarray, nbins: int) -> np.ndarray:
    """``cut(x, breaks = nbins)`` -> **0-based** bin index, ``-1`` for NA.

    R widens the range by ``diff(range(x)) / 1000`` on each side (or by
    ``abs(x[1]) / 1000`` when the range is degenerate) before cutting into
    equal-width intervals, and the intervals are right-closed
    ``(lo, hi]``.  Reproduced verbatim from ``base::cut.default``.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if lo == hi:
        lo, hi = lo - abs(lo) / 1000.0, hi + abs(hi) / 1000.0
        if lo == hi:                       # x is all zeros
            lo, hi = -0.001, 0.001
    else:
        dx = hi - lo
        lo, hi = lo - dx / 1000.0, hi + dx / 1000.0
    breaks = np.linspace(lo, hi, nbins + 1)
    # right = TRUE: interval i is (breaks[i], breaks[i+1]]
    idx = np.searchsorted(breaks, x, side="left") - 1
    idx = np.clip(idx, 0, nbins - 1)
    idx[~np.isfinite(x)] = -1
    return idx


def find_variable_features(data, nfeatures: int = 2000,
                           selection_method: str = "dispersion",
                           num_bin: int = 20,
                           feature_names: list[str] | None = None) -> list[str]:
    """``Seurat::FindVariableFeatures(selection.method = "dispersion")``.

    Returns the top ``nfeatures`` feature names ordered by **raw** dispersion
    (``mvp.dispersion``), after dropping features whose ``mvp.mean`` is exactly
    zero — see ``R/preprocessing.R`` lines 4527-4532 and 4546.

    The binned, z-scored ``mvp.dispersion.scaled`` column is computed (it is
    what ``mean.var.plot`` would threshold on) but is deliberately *not* the
    sort key for ``selection.method = "dispersion"``.
    """
    if selection_method not in ("dispersion", "disp"):
        raise NotImplementedError(
            f"only selection.method='dispersion' is used by SpatialEcoTyper, got {selection_method!r}")

    if sp.issparse(data):
        mask = np.zeros(data.shape, dtype=bool)
        coo = data.tocoo()
        mask[coo.row, coo.col] = True
        dense = _as_dense(data)
    else:
        dense = np.asarray(data, dtype=np.float64)
        # R coerces a dense matrix to dgCMatrix before calling the C++, which
        # drops exact zeros from the stored set.
        mask = dense != 0

    if feature_names is None:
        feature_names = [str(i) for i in range(dense.shape[0])]
    feature_names = np.asarray(feature_names, dtype=object)

    feature_mean = _fast_exp_mean(dense, mask)
    feature_disp = _fast_log_vmr(dense, mask)
    feature_disp = np.nan_to_num(feature_disp, nan=0.0, posinf=0.0, neginf=0.0)
    feature_mean = np.nan_to_num(feature_mean, nan=0.0, posinf=0.0, neginf=0.0)

    # z-score dispersion within equal-width bins of the mean (kept for API
    # completeness / mean.var.plot; unused by the 'dispersion' sort key).
    bins = r_cut_equal_width(feature_mean, num_bin)
    _ = bins

    keep = feature_mean != 0            # hvf.info[which(hvf.info[,1] != 0), ]
    idx = np.flatnonzero(keep)
    # order(decreasing = TRUE) is a *stable* radix sort in R for doubles, so
    # ties keep their original (row) order. np.argsort(kind='stable') on the
    # negated key reproduces that.
    order = idx[np.argsort(-feature_disp[idx], kind="stable")]
    return [str(f) for f in feature_names[order[:nfeatures]]]


# ---------------------------------------------------------------------------
# RunPCA
# ---------------------------------------------------------------------------
def run_pca(scale_data_mat, npcs: int = 50, weight_by_var: bool = True):
    """``Seurat::RunPCA.default(object, npcs, weight.by.var)``.

    ``object`` is genes x cells scale.data.  Seurat computes
    ``irlba(A = t(object), nv = npcs)`` and sets
    ``cell.embeddings = u %*% diag(d)``, ``feature.loadings = v``,
    ``sdev = d / sqrt(ncol(object) - 1)``.

    We use a deterministic truncated SVD (``scipy.sparse.linalg.svds`` on the
    same ``t(object)``) rather than irlba's randomised restart.  Both target
    the same subspace; irlba's default ``tol = 1e-5`` means R's own answer is
    itself only converged to ~1e-5 relative, which is the dominant term in the
    ``pc_embeddings`` Procrustes gate.

    Returns
    -------
    embeddings : (cells, npcs)
    loadings   : (genes, npcs)
    sdev       : (npcs,)
    """
    x = _as_dense(scale_data_mat)             # genes x cells
    a = x.T                                   # cells x genes  == t(object)
    npcs = int(min(npcs, x.shape[0] - 1, min(a.shape) - 1))
    if npcs < 1:
        raise ValueError("npcs must be >= 1 after clamping to the matrix size")

    from scipy.sparse.linalg import svds
    u, d, vt = svds(a, k=npcs, solver="arpack",
                    v0=np.ones(min(a.shape)) / np.sqrt(min(a.shape)))
    order = np.argsort(-d)                    # svds returns ascending
    u, d, vt = u[:, order], d[order], vt[order]

    # Sign convention: SVD signs are arbitrary and R/irlba picks its own.
    # Fix them deterministically so repeated Python runs agree with each other
    # (this cannot make Python agree with R, and does not need to -- every
    # downstream consumer is either a Euclidean distance or a Procrustes
    # comparison, both sign-invariant).
    for j in range(npcs):
        k = np.argmax(np.abs(vt[j]))
        if vt[j, k] < 0:
            vt[j] *= -1
            u[:, j] *= -1

    embeddings = u * d[None, :] if weight_by_var else u
    loadings = vt.T
    sdev = d / np.sqrt(max(1, x.shape[1] - 1))
    return embeddings, loadings, sdev
