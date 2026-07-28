"""Downstream statistics of SpatialEcoTyper, ported one-for-one from R.

Every public function here is a direct translation of a single function in
``SpatialEcoTyper-ref/R/``:

============================  ==========================================
R                             Python
============================  ==========================================
``Coassociation``             :func:`coassociation`
``CoassociationTest``         :func:`coassociation_test`
``.colocalization``           :func:`_colocalization`
``Colocalization``            :func:`colocalization`
``ColocalizationMetaAnalysis``:func:`colocalization_meta_analysis`
``ComputeMetrics``            :func:`compute_metrics`
``.moran``                    :func:`_moran`
``ComputeNormalizedMoranI``   :func:`compute_normalized_moran_i`
``buildKNNWeights``           :func:`build_knn_weights`
``aggregateByWeights``        :func:`aggregate_by_weights`
``ComputeSEAbundanceBySN``    :func:`compute_se_abundance_by_sn`
``SmoothSEAbundances``        :func:`smooth_se_abundances`
``CreatePseudobulks``         :func:`create_pseudobulks`
``InferNCells``               :func:`infer_ncells`
``PartitionTissue``           :func:`partition_tissue`
============================  ==========================================

Randomness
----------
R draws its permutations from the *global* ``.Random.seed`` stream.  These
functions do the same, through :mod:`pyspatialecotyper.rrandom`, whose
Mersenne-Twister is bit-identical to R 4.4.3.  So the Python analogue of::

    set.seed(1); Colocalization(scmeta, nperm = 200, ncores = 1)

is::

    rrandom.set_seed(1); colocalization(scmeta, nperm=200, ncores=1)

There is deliberately no ``seed=`` / ``rng=`` argument: adding one would let
callers desynchronise from R.

Two caveats that come straight from the R side:

* ``Colocalization`` runs its permutations under ``parallel::mclapply``.  With
  ``mc.cores > 1`` mclapply forks and each child reseeds itself from its own
  PID, so **R itself is not reproducible run-to-run for ncores > 1** (verified
  empirically on R 4.4.3).  Only ``ncores = 1`` — where ``mclapply`` short-
  circuits to ``lapply`` — is reproducible.  This port always consumes the
  stream sequentially, i.e. it reproduces the ``ncores = 1`` behaviour; the
  ``ncores`` argument is accepted for signature parity and otherwise ignored.
* ``ComputeNormalizedMoranI`` and ``CoassociationTest`` use plain ``lapply``
  and are fully reproducible.

Sorting
-------
R's ``sort()`` on character vectors collates in ``LC_COLLATE`` (here
``en_US.UTF-8``), while dplyr's ``group_by()``/``arrange()``/``summarise()``
order groups in the **C locale**.  Python's :func:`sorted` is code-point order,
i.e. exactly the C locale.  For ASCII labels that differ before any
case/punctuation tie-break (``SE1_CD4T``, ``NonSE_B``, ``X38_Y-127`` …) the two
agree; :func:`_sort` is the single place that would need a locale hook if a
caller used labels where they do not (e.g. ``SE1_x`` vs ``SE10_x``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import norm

from . import rrandom

__all__ = [
    "coassociation",
    "coassociation_test",
    "colocalization",
    "colocalization_meta_analysis",
    "compute_metrics",
    "compute_normalized_moran_i",
    "build_knn_weights",
    "aggregate_by_weights",
    "compute_se_abundance_by_sn",
    "smooth_se_abundances",
    "create_pseudobulks",
    "infer_ncells",
    "partition_tissue",
    "NamedSparse",
]


# ======================================================================
# Small shared helpers
# ======================================================================

def _sort(values):
    """R's ``sort()`` on a character vector (see the module docstring)."""
    return sorted(values)


def _unique(values):
    """R's ``unique()`` — first-appearance order, *not* sorted."""
    seen = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen)


def _sort_unique(values):
    """R's ``sort(unique(x))``."""
    return _sort(_unique(values))


def _match(needles, haystack):
    """R's ``match(needles, haystack)`` → 0-based positions, ``-1`` for no match.

    R returns 1-based positions and ``NA`` for misses; the ``-1`` sentinel is
    the caller's cue to emit an all-``NA`` row/column, which is what indexing
    an R matrix with ``NA`` does.
    """
    pos = {}
    for i, h in enumerate(haystack):
        pos.setdefault(h, i)
    return np.array([pos.get(n, -1) for n in needles], dtype=np.int64)


def _reindex_matrix(mat, rows, cols, want_rows, want_cols):
    """``mat[match(want_rows, rows), match(want_cols, cols)]`` with R's NA fill."""
    ri = _match(want_rows, rows)
    ci = _match(want_cols, cols)
    out = np.full((len(want_rows), len(want_cols)), np.nan, dtype=np.float64)
    rok = ri >= 0
    cok = ci >= 0
    if rok.any() and cok.any():
        sub = mat[np.ix_(ri[rok], ci[cok])]
        out[np.ix_(np.flatnonzero(rok), np.flatnonzero(cok))] = sub
    return out


def _r_sd(a, axis=0):
    """R's ``sd()`` — the *sample* standard deviation (denominator ``n - 1``)."""
    return np.std(a, axis=axis, ddof=1)


def _r_median(a):
    """R's ``median()`` — for even length, the mean of the two middle values."""
    return float(np.median(np.asarray(a, dtype=np.float64)))


def _r_round(a):
    """R's ``round()`` — round half to *even*, same rule as :func:`numpy.round`."""
    return np.round(a)


def _cor_pairwise_complete(mat):
    """``cor(mat, method = "pearson", use = "pairwise.complete.obs")``.

    For every column pair the mean/sd are recomputed on just the rows where
    both columns are observed (R's ``COV_PAIRWISE``).  Fewer than two complete
    rows, or a zero standard deviation in either column, yields ``NA`` — R
    warns ("the standard deviation is zero") and returns ``NA``; the warning is
    suppressed by the caller in ``Coassociation``.  The result is clamped into
    ``[-1, 1]`` exactly as R's ``CLAMP`` macro does.
    """
    mat = np.asarray(mat, dtype=np.float64)
    p = mat.shape[1]
    out = np.full((p, p), np.nan, dtype=np.float64)
    ok = np.isfinite(mat)
    for i in range(p):
        for j in range(i, p):
            m = ok[:, i] & ok[:, j]
            n = int(m.sum())
            if n < 2:
                continue
            x = mat[m, i]
            y = mat[m, j]
            x = x - x.mean()
            y = y - y.mean()
            # R divides both the covariance and the two sds by (n - 1); the
            # factor cancels, so it is dropped here.
            sxx = float(x @ x)
            syy = float(y @ y)
            if sxx <= 0.0 or syy <= 0.0:      # R: sd == 0 -> NA + warning
                continue
            r = float(x @ y) / np.sqrt(sxx * syy)
            r = 1.0 if r >= 1.0 else (-1.0 if r <= -1.0 else r)   # CLAMP
            out[i, j] = out[j, i] = r
    return out


class NamedSparse:
    """A CSR matrix carrying R-style ``dimnames``.

    R's ``buildKNNWeights`` returns a ``dgCMatrix`` whose row/column names are
    load-bearing — ``aggregateByWeights`` matches on them.  ``scipy.sparse`` has
    no dimnames, so this thin wrapper carries them alongside.
    """

    __slots__ = ("mat", "index", "columns")

    def __init__(self, mat, index=None, columns=None):
        self.mat = sp.csr_matrix(mat)
        n, m = self.mat.shape
        self.index = list(index) if index is not None else list(range(n))
        self.columns = list(columns) if columns is not None else list(range(m))

    @property
    def shape(self):
        return self.mat.shape

    def toarray(self):
        return self.mat.toarray()

    def to_frame(self):
        return pd.DataFrame(self.toarray(), index=self.index, columns=self.columns)

    def __repr__(self):                                     # pragma: no cover
        return (f"<NamedSparse {self.mat.shape[0]}x{self.mat.shape[1]} "
                f"nnz={self.mat.nnz}>")


def _knn(ref, query, k):
    """``RANN::nn2(data = ref, query = query, k = k)`` → (indices, distances).

    RANN searches an exact (``eps = 0``) kd-tree and returns *Euclidean*
    distances sorted ascending, with the 0-distance self-match first when
    ``query`` is ``ref``.  ``scipy.spatial.cKDTree`` has the same contract.
    Indices come back 0-based here; R's are 1-based.
    """
    tree = cKDTree(np.asarray(ref, dtype=np.float64))
    dist, idx = tree.query(np.asarray(query, dtype=np.float64), k=int(k))
    if np.ndim(idx) == 1:                    # k == 1 → scipy drops the axis
        idx = idx[:, None]
        dist = dist[:, None]
    return idx, dist


def _index_names(obj, n):
    """Row names of a data.frame-like, defaulting to R's ``NULL`` rownames."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return [str(v) for v in obj.index]
    return [str(i) for i in range(n)]


# ======================================================================
# PartitionTissue.R
# ======================================================================

def partition_tissue(meta, nrow=2, ncol=2, x="X", y="Y"):
    """``PartitionTissue`` — label each cell with its tissue subregion.

    Parameters
    ----------
    meta : pandas.DataFrame
        Cell metadata; must contain the ``x`` and ``y`` coordinate columns.
    nrow : int
        Number of divisions along the **X** axis (R line 34 uses ``X`` for the
        "row" group, which reads backwards but is reproduced verbatim).
    ncol : int
        Number of divisions along the **Y** axis.
    x, y : str
        Column names holding the x/y coordinates.

    Returns
    -------
    pandas.DataFrame
        A copy of ``meta`` with an extra ``Partition`` column of the form
        ``"<row>_<col>"``.
    """
    out = meta.copy()
    xv = np.asarray(meta[x], dtype=np.float64)
    yv = np.asarray(meta[y], dtype=np.float64)
    # R lines 30-31: the 1.000001 fudge keeps the maximum coordinate inside the
    # last bin instead of spilling into bin `nrow`.
    xlen = (xv.max() - xv.min()) * 1.000001
    ylen = (yv.max() - yv.min()) * 1.000001
    row_width = xlen / nrow
    col_width = ylen / ncol
    row_group = np.floor((xv - xv.min()) / row_width)
    col_group = np.floor((yv - yv.min()) / col_width)
    # R lines 34-36: `floor` yields a double, and paste0 renders e.g. 0 as "0".
    out["Partition"] = [f"{int(r)}_{int(c)}" for r, c in zip(row_group, col_group)]
    return out


# ======================================================================
# InferNCells.R
# ======================================================================

def infer_ncells(normdat, avg_number=5):
    """``InferNCells`` — infer a per-spot cell count from total expression.

    Parameters
    ----------
    normdat : numpy.ndarray | pandas.DataFrame | scipy.sparse matrix
        Genes (rows) x spots (columns) expression.
    avg_number : float
        Target mean number of cells per spot.  Default 5 (Visium).

    Returns
    -------
    numpy.ndarray of int64
        One inferred cell count per column of ``normdat``, floored at 1.
    """
    if not np.isscalar(avg_number) or isinstance(avg_number, (bool, str)) \
            or not np.isreal(avg_number) or float(avg_number) <= 0:
        raise ValueError("'avg_number' must be a single positive numeric value.")
    avg_number = float(avg_number)

    if sp.issparse(normdat):
        total = np.asarray(normdat.sum(axis=0)).ravel()
    else:
        arr = np.asarray(normdat.values if isinstance(normdat, pd.DataFrame)
                         else normdat, dtype=np.float64)
        # R line 25: colSums(..., na.rm = TRUE)
        total = np.nansum(arr, axis=0)

    if total.size == 0:
        raise ValueError("Input contains no columns.")

    if np.all(total == total[0]):
        # R lines 31-33: a flat profile gets the target count everywhere.
        return np.full(total.size, int(_r_round(avg_number)), dtype=np.int64)

    slope = (avg_number - 1.0) / (total.mean() - total.min())
    intercept = 1.0 - slope * total.min()
    ncells = _r_round(slope * total + intercept)
    ncells[ncells < 1] = 1
    # R line 43: as.integer() truncates towards zero; the values are already
    # whole numbers after round(), so the cast is exact.
    return ncells.astype(np.int64)


# ======================================================================
# ComputeMetrics.R
# ======================================================================

def _row_normalised_crosstab(row_labels, col_labels, levels):
    """``group_by(A) %>% count(B) %>% mutate(n / sum(n)) %>% pivot_wider``.

    Reindexed to ``levels`` on both axes with R's ``NA``-for-missing rule,
    the ``NA``s then replaced by 0 (R lines 57/69) and rows renormalised
    (R lines 58/70) — an all-absent level leaves a 0/0 = ``NaN`` row.
    """
    rl = np.asarray(row_labels, dtype=object)
    cl = np.asarray(col_labels, dtype=object)
    ct = pd.crosstab(pd.Series(rl), pd.Series(cl))
    mat = _reindex_matrix(ct.values.astype(np.float64),
                          [str(v) for v in ct.index], [str(v) for v in ct.columns],
                          levels, levels)
    mat[np.isnan(mat)] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def compute_metrics(scmeta, se="SE", pred="cvPred", cell_type=None,
                    sample="Sample", metric="F1"):
    """``ComputeMetrics`` — concordance between true and predicted SE labels.

    Parameters
    ----------
    scmeta : pandas.DataFrame
        Single-cell metadata.
    se : str
        Column holding the *true* SE labels.
    pred : str
        Column holding the *predicted* SE labels.
    cell_type : str or None
        Optional cell-type column.  When given, ``scmeta`` is first sorted by
        ``(cell_type, se)`` and both label columns are suffixed with the cell
        type (R lines 40-43), so the metric is computed per cell state.
    sample : str or None
        Sample column.  ``None`` pools every cell into one sample.
    metric : {"F1", "F2", "precision", "recall"}
        Which statistic to return.

    Returns
    -------
    pandas.DataFrame
        Levels x levels, averaged across samples over the samples where the
        entry is defined.
    """
    scmeta = scmeta.copy()
    # R lines 34-38
    scmeta["Sample"] = "Sample" if sample is None else scmeta[sample].values

    if cell_type is not None:
        # R line 40: arrange() collates in the C locale (= Python code-point
        # order) and is stable, so ties keep their original relative order.
        # Stable-sort on the *last* key first to get a multi-key stable sort.
        se_key = np.asarray(scmeta[se], dtype=object).astype(str)
        ct_key = np.asarray(scmeta[cell_type], dtype=object).astype(str)
        idx = np.arange(len(scmeta))
        idx = idx[np.argsort(se_key[idx], kind="stable")]
        idx = idx[np.argsort(ct_key[idx], kind="stable")]
        scmeta = scmeta.iloc[idx]
        # R lines 41-42
        ct_vals = scmeta[cell_type].astype(str).values
        scmeta[se] = scmeta[se].astype(str).values + "_" + ct_vals
        scmeta[pred] = scmeta[pred].astype(str).values + "_" + ct_vals

    # R line 44: unique() keeps first-appearance order, which is what fixes the
    # row/column order of the returned matrix.
    ses = _unique(scmeta[se].astype(str).values)

    per_sample = []
    for ss in _unique(scmeta["Sample"].astype(str).values):
        metas = scmeta[scmeta["Sample"].astype(str).values == ss]
        # R lines 47-58: rows = true SE, cols = predicted SE.
        recalls = _row_normalised_crosstab(metas[se].astype(str).values,
                                           metas[pred].astype(str).values, ses)
        # R lines 60-70: rows = predicted SE, cols = true SE.
        precision = _row_normalised_crosstab(metas[pred].astype(str).values,
                                             metas[se].astype(str).values, ses)

        if metric == "recall":
            out = recalls
        elif metric == "precision":
            out = precision
        else:
            # R lines 78-82: the NaN rows are zeroed *before* the F-score, so an
            # absent level gives 0/0 = NaN and drops out of the average below.
            recalls = np.nan_to_num(recalls, nan=0.0)
            precision = np.nan_to_num(precision, nan=0.0)
            beta = 2.0 if metric == "F2" else 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                out = ((1 + beta ** 2) * precision * recalls /
                       (beta ** 2 * precision + recalls))
        per_sample.append(out)

    # R lines 86-88: sum of the defined entries / count of the defined entries.
    stack = np.stack(per_sample, axis=0)
    defined = np.isfinite(stack)
    num = np.where(defined, stack, 0.0).sum(axis=0)
    den = defined.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = num / den
    result[~np.isfinite(result)] = 0.0
    return pd.DataFrame(result, index=ses, columns=ses)


# ======================================================================
# Coassociation.R
# ======================================================================

def _coassoc_state_matrix(meta, sample, cell_type, se, non_se, states,
                          drop_non_se=False, fill_zero=False):
    """``get_state_matrix`` from ``Coassociation`` (R lines 62-75)."""
    if drop_non_se:
        meta = meta[meta[se].astype(str).values != non_se]
    samp = meta[sample].astype(str).values
    ct = meta[cell_type].astype(str).values
    state = meta["CellState"].astype(str).values

    df = pd.DataFrame({"S": samp, "C": ct, "St": state})
    cnt = df.groupby(["S", "C", "St"], sort=False).size().rename("n").reset_index()
    # R line 66: Frac = n / sum(n) within (Sample, CellType)
    cnt["Frac"] = cnt["n"] / cnt.groupby(["S", "C"])["n"].transform("sum")
    # R line 67: pivot_wider(id_cols = Sample) — one row per sample present.
    wide = cnt.pivot(index="S", columns="St", values="Frac")
    # R line 72: match() against the global state list; misses become NA columns.
    mat = _reindex_matrix(wide.values.astype(np.float64),
                          [str(v) for v in wide.index],
                          [str(v) for v in wide.columns],
                          [str(v) for v in wide.index], states)
    if fill_zero:                                    # R line 73
        mat[np.isnan(mat)] = 0.0
    return mat


def coassociation(scmeta, sample="Sample", se="SE", cell_type="CellType",
                  non_se="NonSE", nperm=1000, test=True):
    """``Coassociation`` — cross-sample co-association of cell states.

    Parameters
    ----------
    scmeta : pandas.DataFrame
        Single-cell metadata.
    sample : str
        Column with the sample IDs; correlations are taken across its levels.
    se : str
        Column with the spatial-ecotype labels.
    cell_type : str
        Column with the cell-type annotation.
    non_se : str
        Label marking non-SE cells.  Default ``"NonSE"``.
    nperm : int
        Permutations handed to :func:`coassociation_test` when ``test`` is True.
    test : bool
        If True return both the index and the permutation p-values.

    Returns
    -------
    pandas.DataFrame, or dict with ``"CoassociationIndex"`` and ``"Pval"``.

    Notes
    -----
    Consumes the global R stream only when ``test=True`` (via
    :func:`coassociation_test`); the index itself is deterministic.
    """
    scmeta = scmeta.copy()
    # R line 58
    scmeta["CellState"] = (scmeta[se].astype(str).values + "_" +
                           scmeta[cell_type].astype(str).values)
    states = _sort_unique(scmeta["CellState"].astype(str).values)   # R line 59

    # R lines 78-81: the four inclusion/fill schemes.
    f_list = [
        _coassoc_state_matrix(scmeta, sample, cell_type, se, non_se, states,
                              drop_non_se=False, fill_zero=False),
        _coassoc_state_matrix(scmeta, sample, cell_type, se, non_se, states,
                              drop_non_se=True, fill_zero=False),
        _coassoc_state_matrix(scmeta, sample, cell_type, se, non_se, states,
                              drop_non_se=False, fill_zero=True),
        _coassoc_state_matrix(scmeta, sample, cell_type, se, non_se, states,
                              drop_non_se=True, fill_zero=True),
    ]
    cor_list = [_cor_pairwise_complete(m) for m in f_list]          # R lines 86-90

    # R lines 93-98: average the four correlation matrices over the schemes
    # where the entry is defined; entries defined nowhere become 0.
    num = np.zeros_like(cor_list[0])
    den = np.zeros_like(cor_list[0])
    for c in cor_list:
        ok = np.isfinite(c)
        num += np.where(ok, c, 0.0)
        den += ok.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        scgg = num / np.where(den == 0, np.nan, den)
    scgg[~np.isfinite(scgg)] = 0.0
    scgg = pd.DataFrame(scgg, index=states, columns=states)

    if test:
        pval = coassociation_test(scgg, nperm=nperm)
        return {"CoassociationIndex": scgg, "Pval": pval}
    return scgg


# ======================================================================
# CoassociationTest.R
# ======================================================================

def _se_prefix(names):
    """``gsub("_.*", "", x)`` — everything before the first underscore."""
    return [str(n).split("_", 1)[0] for n in names]


def _avg_indices(mat, ses1, ses2, se_levels):
    """``avgIndices`` (R lines 23-31) — mean within-SE block, NAs dropped."""
    ses1 = np.asarray(ses1, dtype=object)
    ses2 = np.asarray(ses2, dtype=object)
    out = np.empty(len(se_levels), dtype=np.float64)
    for i, s in enumerate(se_levels):
        block = mat[np.ix_(ses1 == s, ses2 == s)]
        vals = block[np.isfinite(block)]
        # R: mean(x, na.rm = TRUE) of an all-NA block is NaN.
        out[i] = vals.mean() if vals.size else np.nan
    return out


def coassociation_test(mat, nperm=1000):
    """``CoassociationTest`` — permutation significance of within-SE co-association.

    Parameters
    ----------
    mat : pandas.DataFrame or numpy.ndarray
        Square co-association / colocalization matrix whose row and column
        names encode ``"<SE>_<CellType>"``.
    nperm : int
        Number of column permutations building the null.  Default 1000.

    Returns
    -------
    pandas.Series
        Two-sided p-value per SE, indexed by SE.  The signed Z-scores are
        attached as ``result.attrs["Zscore"]`` (R's ``attr(Pvals, "Zscore")``).

    Notes
    -----
    Draws ``nperm`` full column permutations from the global R stream via
    ``sample(1:ncol(mat), ncol(mat))`` (R line 36).  The R loop is a plain
    ``lapply``, so this is reproducible under a fixed ``set.seed``.
    """
    if isinstance(mat, pd.DataFrame):
        rownames = [str(v) for v in mat.index]
        colnames = [str(v) for v in mat.columns]
        arr = mat.values.astype(np.float64).copy()
    else:
        arr = np.asarray(mat, dtype=np.float64).copy()
        rownames = colnames = [str(i) for i in range(arr.shape[0])]

    diag = np.diag(arr).copy()
    # R line 20: `all(diag(mat) == 1) | all(is.na(diag(mat)))`.  R's three-valued
    # logic makes `if (NA)` an error, so the ambiguous case is refused here too.
    nan_diag = ~np.isfinite(diag)
    all_one = bool(np.all(diag[~nan_diag] == 1)) if (~nan_diag).any() else True
    if nan_diag.all():
        cond = True
    elif not all_one:
        cond = False
    elif nan_diag.any():
        raise ValueError("missing value where TRUE/FALSE needed "
                         "(diag(mat) mixes NA with 1)")
    else:
        cond = True
    if cond:
        np.fill_diagonal(arr, np.nan)

    ses1 = _se_prefix(rownames)          # R line 33
    ses2 = _se_prefix(colnames)          # R line 34
    se_levels = _unique(ses1)            # R line 29: names(avgs) = unique(ses1)

    obs = _avg_indices(arr, ses1, ses2, se_levels)                  # R line 32

    ncol = arr.shape[1]
    perms = np.empty((int(nperm), len(se_levels)), dtype=np.float64)
    ses2_arr = np.asarray(ses2, dtype=object)
    for i in range(int(nperm)):
        # R line 36: sample(1:ncol(mat), ncol(mat)) — a full permutation.
        idx = rrandom.sample_int(ncol, ncol)          # already 0-based
        newmat = arr[:, idx]
        # R line 38: the *positional* SE labels are kept, so column j still
        # claims to belong to SE `ses2[j]` after the shuffle.
        perms[i] = _avg_indices(newmat, ses1, ses2_arr, se_levels)

    perm_avg = perms.mean(axis=0)                                    # R line 41
    with np.errstate(invalid="ignore"):
        perm_sd = np.array([_r_sd(perms[np.isfinite(perms[:, j]), j])
                            if np.isfinite(perms[:, j]).sum() > 1 else np.nan
                            for j in range(perms.shape[1])])         # R line 42
    # R line 43: zero-variance SEs borrow the median SD (even-length median is
    # the mean of the two middle values, which numpy.median also does).
    zero = perm_sd == 0
    if zero.any():
        perm_sd[zero] = _r_median(perm_sd[np.isfinite(perm_sd)])

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - perm_avg) / perm_sd                               # R line 44
    z[~np.isfinite(z)] = 0.0                                         # R line 45
    pvals = norm.sf(np.abs(z)) * 2.0                                 # R line 46

    out = pd.Series(pvals, index=se_levels)
    out.attrs["Zscore"] = pd.Series(z, index=se_levels)              # R line 52
    return out


# ======================================================================
# Colocalization.R
# ======================================================================

def _colocalization(scmeta, coords=("X", "Y"), cell_state="CellState",
                    radius=50, k=200, min_cell=10, _neighbors=None):
    """``.colocalization`` — cell-state x cell-state spatial colocalization.

    Parameters
    ----------
    scmeta : pandas.DataFrame
        One row per cell; the index supplies the cell IDs.
    coords : sequence of str
        The two coordinate columns.
    cell_state : str
        Column holding the cell-state labels.
    radius : float
        Neighbours farther than this are dropped (R line 47 uses ``> radius``,
        so a neighbour exactly at ``radius`` is **kept**).
    k : int
        Neighbours queried per cell before the radius filter.
    min_cell : int
        Cells with fewer than this many state-assigned neighbours are dropped.
    _neighbors : scipy.sparse matrix, optional
        Private cache of the binary cell-cell adjacency.  The coordinates do
        not change across ``Colocalization``'s permutations, so R recomputes an
        identical ``RANN::nn2`` 1000 times; passing it in skips that.

    Returns
    -------
    pandas.DataFrame
        States x states, rows and columns in ``sort()`` order.
    """
    n = len(scmeta)
    if _neighbors is None:
        xy = np.asarray(scmeta[list(coords)].values, dtype=np.float64)
        idx, dist = _knn(xy, xy, min(int(k), n))
        # R lines 50-51: drop the self column, then keep the pairs inside radius.
        keep = dist[:, 1:] <= radius                 # complement of `> radius`
        rows = np.repeat(np.arange(n), keep.sum(axis=1))
        cols = idx[:, 1:][keep]
        # R line 56: `cellneighbors[idx.array] <- 1` assigns, so duplicate pairs
        # stay at 1 rather than accumulating.
        _neighbors = sp.csr_matrix(
            (np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n))
        _neighbors.data[:] = 1.0

    labels = scmeta[cell_state].astype(str).values
    states = _unique(labels)                                        # R line 59
    code = _match(labels, states)
    cell2state = sp.csr_matrix(
        (np.ones(n, dtype=np.float64), (np.arange(n), code)),
        shape=(n, len(states)))                                     # R lines 60-65

    csn = np.asarray((_neighbors @ cell2state).todense())           # R line 68
    keep_cell = csn.sum(axis=1) >= min_cell                         # R line 69
    csn = csn[keep_cell]
    csn = csn / csn.sum(axis=1, keepdims=True)                      # R line 72

    # R lines 74-77: W is the state x kept-cell membership matrix normalised per
    # state, so `W %*% csn` is the per-state mean of the kept cells' profiles.
    code_keep = code[keep_cell]
    out = np.zeros((len(states), len(states)), dtype=np.float64)
    counts = np.bincount(code_keep, minlength=len(states)).astype(np.float64)
    np.add.at(out, code_keep, csn)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / counts[:, None]        # a state with no kept cell -> NaN row

    order = _match(_sort(states), states)                           # R lines 78-79
    out = out[np.ix_(order, order)]
    names = _sort(states)
    return pd.DataFrame(out, index=names, columns=names)


def _permute_within(labels, groups):
    """``group_by(g) %>% mutate(v = sample(v))`` — R's grouped permutation.

    dplyr evaluates grouped ``mutate`` one group at a time in **sorted group
    order** (verified against dplyr 1.2.1), so the global RNG stream must be
    consumed in that order or every downstream draw desynchronises.  Within a
    group, ``sample()`` is applied to the rows in their original order and the
    result is scattered back to those same row positions.
    """
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)
    out = labels.copy()
    for g in _sort_unique(groups):
        pos = np.flatnonzero(groups == g)
        out[pos] = labels[pos][rrandom.sample_int(pos.size, pos.size)]
    return out


def colocalization(scmeta, coords=("X", "Y"), se="SE", cell_type="CellType",
                   radius=50, k=200, min_cell=10, nperm=1000, test=True,
                   ncores=16):
    """``Colocalization`` — Z-scored cell-state colocalization + significance.

    Parameters
    ----------
    scmeta : pandas.DataFrame
        Cell metadata; the index supplies the cell IDs.
    coords : sequence of str
        The two coordinate columns.
    se : str
        Spatial-ecotype column.
    cell_type : str
        Cell-type column.  Permutations shuffle cell states *within* each cell
        type, preserving cell-type composition.
    radius, k, min_cell
        Passed straight to :func:`_colocalization`.
    nperm : int
        Permutations forming the null.  Default 1000.
    test : bool
        If True also run :func:`coassociation_test` on the Z-scores.
    ncores : int
        Accepted for signature parity with R's ``mc.cores`` and otherwise
        ignored — see the module docstring on mclapply reproducibility.

    Returns
    -------
    pandas.DataFrame, or dict with ``"ColocIndex"`` and ``"Pval"``.
    """
    required = list(coords) + [se, cell_type]
    missing = [c for c in required if c not in scmeta.columns]
    if missing:
        raise ValueError("Missing required columns in scmeta: "
                         + ", ".join(missing) + ".")

    scmeta = scmeta.copy()
    scmeta["CellState"] = (scmeta[se].astype(str).values + "_" +
                           scmeta[cell_type].astype(str).values)   # R line 150
    scmeta["CID"] = [str(v) for v in scmeta.index]                 # R line 155

    n = len(scmeta)
    xy = np.asarray(scmeta[list(coords)].values, dtype=np.float64)
    idx, dist = _knn(xy, xy, min(int(k), n))
    keep = dist[:, 1:] <= radius
    rows = np.repeat(np.arange(n), keep.sum(axis=1))
    cols = idx[:, 1:][keep]
    neighbors = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n))
    neighbors.data[:] = 1.0

    obs = _colocalization(scmeta, coords=coords, cell_state="CellState",
                          radius=radius, k=k, min_cell=min_cell,
                          _neighbors=neighbors)                    # R lines 158-159

    ct_vals = scmeta[cell_type].astype(str).values
    state_vals = scmeta["CellState"].astype(str).values
    tmp = scmeta.copy()
    perms = []
    for _ in range(int(nperm)):                                    # R lines 162-169
        tmp["CellState"] = _permute_within(state_vals, ct_vals)
        perms.append(_colocalization(tmp, coords=coords, cell_state="CellState",
                                     radius=radius, k=k, min_cell=min_cell,
                                     _neighbors=neighbors))

    states = list(obs.index)
    # Every permutation preserves the per-(cell type, SE) counts, so the state
    # set is invariant and the sorted dimnames line up positionally.
    stack = np.stack([p.reindex(index=states, columns=states).values
                      for p in perms], axis=0)
    perm_avg = stack.mean(axis=0)                                  # R line 172
    perm_sd = _r_sd(stack, axis=0)                                 # R lines 173-176
    if not np.isfinite(perm_sd).all():
        # R line 177 (`PermuteSD[PermuteSD == 0] <- 1`) errors on an NA subscript
        # with "NAs are not allowed in subscripted assignments"; reproduce that
        # rather than silently diverging.
        raise ValueError("NAs are not allowed in subscripted assignments: some "
                         "cell state had no cell passing `min_cell` in some "
                         "permutation")
    perm_sd = np.where(perm_sd == 0, 1.0, perm_sd)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs.values - perm_avg) / perm_sd                      # R line 180
    z[~np.isfinite(z)] = 0.0                                       # R line 181
    z = pd.DataFrame(z, index=states, columns=states)

    if test:
        return {"ColocIndex": z, "Pval": coassociation_test(z, nperm=nperm)}
    return z


# ======================================================================
# ColocalizationMetaAnalysis.R
# ======================================================================

def colocalization_meta_analysis(colocalization_results, cap=5, min_samples=1):
    """``ColocalizationMetaAnalysis`` — Stouffer-combine per-sample results.

    Parameters
    ----------
    colocalization_results : sequence of dict
        One entry per sample, each with ``"ColocIndex"`` (a states x states
        DataFrame) and ``"Pval"`` (a per-SE Series, ideally carrying
        ``attrs["Zscore"]`` as :func:`coassociation_test` returns).
    cap : float
        Symmetric clip applied to each sample's ``ColocIndex`` before combining.
    min_samples : int
        A cell state must appear in at least this many samples to be kept.

    Returns
    -------
    dict
        ``{"MetaColocIndex": DataFrame, "MetaPval": Series}``.
    """
    if not isinstance(colocalization_results, (list, tuple)) \
            or len(colocalization_results) == 0:
        raise ValueError("`colocalization_results` must be a non-empty list.")
    if not cap > 0:
        raise ValueError("`cap` must be a positive number.")
    if not min_samples >= 1:
        raise ValueError("`min_samples` must be >= 1.")
    for xx in colocalization_results:
        if xx.get("ColocIndex") is None or xx.get("Pval") is None:
            raise ValueError("Every element of `colocalization_results` must "
                             "contain both `ColocIndex` and `Pval`.")

    mat_list = []
    for xx in colocalization_results:                              # R lines 58-63
        m = xx["ColocIndex"]
        arr = np.array(m.values if isinstance(m, pd.DataFrame) else m,
                       dtype=np.float64, copy=True)
        arr[arr > cap] = cap
        arr[arr < -cap] = -cap
        names = ([str(v) for v in m.index] if isinstance(m, pd.DataFrame)
                 else [str(i) for i in range(arr.shape[0])])
        cnames = ([str(v) for v in m.columns] if isinstance(m, pd.DataFrame)
                  else names)
        mat_list.append((arr, names, cnames))

    # R lines 66-67: table() orders its names by sorted level, so `states` comes
    # out sorted; only states seen in >= min.samples samples survive.
    counts = {}
    for _, names, _c in mat_list:
        for nm in names:
            counts[nm] = counts.get(nm, 0) + 1
    states = [s for s in _sort(counts) if counts[s] >= min_samples]
    if len(states) < 3:
        raise ValueError("Fewer than three cell states are shared across "
                         "samples (after applying `min_samples`).")

    padded = np.stack([_reindex_matrix(a, r, c, states, states)
                       for a, r, c in mat_list], axis=0)           # R lines 76-80
    metagg = np.nansum(padded, axis=0)                             # R line 83
    n_nonmissing = np.isfinite(padded).sum(axis=0).astype(np.float64)  # R line 86
    metagg = metagg / (np.sqrt(n_nonmissing) + 1e-7)               # R line 88
    metagg[n_nonmissing == 0] = np.nan                             # R line 89
    metagg = pd.DataFrame(metagg, index=states, columns=states)

    eps = np.finfo(np.float64).eps                                 # R line 93
    rows = []
    for xx in colocalization_results:                              # R lines 94-101
        pv = xx["Pval"]
        zs = pv.attrs.get("Zscore") if isinstance(pv, pd.Series) else None
        names = ([str(v) for v in pv.index] if isinstance(pv, pd.Series)
                 else [str(i) for i in range(len(pv))])
        if zs is None:
            p = np.clip(np.asarray(pv, dtype=np.float64), eps, 1 - eps)
            z = norm.isf(p / 2.0)          # qnorm(p/2, lower.tail = FALSE)
        else:
            z = np.asarray(zs, dtype=np.float64)
        rows.append(pd.DataFrame({"SE": names, "Zscore": z}))
    pval_df = pd.concat(rows, ignore_index=True)
    pval_df = pval_df[np.isfinite(pval_df["Zscore"].values)]       # R line 103

    # R lines 105-110: dplyr::summarize orders groups by sorted SE.
    meta_rows = []
    for se_name in _sort(pd.unique(pval_df["SE"].values)):
        zz = pval_df.loc[pval_df["SE"].values == se_name, "Zscore"].values
        nz = float(len(zz))
        zsum = float(zz.sum()) / (np.sqrt(nz) + 1e-7)
        meta_rows.append((se_name, zsum, float(norm.sf(abs(zsum)) * 2.0)))
    meta_pval = pd.Series([r[2] for r in meta_rows],
                          index=[r[0] for r in meta_rows])
    meta_pval.attrs["Zscore"] = pd.Series([r[1] for r in meta_rows],
                                          index=[r[0] for r in meta_rows])
    return {"MetaColocIndex": metagg, "MetaPval": meta_pval}


# ======================================================================
# ComputeNormalizedMoranI.R
# ======================================================================

def _moran(ses, listw, ncores=1):
    """``.moran`` — Moran's I of each SE's binary indicator.

    Parameters
    ----------
    ses : sequence
        Per-cell SE (or cell-state) labels.
    listw : scipy.sparse matrix
        Row-standardised spatial weights, i.e. ``spdep::nb2listw(style = "W")``
        rendered as a matrix.
    ncores : int
        Accepted for parity with R's ``mc.cores``; no RNG is consumed here, so
        the value cannot change the result.

    Returns
    -------
    pandas.Series
        Moran's I per SE, indexed by ``sort(unique(ses))``.

    Notes
    -----
    ``spdep::moran(x, listw, n = length(x), S0 = Szero(listw))`` is the closed
    form ``I = (n / S0) * (z' W z) / sum(z^2)`` with ``z = x - mean(x)``; under
    ``style = "W"`` every row of ``W`` sums to 1 so ``S0 = n`` and the leading
    factor is 1 (checked against spdep 1.4.2: ``Szero`` returned exactly 4000
    for a 4000-cell graph).
    """
    ses = np.asarray(ses, dtype=object)
    levels = _sort_unique(ses)                                     # R line 22
    n = len(ses)
    s0 = float(listw.sum())
    out = np.empty(len(levels), dtype=np.float64)
    for i, s in enumerate(levels):
        x = (ses == s).astype(np.float64)
        z = x - x.mean()
        zz = float(z @ z)
        lz = listw @ z                                # spdep::lag.listw
        out[i] = (n / s0) * (float(z @ lz) / zz)
    return pd.Series(out, index=levels)


def compute_normalized_moran_i(scmeta, coords=("X", "Y"), se="SE",
                               cell_type="CellType", nperm=1000, k=3, ncores=1):
    """``ComputeNormalizedMoranI`` — permutation-normalised Moran's I per SE.

    Parameters
    ----------
    scmeta : pandas.DataFrame
        Cell metadata.
    coords : sequence of str
        The two coordinate columns.
    se : str
        SE / cell-state column whose spatial autocorrelation is measured.
    cell_type : str
        Cell-type column; the null permutes SE labels *within* each cell type.
    nperm : int
        Permutations forming the null.  Default 1000.
    k : int
        Neighbours in the ``spdep::knearneigh`` graph.  Default 3.
    ncores : int
        Passed to :func:`_moran`; has no effect on the value.

    Returns
    -------
    pandas.Series
        Z-score per SE.

    Notes
    -----
    The permutation loop is a plain ``lapply`` in R, so with the ported R RNG
    this is bit-reproducible under a fixed ``set.seed``.
    """
    required = list(coords) + [se, cell_type]
    missing = [c for c in required if c not in scmeta.columns]
    if missing:
        raise ValueError("Missing required columns in scmeta: "
                         + ", ".join(missing) + ".")

    xy = np.asarray(scmeta[list(coords)].values, dtype=np.float64)
    n = xy.shape[0]
    # R lines 89-91: knearneigh() returns the k nearest neighbours *excluding*
    # the point itself, so query k+1 and strike out the self-match.
    idx, _dist = _knn(xy, xy, min(int(k) + 1, n))
    self_col = idx == np.arange(n)[:, None]
    keep = np.ones_like(self_col, dtype=bool)
    first_self = np.argmax(self_col, axis=1)
    has_self = self_col.any(axis=1)
    keep[np.arange(n)[has_self], first_self[has_self]] = False
    # Points with no self-match (exact duplicate coordinates) drop their last
    # neighbour instead, keeping exactly k per row.
    keep[np.arange(n)[~has_self], -1] = False
    nbr = idx[keep].reshape(n, -1)[:, :int(k)]

    rows = np.repeat(np.arange(n), nbr.shape[1])
    # style = "W": every neighbour of i gets weight 1 / k.
    w = sp.csr_matrix((np.full(nbr.size, 1.0 / nbr.shape[1]),
                       (rows, nbr.ravel())), shape=(n, n))

    obs = _moran(scmeta[se].astype(str).values, w, ncores=ncores)   # R line 94

    se_vals = scmeta[se].astype(str).values
    ct_vals = scmeta[cell_type].astype(str).values
    bgs = np.empty((int(nperm), len(obs)), dtype=np.float64)
    for i in range(int(nperm)):                                     # R lines 97-104
        perm_se = _permute_within(se_vals, ct_vals)
        bgs[i] = _moran(perm_se, w, ncores=ncores).reindex(obs.index).values

    mu = bgs.mean(axis=0)                                           # R line 108
    sigma = _r_sd(bgs, axis=0)                                      # R line 109
    sigma = np.where(sigma == 0, 1.0, sigma)                        # R line 110
    return pd.Series((obs.values - mu) / sigma, index=obs.index)    # R line 112


# ======================================================================
# ComputeSEAbundanceBySN.R
# ======================================================================

def build_knn_weights(ref_coords, query_coords=None, k=10, radius=np.inf,
                      include_self=True):
    """``buildKNNWeights`` — binary k-NN neighbour matrix, optionally radius-capped.

    Parameters
    ----------
    ref_coords : pandas.DataFrame or numpy.ndarray
        Coordinates of the source units (the matrix columns).
    query_coords : pandas.DataFrame or numpy.ndarray, optional
        Coordinates of the output units (the matrix rows).  Defaults to
        ``ref_coords`` (R's default argument).
    k : int
        Neighbours searched per query point; clamped to ``nrow(ref_coords)``.
    radius : float
        Neighbours at distance ``>= radius`` are dropped.  R line 33 uses a
        **strict** ``<``, unlike ``.colocalization``'s ``<=``.
    include_self : bool
        If False, a query point's own zero-distance match is removed (R lines
        35-38 compare the *positional* index, so this only means anything when
        ``query_coords is ref_coords``).

    Returns
    -------
    NamedSparse
        ``nrow(query_coords)`` x ``nrow(ref_coords)``, entries 0/1.
    """
    if query_coords is None:
        query_coords = ref_coords
    qnames = _index_names(query_coords, len(query_coords))
    rnames = _index_names(ref_coords, len(ref_coords))
    q = np.asarray(query_coords.values if isinstance(query_coords, pd.DataFrame)
                   else query_coords, dtype=np.float64)
    r = np.asarray(ref_coords.values if isinstance(ref_coords, pd.DataFrame)
                   else ref_coords, dtype=np.float64)
    k = min(int(k), r.shape[0])                                     # R line 22

    idx, dist = _knn(r, q, k)
    keep = dist < radius                                            # R line 33
    if not include_self:
        # R lines 35-38: strike the self-match, identified by index AND a zero
        # distance (so a coincident but distinct point is retained).
        self_hit = (idx == np.arange(q.shape[0])[:, None]) & (dist == 0)
        keep &= ~self_hit

    rows = np.repeat(np.arange(q.shape[0]), keep.sum(axis=1))
    cols = idx[keep]
    mat = sp.csr_matrix((np.ones(rows.size, dtype=np.float64), (rows, cols)),
                        shape=(q.shape[0], r.shape[0]))
    mat.data[:] = 1.0                       # assignment, not accumulation
    mat.eliminate_zeros()                   # Matrix::drop0 (R line 45)
    return NamedSparse(mat, index=qnames, columns=rnames)


def aggregate_by_weights(cell2se, weights, sum2one=True, min_cells=5):
    """``aggregateByWeights`` — push a value matrix through a neighbour matrix.

    Parameters
    ----------
    cell2se : pandas.DataFrame, NamedSparse or scipy.sparse matrix
        Source units (rows) x features (columns); the row names must match
        ``weights``' column names.
    weights : NamedSparse
        Typically from :func:`build_knn_weights`.
    sum2one : bool
        Renormalise each output row to sum to 1 after aggregation.
    min_cells : int
        Output units backed by fewer than this many source units are dropped.

    Returns
    -------
    pandas.DataFrame
        Kept output units x features.
    """
    if isinstance(cell2se, NamedSparse):
        c_index, c_cols, cmat = cell2se.index, cell2se.columns, cell2se.mat
    elif isinstance(cell2se, pd.DataFrame):
        c_index = [str(v) for v in cell2se.index]
        c_cols = [str(v) for v in cell2se.columns]
        cmat = cell2se.values.astype(np.float64)
    else:
        cmat = cell2se
        c_index = [str(i) for i in range(cmat.shape[0])]
        c_cols = [str(i) for i in range(cmat.shape[1])]

    w = weights.mat
    w_cols = weights.columns
    # R lines 62-64
    if not (w.shape[1] == len(c_index) and set(w_cols) <= set(c_index)):
        raise ValueError("The column names of `weights` do not match row names "
                         "of `cell2se`.")
    order = _match(c_index, w_cols)                                 # R line 65
    w = w[:, order]

    counts = np.asarray((w > 0).sum(axis=1)).ravel()
    keep = counts >= min_cells                                      # R line 66
    if not keep.any():
        raise ValueError("No SNs have at least `min_cells` cells."
                         "Try increasing the search radius/k or lowering "
                         "`min_cells`.")
    w = w[keep]
    rs = np.asarray(w.sum(axis=1)).ravel()
    w = sp.diags(1.0 / rs) @ w                                      # R line 72

    aggr = np.asarray((w @ cmat).todense()) if sp.issparse(cmat) else np.asarray(w @ cmat)
    if sum2one:                                                     # R line 76
        with np.errstate(invalid="ignore", divide="ignore"):
            aggr = aggr / aggr.sum(axis=1, keepdims=True)
    kept_names = [weights.index[i] for i in np.flatnonzero(keep)]
    return pd.DataFrame(aggr, index=kept_names, columns=c_cols)


def compute_se_abundance_by_sn(df, spot_coords=None, radius=50, grid_size=50,
                               k=None, x="X", y="Y", se="SE", min_cells=5):
    """``ComputeSEAbundanceBySN`` — SE abundances inside spatial neighbourhoods.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per cell, with coordinate columns ``x``/``y`` and an SE column.
    spot_coords : pandas.DataFrame, optional
        SN centres with literal ``"X"``/``"Y"`` columns (R hard-codes those two
        names here, unlike ``smooth_se_abundances``) and SN IDs as the index.
        If None, a regular grid spaced ``grid_size`` apart is generated.
    radius : float
        Cells farther than this from an SN centre are excluded.
    grid_size : float
        Spacing of the auto-generated SN grid.
    k : int, optional
        Neighbours queried per SN.  R's default is the lazily evaluated
        ``min(200, nrow(df))``, reproduced here by the ``None`` sentinel.
    x, y, se : str
        Column names in ``df``.
    min_cells : int
        SNs with fewer cells are dropped.

    Returns
    -------
    pandas.DataFrame
        ``X``, ``Y``, then one column per SE level, indexed by SN ID.
    """
    if k is None:
        k = min(200, len(df))                                       # R line 102

    se_levels = _sort_unique(df[se].astype(str).values)             # R line 107
    code = _match(df[se].astype(str).values, se_levels)
    cell2se = pd.DataFrame(
        sp.csr_matrix((np.ones(len(df)), (np.arange(len(df)), code)),
                      shape=(len(df), len(se_levels))).toarray(),
        index=_index_names(df, len(df)), columns=se_levels)         # R lines 108-112

    sc_coords = pd.DataFrame({"__x": np.asarray(df[x], dtype=np.float64),
                              "__y": np.asarray(df[y], dtype=np.float64)},
                             index=_index_names(df, len(df)))

    if spot_coords is None:
        # R lines 118-126: bin on a `grid.size` lattice, then take the component-
        # wise MEDIAN of the member cells as the SN centre.  R's round() is
        # half-to-even and its median of an even-length vector is the mean of
        # the two middle values -- numpy matches both.
        rx = _r_round(sc_coords["__x"].values / grid_size).astype(np.int64)
        ry = _r_round(sc_coords["__y"].values / grid_size).astype(np.int64)
        spot_id = np.array([f"X{a}_Y{b}" for a, b in zip(rx, ry)], dtype=object)
        # dplyr::summarize returns groups in C-locale sorted order.
        ids = _sort(set(spot_id.tolist()))
        xs = np.empty(len(ids)); ys = np.empty(len(ids))
        for i, sid in enumerate(ids):
            m = spot_id == sid
            xs[i] = _r_median(sc_coords["__x"].values[m])
            ys[i] = _r_median(sc_coords["__y"].values[m])
        spot_coords = pd.DataFrame({"X": xs, "Y": ys}, index=ids)

    weights = build_knn_weights(ref_coords=sc_coords,
                                query_coords=spot_coords[["X", "Y"]],
                                k=k, radius=radius, include_self=True)
    seabunds = aggregate_by_weights(cell2se=cell2se, weights=weights,
                                    sum2one=True, min_cells=min_cells)
    # R lines 136-137
    coords_block = spot_coords.loc[seabunds.index, ["X", "Y"]]
    return pd.concat([coords_block, seabunds], axis=1)


def smooth_se_abundances(se_mat, spot_coords, k=7, x="X", y="Y",
                         include_self=True, min_neighbors=3):
    """``SmoothSEAbundances`` — average spot-level SE abundances over k-NN spots.

    Parameters
    ----------
    se_mat : pandas.DataFrame
        Spot x SE abundances, indexed by spot ID.
    spot_coords : pandas.DataFrame
        Spot coordinates indexed by spot ID, with columns ``x``/``y``.
    k : int
        Neighbouring spots averaged over.  Default 7 = self + 6 Visium
        hex neighbours when ``include_self`` is True.
    x, y : str
        Coordinate column names in ``spot_coords``.
    include_self : bool
        Whether a spot's own value is one of its k neighbours.
    min_neighbors : int
        Spots with fewer neighbours are dropped.

    Returns
    -------
    pandas.DataFrame
        Coordinates then smoothed SE abundances, indexed by spot ID.
    """
    spot_coords = pd.DataFrame(spot_coords)
    # R lines 168-177: intersect() keeps the order of its FIRST argument, so the
    # retained spots follow `se_mat`'s row order.
    coord_ids = set(str(v) for v in spot_coords.index)
    common = [str(v) for v in se_mat.index if str(v) in coord_ids]
    if len(common) == 0:
        raise ValueError("rownames do not match in `se_mat` and `spot_coords`.")
    if len(common) < len(se_mat):
        import warnings
        warnings.warn("Some spots in `se_mat` have no matching coordinates in "
                      "`spot_coords`; those spots will be dropped.")
    se_mat = se_mat.loc[common]
    spot_coords = spot_coords.loc[common]

    weights = build_knn_weights(ref_coords=spot_coords[[x, y]],
                                query_coords=spot_coords[[x, y]],
                                k=k, radius=np.inf, include_self=include_self)
    seabunds = aggregate_by_weights(cell2se=se_mat, weights=weights,
                                    sum2one=False, min_cells=min_neighbors)
    coords_block = spot_coords.loc[seabunds.index, [x, y]]
    return pd.concat([coords_block, seabunds], axis=1)


# ======================================================================
# CreatePseudobulks.R
# ======================================================================

def create_pseudobulks(data=None, groups=None, counts=None, n_mixtures=100):
    """``CreatePseudobulks`` — build pseudobulk mixtures from single cells.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Normalised genes x cells expression.  If None it is derived from
        ``counts`` as ``counts / colSums(counts) * 10000``.
    groups : pandas.Series or sequence
        Group (e.g. SE) label per cell.  If it carries an index, it is
        reordered onto ``data``'s columns; otherwise the column order is
        assumed.
    counts : pandas.DataFrame, optional
        Raw genes x cells counts, used when ``data`` is None.
    n_mixtures : int
        Number of pseudobulks to build.  Default 100.

    Returns
    -------
    dict
        ``{"Fracs": DataFrame (pseudobulk x group),
           "Mixtures": DataFrame (gene x pseudobulk)}``.

    Notes
    -----
    Consumes the global R stream twice per call: ``sample(rnorm(10000, mean=2),
    n_mixtures)`` once per group (R line 47), then one
    ``sample(..., replace = TRUE)`` per (pseudobulk, group) cell draw
    (R line 61).  Both loops are plain ``lapply``/``apply``, so a fixed
    ``set.seed`` reproduces R exactly.
    """
    if data is None:                                                # R line 39
        cs = np.asarray(counts.sum(axis=0), dtype=np.float64)
        data = counts / cs * 10000.0
    genes = [str(g) for g in data.index]
    cells = [str(c) for c in data.columns]
    arr = np.asarray(data.values, dtype=np.float64)
    if arr.max() < 80:                                              # R line 40
        arr = 2.0 ** arr - 1.0

    if isinstance(groups, pd.Series):
        groups = groups.astype(str)
        if groups.index.equals(pd.Index(range(len(groups)))):
            gvals = groups.values
        else:
            gvals = groups.reindex(cells).values                    # R line 44
    else:
        gvals = np.asarray(groups, dtype=object).astype(str)
    gvals = np.asarray(gvals, dtype=object)

    levels = _sort_unique(gvals)                                    # R line 46
    cols = []
    for _ in levels:
        pool = rrandom.rnorm(10000, mean=2.0)                       # R line 47
        cols.append(rrandom.sample(pool, int(n_mixtures)))
    fracs = np.column_stack(cols)
    rownames = [f"Pseudobulk{i}" for i in range(1, int(n_mixtures) + 1)]
    fracs[fracs < 0] = 0                                            # R line 52
    keep = (fracs > 0).sum(axis=1) > 2                              # R line 53
    fracs = fracs[keep]
    rownames = [r for r, kk in zip(rownames, keep) if kk]
    fracs = fracs / fracs.sum(axis=1, keepdims=True)                # R line 54
    ncells = _r_round(fracs * 1000)                                 # R line 55

    cell_pos = {c: i for i, c in enumerate(cells)}
    per_level = {lv: np.array([c for c, g in zip(cells, gvals) if g == lv],
                              dtype=object) for lv in levels}

    mixtures = np.empty((arr.shape[0], fracs.shape[0]), dtype=np.float64)
    for i in range(fracs.shape[0]):                                 # R lines 57-64
        picked = []
        for j, lv in enumerate(levels):
            size = int(ncells[i, j])
            if size == 0:
                continue                    # sample(v, 0) draws nothing in R
            picked.append(rrandom.sample(per_level[lv], size, replace=True))
        chosen = np.concatenate(picked) if picked else np.array([], dtype=object)
        colidx = np.array([cell_pos[c] for c in chosen], dtype=np.int64)
        mixtures[:, i] = arr[:, colidx].mean(axis=1)

    # R line 65: Seurat::NormalizeData -> LogNormalize, log1p(x / colSum * 1e4).
    mixtures = np.log1p(mixtures / mixtures.sum(axis=0, keepdims=True) * 1e4)
    return {
        "Fracs": pd.DataFrame(fracs, index=rownames, columns=levels),
        "Mixtures": pd.DataFrame(mixtures, index=genes, columns=rownames),
    }
