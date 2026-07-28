"""Small R semantics that NumPy/pandas do not give you for free.

Every function here exists because a naive Python equivalent silently differs
from R: ``rank`` averages ties, ``order`` is a *stable* radix sort, ``table``
orders by sorted level, ``median`` of an even-length vector averages the two
middle values, and ``sort`` on a sparse row has to preserve the original
positions of equal values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "most_frequent",
    "rank_sparse",
    "matrix_multiply",
    "r_rank",
    "r_order",
    "r_table",
]


def most_frequent(x) -> str:
    """``mostFrequent`` (``R/util.R``).

    R does ``table(as.character(x)) |> as.data.frame() |> arrange(desc(Freq)) |>
    slice(1) |> pull(1)``.  ``table`` returns levels in **sorted** order and
    ``dplyr::arrange(desc(Freq))`` is a stable sort, so ties are broken by the
    lexicographically smallest level -- not by first appearance.
    """
    s = pd.Series([str(v) for v in np.asarray(x).ravel()])
    counts = s.value_counts()
    # value_counts() sorts by count desc but ties are in encounter order;
    # re-sort stably on the sorted-level order to match table() + arrange().
    counts = counts.reindex(sorted(counts.index))
    return str(counts.index[np.argmax(counts.to_numpy())])


def r_rank(x: np.ndarray) -> np.ndarray:
    """``base::rank(x)`` with the default ``ties.method = "average"``."""
    from scipy.stats import rankdata
    return rankdata(np.asarray(x, dtype=np.float64), method="average")


def r_order(x: np.ndarray, decreasing: bool = False) -> np.ndarray:
    """``base::order(x, decreasing)`` -> 0-based indices.

    R's default method for numeric vectors shorter than 2^31 is ``"radix"``,
    which is *stable*: equal values keep their original relative order, in
    both directions.  ``np.argsort(kind="stable")`` on the negated key
    reproduces the decreasing case (negating preserves stability, whereas
    reversing an ascending argsort would not).
    """
    x = np.asarray(x)
    key = -x if decreasing else x
    return np.argsort(key, kind="stable")


def r_table(x) -> pd.Series:
    """``base::table(x)`` -- counts indexed by **sorted** unique level."""
    s = pd.Series([str(v) for v in np.asarray(x).ravel()])
    return s.value_counts().reindex(sorted(s.unique())).fillna(0).astype(int)


def rank_sparse(mat):
    """``rankSparse`` (``R/util.R``).

    Ranks the non-zero elements **within each column** and divides by
    ``nrow``.  The R implementation is::

        sparseMat@x <- ave(non_zero_values, non_zero_indices[, "col"], FUN = rank)
        sparseMat <- sparseMat / nrow(sparseMat)

    ``sparseMat@x`` is in CSC order, so grouping by column and applying
    ``rank`` (average ties) column-block by column-block is exactly what
    ``ave`` does.  Structural zeros are untouched and stay zero, so the
    output is *not* a full 1..n ranking of the column -- only of its support.
    """
    m = sp.csc_matrix(mat, copy=True).astype(np.float64)
    m.eliminate_zeros()                       # drop0()
    indptr, data = m.indptr, m.data
    out = np.empty_like(data)
    for j in range(m.shape[1]):
        lo, hi = indptr[j], indptr[j + 1]
        if hi > lo:
            out[lo:hi] = r_rank(data[lo:hi])
    m.data = out / m.shape[0]
    return m


def matrix_multiply(mat1, mat2, minibatch: int = 5000, ncores: int = 1):
    """``matrixMultiply`` (``R/SNF2.R``).

    Column-blocked ``mat1 %*% mat2``.  The blocking is a memory optimisation
    only; the result is mathematically identical to a single product (floating
    point addition inside each dot product is not reordered by the blocking,
    since blocks partition the *columns* of ``mat2``, not the summation index).
    """
    n = mat2.shape[1]
    cycles = int(np.ceil(n / minibatch))
    if cycles <= 1:
        return mat1 @ mat2
    parts = []
    for ii in range(cycles):
        lo = ii * minibatch
        hi = min((ii + 1) * minibatch, n)
        parts.append(mat1 @ mat2[:, lo:hi])
    if sp.issparse(parts[0]):
        return sp.hstack(parts, format="csc")
    return np.hstack(parts)
