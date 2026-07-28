"""Spatial metacells -- ``GetSpatialMetacells`` and its ``GetKnnWeights`` helper.

A "spatial metacell" is the average expression profile of the ``k`` cells of one
cell type that lie nearest a spatial-neighbourhood centre, restricted to those
within ``radius``.  The output is one column per (neighbourhood, cell type)
pair, which is what makes the downstream similarity networks cell-type-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

__all__ = ["get_knn_weights", "get_spatial_metacells"]


def _nn2(data: np.ndarray, query: np.ndarray, k: int):
    """``RANN::nn2(data, query, k)`` -> (idx, dists), idx **0-based**.

    ``nn2`` defaults to ``eps = 0`` (exact search) and ``treetype = "kd"``, so
    an exact cKDTree query returns the same neighbour sets and identical
    distances.  Where fewer than ``k`` points exist, RANN pads ``nn.idx`` with
    0 (1-based) and ``nn.dists`` with Inf; we mark those with ``-1``.
    """
    tree = cKDTree(data)
    dists, idx = tree.query(query, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    idx = np.where(np.isfinite(dists), idx, -1)
    return idx, dists


def get_knn_weights(scmeta: pd.DataFrame, spotmeta: pd.DataFrame,
                    k: int = 20, radius: float = 50,
                    X: str = "X", Y: str = "Y",
                    min_cells_per_region: int = 1):
    """``GetKnnWeights`` (``R/GetSpatialMetacells.R``).

    Returns ``(weights, kept_spot_names)`` with ``weights`` a
    ``cells x kept_spots`` CSC matrix whose columns sum to 1, or
    ``(None, None)`` where R returns ``NULL``.

    The R code assigns ``nn.dists + 1`` into the sparse matrix and then zeros
    anything above ``radius + 1``; the ``+1`` shift exists purely so that a
    zero *distance* (a cell sitting exactly on the neighbourhood centre) is not
    confused with a structural zero.  Immediately afterwards every surviving
    entry is overwritten with 1 (``weights@x <- rep(1, ...)``), so the
    distances only ever act as a radius mask -- the metacell is an *unweighted*
    mean of the in-radius neighbours, not a distance-weighted one.
    """
    locs = np.asarray(scmeta[[X, Y]], dtype=np.float64)
    spot_locs = np.asarray(spotmeta[[X, Y]], dtype=np.float64)
    if spot_locs.shape[0] < 3:
        return None, None

    if k > locs.shape[0]:
        k = locs.shape[0] - 1
    if k < 1:
        return None, None

    idx, dists = _nn2(locs, spot_locs, k)

    rows, cols, vals = [], [], []
    for i in range(k):
        col_ok = idx[:, i] >= 0
        d = dists[col_ok, i] + 1.0
        ok = np.isfinite(d)
        rows.append(idx[col_ok, i][ok])
        cols.append(np.flatnonzero(col_ok)[ok])
        vals.append(d[ok])
    rows = np.concatenate(rows) if rows else np.array([], dtype=int)
    cols = np.concatenate(cols) if cols else np.array([], dtype=int)
    vals = np.concatenate(vals) if vals else np.array([], dtype=float)

    w = sp.coo_matrix((vals, (rows, cols)),
                      shape=(locs.shape[0], spot_locs.shape[0])).tocsc()
    # weights@x[weights@x > (radius + 1)] <- 0 ; drop0()
    w.data[w.data > (radius + 1)] = 0.0
    w.eliminate_zeros()
    # Unweight
    w.data[:] = 1.0

    colsums = np.asarray(w.sum(axis=0)).ravel()
    keep = colsums >= min_cells_per_region
    w = w[:, keep]
    if w.shape[1] < 5:
        return None, None
    kept = np.asarray(spotmeta.index, dtype=object)[keep]

    # t(t(weights) / colSums(weights))
    colsums = np.asarray(w.sum(axis=0)).ravel()
    w = w @ sp.diags(1.0 / colsums)
    return w.tocsc(), list(kept)


def get_spatial_metacells(normdata, metadata: pd.DataFrame,
                          X: str = "X", Y: str = "Y",
                          CellType: str = "CellType",
                          spotCoord: pd.DataFrame | None = None,
                          k: int = 20, radius: float = 50,
                          min_cells_per_region: int = 1,
                          ncores: int = 4,
                          gene_names: list[str] | None = None,
                          verbose: bool = True):
    """``GetSpatialMetacells(normdata, metadata, X, Y, CellType, spotCoord, k, radius, ...)``.

    Parameters
    ----------
    normdata
        genes x cells, dense or sparse, in the same column order as ``metadata``.
    metadata
        Must carry ``X``, ``Y`` and ``CellType`` columns; index = cell IDs.
    spotCoord
        Spatial-neighbourhood centres, index = neighbourhood IDs.  When ``None``
        a grid of side ``round(radius * 1.4)`` is derived from the cells, exactly
        as R does.

    Returns
    -------
    (matrix, colnames)
        ``matrix`` is genes x metacells; ``colnames`` are
        ``"<SpotID>..<CellType>"``, the ``..`` separator that every downstream
        function splits on with ``gsub("\\\\.+.*", "", x)``.

    Cell types are only used when they have **more than** ``k`` cells
    (``celltypes[celltypes > k]``, strictly greater), and a cell type is dropped
    again if it yields <= 2 metacell columns (``ncols > 2``).
    """
    metadata = pd.DataFrame(metadata).copy()
    metadata["CellType"] = metadata[CellType].to_numpy()
    metadata["X"] = metadata[X].to_numpy()
    metadata["Y"] = metadata[Y].to_numpy()

    if gene_names is None:
        gene_names = [str(i) for i in range(normdata.shape[0])]

    if spotCoord is None:
        binsize = round(radius * 1.4)
        tmp = metadata.copy()
        tmp["SpotID"] = ["X%d_Y%d" % (round(a / binsize), round(b / binsize))
                         for a, b in zip(tmp["X"], tmp["Y"])]
        spotCoord = tmp.groupby("SpotID", sort=True).agg(X=("X", "median"),
                                                         Y=("Y", "median"))
    else:
        spotCoord = pd.DataFrame(spotCoord).copy()
        spotCoord["X"] = spotCoord[X].to_numpy()
        spotCoord["Y"] = spotCoord[Y].to_numpy()

    ct_counts = metadata["CellType"].value_counts()
    # names(celltypes)[celltypes > k] -- table() order is sorted level order
    celltypes = [ct for ct in sorted(ct_counts.index) if ct_counts[ct] > k]

    blocks, names, kept_cts = [], [], []
    for ct in celltypes:
        sel = (metadata["CellType"] == ct).to_numpy()
        tmpmeta = metadata.loc[sel]
        tmpgcm = normdata[:, sel] if sp.issparse(normdata) \
            else np.asarray(normdata)[:, sel]
        w, spots = get_knn_weights(tmpmeta, spotCoord, k=k, radius=radius,
                                   min_cells_per_region=min_cells_per_region)
        if w is None:
            continue
        mc = tmpgcm @ w
        if mc.shape[1] <= 2:                # metacell_list[ncols > 2]
            continue
        blocks.append(mc)
        names.extend([f"{s}..{ct}" for s in spots])
        kept_cts.append(ct)

    excluded = sorted(set(metadata["CellType"].unique()) - set(kept_cts))
    if verbose and excluded:
        print("Excluding cell types due to insufficient spatial metacells: "
              + ", ".join(map(str, excluded)) + ".")

    if not blocks:
        raise ValueError("No cell type produced spatial metacells.")
    if any(sp.issparse(b) for b in blocks):
        merged = sp.hstack([sp.csc_matrix(b) for b in blocks], format="csc")
    else:
        merged = np.hstack(blocks)
    return merged, names
