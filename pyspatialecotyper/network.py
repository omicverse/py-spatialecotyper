"""Cell-type-specific similarity networks and their fusion.

``get_pc_list`` -> ``get_sn_list`` -> ``snf2`` is the numerical spine of
SpatialEcoTyper: one KNN similarity graph over spatial neighbourhoods per cell
type, then a cross-diffusion that fuses them into a single consensus graph.

``snf2`` is upstream's enhanced Similarity Network Fusion (Wang et al. 2014),
adapted from SNFtool to tolerate sparse inputs and missing neighbourhoods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .metacells import _nn2
from .seuratcompat import find_variable_features, run_pca, scale_data
from .utils import matrix_multiply

__all__ = ["get_pc_list", "get_sn", "get_sn_list", "fillspots", "snf2",
           "dominateset"]


# ---------------------------------------------------------------------------
# GetPCList
# ---------------------------------------------------------------------------
def get_pc_list(mergedncem, colnames: list[str], gene_names: list[str],
                min_cells: int = 3, min_features: int = 5,
                nfeatures: int = 3000, ncores: int = 1,
                do_scale: bool = True, verbose: bool = True):
    """``GetPCList(mergedncem, min.cells, min.features, nfeatures, ncores, do.scale)``.

    One PCA per cell type over that cell type's spatial-metacell profiles.

    Three filters, in R's order (``R/GetPCList.R``):

    1. genes detected in ``>= min.cells`` metacells; bail out (``NULL``) if
       fewer than ``min_features`` genes survive;
    2. metacells with ``>= min_features`` detected genes; bail out if fewer
       than ``min_cells`` metacells survive;
    3. ``npcs = min(ncol(ncem) - 3, 30)``.

    The Seurat object is built on ``ncem - rowMeans(ncem)`` -- i.e. the data are
    already gene-centred before ``FindVariableFeatures`` sees them, which is why
    the dispersion values can be negative and why ``FastLogVMR`` returns NaN for
    a good fraction of genes (mapped to 0 by Seurat).

    Returns ``{celltype: (embedding_pcs_x_spots, spot_names)}``.
    """
    colnames = np.asarray(colnames, dtype=object)
    # celltypes <- gsub(".*\\.+", "", colnames)  -- text after the last '.'
    celltypes = np.array([c.rsplit("..", 1)[-1] for c in colnames], dtype=object)
    spots_all = np.array([c.rsplit("..", 1)[0] for c in colnames], dtype=object)

    out = {}
    dropped = []
    for ct in pd.unique(celltypes):
        sel = celltypes == ct
        ncem = mergedncem[:, sel] if sp.issparse(mergedncem) \
            else np.asarray(mergedncem)[:, sel]
        spots = spots_all[sel]
        genes = np.asarray(gene_names, dtype=object)

        nz_row = np.asarray((ncem > 0).sum(axis=1)).ravel()
        keep_g = nz_row >= min_cells
        if keep_g.sum() < min_features:
            dropped.append(ct)
            continue
        ncem = ncem[keep_g, :]
        genes = genes[keep_g]

        nz_col = np.asarray((ncem > 0).sum(axis=0)).ravel()
        keep_c = nz_col >= min_features
        if keep_c.sum() < min_cells:
            dropped.append(ct)
            continue
        ncem = ncem[:, keep_c]
        spots = spots[keep_c]

        dense = np.asarray(ncem.todense()) if sp.issparse(ncem) else np.asarray(ncem, float)
        centred = dense - dense.mean(axis=1, keepdims=True)

        hvf = find_variable_features(centred, nfeatures=nfeatures,
                                     selection_method="dispersion",
                                     feature_names=list(genes))
        gi = {g: i for i, g in enumerate(genes)}
        rows = np.array([gi[g] for g in hvf], dtype=int)
        scaled = scale_data(centred[rows], do_scale=do_scale, do_center=True)

        npcs = min(centred.shape[1] - 3, 30)
        if npcs < 1:
            dropped.append(ct)
            continue
        emb, _, _ = run_pca(scaled, npcs=npcs)
        out[ct] = (emb.T, list(spots))          # t(Embeddings(obj, "pca"))

    if verbose and dropped:
        print("Excluding cell types with insufficient spatial metacells: "
              + ", ".join(map(str, dropped)) + ".")
    return out


# ---------------------------------------------------------------------------
# getSN / GetSNList / fillspots
# ---------------------------------------------------------------------------
def get_sn(emb: np.ndarray, k: int = 50):
    """``getSN(emb, k)`` (``R/GetSNList.R``).

    ``emb`` is PCs x spots.  Builds an exact ``tmpK = min(k, ncol(emb) - 1)``
    nearest-neighbour graph over the spot columns and converts distance to
    similarity as ``1 / (d + 2)`` -- R writes it in two steps
    (``W@x <- d + 1`` then ``W@x <- 1 / (W@x + 1)``), so a spot's self-entry,
    at distance 0, gets the maximal similarity 0.5 rather than 1.

    The graph is directed: ``W[neighbour, query] = similarity``, because R fills
    with the column-major linear index ``nrow(W) * (0:(ncol-1)) + nn.idx[, i]``.
    """
    n = emb.shape[1]
    tmp_k = min(k, n - 1)
    pts = np.ascontiguousarray(emb.T)          # spots x PCs  == t(emb)
    idx, dists = _nn2(pts, pts, tmp_k)

    rows, cols, vals = [], [], []
    for i in range(tmp_k):
        ok = idx[:, i] >= 0
        d = dists[ok, i] + 1.0
        fin = np.isfinite(d)
        rows.append(idx[ok, i][fin])
        cols.append(np.flatnonzero(ok)[fin])
        vals.append(d[fin])
    w = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsc()
    w.eliminate_zeros()                        # drop0()
    w.data = 1.0 / (w.data + 1.0)
    return w


def fillspots(snlist: dict):
    """``fillspots(snlist)`` -- pad every network to the union of spot sets.

    ``spots <- unique(unlist(lapply(snlist, colnames)))`` is *first-appearance*
    order across the list, not sorted order; every matrix is then reindexed to
    that order so the fusion can add them elementwise.
    """
    spots = []
    seen = set()
    for _, (_, names) in snlist.items():
        for s in names:
            if s not in seen:
                seen.add(s)
                spots.append(s)
    out = {}
    for ct, (w, names) in snlist.items():
        pos = {s: i for i, s in enumerate(names)}
        n_old, n_new = len(names), len(spots)
        # map old index -> new index; missing spots get an all-zero row/col
        take = np.array([pos.get(s, -1) for s in spots])
        p = sp.csr_matrix(
            (np.ones((take >= 0).sum()),
             (np.flatnonzero(take >= 0), take[take >= 0])),
            shape=(n_new, n_old))
        out[ct] = (sp.csc_matrix(p @ w @ p.T), spots)
    return out, spots


def get_sn_list(emb_list: dict, npcs: int = 20, k: int = 50,
                min_cts_per_region: int = 1, ncores: int = 1,
                verbose: bool = True):
    """``GetSNList(emb_list, npcs, k, min.cts.per.region, ncores)``.

    ``min_cts_per_region`` filters *spatial neighbourhoods*, not cell types: a
    neighbourhood must appear in at least that many cell-type embeddings to be
    kept.  ``table()`` counts appearances across the whole list.
    """
    trimmed = {ct: (emb[:min(npcs, emb.shape[0])], names)
               for ct, (emb, names) in emb_list.items()}

    counts: dict[str, int] = {}
    for _, names in trimmed.values():
        for s in names:
            counts[s] = counts.get(s, 0) + 1
    n_drop = sum(1 for v in counts.values() if v < min_cts_per_region)
    if n_drop:
        if verbose:
            print(f"Removing {n_drop} spatial neighborhoods with fewer than "
                  f"{min_cts_per_region} distinct cell types.")
        allow = {s for s, v in counts.items() if v >= min_cts_per_region}
        trimmed = {ct: (emb[:, [i for i, s in enumerate(names) if s in allow]],
                        [s for s in names if s in allow])
                   for ct, (emb, names) in trimmed.items()}

    snlist = {ct: (get_sn(emb, k=k), names) for ct, (emb, names) in trimmed.items()}
    return fillspots(snlist)


# ---------------------------------------------------------------------------
# SNF2
# ---------------------------------------------------------------------------
def _normalize(x):
    """``normalize(X)`` from ``R/SNF2.R``.

    ``row.sum.mdiag <- rowSums(X) - diag(X)``; zeros are replaced by 1 to avoid
    a division by zero; ``X <- X / (2 * row.sum.mdiag)`` divides **row-wise**
    (R recycles the vector down the columns of a column-major matrix);
    ``diag(X) <- 0.5``; then symmetrise.
    """
    x = sp.csr_matrix(x, copy=True).astype(np.float64)
    rs = np.asarray(x.sum(axis=1)).ravel() - x.diagonal()
    rs[rs == 0] = 1.0
    x = sp.diags(1.0 / (2.0 * rs)) @ x
    x = sp.lil_matrix(x)
    x.setdiag(0.5)
    x = sp.csr_matrix(x)
    return ((x + x.T) * 0.5).tocsr()


def dominateset(xx, KK: int = 20, ncores: int = 8):
    """``.dominateset(xx, KK)`` -- keep only the ``KK`` largest entries per row.

    R's ``zero()`` sorts the *dense* row ascending and zeroes the first
    ``length(x) - KK`` positions.  With many tied zeros the choice of which ties
    survive is decided by ``sort``'s stability (R uses a radix sort for numeric
    vectors, which is stable), so ``np.argsort(kind="stable")`` reproduces it.

    Two edge cases are inherited verbatim from R's ``x[s$ix[1:(length(x)-KK)]]``:

    * ``n == KK``: ``1:0`` is ``c(1, 0)`` in R, and index 0 is dropped, so R
      zeroes the **single smallest** element rather than none of them.
    * ``n < KK``: ``1:negative`` mixes positive and negative subscripts and R
      aborts with "only 0's may be mixed with negative subscripts".  We raise
      the same error rather than silently doing something different.
    """
    x = sp.csr_matrix(xx).astype(np.float64)
    n = x.shape[0]
    if x.shape[1] < KK:
        raise ValueError(
            "only 0's may be mixed with negative subscripts -- .dominateset "
            f"needs at least KK={KK} columns, got {x.shape[1]}")
    rows, cols, vals = [], [], []
    for i in range(n):
        dense = np.asarray(x[i].todense()).ravel()
        if dense.size == KK:
            order = np.argsort(dense, kind="stable")
            keep_pos = order[1:]
            keep = keep_pos[dense[keep_pos] != 0]
        else:
            order = np.argsort(dense, kind="stable")   # ascending, stable
            keep_pos = order[dense.size - KK:]
            keep = keep_pos[dense[keep_pos] != 0]
        if keep.size:
            rows.append(np.full(keep.size, i))
            cols.append(keep)
            vals.append(dense[keep])
    if not rows:
        return sp.csr_matrix(x.shape)
    return sp.coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=x.shape).tocsr()


def snf2(Wall: list, K: int = 10, t: int = 10, minibatch: int = 5000,
         ncores: int = 4, verbose: bool = False):
    """``SNF2(Wall, K, t, minibatch, ncores, verbose)`` (``R/SNF2.R``).

    Enhanced Similarity Network Fusion.  Each view is row-normalised, reduced
    to its ``K`` dominant neighbours, and then cross-diffused for ``t`` rounds::

        W_j <- normalize( P_j  x  (mean of the other views)  x  P_j^T )

    The final matrix averages the views **only over the views that actually
    observed each pair** (``Reduce("+", Wall) / Reduce("+", Wall > 0)``), which
    is the upstream modification that lets a neighbourhood be missing from some
    cell types without dragging its fused similarity toward zero.
    """
    if len(Wall) < 2:
        raise ValueError(">=2 similarity matrices are required.")

    Wall = [_normalize(w) for w in Wall]
    newW = [_normalize(dominateset(w, K, ncores)) for w in Wall]

    LW = len(Wall)
    for it in range(t):
        if verbose:
            print(f"\tIteration: {it + 1}")
        nxt = []
        for j in range(LW):
            acc = None
            for m in range(LW):
                if m == j:
                    continue
                acc = Wall[m] if acc is None else acc + Wall[m]
            sum_wj = acc * (1.0 / (LW - 1))
            x = matrix_multiply(newW[j], sum_wj, minibatch=minibatch)
            x = matrix_multiply(x, newW[j].T, minibatch=minibatch)
            nxt.append(_normalize(x))
        Wall = nxt

    counts = None
    for w in Wall:
        c = (w > 0).astype(np.float64)
        counts = c if counts is None else counts + c
    total = None
    for w in Wall:
        total = w if total is None else total + w

    total = sp.csr_matrix(total)
    counts = sp.csr_matrix(counts)
    # Elementwise total/counts on the union pattern; 0/0 -> NaN -> 0 and
    # x/0 -> Inf -> 0, exactly as R's
    #   Wall@x[is.infinite(Wall@x)] <- 0; Wall@x[is.nan(Wall@x)] <- 0
    den = counts.toarray()
    num = total.toarray()
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)
    res[~np.isfinite(res)] = 0.0
    return sp.csr_matrix(res)
