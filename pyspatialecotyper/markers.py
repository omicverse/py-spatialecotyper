"""``AverageMarkerExpression`` -- SE-level marker-set scoring.

The R function takes a Seurat object, pseudo-bulks it by SE, scores a list of
gene sets, z-scores the resulting matrix and hands it to ``HeatmapView``.
Here the Seurat object is replaced by an explicit ``(matrix, group)`` pair, so
the function works on any expression matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .seuratcompat import scale_data

__all__ = ["average_expression", "average_marker_expression"]


def average_expression(data, groups, gene_names=None) -> pd.DataFrame:
    """``Seurat::AverageExpression(obj, group.by, layer = "data")``.

    Seurat averages on the **linear** scale and returns
    ``expm1`` -> mean -> (no log), i.e. ``rowMeans(expm1(data))`` per group for
    a log-normalised ``data`` layer.  Groups come out in sorted-factor order.
    """
    x = np.asarray(data.todense()) if sp.issparse(data) else np.asarray(data, float)
    groups = np.asarray([str(g) for g in np.asarray(groups).ravel()])
    if gene_names is None:
        gene_names = [f"G{i}" for i in range(x.shape[0])]
    levels = sorted(pd.unique(groups))
    out = np.column_stack([np.expm1(x[:, groups == g]).mean(axis=1) for g in levels])
    return pd.DataFrame(out, index=list(gene_names), columns=levels)


def average_marker_expression(data, groups, gene_names=None,
                              genesets: dict | None = None,
                              plot: bool = True):
    """``AverageMarkerExpression(obj, group.by, genesets)``.

    Parameters
    ----------
    data, groups, gene_names
        Expression matrix (genes x cells), per-cell SE labels, gene names.
    genesets
        ``{set_name: [gene, ...]}``.  R defaults to the SE consensus markers
        shipped in the package's ``inst/extdata``; those ``.rds`` files are not
        redistributable here, so ``genesets`` is required.

    Returns
    -------
    ``(avg_exprs, fig)`` where ``avg_exprs`` is a genesets x SE DataFrame,
    z-scored across SEs, and ``fig`` is the heatmap (``None`` if ``plot`` is
    ``False``).

    The R implementation drops the ``"NonSE"`` column, log2(x+1)-transforms
    the pseudo-bulk, and z-scores only the rows that have at least one
    non-missing value (``ses_not_na``).
    """
    if genesets is None:
        raise ValueError(
            "genesets is required: the default SE consensus markers live in the "
            "R package's inst/extdata and are not redistributed with this port. "
            "Load them from a SpatialEcoTyper checkout and pass them here.")

    pseudobulk = average_expression(data, groups, gene_names)
    pseudobulk = pseudobulk.loc[(pseudobulk > 0).sum(axis=1) > 0]
    pseudobulk = np.log2(pseudobulk + 1)
    pseudobulk = pseudobulk.loc[:, [c for c in pseudobulk.columns if c != "NonSE"]]

    rows, names = [], []
    for name, gs in genesets.items():
        hit = [g for g in gs if g in pseudobulk.index]
        names.append(name)
        if not hit:
            rows.append(np.full(pseudobulk.shape[1], np.nan))
        else:
            rows.append(pseudobulk.loc[pseudobulk.index.isin(hit)].mean(axis=0).to_numpy())
    avg = np.vstack(rows)

    ok = (~np.isnan(avg)).sum(axis=1) > 0
    avg[ok] = scale_data(avg[ok])
    avg = pd.DataFrame(avg, index=names, columns=list(pseudobulk.columns))

    fig = None
    if plot:
        from .palettes import magma
        from .plotting import heatmap_view
        colors = [magma(7)[i] for i in (0, 1, 2, 4, 6)]
        breaks = np.nanquantile(avg.to_numpy(), [0.3, 0.5, 0.7, 0.85, 0.95])
        fig = heatmap_view(avg, breaks=tuple(breaks), colors=tuple(colors),
                           legend_height=1, na_col="#666666",
                           column_title="Spatial ecotypes",
                           column_title_side="bottom",
                           row_title="SE consensus markers",
                           show_row_names=True, show_column_names=True)
    return avg, fig
