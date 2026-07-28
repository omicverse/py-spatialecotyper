"""Matplotlib port of SpatialEcoTyper's visualisation surface.

Covers four R entry points:

============================== ==========================================
R                              Python
============================== ==========================================
``SpatialView`` (SpatialView.R)      :func:`spatial_view`
``HeatmapView`` (HeatmapView.R)      :func:`heatmap_view`
``drawRectangleAnnotation``          :func:`draw_rectangle_annotation`
``CooccurrenceHeatmapView``          :func:`cooccurrence_heatmap_view`
============================== ==========================================

``SpatialView`` is a ggplot2 scatter plot and ports cleanly.  ``HeatmapView``
is a wrapper around ``ComplexHeatmap::Heatmap``; there is no ComplexHeatmap in
Python, so the *semantics* are reproduced on top of matplotlib and
``scipy.cluster.hierarchy``: ``circlize::colorRamp2`` LAB interpolation with
clamping, row/column dendrograms, four annotation tracks with their own colour
mappings, ``row_split`` / ``column_split`` slicing, and a legend stack.  Every
R argument survives with the same name (up to PEP-8) and the same default.
The rendering is a faithful equivalent, not a pixel-for-pixel copy.

No function calls ``plt.show()``; each returns the :class:`~matplotlib.figure.Figure`
it drew into so the caller decides what to do with it.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

# Only claim the backend when the process has not already chosen one, so that
# importing this module inside a notebook or a GUI session is harmless.
if "matplotlib.pyplot" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .palettes import get_colors  # noqa: E402
from . import rrandom  # noqa: E402

__all__ = [
    "spatial_view",
    "heatmap_view",
    "cooccurrence_heatmap_view",
    "draw_rectangle_annotation",
    "color_ramp2",
    "BlockAnnotation",
    "HeatmapViewResult",
]


# ===========================================================================
# circlize::colorRamp2
# ===========================================================================

# colorspace's D65 white point, which is what circlize inherits when it asks
# for space = "LAB" (the colorRamp2 default).
_WHITE_X, _WHITE_Y, _WHITE_Z = 95.047, 100.000, 108.883


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIE L*a*b*, matching ``colorspace``'s constants."""
    rgb = np.asarray(rgb, dtype=float)
    # gtrans(): colorspace uses the 0.03928 threshold, not the later 0.04045.
    lin = np.where(rgb > 0.03928, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 100.0 * (0.412453 * r + 0.357580 * g + 0.180423 * b)
    y = 100.0 * (0.212671 * r + 0.715160 * g + 0.072169 * b)
    z = 100.0 * (0.019334 * r + 0.119193 * g + 0.950227 * b)
    xr, yr, zr = x / _WHITE_X, y / _WHITE_Y, z / _WHITE_Z

    def f(t):
        return np.where(t > 0.008856, np.cbrt(np.maximum(t, 0.0)),
                        7.787 * t + 16.0 / 116.0)

    ll = np.where(yr > 0.008856, 116.0 * np.cbrt(np.maximum(yr, 0.0)) - 16.0,
                  903.3 * yr)
    aa = 500.0 * (f(xr) - f(yr))
    bb = 200.0 * (f(yr) - f(zr))
    return np.stack([ll, aa, bb], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_srgb_to_lab`, clamped to the sRGB cube."""
    lab = np.asarray(lab, dtype=float)
    ll, aa, bb = lab[..., 0], lab[..., 1], lab[..., 2]
    y = np.where(ll <= 8.0, ll * _WHITE_Y / 903.3,
                 _WHITE_Y * ((ll + 16.0) / 116.0) ** 3)
    fy = np.where(y / _WHITE_Y > 0.008856, np.cbrt(np.maximum(y / _WHITE_Y, 0.0)),
                  7.787 * (y / _WHITE_Y) + 16.0 / 116.0)
    fx = fy + aa / 500.0
    fz = fy - bb / 200.0

    def finv(t):
        return np.where(t ** 3 > 0.008856, t ** 3, (t - 16.0 / 116.0) / 7.787)

    x = _WHITE_X * finv(fx)
    z = _WHITE_Z * finv(fz)
    xn, yn, zn = x / 100.0, y / 100.0, z / 100.0
    r = 3.240479 * xn - 1.537150 * yn - 0.498535 * zn
    g = -0.969256 * xn + 1.875992 * yn + 0.041556 * zn
    b = 0.055648 * xn - 0.204043 * yn + 1.057311 * zn
    lin = np.stack([r, g, b], axis=-1)
    # ftrans()
    out = np.where(lin > 0.0031308, 1.055 * np.power(np.maximum(lin, 0.0), 1 / 2.4) - 0.055,
                   12.92 * lin)
    return np.clip(out, 0.0, 1.0)


class ColorRamp2:
    """``circlize::colorRamp2(breaks, colors, space = "LAB")``.

    Values outside ``[min(breaks), max(breaks)]`` are clamped to the end
    colours, which is what makes ``HeatmapView``'s explicit clamp on lines
    121-122 of ``HeatmapView.R`` redundant but harmless.
    """

    def __init__(self, breaks: Sequence[float], colors: Sequence[str],
                 space: str = "LAB"):
        breaks = np.asarray(breaks, dtype=float)
        colors = list(colors)
        if len(breaks) != len(colors):
            raise ValueError("'breaks' and 'colors' must have the same length")
        order = np.argsort(breaks, kind="stable")          # colorRamp2 sorts
        breaks = breaks[order]
        colors = [colors[i] for i in order]
        keep = np.concatenate([[True], np.diff(breaks) != 0])  # duplicated()
        self.breaks = breaks[keep]
        self.colors = [c for c, k in zip(colors, keep) if k]
        if len(self.breaks) < 2:
            raise ValueError("'breaks' should have at least two distinct values")
        self.space = space
        rgb = np.array([to_rgb(c) for c in self.colors], dtype=float)
        self._anchor = _srgb_to_lab(rgb) if space.upper() == "LAB" else rgb

    def __call__(self, x) -> np.ndarray:
        """Map values to an RGB array (``...`` x 3, floats in [0, 1])."""
        x = np.asarray(x, dtype=float)
        shape = x.shape
        flat = np.clip(x.ravel(), self.breaks[0], self.breaks[-1])
        idx = np.searchsorted(self.breaks, flat, side="left") - 1
        idx = np.clip(idx, 0, len(self.breaks) - 2)
        lo, hi = self.breaks[idx], self.breaks[idx + 1]
        w = np.where(hi > lo, (flat - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
        mixed = (self._anchor[idx] * (1.0 - w[:, None])
                 + self._anchor[idx + 1] * w[:, None])
        rgb = _lab_to_srgb(mixed) if self.space.upper() == "LAB" else np.clip(mixed, 0, 1)
        return rgb.reshape(shape + (3,))

    def to_hex(self, x) -> Any:
        rgb = self(x)
        flat = rgb.reshape(-1, 3)
        hexes = ["#%02X%02X%02X" % tuple(int(round(c * 255)) for c in row)
                 for row in flat]
        if np.asarray(x).ndim == 0:
            return hexes[0]
        return np.array(hexes).reshape(np.asarray(x).shape)

    def to_cmap(self, n: int = 256) -> LinearSegmentedColormap:
        """A matplotlib colormap sampling this ramp, for colorbars."""
        grid = np.linspace(self.breaks[0], self.breaks[-1], n)
        return LinearSegmentedColormap.from_list("colorRamp2", self(grid), N=n)


def color_ramp2(breaks: Sequence[float], colors: Sequence[str],
                space: str = "LAB") -> ColorRamp2:
    """Functional spelling of :class:`ColorRamp2`."""
    return ColorRamp2(breaks, colors, space=space)


# ===========================================================================
# small R compatibility shims
# ===========================================================================

_R_PCH_TO_MARKER = {
    0: "s", 1: "o", 2: "^", 3: "+", 4: "x", 5: "D", 6: "v", 7: "x",
    8: "*", 15: "s", 16: "o", 17: "^", 18: "D", 19: "o", 20: "o",
    21: "o", 22: "s", 23: "D", 24: "^", 25: "v",
}

_MM_PER_GG_SIZE = 2.845276  # ggplot2 point size is a diameter in mm; 1 mm = 2.845 pt


def _r_color(col: str) -> str:
    """Translate the handful of R colour names matplotlib does not know."""
    if not isinstance(col, str):
        return col
    m = re.fullmatch(r"(?:gray|grey)(\d{1,3})", col)
    if m:
        level = int(m.group(1))
        v = int(level / 100.0 * 255 + 0.5)
        return "#%02X%02X%02X" % (v, v, v)
    return col


def _gpar(gp: Mapping[str, Any] | None) -> dict:
    """``grid::gpar(...)`` -> matplotlib text kwargs."""
    out: dict[str, Any] = {}
    if not gp:
        return out
    if "fontsize" in gp:
        out["fontsize"] = gp["fontsize"]
    if "col" in gp:
        out["color"] = _r_color(gp["col"])
    if "fontfamily" in gp:
        out["family"] = gp["fontfamily"]
    face = gp.get("fontface")
    if face in (2, "bold"):
        out["fontweight"] = "bold"
    elif face in (3, "italic"):
        out["fontstyle"] = "italic"
    elif face in (4, "bold.italic"):
        out["fontweight"] = "bold"
        out["fontstyle"] = "italic"
    return out


def _first(x):
    """R's ``match.arg`` on a default vector: take the first element."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def _unique_in_order(values) -> list:
    """R ``unique()``: first-appearance order, NAs kept as-is."""
    seen: list = []
    known = set()
    for v in values:
        key = ("__nan__" if (isinstance(v, float) and np.isnan(v)) else v)
        if key in known:
            continue
        known.add(key)
        seen.append(v)
    return seen


# ===========================================================================
# SpatialView
# ===========================================================================

def spatial_view(obj, by, x="X", y="Y",
                 pt_shape=20,
                 pt_size=0.5,
                 pt_alpha=1,
                 jitter=False,
                 slot="data",
                 coord_fix=False,
                 highlight_cells=None,
                 control_cells=None,
                 bg_downsample=2000,
                 bg_color="gray80",
                 bg_size=0.5,
                 bg_alpha=0.7,
                 ax=None,
                 figsize=None):
    """Visualise the spatial landscape of cells / spots -- R ``SpatialView``.

    Parameters mirror ``SpatialView.R`` line 43 onwards; ``X``/``Y`` become
    ``x``/``y`` and the dotted R names become underscored.

    Parameters
    ----------
    obj
        A :class:`pandas.DataFrame` indexed by cell name, or an AnnData-like
        object (anything exposing ``.obs``).  For AnnData, ``by`` is looked up
        in ``.obs`` first and then among the variables, in which case the
        expression is pulled from the layer named by ``slot`` (``"data"`` and
        ``"X"`` both mean ``adata.X``).
    by
        Feature to colour by: cell type, region, gene expression, ...
    x, y
        Column names holding the spatial coordinates.
    pt_shape
        R ``pch`` code; translated to a matplotlib marker.
    pt_size, pt_alpha
        Size (ggplot2 mm) and alpha of the non-control points.
    jitter
        Add uniform jitter of one coordinate step, as on lines 62-67.
    coord_fix
        ``coord_fixed()`` -- equal aspect ratio.
    highlight_cells, control_cells
        Cell-name subsets.  Control cells are drawn flat in ``bg_color``.
    bg_downsample
        Cap on the number of control cells drawn.
    bg_color, bg_size, bg_alpha
        Styling of the control cells.
    ax, figsize
        Optional target axes / figure size.

    Returns
    -------
    (Figure, Axes)
    """
    data = _fetch_data(obj, x, y, by, slot)

    if jitter:
        # SpatialView.R lines 62-67: jitter by the smallest non-zero spacing.
        for col in ("X", "Y"):
            vals = np.sort(np.asarray(data[col], dtype=float))
            diffs = np.unique(np.diff(vals))
            diffs = np.sort(diffs[diffs != 0])
            if diffs.size:
                interval = float(diffs[0])
                # R uses the global stream here without seeding; rrandom's
                # module-global stream keeps the result deterministic.
                noise = rrandom.runif(len(data), -interval, interval)
                data[col] = np.asarray(data[col], dtype=float) + noise

    marker = _R_PCH_TO_MARKER.get(pt_shape, pt_shape if isinstance(pt_shape, str) else "o")
    s_fg = (pt_size * _MM_PER_GG_SIZE) ** 2
    s_bg = (bg_size * _MM_PER_GG_SIZE) ** 2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6.0, 5.5))
    else:
        fig = ax.figure

    bgdata = None
    if highlight_cells is not None and len(highlight_cells) > 1:
        # SpatialView.R lines 72-80
        highlightdata = data.loc[data.index.isin(list(highlight_cells))]
        if control_cells is None or len(control_cells) < 2:
            control_cells = [i for i in data.index if i not in set(highlight_cells)]
        bgdata = data.loc[data.index.isin(list(control_cells))]
    elif control_cells is not None and len(control_cells) > 1:
        # SpatialView.R lines 81-88
        bgdata = data.loc[data.index.isin(list(control_cells))]
        highlightdata = data.loc[~data.index.isin(list(control_cells))]
    else:
        highlightdata = data

    if bgdata is not None and len(bgdata) > bg_downsample:
        idx = rrandom.sample_int(len(bgdata), int(bg_downsample), replace=False)
        bgdata = bgdata.iloc[np.asarray(idx)]

    if bgdata is not None:
        ax.scatter(bgdata["X"], bgdata["Y"], c=_r_color(bg_color), marker=marker,
                   s=s_bg, alpha=bg_alpha, linewidths=0)

    group = highlightdata["group.by"]
    if pd.api.types.is_numeric_dtype(group):
        # SpatialView.R line 95: scale_color_gradient(low, high)
        cmap = LinearSegmentedColormap.from_list("spatialview", ["#00204C", "#FFE945"])
        sc = ax.scatter(highlightdata["X"], highlightdata["Y"], c=np.asarray(group, dtype=float),
                        cmap=cmap, marker=marker, s=s_fg, alpha=pt_alpha, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.6)
    else:
        levels = _unique_in_order(list(group))
        if len(levels) <= 45:
            # SpatialView.R lines 97-101: kelly minus its first entry, then cols25.
            from .palettes import _KELLY, _COLS25
            palette = list(_KELLY[1:]) + list(_COLS25)
            colmap = dict(zip(levels, palette[:len(levels)]))
        else:
            colmap = dict(zip(levels, get_colors(len(levels), palette=1)))
        for lev in levels:
            sub = highlightdata[group.astype(object) == lev]
            ax.scatter(sub["X"], sub["Y"], c=colmap[lev], marker=marker, s=s_fg,
                       alpha=pt_alpha, linewidths=0, label=str(lev))
        if len(levels) <= 45:
            handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                              markerfacecolor=colmap[l], markeredgecolor="none",
                              label=str(l)) for l in levels]
            # labs(color = NULL) -> no legend title (line 107)
            ax.legend(handles=handles, title=None, frameon=False, fontsize=9,
                      loc="center left", bbox_to_anchor=(1.01, 0.5))

    # theme_void(base_size = 12) -- line 93
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if coord_fix:
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def _fetch_data(obj, x, y, by, slot) -> pd.DataFrame:
    """``FetchData(obj, c(X, Y, by), slot = slot)`` for DataFrame / AnnData."""
    if isinstance(obj, pd.DataFrame):
        df = obj
        vals = df[by]
    elif hasattr(obj, "obs"):
        df = obj.obs
        if by in df.columns:
            vals = df[by]
        else:
            var_names = list(getattr(obj, "var_names", []))
            if by not in var_names:
                raise KeyError("%r is neither an .obs column nor a variable" % by)
            j = var_names.index(by)
            mat = obj.X if slot in ("data", "X", None) else obj.layers[slot]
            col = mat[:, j]
            col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
            vals = pd.Series(col, index=df.index, name=by)
    else:
        raise TypeError("obj must be a pandas DataFrame or an AnnData-like object")

    if x in df.columns and y in df.columns:
        xs, ys = df[x], df[y]
    elif hasattr(obj, "obsm") and "spatial" in getattr(obj, "obsm", {}):
        spatial = np.asarray(obj.obsm["spatial"])
        xs = pd.Series(spatial[:, 0], index=df.index)
        ys = pd.Series(spatial[:, 1], index=df.index)
    else:
        raise KeyError("coordinates %r / %r not found" % (x, y))

    # SpatialView.R lines 68-70: rename to the canonical X / Y / group.by
    out = pd.DataFrame({"X": np.asarray(xs, dtype=float),
                        "Y": np.asarray(ys, dtype=float),
                        "group.by": np.asarray(vals)}, index=df.index)
    return out


# ===========================================================================
# HeatmapView
# ===========================================================================

class BlockAnnotation:
    """``ComplexHeatmap::anno_block(align_to = ..., labels = ...)``.

    ``CooccurrenceHeatmapView`` builds its right-hand annotation this way
    (``Colocalization.R`` lines 268-272): one text label per group of rows.
    """

    def __init__(self, align_to: Mapping[Any, Sequence[int]],
                 labels: Sequence[str], name: str = "",
                 gp: Mapping[str, Any] | None = None):
        self.align_to = {k: np.asarray(v, dtype=int) for k, v in align_to.items()}
        self.labels = list(labels)
        self.name = name
        self.gp = dict(gp or {})


class HeatmapViewResult(Figure):
    """Marker class -- :func:`heatmap_view` returns a plain Figure carrying
    ``.heatmap_view_info``; this alias exists only for documentation."""


def _as_frame(ann) -> pd.DataFrame | None:
    if ann is None:
        return None
    if isinstance(ann, BlockAnnotation):
        return ann
    if isinstance(ann, pd.DataFrame):
        return ann
    if isinstance(ann, pd.Series):
        return ann.to_frame()
    return pd.DataFrame(ann)


def _annotation_colors(df: pd.DataFrame, palettes: Mapping[str, Any] | None):
    """``heatmap_annotation`` from ``HeatmapView.R`` line 184.

    For every column without a user-supplied mapping: numeric columns get a
    blue/white/red ``colorRamp2`` at the 10 / 50 / 90 % quantiles (line 195);
    everything else gets ``getColors(n_unique, palette = which(columns == col))``
    named by ``unique(df[, col])`` (lines 197-198).  The palette index is the
    column's position among the *remaining* columns, exactly as in R.
    """
    palettes = dict(palettes or {})
    if len(df.columns) > len(palettes):
        columns = [c for c in df.columns if c not in palettes]
        for i, col in enumerate(columns, start=1):
            series = df[col]
            if series.isna().all():
                continue
            if pd.api.types.is_numeric_dtype(series):
                q = np.nanquantile(np.asarray(series, dtype=float), [0.1, 0.5, 0.9])
                palettes[col] = color_ramp2(q, ["blue", "white", "red"])
            else:
                levels = _unique_in_order(list(series))
                cols = get_colors(len(levels), palette=i)
                palettes[col] = dict(zip(levels, cols))
    return palettes


def _ann_rgb(series: pd.Series, mapping, na_col: str) -> np.ndarray:
    """Render one annotation column to an (n, 3) RGB array."""
    n = len(series)
    out = np.empty((n, 3), dtype=float)
    na_rgb = to_rgb(_r_color(na_col))
    if isinstance(mapping, ColorRamp2):
        vals = np.asarray(series, dtype=float)
        rgb = mapping(np.nan_to_num(vals, nan=mapping.breaks[0]))
        out[:] = rgb
        out[np.isnan(vals)] = na_rgb
    else:
        for i, v in enumerate(series):
            c = mapping.get(v) if isinstance(mapping, dict) else None
            out[i] = to_rgb(_r_color(c)) if c is not None else na_rgb
    return out


def _leaf_order(sub: np.ndarray, cluster, axis: int):
    """Dendrogram leaf order for one slice; returns (order, linkage or None)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    m = sub if axis == 0 else sub.T
    if m.shape[0] < 2:
        return np.arange(m.shape[0]), None
    data = np.nan_to_num(m.astype(float), nan=0.0)
    if isinstance(cluster, str):
        method = cluster
    else:
        method = "complete"          # ComplexHeatmap's clustering_method default
    z = linkage(pdist(data, metric="euclidean"), method=method)
    order = dendrogram(z, no_plot=True)["leaves"]
    return np.asarray(order, dtype=int), z


def _make_slices(mat: np.ndarray, axis: int, cluster, split):
    """Compute the slice structure for one axis.

    Returns a list of ``(title, indices)`` where ``indices`` are positions into
    the original matrix, already ordered.  ``split`` may be ``None``, a label
    vector, or -- when clustering is on -- an integer ``k`` meaning ``cutree``.
    """
    n = mat.shape[axis]
    if split is None:
        groups = [(None, np.arange(n))]
    elif np.isscalar(split) and not isinstance(split, str):
        from scipy.cluster.hierarchy import fcluster
        if not cluster:
            raise ValueError("an integer split requires clustering on that axis")
        _, z = _leaf_order(mat, cluster, axis)
        if z is None:
            groups = [(None, np.arange(n))]
        else:
            lab = fcluster(z, int(split), criterion="maxclust")
            groups = [(str(g), np.where(lab == g)[0]) for g in _unique_in_order(list(lab))]
    else:
        lab = np.asarray(pd.Series(list(split)).astype(str))
        levels = sorted(set(lab))          # R factor levels are sorted
        groups = [(lev, np.where(lab == lev)[0]) for lev in levels]

    out = []
    for title, idx in groups:
        z = None
        if cluster is not False and len(idx) > 1:
            sub = mat[idx, :] if axis == 0 else mat[:, idx]
            order, z = _leaf_order(sub, cluster, axis)
            idx = idx[order]
        # ``z`` is the linkage of the *unordered* slice, so scipy's own
        # dendrogram leaf order is exactly the display order of ``idx``.
        out.append((title, np.asarray(idx, dtype=int), z))
    return out


def _draw_dendrogram(ax, z, n_leaves, orientation):
    """Plot a scipy linkage in [0, 1] leaf coordinates, matching the body."""
    from scipy.cluster.hierarchy import dendrogram
    if z is None or n_leaves < 2:
        ax.set_axis_off()
        return
    d = dendrogram(z, no_plot=True)
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        pos = [(v - 5.0) / 10.0 for v in xs]        # scipy leaf centres: 5, 15, ...
        pos = [(p + 0.5) / n_leaves for p in pos]
        if orientation == "top":
            ax.plot(pos, ys, color="black", linewidth=0.8)
        elif orientation == "bottom":
            ax.plot(pos, [-v for v in ys], color="black", linewidth=0.8)
        elif orientation == "left":
            ax.plot([-v for v in ys], [1.0 - p for p in pos], color="black", linewidth=0.8)
        else:  # right
            ax.plot(ys, [1.0 - p for p in pos], color="black", linewidth=0.8)
    if orientation in ("top", "bottom"):
        ax.set_xlim(0, 1)
    else:
        ax.set_ylim(0, 1)
    ax.set_axis_off()


def heatmap_view(mat,
                 breaks=(0, 0.6, 1.2),
                 colors=("#ffffd9", "#edf8b1", "#225ea8"),
                 na_col="grey",
                 name="hmap",
                 cluster_rows=False,
                 row_dend_side=("left", "right"),
                 cluster_cols=False,
                 column_dend_side=("top", "bottom"),
                 show_row_names=True,
                 row_names_side="left",
                 row_names_gp=None,
                 row_names_rot=0,
                 show_column_names=True,
                 column_names_side="bottom",
                 column_names_gp=None,
                 column_names_rot=90,
                 show_legend=True,
                 top_ann=None,
                 top_ann_col=None,
                 show_top_legend=None,
                 bott_ann=None,
                 bott_ann_col=None,
                 show_bott_legend=None,
                 left_ann=None,
                 left_ann_col=None,
                 show_left_legend=None,
                 right_ann=None,
                 right_ann_col=None,
                 show_right_legend=None,
                 show_ann_name=True,
                 annotation_legend_param=None,
                 row_split=None,
                 column_split=None,
                 show_heatmap_legend=True,
                 legend_title=None,
                 legend_title_position="lefttop",
                 legend_direction="vertical",
                 legend_title_gp=None,
                 legend_labels_gp=None,
                 legend_height=2,
                 legend_width=0.3,
                 legend_side="right",
                 fig=None,
                 figsize=None,
                 cell_size=0.22,
                 **kwargs):
    """Draw a heatmap -- R ``HeatmapView`` (``HeatmapView.R`` line 77).

    Every R argument is present with its R default.  ``row_names_gp`` and
    friends take a dict standing in for ``grid::gpar`` (``{"fontsize": 12}``,
    which is the R default and what ``None`` resolves to).  ``row_dend_side``
    and ``column_dend_side`` keep R's ``c("left", "right")`` /
    ``c("top", "bottom")`` defaults and are resolved ``match.arg``-style by
    taking the first element.

    Parameters
    ----------
    mat
        Matrix-like: :class:`pandas.DataFrame` (row/column names are used) or
        a 2-D array.
    breaks, colors
        Passed straight to :func:`color_ramp2`; ``mat`` is clamped to
        ``[min(breaks), max(breaks)]`` first, as on lines 121-122.
    cluster_rows, cluster_cols
        ``False``, ``True``, or a linkage method name such as ``"average"``.
    row_split, column_split
        A label vector, or an integer ``k`` when the matching axis is clustered.
    top_ann, bott_ann, left_ann, right_ann
        DataFrames of annotations (one column per track).  ``right_ann`` also
        accepts a :class:`BlockAnnotation`.
    top_ann_col, ...
        ``{column_name: {level: colour}}`` or ``{column_name: ColorRamp2}``.
        Missing entries are filled in by the port of ``heatmap_annotation``.
    fig, figsize, cell_size
        Rendering controls; ``cell_size`` (inches) drives the auto figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, with a ``heatmap_view_info`` attribute holding ``name``,
        ``row_order``, ``column_order`` and the per-slice body axes.  That is
        what :func:`draw_rectangle_annotation` consumes, standing in for
        ``ComplexHeatmap::row_order`` / ``column_order``.
    """
    row_names_gp = {"fontsize": 12} if row_names_gp is None else row_names_gp
    column_names_gp = {"fontsize": 12} if column_names_gp is None else column_names_gp
    legend_title_gp = {"fontsize": 12} if legend_title_gp is None else legend_title_gp
    legend_labels_gp = {"fontsize": 12} if legend_labels_gp is None else legend_labels_gp
    show_top_legend = show_legend if show_top_legend is None else show_top_legend
    show_bott_legend = show_legend if show_bott_legend is None else show_bott_legend
    show_left_legend = show_legend if show_left_legend is None else show_left_legend
    show_right_legend = show_legend if show_right_legend is None else show_right_legend
    annotation_legend_param = dict(annotation_legend_param or {})

    row_dend_side = _first(row_dend_side)
    column_dend_side = _first(column_dend_side)

    if isinstance(mat, pd.DataFrame):
        row_labels = [str(v) for v in mat.index]
        col_labels = [str(v) for v in mat.columns]
        values = mat.to_numpy(dtype=float)
    else:
        values = np.asarray(mat, dtype=float)
        row_labels = [str(i) for i in range(values.shape[0])]
        col_labels = [str(i) for i in range(values.shape[1])]

    breaks = np.asarray(breaks, dtype=float)
    # HeatmapView.R lines 121-122
    values = np.where(values > breaks.max(), breaks.max(), values)
    values = np.where(values < breaks.min(), breaks.min(), values)
    col_pal = color_ramp2(breaks, list(colors))

    row_slices = _make_slices(values, 0, cluster_rows, row_split)
    col_slices = _make_slices(values, 1, cluster_cols, column_split)

    # -- annotations -------------------------------------------------------
    anns = {}
    for key, df, palettes, show in (
            ("top", top_ann, top_ann_col, show_top_legend),
            ("bott", bott_ann, bott_ann_col, show_bott_legend),
            ("left", left_ann, left_ann_col, show_left_legend),
            ("right", right_ann, right_ann_col, show_right_legend)):
        df = _as_frame(df)
        if df is None:
            continue
        if isinstance(df, BlockAnnotation):
            anns[key] = (df, None, show)
        else:
            anns[key] = (df, _annotation_colors(df, palettes), show)

    def _ntracks(key):
        entry = anns.get(key)
        if entry is None:
            return 0
        df = entry[0]
        return 1 if isinstance(df, BlockAnnotation) else df.shape[1]

    # -- geometry ----------------------------------------------------------
    n_rs, n_cs = len(row_slices), len(col_slices)
    ann_h = 0.16          # inches per column-annotation track
    ann_w = 0.16          # inches per row-annotation track
    dend_sz = 0.55
    gap = 0.06

    body_w = max(cell_size * values.shape[1], 1.2)
    body_h = max(cell_size * values.shape[0], 1.2)

    left_extra = (_ntracks("left") * ann_w
                  + (dend_sz if (cluster_rows is not False and row_dend_side == "left") else 0.0))
    right_extra = (_ntracks("right") * ann_w
                   + (dend_sz if (cluster_rows is not False and row_dend_side == "right") else 0.0))
    top_extra = (_ntracks("top") * ann_h
                 + (dend_sz if (cluster_cols is not False and column_dend_side == "top") else 0.0))
    bott_extra = (_ntracks("bott") * ann_h
                  + (dend_sz if (cluster_cols is not False and column_dend_side == "bottom") else 0.0))

    # Space for the row / column labels themselves.  A rotated label needs one
    # line's worth; an unrotated one needs room for its longest string.
    char_w = 0.085
    if show_row_names:
        longest = max((len(s) for s in row_labels), default=1)
        name_w = 0.3 if row_names_rot in (90, 270) else min(char_w * longest + 0.2, 2.5)
    else:
        name_w = 0.05
    if show_column_names:
        longest = max((len(s) for s in col_labels), default=1)
        name_h = 0.3 if column_names_rot in (0, 180) else min(char_w * longest + 0.2, 2.5)
    else:
        name_h = 0.05

    # show_ann_name puts the track names to the right of column annotations and
    # below row annotations, so those margins have to be reserved too.
    ann_name_w = 0.75 if (show_ann_name and (_ntracks("top") or _ntracks("bott"))) else 0.0
    ann_name_h = 0.65 if (show_ann_name and (_ntracks("left") or _ntracks("right"))) else 0.0
    legend_w = 1.7 if (show_heatmap_legend or show_legend) else 0.05

    fig_w = (0.35 + name_w + left_extra + body_w + right_extra + ann_name_w + legend_w)
    fig_h = (0.40 + name_h + top_extra + body_h + bott_extra + ann_name_h + 0.25)
    if figsize is not None:
        fig_w, fig_h = figsize
    if fig is None:
        fig = plt.figure(figsize=(fig_w, fig_h))

    # Column layout (fractions of the figure width).
    x0 = (0.25 + (name_w if row_names_side == "left" else 0.0)) / fig_w
    x1 = 1.0 - (0.15 + legend_w + ann_name_w
                + (name_w if row_names_side == "right" else 0.0)) / fig_w
    y0 = (0.25 + ann_name_h + (name_h if column_names_side == "bottom" else 0.0)) / fig_h
    y1 = 1.0 - (0.40 + (name_h if column_names_side == "top" else 0.0)) / fig_h

    fx_left = left_extra / fig_w
    fx_right = right_extra / fig_w
    fy_top = top_extra / fig_h
    fy_bott = bott_extra / fig_h

    bx0, bx1 = x0 + fx_left, x1 - fx_right
    by0, by1 = y0 + fy_bott, y1 - fy_top

    col_sizes = np.array([len(idx) for _, idx, _ in col_slices], dtype=float)
    row_sizes = np.array([len(idx) for _, idx, _ in row_slices], dtype=float)
    gx = (gap / fig_w) if n_cs > 1 else 0.0
    gy = (gap / fig_h) if n_rs > 1 else 0.0
    col_w = (bx1 - bx0 - gx * (n_cs - 1)) * col_sizes / col_sizes.sum()
    row_h = (by1 - by0 - gy * (n_rs - 1)) * row_sizes / row_sizes.sum()

    col_x = []
    cur = bx0
    for w in col_w:
        col_x.append(cur)
        cur += w + gx
    row_y = []
    cur = by1
    for h in row_h:
        cur -= h
        row_y.append(cur)
        cur -= gy

    # -- body --------------------------------------------------------------
    body_axes: list[list] = []
    for i, (rtitle, ridx, _) in enumerate(row_slices):
        row_axes = []
        for j, (ctitle, cidx, _) in enumerate(col_slices):
            ax = fig.add_axes([col_x[j], row_y[i], col_w[j], row_h[i]])
            sub = values[np.ix_(ridx, cidx)]
            rgb = col_pal(np.nan_to_num(sub, nan=breaks.min()))
            rgb[np.isnan(sub)] = to_rgb(_r_color(na_col))
            ax.imshow(rgb, aspect="auto", interpolation="nearest",
                      extent=(0, 1, 0, 1), origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():       # border = T
                spine.set_linewidth(0.8)
                spine.set_color("black")
            if rtitle is not None and j == n_cs - 1:
                ax.text(1.01, 0.5, str(rtitle), transform=ax.transAxes,
                        rotation=270, va="center", ha="left", fontsize=10)
            if ctitle is not None and i == 0:
                ax.text(0.5, 1.01, str(ctitle), transform=ax.transAxes,
                        va="bottom", ha="center", fontsize=10)
            row_axes.append(ax)
        body_axes.append(row_axes)

    # -- row / column names -------------------------------------------------
    if show_row_names:
        jj = 0 if row_names_side == "left" else n_cs - 1
        for i, (_, ridx, _rz) in enumerate(row_slices):
            ax = body_axes[i][jj]
            pos = (np.arange(len(ridx)) + 0.5) / len(ridx)
            ax.set_yticks(1.0 - pos)
            ax.set_yticklabels([row_labels[k] for k in ridx],
                               rotation=row_names_rot, **_gpar(row_names_gp))
            ax.yaxis.set_ticks_position(row_names_side)
            ax.yaxis.set_label_position(row_names_side)
            # Push the labels clear of the row-annotation strips, which are
            # added to the figure later and would otherwise paint over them.
            pad = (left_extra if row_names_side == "left" else right_extra) * 72.0
            ax.tick_params(axis="y", length=0, pad=pad + 2.0)
    if show_column_names:
        ii = n_rs - 1 if column_names_side == "bottom" else 0
        for j, (_, cidx, _cz) in enumerate(col_slices):
            ax = body_axes[ii][j]
            pos = (np.arange(len(cidx)) + 0.5) / len(cidx)
            ax.set_xticks(pos)
            ax.set_xticklabels([col_labels[k] for k in cidx],
                               rotation=column_names_rot,
                               ha="right" if column_names_rot not in (0, 90) else
                               ("center" if column_names_rot == 0 else "center"),
                               **_gpar(column_names_gp))
            ax.xaxis.set_ticks_position(column_names_side)
            ax.xaxis.set_label_position(column_names_side)
            pad = (bott_extra if column_names_side == "bottom" else top_extra) * 72.0
            ax.tick_params(axis="x", length=0, pad=pad + 2.0)

    # -- dendrograms --------------------------------------------------------
    if cluster_cols is not False:
        dh = dend_sz / fig_h
        for j, (_, cidx, cz) in enumerate(col_slices):
            # Sit outside the column-annotation tracks, not on top of them.
            yy = (by1 + _ntracks("top") * ann_h / fig_h if column_dend_side == "top"
                  else by0 - fy_bott)
            ax = fig.add_axes([col_x[j], yy, col_w[j], dh])
            _draw_dendrogram(ax, cz, len(cidx), column_dend_side)
    if cluster_rows is not False:
        dw = dend_sz / fig_w
        for i, (_, ridx, rz) in enumerate(row_slices):
            xx = (bx0 - fx_left if row_dend_side == "left"
                  else bx1 + fx_right - dw)
            ax = fig.add_axes([xx, row_y[i], dw, row_h[i]])
            _draw_dendrogram(ax, rz, len(ridx), row_dend_side)

    # -- annotation tracks ---------------------------------------------------
    legend_entries: list[tuple[str, Any]] = []

    def _draw_col_ann(key, side):
        entry = anns.get(key)
        if entry is None:
            return
        df, palettes, show = entry
        if isinstance(df, BlockAnnotation):
            return
        ntr = df.shape[1]
        h = ann_h / fig_h
        for t, colname in enumerate(df.columns):
            for j, (_, cidx, _cz) in enumerate(col_slices):
                if side == "top":
                    yy = by1 + (ntr - 1 - t) * h + (gy if n_rs > 1 else 0.0) * 0
                else:
                    yy = by0 - (t + 1) * h
                ax = fig.add_axes([col_x[j], yy, col_w[j], h * 0.92])
                rgb = _ann_rgb(df[colname].iloc[cidx], palettes.get(colname), na_col)
                ax.imshow(rgb[None, :, :], aspect="auto", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                if show_ann_name and j == n_cs - 1:
                    ax.text(1.01, 0.5, str(colname), transform=ax.transAxes,
                            va="center", ha="left", fontsize=9)
            if show:
                legend_entries.append((str(colname), palettes.get(colname)))

    def _draw_row_ann(key, side):
        entry = anns.get(key)
        if entry is None:
            return
        df, palettes, show = entry
        w = ann_w / fig_w
        if isinstance(df, BlockAnnotation):
            for i, (_, ridx, _rz) in enumerate(row_slices):
                ax = fig.add_axes([bx1 + 0.002, row_y[i], w, row_h[i]])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
                pos = {k: set(v.tolist()) for k, v in df.align_to.items()}
                for lab, (k, members) in zip(df.labels, pos.items()):
                    hits = [p for p, r in enumerate(ridx) if r in members]
                    if not hits:
                        continue
                    lo, hi = min(hits), max(hits)
                    ycen = 1.0 - (lo + hi + 1) / (2.0 * len(ridx))
                    ax.text(0.5, ycen, str(lab), transform=ax.transAxes,
                            va="center", ha="center", **_gpar(df.gp))
            return
        ntr = df.shape[1]
        for t, colname in enumerate(df.columns):
            for i, (_, ridx, _rz) in enumerate(row_slices):
                if side == "left":
                    xx = bx0 - (ntr - t) * w
                else:
                    xx = bx1 + t * w
                ax = fig.add_axes([xx, row_y[i], w * 0.92, row_h[i]])
                rgb = _ann_rgb(df[colname].iloc[ridx], palettes.get(colname), na_col)
                ax.imshow(rgb[:, None, :], aspect="auto", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                if show_ann_name and i == n_rs - 1:
                    ax.text(0.5, -0.01, str(colname), transform=ax.transAxes,
                            va="top", ha="center", rotation=90, fontsize=9)
            if show:
                legend_entries.append((str(colname), palettes.get(colname)))

    _draw_col_ann("top", "top")
    _draw_col_ann("bott", "bottom")
    _draw_row_ann("left", "left")
    _draw_row_ann("right", "right")

    # -- legends -------------------------------------------------------------
    if legend_side == "right":
        lx = 1.0 - (legend_w - 0.15) / fig_w
    elif legend_side == "left":
        lx = 0.01
    else:
        lx = 1.0 - (legend_w - 0.15) / fig_w
    ly = y1

    import matplotlib.colorbar as mcbar

    if show_heatmap_legend:
        h = min(legend_height / fig_h, 0.6)
        w = legend_width / fig_w
        if legend_direction == "horizontal":
            cax = fig.add_axes([lx, ly - w, h, w])
            orient = "horizontal"
            ly -= (w + 0.06)
        else:
            cax = fig.add_axes([lx, ly - h, w, h])
            orient = "vertical"
            ly -= (h + 0.06)
        norm = Normalize(vmin=float(breaks.min()), vmax=float(breaks.max()))
        cb = mcbar.ColorbarBase(cax, cmap=col_pal.to_cmap(), norm=norm,
                                orientation=orient)
        cb.ax.tick_params(labelsize=legend_labels_gp.get("fontsize", 12))
        title = legend_title if legend_title is not None else name
        if title:
            # legend_title_position = "lefttop": title above, left aligned.
            cax.set_title(str(title), loc="left", **_gpar(legend_title_gp))

    seen_titles = set()
    lab_fs = max(legend_labels_gp.get("fontsize", 12) - 2, 6)
    tit_fs = max(legend_title_gp.get("fontsize", 12) - 2, 6)
    for title, mapping in legend_entries:
        if title in seen_titles or mapping is None:
            continue
        seen_titles.add(title)
        if isinstance(mapping, ColorRamp2):
            # A numeric annotation column gets a continuous legend, as
            # ComplexHeatmap does for a colorRamp2 mapping.
            h = min(0.9 / fig_h, 0.28)
            w = 0.22 / fig_w
            cax = fig.add_axes([lx, max(ly - h, 0.02), w, h])
            norm = Normalize(vmin=float(mapping.breaks[0]),
                             vmax=float(mapping.breaks[-1]))
            cbn = mcbar.ColorbarBase(cax, cmap=mapping.to_cmap(), norm=norm,
                                     orientation="vertical")
            cbn.ax.tick_params(labelsize=lab_fs)
            cax.set_title(title, loc="left", fontsize=tit_fs)
            ly -= (h + 0.09)
            continue
        if not isinstance(mapping, dict):
            continue
        handles = [Patch(facecolor=_r_color(c), edgecolor="none", label=str(k))
                   for k, c in mapping.items()]
        leg_ax = fig.add_axes([lx, max(ly - 0.02, 0.02), 0.01, 0.01])
        leg_ax.set_axis_off()
        leg = leg_ax.legend(handles=handles, title=title, loc="upper left",
                            bbox_to_anchor=(0, 1), frameon=False,
                            fontsize=lab_fs, title_fontsize=tit_fs,
                            handlelength=1.0, handleheight=1.0, labelspacing=0.25)
        leg._legend_box.align = "left"
        ly -= (0.055 * (len(handles) + 1.4))

    fig.heatmap_view_info = {
        "name": name,
        "row_order": [idx.tolist() for _, idx, _ in row_slices],
        "column_order": [idx.tolist() for _, idx, _ in col_slices],
        "row_slice_titles": [t for t, _, _ in row_slices],
        "column_slice_titles": [t for t, _, _ in col_slices],
        "body_axes": body_axes,
        "col_pal": col_pal,
        "row_labels": row_labels,
        "column_labels": col_labels,
    }
    return fig


# ===========================================================================
# drawRectangleAnnotation
# ===========================================================================

def draw_rectangle_annotation(ht, rows, columns, col="black",
                              heatmap_name="hmap", include_na=False):
    """Outline matching row/column annotation blocks -- R
    ``drawRectangleAnnotation`` (``HeatmapView.R`` line 280).

    Parameters
    ----------
    ht
        The :class:`~matplotlib.figure.Figure` returned by :func:`heatmap_view`
        (or its ``heatmap_view_info`` dict directly).  This stands in for the
        drawn ``HeatmapList`` that the R version decorates.
    rows, columns
        Annotation labels, one per matrix row / column *in the original order*
        -- exactly as in R, where they are indexed by ``row_order(ht)``.
    col
        Rectangle border colour.
    heatmap_name
        Checked against the heatmap's ``name``, mirroring
        ``decorate_heatmap_body(heatmap_name, ...)``.
    include_na
        Treat ``NA`` as its own group instead of skipping it (lines 289-292).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, with the rectangles added.
    """
    info = ht.heatmap_view_info if hasattr(ht, "heatmap_view_info") else ht
    fig = ht if isinstance(ht, Figure) else None
    if info.get("name") not in (None, heatmap_name):
        raise ValueError("no heatmap named %r in this figure" % heatmap_name)

    if len(rows) == 0 or len(columns) == 0:
        raise ValueError("'rows' and 'columns' must both be non-empty")

    rows = ["NA" if pd.isna(v) else str(v) for v in rows] if include_na else \
        [None if pd.isna(v) else str(v) for v in rows]
    columns = ["NA" if pd.isna(v) else str(v) for v in columns] if include_na else \
        [None if pd.isna(v) else str(v) for v in columns]

    def get_blocks(vec):
        """Run-length encode into contiguous blocks (lines 295-302)."""
        if len(vec) == 0:
            return []
        blocks = []
        start = 0
        for i in range(1, len(vec) + 1):
            if i == len(vec) or vec[i] != vec[start]:
                blocks.append((vec[start], start, i - 1))   # 0-based inclusive
                start = i
        return blocks

    def same_label(a, b):
        if include_na:
            return a == b
        return a is not None and b is not None and a == b

    ro_list = info["row_order"]
    co_list = info["column_order"]
    body_axes = info["body_axes"]

    for si, r_ind in enumerate(ro_list):
        for sj, c_ind in enumerate(co_list):
            if len(r_ind) == 0 or len(c_ind) == 0:
                continue
            rows_slice = [rows[k] for k in r_ind]
            cols_slice = [columns[k] for k in c_ind]
            row_blocks = get_blocks(rows_slice)
            col_blocks = get_blocks(cols_slice)
            n_row, n_col = len(rows_slice), len(cols_slice)
            ax = body_axes[si][sj]
            for rlab, rs, re_ in row_blocks:
                ymin = 1.0 - (re_ + 1) / n_row
                ymax = 1.0 - rs / n_row
                for clab, cs, ce in col_blocks:
                    if not same_label(rlab, clab):
                        continue
                    xmin = cs / n_col
                    xmax = (ce + 1) / n_col
                    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           transform=ax.transAxes,
                                           facecolor="none",
                                           edgecolor=_r_color(col),
                                           linewidth=2, linestyle="-",
                                           zorder=5, clip_on=False))
    return fig if fig is not None else info


# ===========================================================================
# CooccurrenceHeatmapView
# ===========================================================================

def cooccurrence_heatmap_view(coloc_index, pval=None,
                              breaks=None,
                              colors=("#f2f0f7", "#cbc9e2", "#54278f"),
                              legend_height=1,
                              cluster_rows=False, cluster_cols=False,
                              show_left_legend=False, show_column_names=False,
                              show_row_names=False, **kwargs):
    """Colocalization heatmap -- R ``CooccurrenceHeatmapView``
    (``Colocalization.R`` line 229).

    Parameters
    ----------
    coloc_index
        Square matrix of colocalization indices; row names must read
        ``"<SE>_<CellType>"``.
    pval
        Optional mapping (or Series) of SE -> p-value.  When given, a right
        annotation carries the significance stars.
    breaks
        Defaults to the 55 / 75 / 90 % quantiles of ``coloc_index``, matching
        the R default argument (line 230).  R's ``quantile`` type 7 is
        NumPy's default, so the numbers agree.
    colors, legend_height, cluster_rows, cluster_cols, show_left_legend,
    show_column_names, show_row_names
        As in R.
    **kwargs
        Forwarded to :func:`heatmap_view`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    gg = coloc_index if isinstance(coloc_index, pd.DataFrame) else pd.DataFrame(coloc_index)
    names = [str(v) for v in gg.index]

    if breaks is None:
        breaks = np.nanquantile(gg.to_numpy(dtype=float), [0.55, 0.75, 0.9])

    # Colocalization.R lines 238-240: gsub("_.*", "") / gsub(".*_", "")
    se = [re.sub(r"_.*", "", v) for v in names]
    celltype = [re.sub(r".*_", "", v) for v in names]
    rowann = pd.DataFrame({"SE": se, "CellType": celltype}, index=names)

    se_levels = _unique_in_order(se)
    ct_levels = _unique_in_order(celltype)
    se_colors = dict(zip(se_levels, get_colors(len(se_levels), palette=1)))
    celltype_cols = dict(zip(ct_levels, get_colors(len(ct_levels), palette=2)))

    right_ann = None
    if pval is not None:
        pv = pd.Series(pval)
        rowann = rowann.copy()
        rowann["Pval"] = [float(pv.get(s, np.nan)) for s in se]
        signif = []
        for p in rowann["Pval"]:
            s = "ns"
            if p < 0.05:
                s = "*"
            if p < 0.01:
                s = "**"
            if p < 0.001:
                s = "***"
            if p < 0.0001:
                s = "****"
            signif.append(s)
        rowann["Signif"] = signif
        # lines 253-255: one label per SE, aligned to that SE's rows.
        align_to = {s: [i for i, v in enumerate(se) if v == s] for s in se_levels}
        labels = [rowann["Signif"].iloc[align_to[s][0]] for s in se_levels]
        right_ann = BlockAnnotation(align_to, labels, name="SE")

    ann = rowann[["SE", "CellType"]]        # rowann[, -c(3, 4)]
    ann_col = {"SE": se_colors, "CellType": celltype_cols}

    return heatmap_view(gg, breaks=breaks, colors=colors,
                        legend_height=legend_height,
                        cluster_rows=cluster_rows,
                        cluster_cols=cluster_cols,
                        show_left_legend=show_left_legend,
                        show_column_names=show_column_names,
                        show_row_names=show_row_names,
                        top_ann=ann,
                        left_ann=ann,
                        top_ann_col=ann_col,
                        left_ann_col=ann_col,
                        right_ann=right_ann,
                        **kwargs)
