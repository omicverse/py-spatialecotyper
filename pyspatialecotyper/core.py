"""The single-sample entry point: ``SpatialEcoTyper`` and ``AnnotateCells``.

Pipeline (``R/SpatialEcoTyper.R``)::

    PreprocessST
      -> spatial-neighbourhood grid (SpotID)
      -> GetSpatialMetacells      (per cell type)
      -> GetPCList                (per cell type)
      -> GetSNList                (per cell type KNN similarity graph)
      -> SNF2                     (cross-diffusion fusion)
      -> rankSparse
      -> Seurat: ScaleData -> RunPCA -> FindNeighbors -> FindClusters
      -> AnnotateCells            (neighbourhood label -> single cell)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .metacells import _nn2, get_spatial_metacells
from .network import get_pc_list, get_sn_list, snf2
from .preprocessing import preprocess_st
from .seuratcompat import run_pca, scale_data
from .utils import most_frequent, rank_sparse

__all__ = ["spatial_ecotyper", "annotate_cells", "SpatialEcoTyper",
           "SpatialEcoTyperResult"]


@dataclass
class SpatialEcoTyperResult:
    """What R returns as ``list(obj = <Seurat>, metadata = <data.frame>)``.

    ``obj`` has no Python equivalent, so its contents are exposed directly:
    the fused rank-transformed graph, the neighbourhood-level metadata with an
    ``SE`` column, the PCA embedding, and the SNN graph.
    """
    fused: sp.spmatrix
    spot_names: list
    spot_metadata: pd.DataFrame
    metadata: pd.DataFrame
    pca: np.ndarray = field(default=None, repr=False)
    snn: sp.spmatrix = field(default=None, repr=False)
    clusters: np.ndarray = field(default=None, repr=False)
    project_name: str = ""

    @property
    def obs(self) -> pd.DataFrame:
        return self.spot_metadata


def _spot_metadata(metadata: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """Collapse cell metadata onto the spatial-neighbourhood grid.

    R (``R/SpatialEcoTyper.R`` lines 107-119)::

        SpotID <- paste0("X", round(X / grid.size), "_Y", round(Y / grid.size))
        group_by(SpotID) |> mutate(X = median(X), Y = median(Y))
        # numeric columns -> median ; character/factor/logical -> mostFrequent
        merge(medians, freqs, by = c("SpotID", "X", "Y"))

    Two R behaviours are load-bearing:

    * ``round()`` is banker's rounding (half to even) -- ``np.round`` agrees.
    * ``median()`` of an even-length vector is the mean of the two middle
      values -- ``np.median`` agrees.
    * ``merge()`` sorts by the ``by`` columns, so the resulting row order is
      lexicographic in ``(SpotID, X, Y)``.  Downstream code always indexes by
      name, but the order determines the metacell column order, so it matters.
    """
    m = metadata.copy()
    spot = ["X%d_Y%d" % (int(np.round(a / grid_size)), int(np.round(b / grid_size)))
            for a, b in zip(m["X"].to_numpy(float), m["Y"].to_numpy(float))]
    m["SpotID"] = spot
    grp = m.groupby("SpotID", sort=False)
    m["X"] = grp["X"].transform("median")
    m["Y"] = grp["Y"].transform("median")
    if "CID" in m.columns:
        m = m.drop(columns=["CID"])

    num_cols = [c for c in m.columns
                if c not in ("SpotID",) and pd.api.types.is_numeric_dtype(m[c])
                and not pd.api.types.is_bool_dtype(m[c])]
    cat_cols = [c for c in m.columns
                if c not in ("SpotID", "X", "Y") and c not in num_cols]

    keys = ["SpotID", "X", "Y"]
    med = m.groupby(keys, sort=True)[[c for c in num_cols if c not in ("X", "Y")]].median()
    if cat_cols:
        frq = m.groupby(keys, sort=True)[cat_cols].agg(most_frequent)
        out = med.join(frq)
    else:
        out = med
    out = out.reset_index()
    # merge(..., by = c("SpotID","X","Y")) sorts on the character SpotID first
    out = out.sort_values(by=keys, kind="stable", key=lambda s: s.astype(str)
                          if s.name == "SpotID" else s)
    out.index = out["SpotID"].to_numpy()
    return out


def annotate_cells(scmeta: pd.DataFrame, spot_metadata: pd.DataFrame,
                   radius: float, col: str = "SE", dropcell: bool = True):
    """``AnnotateCells(scmeta, obj, col, dropcell)`` (``R/AnnotateCells.R``).

    Assigns each single cell the label of its **single** nearest spatial
    neighbourhood, then blanks the assignment when that neighbourhood is
    further away than ``radius``.  R reads ``radius`` out of the Seurat
    object's ``project.name`` string; here it is an explicit argument.
    """
    ref = np.asarray(spot_metadata[["X", "Y"]], dtype=np.float64)
    qry = np.asarray(scmeta[["X", "Y"]], dtype=np.float64)
    idx, dists = _nn2(ref, qry, 1)
    labels = np.asarray(spot_metadata[col], dtype=object)[idx[:, 0]]
    labels = np.where(dists[:, 0] > radius, None, labels)
    out = scmeta.copy()
    out[col] = labels
    if dropcell:
        out = out[~pd.isna(out[col])]
    return out


def spatial_ecotyper(normdata, metadata: pd.DataFrame,
                     gene_names: list[str] | None = None,
                     outprefix: str | None = None,
                     radius: float = 50,
                     resolution: float = 0.5,
                     nfeatures: int = 300,
                     min_cts_per_region: int = 2,
                     npcs: int = 20,
                     min_cells: int = 5,
                     min_features: int = 10,
                     iterations: int = 10,
                     minibatch: int = 5000,
                     ncores: int = 4,
                     grid_size: float | None = None,
                     filter_region_by_celltypes=None,
                     k: int = 20,
                     k_sn: int = 50,
                     dropcell: bool = True,
                     verbose: bool = True) -> SpatialEcoTyperResult:
    """``SpatialEcoTyper(normdata, metadata, ...)`` -- discover spatial ecotypes
    in a single single-cell spatial-transcriptomics sample.

    Parameters mirror R one-for-one (``grid.size`` defaults to
    ``round(radius * 1.4)``; ``npcs`` is the number of PCs carried into the
    similarity networks, not the number computed).

    ``normdata`` is genes x cells; ``metadata`` needs ``X``, ``Y`` and
    ``CellType`` columns and an index matching the columns of ``normdata``.
    """
    from ._modularity import find_clusters, find_neighbors

    if normdata.shape[1] != metadata.shape[0]:
        raise ValueError("The number of columns in normdata does not match the "
                         "number of rows in metadata.")
    if grid_size is None:
        grid_size = round(radius * 1.4)

    normdata, metadata, gene_names, cell_names = preprocess_st(
        normdata, metadata, min_cells=min_cells, min_features=min_features,
        genes=gene_names, verbose=verbose)

    for req in ("CellType", "X", "Y"):
        if req not in metadata.columns:
            raise ValueError(f"Metadata must include a '{req}' column.")

    ncmeta = _spot_metadata(metadata, grid_size)

    # if(max(normdata) > 50) normdata <- log1p(normdata)
    mx = normdata.data.max() if sp.issparse(normdata) else np.max(normdata)
    if mx > 50:
        if sp.issparse(normdata):
            normdata = normdata.copy()
            normdata.data = np.log1p(normdata.data)
        else:
            normdata = np.log1p(normdata)

    if verbose:
        print("Construct spatial meta cells for each cell type")
    ncem, ncem_cols = get_spatial_metacells(
        normdata, metadata, spotCoord=ncmeta, k=k, radius=radius,
        ncores=ncores, gene_names=gene_names, verbose=verbose)

    if filter_region_by_celltypes:
        cts = list(filter_region_by_celltypes)
        per_spot: dict[str, set] = {}
        for name in ncem_cols:
            spot, ct = name.rsplit("..", 1)
            if ct in cts:
                per_spot.setdefault(spot, set()).add(ct)
        keep_spots = {s for s, v in per_spot.items() if len(v) == len(cts)}
        sel = np.array([c.rsplit("..", 1)[0] in keep_spots for c in ncem_cols])
        if sel.sum() < 20:
            raise ValueError("Fewer than 20 spatial neighborhoods contain "
                             + ", ".join(cts))
        ncem = ncem[:, sel]
        ncem_cols = [c for c, s in zip(ncem_cols, sel) if s]

    spots = pd.unique(np.asarray([c.rsplit("..", 1)[0] for c in ncem_cols], dtype=object))
    if verbose:
        print(f"Total spatial neighborhoods: {len(spots)}; "
              f"total spatial meta cells: {len(ncem_cols)}")

    if verbose:
        print("PCA for each cell type")
    emb_list = get_pc_list(ncem, ncem_cols, gene_names, nfeatures=nfeatures,
                           min_cells=min_cells, min_features=min_features,
                           ncores=ncores, verbose=verbose)

    if verbose:
        print("Construct cell-type-specific similarity network")
    snlist, sn_spots = get_sn_list(emb_list, npcs=npcs,
                                   min_cts_per_region=min_cts_per_region,
                                   k=k_sn, ncores=ncores, verbose=verbose)

    if verbose:
        print("Similarity network fusion")
    fused = snf2([w for w, _ in snlist.values()], t=iterations,
                 minibatch=minibatch, ncores=ncores, verbose=verbose)
    fused = rank_sparse(fused)

    # rownames(obj) <- gsub("_", "-", rownames(obj)); colnames stay as-is.
    spot_meta = ncmeta.loc[list(sn_spots)]

    if verbose:
        print("Create Seurat object and perform clustering analysis")
    scaled = scale_data(fused, do_scale=False, do_center=True)
    n_pc = min(50, scaled.shape[0] - 1)
    emb, _, _ = run_pca(scaled, npcs=n_pc)
    _, snn = find_neighbors(emb[:, :10], k_param=20)
    clusters = find_clusters(snn, resolution=resolution, random_seed=0)

    se = np.array([f"NewSE{c + 1}" for c in clusters], dtype=object)
    spot_meta = spot_meta.copy()
    spot_meta["SE"] = se

    project_name = (f"{outprefix or 'Denovo'}_radius{radius}_nfeatures{nfeatures}"
                    f"_npcs{npcs}_k.sn{k_sn}_k{k}_min.cells{min_cells}"
                    f"_min.features{min_features}"
                    f"_min.cts.per.region{min_cts_per_region}"
                    f"_iterations{iterations}_grid.size{grid_size}")

    cellmeta = annotate_cells(metadata, spot_meta, radius=radius,
                              col="SE", dropcell=dropcell)

    return SpatialEcoTyperResult(fused=fused, spot_names=list(sn_spots),
                                 spot_metadata=spot_meta, metadata=cellmeta,
                                 pca=emb, snn=snn, clusters=clusters,
                                 project_name=project_name)


class SpatialEcoTyper:
    """Class API over :func:`spatial_ecotyper`, for method chaining.

    ``SpatialEcoTyper(normdata, metadata).run()`` is equivalent to calling the
    function; the object keeps the staged intermediates so a user can inspect
    the metacells or the per-cell-type networks without re-running.
    """

    def __init__(self, normdata, metadata, gene_names=None, **kwargs):
        self.normdata = normdata
        self.metadata = pd.DataFrame(metadata)
        self.gene_names = gene_names
        self.kwargs = kwargs
        self.result: SpatialEcoTyperResult | None = None

    def run(self, **kwargs) -> "SpatialEcoTyper":
        opts = dict(self.kwargs)
        opts.update(kwargs)
        self.result = spatial_ecotyper(self.normdata, self.metadata,
                                       gene_names=self.gene_names, **opts)
        return self

    @property
    def se(self) -> pd.Series:
        if self.result is None:
            raise RuntimeError("call .run() first")
        return self.result.metadata["SE"]

    def __repr__(self):
        if self.result is None:
            return (f"SpatialEcoTyper(<{self.normdata.shape[0]} genes x "
                    f"{self.normdata.shape[1]} cells>, not run)")
        r = self.result
        return (f"SpatialEcoTyper({len(r.spot_names)} spatial neighborhoods, "
                f"{len(set(r.spot_metadata['SE']))} SEs)")
