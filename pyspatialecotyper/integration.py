"""Multi-sample integration: ``ComputeFCs``, ``Integrate``,
``IntegrateSpatialEcoTyper`` and ``MultiSpatialEcoTyper``.

Stage two of the method.  Each sample's spatial neighbourhoods are
over-clustered into fine-grained "InitSE" spatial clusters, every cluster is
represented by the cell-type-specific expression signature of its cells, and
those signatures are fused across samples (rank-correlation networks + SNF)
before a consensus NMF picks out the *conserved* spatial ecotypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import annotate_cells
from .network import fillspots, snf2
from .nmf import nmf_clustering
from .preprocessing import znorm
from .rrandom import RRandom
from .seuratcompat import scale_data
from .utils import most_frequent, r_order, r_rank, rank_sparse

__all__ = ["compute_fcs", "integrate", "integrate_spatial_ecotyper",
           "multi_spatial_ecotyper"]


def compute_fcs(normdata, scmeta: pd.DataFrame, gene_names,
                cluster: str = "SE", Region: str | None = None,
                scale: bool = False, ncores: int = 4):
    """``ComputeFCs(normdata, scmeta, cluster, Region, scale, ncores)``.

    Per cell type, the log-fold-change of every gene in each spatial cluster
    versus **all other cells of that cell type** (not versus all cells).  Row
    names become ``"<CellType>..<gene>"``, which is the key the integration
    step splits on.

    Cell types with fewer than 10 cells, or with fewer than 2 spatial clusters,
    are dropped (``R/ComputeFCs.R`` lines 12 and 22).
    """
    scmeta = pd.DataFrame(scmeta).copy()
    if "CellType" not in scmeta.columns:
        raise ValueError("Metadata must include a column named 'CellType' for "
                         "cell type annotations.")
    scmeta["SE"] = scmeta[cluster].to_numpy()
    keep = ~pd.isna(scmeta["SE"])
    scmeta = scmeta.loc[keep]
    normdata = normdata[:, np.flatnonzero(keep.to_numpy())]

    all_ses = list(pd.unique(scmeta["SE"].to_numpy()))
    blocks, rows = [], []
    for ct in pd.unique(scmeta["CellType"].to_numpy()):
        sel = (scmeta["CellType"] == ct).to_numpy()
        if sel.sum() < 10:
            continue
        tmpmeta = scmeta.loc[sel]
        tmpdata = normdata[:, np.flatnonzero(sel)]
        tmpdata = np.asarray(tmpdata.todense()) if sp.issparse(tmpdata) \
            else np.asarray(tmpdata, dtype=np.float64)
        if scale:
            if Region is not None and Region in scmeta.columns:
                tmpdata = znorm(tmpdata, groups=tmpmeta[Region].to_numpy())
            else:
                tmpdata = scale_data(tmpdata)
        ses = list(pd.unique(tmpmeta["SE"].to_numpy()))
        if len(ses) < 2:
            continue
        lfc = np.zeros((tmpdata.shape[0], len(all_ses)))
        lfc[:] = np.nan
        for s in ses:
            m = (tmpmeta["SE"] == s).to_numpy()
            lfc[:, all_ses.index(s)] = tmpdata[:, m].mean(axis=1) - tmpdata[:, ~m].mean(axis=1)
        lfc = np.nan_to_num(lfc, nan=0.0)
        blocks.append(lfc)
        rows.extend([f"{ct}..{g}" for g in gene_names])
    if not blocks:
        raise ValueError("No cell type yielded fold changes.")
    return np.vstack(blocks), rows, all_ses


def integrate(avgexprs, row_names, col_names, Region=None,
              downsample_by_region: bool = True, nfeatures: int = 200,
              min_features: int = 5, minibatch: int = 5000,
              ncores: int = 1, seed: int = 1, verbose: bool = False):
    """``Integrate(avgexprs, Region, downsample.by.region, nfeatures, ...)``.

    Builds one *rank-normalised correlation* network per cell type across all
    samples' spatial clusters, then fuses them with :func:`snf2`.

    The rank normalisation (R lines 60-66) is the detail that makes the
    networks comparable across samples: within each source sample, every column
    of the correlation sub-matrix is converted to ranks and divided by the
    column sum, so a cluster's similarity profile becomes a probability
    distribution over that sample's clusters rather than a raw correlation.

    Variable-feature selection ranks genes by ``mean(log(rank of -variance))``
    computed **per sample**, which favours genes that are variable in *every*
    sample rather than very variable in one.
    """
    avg = np.asarray(avgexprs, dtype=np.float64)
    col_names = list(col_names)
    row_names = list(row_names)

    if Region is not None:
        Region = np.asarray(Region, dtype=object)
        if downsample_by_region:
            rng = RRandom(seed)
            samples = np.array([c.split("..")[0] for c in col_names], dtype=object)
            ids = []
            for s in pd.unique(samples):
                m = samples == s
                sub_regions = Region[m]
                sizes = pd.Series(sub_regions).value_counts()
                n = int(sizes.min())
                sub_ids = np.asarray(col_names, dtype=object)[m]
                # dplyr::slice_sample(n = n) within each region group,
                # groups visited in sorted level order.
                for g in sorted(pd.unique(sub_regions)):
                    gi = np.flatnonzero(sub_regions == g)
                    pick = rng.sample_int(len(gi), n)
                    ids.extend(sub_ids[gi[pick]])
            pos = {c: i for i, c in enumerate(col_names)}
            idx = [pos[c] for c in ids]
            avg = avg[:, idx]
            Region = Region[idx]
            col_names = list(ids)

    celltypes = np.array([r.split("..")[0] for r in row_names], dtype=object)
    nets = {}
    for ct in pd.unique(celltypes):
        m = celltypes == ct
        tmp = avg[m]
        cols = np.asarray(col_names, dtype=object)
        if tmp.shape[0] <= min_features:
            continue
        keep = (~np.isnan(tmp)).sum(axis=0) > min_features
        tmp, cols = tmp[:, keep], cols[keep]
        keep = np.nansum(tmp != 0, axis=0) > min_features
        tmp, cols = tmp[:, keep], cols[keep]

        samples = np.array([c.split("..")[0] for c in cols], dtype=object)
        vc = pd.Series(samples).value_counts()
        if len(vc) < 2 or vc.min() < 3:
            continue
        variances = np.column_stack([
            np.nanvar(tmp[:, samples == s], axis=1, ddof=1)
            for s in pd.unique(samples)])
        pos_var = variances.min(axis=1) > 0
        variances = variances[pos_var]
        sub = tmp[pos_var]
        var_ranks = np.column_stack([r_rank(-variances[:, j])
                                     for j in range(variances.shape[1])])
        order = r_order(np.log(var_ranks).mean(axis=1))
        order = order[:min(len(order), nfeatures)]
        if len(order) < min_features:
            continue
        sub = sub[order]

        with np.errstate(invalid="ignore"):
            cor = np.corrcoef(sub, rowvar=False)
        cor = np.nan_to_num(cor, nan=0.0)

        if Region is not None and not downsample_by_region:
            reg = np.asarray(Region, dtype=object)
            for g in pd.unique(reg):
                gi = np.flatnonzero(reg == g)
                cor[np.ix_(gi, gi)] *= 1.1

        blocks, order_rows = [], []
        for s in pd.unique(samples):
            gi = np.flatnonzero(samples == s)
            sscor = np.column_stack([r_rank(cor[gi, j]) for j in range(cor.shape[1])])
            sscor = sscor / sscor.sum(axis=0)[None, :]
            blocks.append(sscor)
            order_rows.append(gi)
        cor = np.vstack(blocks)
        rows_order = np.concatenate(order_rows)
        cor = cor[:, rows_order]                 # tmpcor[, rownames(tmpcor)]
        cor = cor + cor.T
        nets[ct] = (sp.csr_matrix(cor), [cols[i] for i in rows_order])

    if len(nets) < 2:
        raise ValueError(">=2 cell-type networks are required for integration.")
    filled, spots = fillspots(nets)
    obj = snf2([w for w, _ in filled.values()], ncores=ncores,
               minibatch=minibatch, verbose=verbose)
    obj = rank_sparse(obj)
    obj = (obj + obj.T) * 0.5
    return obj, spots


def integrate_spatial_ecotyper(result_list: dict, data_list: dict,
                               gene_names: dict | list,
                               outdir: str | None = None,
                               normalization_method: str = "None",
                               nmf_ranks=10, nrun_per_rank: int = 30,
                               min_coph: float = 0.95, nfeatures: int = 300,
                               min_features: int = 10, Region: str | None = None,
                               downsample_by_region: bool = True,
                               subresolution: float = 30,
                               minibatch: int = 5000, ncores: int = 4,
                               seed: int = 1, verbose: bool = True):
    """``IntegrateSpatialEcoTyper(SpatialEcoTyper_list, data_list, ...)``.

    Parameters
    ----------
    result_list
        ``{sample: SpatialEcoTyperResult}`` from :func:`spatial_ecotyper`.
    data_list
        ``{sample: genes x cells matrix}`` -- the *same* matrices fed to the
        single-sample runs.
    gene_names
        ``{sample: [gene, ...]}`` or one shared list.

    Returns a dict with the integrated similarity matrix, the NMF result, the
    per-spatial-cluster conserved SE assignment and the updated single-cell
    metadata -- the Python equivalent of the files R writes into ``outdir``.

    Only ``normalization.method = "None"`` is supported; the R "SCT" branch
    would require a port of ``SCTransform``, which is out of scope and raises.
    """
    from ._modularity import find_clusters

    if normalization_method != "None":
        raise NotImplementedError(
            "only normalization.method='None' is ported; normalise the "
            "matrices yourself with pyspatialecotyper.seuratcompat.normalize_data")
    samples = list(result_list)
    if isinstance(gene_names, (list, tuple, np.ndarray)):
        gene_names = {s: list(gene_names) for s in samples}

    # --- per-sample over-clustering into InitSE -------------------------
    metadata_list = {}
    for s in samples:
        res = result_list[s]
        cl = find_clusters(res.snn, resolution=subresolution, random_seed=0)
        spot_meta = res.spot_metadata.copy()
        spot_meta["SE"] = [f"InitSE{c}" for c in cl]
        md = annotate_cells(res.metadata.drop(columns=["SE"], errors="ignore"),
                            spot_meta, radius=50, col="SE", dropcell=True)
        md = md.rename(columns={"SE": "InitSE"})
        md["CID"] = md.index
        md["Sample"] = s
        md["InitSE"] = [f"{s}..{v}" for v in md["InitSE"]]
        metadata_list[s] = md

    # --- restrict expression matrices to the annotated, detected cells ---
    data_sub, meta_sub = {}, {}
    for s in samples:
        cols = list(result_list[s].metadata.index)
        keep_names = set(metadata_list[s].index)
        mat = data_list[s]
        idx = [i for i, c in enumerate(cols) if c in keep_names]
        sub = mat[:, idx]
        nz = np.asarray((sub > 0).sum(axis=0)).ravel()
        keep2 = nz >= min_features
        data_sub[s] = sub[:, np.flatnonzero(keep2)]
        names = [cols[i] for i, k in zip(idx, keep2) if k]
        meta_sub[s] = metadata_list[s].loc[names]

    metadatas = pd.concat([meta_sub[s] for s in samples])

    if verbose:
        print("Construct cell-type-specific gene expression signatures of "
              "spatial clusters")
    avg_blocks, avg_rows, avg_cols = [], None, []
    per_sample = {}
    for s in samples:
        fc, rows, ses = compute_fcs(data_sub[s], meta_sub[s], gene_names[s],
                                    cluster="InitSE", Region=Region,
                                    scale=True, ncores=ncores)
        per_sample[s] = (fc, rows, ses)
    common = None
    for s in samples:
        rs = set(per_sample[s][1])
        common = rs if common is None else (common & rs)
    common = [r for r in per_sample[samples[0]][1] if r in common]
    for s in samples:
        fc, rows, ses = per_sample[s]
        pos = {r: i for i, r in enumerate(rows)}
        avg_blocks.append(fc[[pos[r] for r in common]])
        avg_cols.extend(ses)
    avgexprs = np.hstack(avg_blocks)
    avg_rows = common

    clustmetas = (metadatas.drop(columns=["CID"])
                  .groupby("InitSE", sort=True)
                  .agg({c: most_frequent for c in metadatas.columns
                        if c not in ("CID", "InitSE")
                        and not pd.api.types.is_numeric_dtype(metadatas[c])}))
    clustmetas = clustmetas.reindex(avg_cols)

    if verbose:
        print("Start integrating SEs across samples")
    region_vec = clustmetas[Region].to_numpy() if Region else None
    integrated, spots = integrate(avgexprs, avg_rows, avg_cols,
                                  Region=region_vec,
                                  downsample_by_region=downsample_by_region,
                                  nfeatures=nfeatures, min_features=min_features,
                                  minibatch=minibatch, ncores=ncores, seed=seed,
                                  verbose=verbose)

    if verbose:
        print("Identify conserved SEs via NMF clustering")
    dense = np.asarray(integrated.todense()) if sp.issparse(integrated) else integrated
    nmf_res = nmf_clustering(dense, row_names=spots, col_names=spots,
                             ranks=nmf_ranks, nrun_per_rank=nrun_per_rank,
                             min_coph=min_coph, ncores=ncores, seed=seed,
                             verbose=verbose)

    ses = pd.Series(nmf_res["labels"], index=spots)
    ann = pd.DataFrame({"Sample": [s.split("..")[0] for s in spots],
                        "SE": [f"NewSE{int(v)}" for v in ses]}, index=spots)
    metadatas = metadatas.copy()
    metadatas["SE"] = ann["SE"].reindex(metadatas["InitSE"].to_numpy()).to_numpy()

    out = {"integrated": integrated, "spots": spots, "nmf": nmf_res,
           "cluster_SE": ann, "metadata": metadatas,
           "avgexprs": avgexprs, "avg_rows": avg_rows, "avg_cols": avg_cols}
    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        metadatas.to_csv(os.path.join(outdir, "MultiSE_metadata_final.tsv"),
                         sep="\t", index=False)
    return out


def multi_spatial_ecotyper(data_list: dict, metadata_list: dict,
                           gene_names, outdir: str | None = None,
                           normalization_method: str = "None",
                           nmf_ranks=10, nrun_per_rank: int = 30,
                           min_coph: float = 0.95, radius: float = 50,
                           min_cts_per_region: int = 1, nfeatures: int = 300,
                           min_features: int = 10, Region: str | None = None,
                           downsample_by_region: bool = True,
                           subresolution: float = 30, minibatch: int = 5000,
                           ncores: int = 1, seed: int = 1,
                           filter_region_by_celltypes=None, verbose: bool = True,
                           **kwargs):
    """``MultiSpatialEcoTyper(data_list, metadata_list, ...)`` -- end-to-end
    multi-sample pipeline: :func:`spatial_ecotyper` per sample, then
    :func:`integrate_spatial_ecotyper`.
    """
    from .core import spatial_ecotyper

    samples = list(data_list)
    if isinstance(gene_names, (list, tuple, np.ndarray)):
        gene_names = {s: list(gene_names) for s in samples}
    results = {}
    for s in samples:
        if verbose:
            print(f"SpatialEcoTyper analysis for {s}")
        results[s] = spatial_ecotyper(
            data_list[s], metadata_list[s], gene_names=gene_names[s],
            radius=radius, nfeatures=nfeatures, min_features=min_features,
            min_cts_per_region=min_cts_per_region, minibatch=minibatch,
            ncores=ncores, filter_region_by_celltypes=filter_region_by_celltypes,
            verbose=verbose, **kwargs)
    return integrate_spatial_ecotyper(
        results, data_list, gene_names, outdir=outdir,
        normalization_method="None", nmf_ranks=nmf_ranks,
        nrun_per_rank=nrun_per_rank, min_coph=min_coph, nfeatures=nfeatures,
        min_features=min_features, Region=Region,
        downsample_by_region=downsample_by_region, subresolution=subresolution,
        minibatch=minibatch, ncores=ncores, seed=seed, verbose=verbose)
