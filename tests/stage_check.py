"""Stage-by-stage diff of the Python port against the R reference dump.

Not a pytest module -- a development harness.  Run it as::

    python tests/stage_check.py reference_out/ci

Each stage feeds the Python function the *R* input for that stage, so a
failure localises to one function instead of propagating.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The rebuildr kit is a development-time dependency living outside this repo.
# Point at it with OMICVERSE_REBUILDR when running the staged parity check; the
# hardcoded path this replaced only existed on one machine.
_kit = os.environ.get("OMICVERSE_REBUILDR")
if _kit:
    sys.path.insert(0, _kit)

import rio
from pyspatialecotyper import metacells, network, preprocessing, seuratcompat, utils, core

RESULTS = []


def report(stage, metric, value, threshold, lower_is_better=True):
    ok = (value < threshold) if lower_is_better else (value >= threshold)
    RESULTS.append((stage, metric, value, threshold, ok))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {stage:<34s} {metric:<26s} = {value:.6g} "
          f"({'<' if lower_is_better else '>='} {threshold:g})")
    return ok


def maxabs(a, b):
    a = np.asarray(a.todense()).ravel() if sp.issparse(a) else np.asarray(a).ravel()
    b = np.asarray(b.todense()).ravel() if sp.issparse(b) else np.asarray(b).ravel()
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)))


def main(outdir):
    print(f"=== stage check against {outdir} ===")
    params = rio.read_json(outdir, "params")
    grid_size = params["grid.size"]
    radius = params["radius"]

    # ---- stage 1: PreprocessST ----------------------------------------
    norm = rio.read_sparse_raw(outdir, "00_normdata").tocsc()
    norm_genes = rio.read_lines(outdir, "00_normdata.rows")
    norm_cells = rio.read_lines(outdir, "00_normdata.cols")
    scmeta = rio.read_df(outdir, "00_scmeta")

    ref_exp = rio.read_sparse_raw(outdir, "01_preprocess_expdat").tocsc()
    ref_genes = rio.read_lines(outdir, "01_preprocess_expdat.rows")
    ref_cells = rio.read_lines(outdir, "01_preprocess_expdat.cols")

    exp, meta, genes, cells = preprocessing.preprocess_st(
        norm, scmeta, min_cells=params["min.cells"],
        min_features=params["min.features"],
        genes=norm_genes, cells=norm_cells, verbose=False)
    assert genes == ref_genes, "gene filter mismatch"
    assert cells == ref_cells, "cell filter mismatch"
    report("PreprocessST", "max|dR - dPy|", maxabs(ref_exp, exp), 1e-13)

    # ---- stage 2: spot metadata ---------------------------------------
    ref_ncmeta = rio.read_df(outdir, "02_ncmeta")
    ncmeta = core._spot_metadata(meta, grid_size)
    same_ids = set(ncmeta.index) == set(ref_ncmeta.index)
    report("spot grid (SpotID set)", "jaccard",
           len(set(ncmeta.index) & set(ref_ncmeta.index)) /
           len(set(ncmeta.index) | set(ref_ncmeta.index)), 1.0, False)
    if same_ids:
        a = ncmeta.loc[ref_ncmeta.index]
        # 1e-9, not 1e-12: the coordinates are O(1e3-1e4) and `data.table::fwrite`
        # writes 15 significant digits, so the dump itself carries ~5e-12 of
        # absolute error at this magnitude. That is a serialisation floor, not a
        # computation difference -- relative error is ~1e-15.
        report("spot grid X", "max|d|", maxabs(ref_ncmeta["X"], a["X"]), 1e-9)
        report("spot grid Y", "max|d|", maxabs(ref_ncmeta["Y"], a["Y"]), 1e-9)
        for c in ("CellType", "Region"):
            if c in ref_ncmeta.columns and c in a.columns:
                agree = float((ref_ncmeta[c].astype(str).to_numpy()
                               == a[c].astype(str).to_numpy()).mean())
                report(f"spot grid {c}", "agreement", agree, 1.0, False)

    # ---- stage 3: GetSpatialMetacells ---------------------------------
    ref_mc = rio.read_sparse_raw(outdir, "03_metacells").tocsc()
    ref_mc_cols = rio.read_lines(outdir, "03_metacells.cols")
    ref_mc_rows = rio.read_lines(outdir, "03_metacells.rows")

    logexp = exp.copy()
    mx = logexp.data.max() if sp.issparse(logexp) else np.max(logexp)
    if mx > 50:
        logexp.data = np.log1p(logexp.data)
    mc, mc_cols = metacells.get_spatial_metacells(
        logexp, meta, spotCoord=ncmeta, k=params["k"], radius=radius,
        gene_names=genes, verbose=False)
    print(f"    metacells: R {ref_mc.shape} vs Py {mc.shape}")
    if list(mc_cols) == list(ref_mc_cols) and list(genes) == list(ref_mc_rows):
        report("GetSpatialMetacells", "max|d|", maxabs(ref_mc, mc), 1e-8)
    else:
        common = [c for c in ref_mc_cols if c in set(mc_cols)]
        report("GetSpatialMetacells cols", "jaccard",
               len(set(mc_cols) & set(ref_mc_cols)) /
               len(set(mc_cols) | set(ref_mc_cols)), 1.0, False)
        ri = {c: i for i, c in enumerate(ref_mc_cols)}
        pi = {c: i for i, c in enumerate(mc_cols)}
        A = ref_mc[:, [ri[c] for c in common]]
        B = mc[:, [pi[c] for c in common]]
        report("GetSpatialMetacells", "max|d| (common cols)", maxabs(A, B), 1e-8)

    # ---- stage 4: GetPCList -------------------------------------------
    ct_names = rio.read_lines(outdir, "04_emb_celltypes")
    emb_list = network.get_pc_list(mc, mc_cols, genes,
                                   nfeatures=params["nfeatures"],
                                   min_cells=params["min.cells"],
                                   min_features=params["min.features"],
                                   verbose=False)
    from scipy.spatial import procrustes
    for ct in ct_names:
        ref = rio.read_dense(outdir, f"04_emb_{ct}")
        if ct not in emb_list:
            report(f"GetPCList[{ct}]", "present", 0.0, 1.0, False)
            continue
        emb, spots = emb_list[ct]
        if list(spots) != list(ref.columns):
            report(f"GetPCList[{ct}] spots", "jaccard",
                   len(set(spots) & set(ref.columns)) /
                   len(set(spots) | set(ref.columns)), 1.0, False)
            continue
        npc = min(ref.shape[0], emb.shape[0], 20)
        A = ref.to_numpy()[:npc].T
        B = emb[:npc].T
        _, _, disp = procrustes(A, B)
        report(f"GetPCList[{ct}]", "procrustes sim", 1 - disp, 0.95, False)

    # ---- stage 5: getSN on R's OWN embeddings --------------------------
    sn_names = rio.read_lines(outdir, "05_sn_celltypes")
    r_emb = {}
    for ct in ct_names:
        ref = rio.read_dense(outdir, f"04_emb_{ct}")
        r_emb[ct] = (ref.to_numpy(), list(ref.columns))
    r_snlist, r_spots = network.get_sn_list(
        r_emb, npcs=params["npcs"],
        min_cts_per_region=params["min.cts.per.region"],
        k=params["k.sn"], verbose=False)
    for ct in sn_names:
        ref = rio.read_sparse_raw(outdir, f"05_sn_{ct}").tocsr()
        ref_cols = rio.read_lines(outdir, f"05_sn_{ct}.cols")
        w, names = r_snlist[ct]
        if list(names) != list(ref_cols):
            pos = {s: i for i, s in enumerate(names)}
            perm = [pos[s] for s in ref_cols]
            w = w[perm][:, perm]
        report(f"getSN[{ct}]", "max|d|", maxabs(ref, w), 1e-8)

    # ---- stage 6: SNF2 on R's OWN networks -----------------------------
    r_nets = []
    ref_cols0 = rio.read_lines(outdir, f"05_sn_{sn_names[0]}.cols")
    for ct in sn_names:
        m = rio.read_sparse_raw(outdir, f"05_sn_{ct}").tocsr()
        cols = rio.read_lines(outdir, f"05_sn_{ct}.cols")
        assert cols == ref_cols0
        r_nets.append(m)
    fused = network.snf2(r_nets, t=params["iterations"],
                         minibatch=params["minibatch"], verbose=False)
    ref_fused = rio.read_sparse_raw(outdir, "06_snf_fused").tocsr()
    report("SNF2", "max|d|", maxabs(ref_fused, fused), 1e-8)

    # ---- stage 7: rankSparse -------------------------------------------
    ranked = utils.rank_sparse(ref_fused)
    ref_ranked = rio.read_sparse_raw(outdir, "07_rank_sparse").tocsr()
    report("rankSparse", "max|d|", maxabs(ref_ranked, ranked), 1e-13)

    # ---- stage 8/9: ScaleData + RunPCA on R's ranked matrix ------------
    ref_scaled = rio.read_dense(outdir, "08_scaled")
    scaled = seuratcompat.scale_data(ref_ranked, do_scale=False, do_center=True)
    report("ScaleData(do.scale=F)", "max|d|", maxabs(ref_scaled.to_numpy(), scaled), 1e-8)

    ref_pca = rio.read_dense(outdir, "09_pca")
    emb, _, _ = seuratcompat.run_pca(scaled, npcs=min(50, scaled.shape[0] - 1))
    npc = min(ref_pca.shape[1], emb.shape[1], 10)
    _, _, disp = procrustes(ref_pca.to_numpy()[:, :npc], emb[:, :npc])
    report("RunPCA (top 10 PCs)", "procrustes sim", 1 - disp, 0.95, False)

    print("\n=== summary ===")
    npass = sum(1 for r in RESULTS if r[4])
    print(f"{npass}/{len(RESULTS)} checks passed")
    for stage, metric, value, thr, ok in RESULTS:
        if not ok:
            print(f"  FAIL {stage}: {metric} = {value:.6g} (threshold {thr:g})")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reference_out/ci"))
