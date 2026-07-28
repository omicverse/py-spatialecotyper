"""Diff the NMF + integration layer against the R reference dump.

Development harness (not pytest).  Run as::

    python tests/nmf_check.py reference_out/nmf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rio
import pyspatialecotyper.integration as integ_mod
from pyspatialecotyper import nmf as nmf_mod
from pyspatialecotyper import preprocessing

RESULTS = []


def report(stage, metric, value, threshold, lower_is_better=True):
    ok = (value < threshold) if lower_is_better else (value >= threshold)
    RESULTS.append((stage, metric, value, threshold, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {stage:<30s} {metric:<28s} = {value:.6g} "
          f"({'<' if lower_is_better else '>='} {threshold:g})")


def maxabs(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)))


def matched_factor_corr(ref: np.ndarray, cand: np.ndarray) -> float:
    """Hungarian-matched mean |Pearson| over factor columns.

    The ``factorization`` parity class registered in ``data/manifest.yaml``:
    NMF factors carry no canonical order, so columns are matched to the
    reference by maximum-weight assignment on the |correlation| matrix and the
    reported statistic is the mean matched |r|.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr
    k = ref.shape[1]
    c = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a, b = ref[:, i], cand[:, j]
            if np.std(a) == 0 or np.std(b) == 0:
                c[i, j] = 0.0
            else:
                c[i, j] = abs(pearsonr(a, b)[0])
    r, cc = linear_sum_assignment(-c)
    return float(np.mean(c[r, cc]))


def main(outdir):
    print(f"=== NMF layer check against {outdir} ===")
    V = rio.read_dense(outdir, "nmf_V")
    genes, cells = list(V.index), list(V.columns)
    Vm = V.to_numpy()
    meta = rio.read_json(outdir, "nmf_cellse")
    Fracs = rio.read_dense(outdir, "nmf_Fracs")
    ses = list(Fracs.columns)

    rr = rio.read_json(outdir, "nmf_W_R_vs_R")
    print(f"    [R vs R] NMFGenerateW  : feature-set jaccard={rr['jaccard']:.4f}, "
          f"max|dW| on common={rr['max_abs_diff_common']:.3g}, cor={rr['cor_common']:.6f}")
    print(f"    [R vs R] NMFpredict    : max|dH|={rio.read_json(outdir, 'nmf_H_R_vs_R')['max_abs_diff']:.3g}")
    cr = rio.read_json(outdir, "nmf_clustering_R_vs_R")
    print(f"    [R vs R] nmfClustering : identical labels={cr['identical_labels']}, "
          f"max|dconsensus|={cr['max_abs_consensus_diff']:.3g}")

    # ---- Znorm ---------------------------------------------------------
    ref = rio.read_dense(outdir, "nmf_znorm")
    grp = np.array(["R1", "R2"] * (Vm.shape[1] // 2))[:Vm.shape[1]]
    report("Znorm(groups=)", "max|d|", maxabs(ref.to_numpy(), preprocessing.znorm(Vm, grp)), 1e-8)
    ref = rio.read_dense(outdir, "nmf_znorm_nogroup")
    report("Znorm(groups=None)", "max|d|", maxabs(ref.to_numpy(), preprocessing.znorm(Vm)), 1e-8)

    # ---- ComputeFCs ----------------------------------------------------
    for tag, scale in (("nmf_computefcs", True), ("nmf_computefcs_noscale", False)):
        ref = rio.read_dense(outdir, tag)
        scmeta = pd.DataFrame({"CellType": meta["CellType"], "SE": meta["SE"]},
                              index=cells)
        fc, rows, cols = integ_mod.compute_fcs(Vm, scmeta, genes, cluster="SE",
                                               scale=scale)
        if list(rows) == list(ref.index) and list(cols) == list(ref.columns):
            report(f"ComputeFCs(scale={scale})", "max|d|",
                   maxabs(ref.to_numpy(), fc), 1e-8)
        else:
            ri = {r: i for i, r in enumerate(rows)}
            ci = {c: i for i, c in enumerate(cols)}
            sub = fc[[ri[r] for r in ref.index]][:, [ci[c] for c in ref.columns]]
            report(f"ComputeFCs(scale={scale})", "max|d| (reindexed)",
                   maxabs(ref.to_numpy(), sub), 1e-8)

    # ---- NMFGenerateW --------------------------------------------------
    refW = rio.read_dense(outdir, "nmf_W_seed11")
    W, wn, wses = nmf_mod.nmf_generate_w(Fracs.to_numpy(), Vm,
                                         feature_names=genes, se_names=ses,
                                         scale=True, nfeature=300,
                                         nfeature_per_se=50)
    common = [f for f in refW.index if f in set(wn)]
    print(f"    NMFGenerateW: R {refW.shape} features, Py {W.shape}, "
          f"common {len(common)} (jaccard "
          f"{len(common) / len(set(refW.index) | set(wn)):.4f})")
    wi = {f: i for i, f in enumerate(wn)}
    A = refW.loc[common].to_numpy()
    B = W[[wi[f] for f in common]]
    report("NMFGenerateW", "matched |pearson|", matched_factor_corr(A, B), 0.95, False)
    report("NMFGenerateW", "max|d| (common features)", maxabs(A, B), 1e-6)

    # ---- NMFpredict (fed R's own W) ------------------------------------
    for tag, chunk in (("nmf_H", 5000), ("nmf_H_chunked", 25)):
        refH = rio.read_dense(outdir, tag)
        H, se_out, cell_out = nmf_mod.nmf_predict(
            refW.to_numpy(), list(refW.index), list(refW.columns),
            Vm, genes, cells, scale=False, ncell_per_run=chunk, sum2one=True)
        Href = refH.to_numpy()
        if Href.shape != H.shape:
            Href = Href.T
        report(f"NMFpredict(ncell.per.run={chunk})", "matched |pearson|",
               matched_factor_corr(Href, H), 0.95, False)
        report(f"NMFpredict(ncell.per.run={chunk})", "max|d|", maxabs(Href, H), 1e-6)

    # ---- Integrate -----------------------------------------------------
    avg = rio.read_dense(outdir, "nmf_avgexprs")
    refI = rio.read_dense(outdir, "nmf_integrated")
    obj, spots = integ_mod.integrate(avg.to_numpy(), list(avg.index),
                                     list(avg.columns), nfeatures=200,
                                     min_features=5, seed=1)
    dense = np.asarray(obj.todense())
    pos = {s: i for i, s in enumerate(spots)}
    if set(spots) == set(refI.index):
        perm = [pos[s] for s in refI.index]
        report("Integrate", "max|d|", maxabs(refI.to_numpy(), dense[perm][:, perm]), 1e-8)
    else:
        report("Integrate", "spot jaccard",
               len(set(spots) & set(refI.index)) / len(set(spots) | set(refI.index)),
               1.0, False)

    # ---- nmfClustering -------------------------------------------------
    from sklearn.metrics import adjusted_rand_score
    refc = rio.read_json(outdir, "nmf_clustering_k4")
    S = refI.to_numpy()
    res = nmf_mod.nmf_clustering(S, row_names=list(refI.index),
                                 col_names=list(refI.columns), ranks=4,
                                 nrun_per_rank=10, seed=2024)
    ari = adjusted_rand_score(refc["label"], res["labels"])
    report("nmfClustering labels", "ARI", ari, 0.85, False)
    coph_py = res["cophenetic"]["Cophenetic"].iloc[0]
    print(f"    cophenetic: R {refc['cophenetic']:.6f}  Py {coph_py:.6f} "
          f"(|d| = {abs(refc['cophenetic'] - coph_py):.3g})")
    refcons = rio.read_dense(outdir, "nmf_consensus_k4").to_numpy()
    pycons = res["fits"]["K.4"]["consensus"]
    report("nmfClustering consensus", "max|d|", maxabs(refcons, pycons), 1e-6)

    print("\n=== summary ===")
    npass = sum(1 for r in RESULTS if r[4])
    print(f"{npass}/{len(RESULTS)} checks passed")
    for stage, metric, value, thr, ok in RESULTS:
        if not ok:
            print(f"  FAIL {stage}: {metric} = {value:.6g} (threshold {thr:g})")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reference_out/nmf"))
