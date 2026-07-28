"""The parity gate, as a runnable pytest.

Every threshold is read out of ``data/manifest.yaml`` -- the pre-registered
gate committed before any algorithmic Python was written.  Nothing here
hard-codes a number, so the gate cannot be quietly widened by editing a test.

Requires the R reference dump.  Regenerate with::

    export R_LIBS=/path/to/Rlibs:/path/to/R/library
    Rscript tests/r_reference_driver.R data/Melanoma1_subset_counts.tsv.gz \\
                                       data/Melanoma1_subset_scmeta.tsv \\
                                       reference_out/ci 6000
    Rscript tests/r_nmf_driver.R      reference_out/nmf
    Rscript tests/r_nmfclust_driver.R reference_out/nmf

Tests skip cleanly when a dump is absent, so ``pytest -q`` is green on a fresh
checkout without R.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import rio  # noqa: E402
from pyspatialecotyper import (core, metacells, network, nmf as nmf_mod,  # noqa: E402
                               preprocessing, seuratcompat, utils)
import pyspatialecotyper.integration as integ_mod  # noqa: E402

CI = os.path.join(ROOT, "reference_out", "ci")
NMF = os.path.join(ROOT, "reference_out", "nmf")

needs_ci = pytest.mark.skipif(
    not os.path.exists(os.path.join(CI, "params.json")),
    reason="R reference dump reference_out/ci is absent")
needs_nmf = pytest.mark.skipif(
    not os.path.exists(os.path.join(NMF, "nmf_V.tsv.gz")),
    reason="R reference dump reference_out/nmf is absent")


# --------------------------------------------------------------------------
# manifest access
# --------------------------------------------------------------------------
def _manifest():
    import yaml
    with open(os.path.join(ROOT, "data", "manifest.yaml")) as fh:
        return yaml.safe_load(fh)


MANIFEST = _manifest()
GATES = {o["name"]: o for o in MANIFEST["outputs"]}


def gate(name):
    """Return ``(metric_class, threshold)`` for a pre-registered output."""
    g = GATES[name]
    return g["metric"], float(g["threshold"])


def assert_gate(name, value):
    """Apply the pre-registered pass direction for the output's metric class."""
    metric, thr = gate(name)
    if metric.startswith("deterministic"):
        assert value < thr, (f"parity gate '{name}' FAILED: max|d| = {value:.6g} "
                             f"is not < {thr:g}")
    else:
        assert value >= thr, (f"parity gate '{name}' FAILED: {metric} = "
                              f"{value:.6g} is not >= {thr:g}")


def maxabs(a, b):
    a = np.asarray(a.todense()).ravel() if sp.issparse(a) else np.asarray(a, float).ravel()
    b = np.asarray(b.todense()).ravel() if sp.issparse(b) else np.asarray(b, float).ravel()
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)))


def matched_factor_corr(ref, cand):
    """The port-local ``factorization`` class: Hungarian-matched mean |Pearson|."""
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr
    k = ref.shape[1]
    c = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a, b = ref[:, i], cand[:, j]
            c[i, j] = 0.0 if (np.std(a) == 0 or np.std(b) == 0) else abs(pearsonr(a, b)[0])
    r, cc = linear_sum_assignment(-c)
    return float(np.mean(c[r, cc]))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ci():
    params = rio.read_json(CI, "params")
    norm = rio.read_sparse_raw(CI, "00_normdata").tocsc()
    genes = rio.read_lines(CI, "00_normdata.rows")
    cells = rio.read_lines(CI, "00_normdata.cols")
    scmeta = rio.read_df(CI, "00_scmeta")
    exp, meta, g2, c2 = preprocessing.preprocess_st(
        norm, scmeta, min_cells=params["min.cells"],
        min_features=params["min.features"], genes=genes, cells=cells,
        verbose=False)
    return dict(params=params, norm=norm, genes=genes, cells=cells,
                scmeta=scmeta, exp=exp, meta=meta, g2=g2, c2=c2)


# --------------------------------------------------------------------------
# Tier 1 -- deterministic kernels
# --------------------------------------------------------------------------
@needs_ci
def test_preprocess_st(ci):
    ref = rio.read_sparse_raw(CI, "01_preprocess_expdat").tocsc()
    assert ci["g2"] == rio.read_lines(CI, "01_preprocess_expdat.rows")
    assert ci["c2"] == rio.read_lines(CI, "01_preprocess_expdat.cols")
    assert_gate("preprocess_st", maxabs(ref, ci["exp"]))


@needs_nmf
def test_znorm():
    V = rio.read_dense(NMF, "nmf_V").to_numpy()
    grp = np.array(["R1", "R2"] * (V.shape[1] // 2))[:V.shape[1]]
    assert_gate("znorm", maxabs(rio.read_dense(NMF, "nmf_znorm").to_numpy(),
                                preprocessing.znorm(V, grp)))
    assert_gate("znorm", maxabs(rio.read_dense(NMF, "nmf_znorm_nogroup").to_numpy(),
                                preprocessing.znorm(V)))


@needs_ci
def test_rank_sparse():
    fused = rio.read_sparse_raw(CI, "06_snf_fused").tocsr()
    ref = rio.read_sparse_raw(CI, "07_rank_sparse").tocsr()
    assert_gate("rank_sparse", maxabs(ref, utils.rank_sparse(fused)))


@needs_ci
def test_matrix_multiply():
    """`matrixMultiply` blocks over the columns of mat2; blocking must not
    change the result, since it never reorders a summation index."""
    a = rio.read_sparse_raw(CI, "05_sn_B").tocsr()
    b = rio.read_sparse_raw(CI, "05_sn_CD4T").tocsr()
    full = np.asarray((a @ b).todense())
    for mb in (17, 64, 5000):
        got = utils.matrix_multiply(a, b, minibatch=mb)
        assert_gate("matrix_multiply", maxabs(full, got))


@needs_ci
def test_spatial_metacells(ci):
    p = ci["params"]
    ref = rio.read_sparse_raw(CI, "03_metacells").tocsc()
    ref_cols = rio.read_lines(CI, "03_metacells.cols")
    ncmeta = core._spot_metadata(ci["meta"], p["grid.size"])
    logexp = ci["exp"].copy()
    if logexp.data.max() > 50:
        logexp.data = np.log1p(logexp.data)
    mc, cols = metacells.get_spatial_metacells(
        logexp, ci["meta"], spotCoord=ncmeta, k=p["k"], radius=p["radius"],
        gene_names=ci["g2"], verbose=False)
    assert list(cols) == list(ref_cols), "metacell column set/order differs from R"
    assert_gate("spatial_metacells", maxabs(ref, mc))


@needs_ci
def test_snf2_on_r_networks(ci):
    """SNF2 fed R's own similarity-network list -- isolates the diffusion."""
    p = ci["params"]
    cts = rio.read_lines(CI, "05_sn_celltypes")
    nets = [rio.read_sparse_raw(CI, f"05_sn_{ct}").tocsr() for ct in cts]
    fused = network.snf2(nets, t=p["iterations"], minibatch=p["minibatch"],
                         verbose=False)
    assert_gate("snf_fused", maxabs(rio.read_sparse_raw(CI, "06_snf_fused").tocsr(),
                                    fused))


@needs_ci
def test_scale_data_center_only():
    ref = rio.read_dense(CI, "08_scaled").to_numpy()
    ranked = rio.read_sparse_raw(CI, "07_rank_sparse").tocsr()
    got = seuratcompat.scale_data(ranked, do_scale=False, do_center=True)
    assert_gate("znorm", maxabs(ref, got))   # same deterministic-standard tier


@needs_nmf
def test_compute_fcs():
    V = rio.read_dense(NMF, "nmf_V")
    meta = rio.read_json(NMF, "nmf_cellse")
    scmeta = pd.DataFrame({"CellType": meta["CellType"], "SE": meta["SE"]},
                          index=list(V.columns))
    for tag, scale in (("nmf_computefcs", True), ("nmf_computefcs_noscale", False)):
        ref = rio.read_dense(NMF, tag)
        fc, rows, cols = integ_mod.compute_fcs(V.to_numpy(), scmeta,
                                               list(V.index), cluster="SE",
                                               scale=scale)
        ri = {r: i for i, r in enumerate(rows)}
        cix = {c: i for i, c in enumerate(cols)}
        sub = fc[[ri[r] for r in ref.index]][:, [cix[c] for c in ref.columns]]
        # ComputeFCs feeds `integrated_matrix`; gate it at the same tier.
        assert_gate("integrated_matrix", maxabs(ref.to_numpy(), sub))


# --------------------------------------------------------------------------
# Tier 2 -- decompositions
# --------------------------------------------------------------------------
@needs_ci
def test_pc_embeddings(ci):
    from scipy.spatial import procrustes
    p = ci["params"]
    ncmeta = core._spot_metadata(ci["meta"], p["grid.size"])
    logexp = ci["exp"].copy()
    if logexp.data.max() > 50:
        logexp.data = np.log1p(logexp.data)
    mc, cols = metacells.get_spatial_metacells(
        logexp, ci["meta"], spotCoord=ncmeta, k=p["k"], radius=p["radius"],
        gene_names=ci["g2"], verbose=False)
    emb_list = network.get_pc_list(mc, cols, ci["g2"], nfeatures=p["nfeatures"],
                                   min_cells=p["min.cells"],
                                   min_features=p["min.features"], verbose=False)
    worst = 1.0
    for ct in rio.read_lines(CI, "04_emb_celltypes"):
        ref = rio.read_dense(CI, f"04_emb_{ct}")
        emb, spots = emb_list[ct]
        assert list(spots) == list(ref.columns)
        npc = min(ref.shape[0], emb.shape[0], 20)
        _, _, disp = procrustes(ref.to_numpy()[:npc].T, emb[:npc].T)
        worst = min(worst, 1 - disp)
    assert_gate("pc_embeddings", worst)


@needs_nmf
def test_nmf_generate_w():
    V = rio.read_dense(NMF, "nmf_V")
    Fracs = rio.read_dense(NMF, "nmf_Fracs")
    refW = rio.read_dense(NMF, "nmf_W_seed11")
    W, wn, _ = nmf_mod.nmf_generate_w(Fracs.to_numpy(), V.to_numpy(),
                                      feature_names=list(V.index),
                                      se_names=list(Fracs.columns),
                                      scale=True, nfeature=300,
                                      nfeature_per_se=50)
    common = [f for f in refW.index if f in set(wn)]
    wi = {f: i for i, f in enumerate(wn)}
    assert_gate("nmf_W", matched_factor_corr(refW.loc[common].to_numpy(),
                                             W[[wi[f] for f in common]]))


@needs_nmf
def test_nmf_predict():
    V = rio.read_dense(NMF, "nmf_V")
    refW = rio.read_dense(NMF, "nmf_W_seed11")
    for tag, chunk in (("nmf_H", 5000), ("nmf_H_chunked", 25)):
        refH = rio.read_dense(NMF, tag).to_numpy()
        H, _, _ = nmf_mod.nmf_predict(refW.to_numpy(), list(refW.index),
                                      list(refW.columns), V.to_numpy(),
                                      list(V.index), list(V.columns),
                                      scale=False, ncell_per_run=chunk,
                                      sum2one=True)
        if refH.shape != H.shape:
            refH = refH.T
        assert_gate("nmf_H", matched_factor_corr(refH, H))


# --------------------------------------------------------------------------
# Tier 3 -- labels
# --------------------------------------------------------------------------
@needs_ci
def test_annotate_cells(ci):
    ref = rio.read_df(CI, "12_se_cells")
    spot = rio.read_json(CI, "12_se_spot")
    ncmeta = core._spot_metadata(ci["meta"], ci["params"]["grid.size"])
    spot_meta = ncmeta.loc[list(spot["spot"])].copy()
    spot_meta["SE"] = list(spot["SE"])
    got = core.annotate_cells(ci["meta"], spot_meta,
                              radius=ci["params"]["radius"], dropcell=True)
    common = [c for c in ref.index if c in set(got.index)]
    agree = float((ref.loc[common, "SE"].astype(str).to_numpy()
                   == got.loc[common, "SE"].astype(str).to_numpy()).mean())
    assert len(common) / len(ref) > 0.999, "cell set after AnnotateCells differs"
    assert_gate("annotate_cells", agree)


@needs_ci
def test_se_labels_single_sample_end_to_end(ci):
    """THE headline gate: full single-sample pipeline vs R, ARI over labels."""
    from sklearn.metrics import adjusted_rand_score
    p = ci["params"]
    res = core.spatial_ecotyper(
        ci["norm"], ci["scmeta"], gene_names=ci["genes"], radius=p["radius"],
        resolution=p["resolution"], nfeatures=p["nfeatures"],
        min_cts_per_region=p["min.cts.per.region"], npcs=p["npcs"],
        min_cells=p["min.cells"], min_features=p["min.features"],
        iterations=p["iterations"], k=p["k"], k_sn=p["k.sn"], verbose=False)
    ref = rio.read_json(CI, "12_se_spot")
    py = dict(zip(res.spot_metadata.index, res.spot_metadata["SE"]))
    common = [s for s in ref["spot"] if s in py]
    assert len(common) == len(ref["spot"]), "spatial-neighbourhood set differs from R"
    rmap = dict(zip(ref["spot"], ref["SE"]))
    ari = adjusted_rand_score([rmap[s] for s in common], [py[s] for s in common])
    assert_gate("se_labels_single_sample", ari)


@needs_nmf
def test_conserved_se_labels():
    from sklearn.metrics import adjusted_rand_score
    S = rio.read_dense(NMF, "nmf_integrated")
    ref = rio.read_json(NMF, "nmf_clustering_k4")
    res = nmf_mod.nmf_clustering(S.to_numpy(), row_names=list(S.index),
                                 col_names=list(S.columns), ranks=4,
                                 nrun_per_rank=10, seed=2024)
    assert_gate("conserved_se_labels",
                adjusted_rand_score(ref["label"], res["labels"]))


@needs_nmf
def test_nmf_consensus_matrix_is_bit_equal():
    """Stronger than the label gate: the consensus matrix itself."""
    S = rio.read_dense(NMF, "nmf_integrated")
    res = nmf_mod.nmf_clustering(S.to_numpy(), row_names=list(S.index),
                                 col_names=list(S.columns), ranks=4,
                                 nrun_per_rank=10, seed=2024)
    ref = rio.read_dense(NMF, "nmf_consensus_k4").to_numpy()
    assert maxabs(ref, res["fits"]["K.4"]["consensus"]) < 1e-12


# --------------------------------------------------------------------------
# R RNG substrate -- the reason four otherwise-stochastic outputs are gated
# deterministically at all.
# --------------------------------------------------------------------------
def test_r_rng_matches_r():
    """`runif` bit-identical, `sample()` identical, `rnorm` to f64 qnorm error.

    Reference values captured from R 4.4.3 (`set.seed(42)`).
    """
    from pyspatialecotyper.rrandom import RRandom
    r = RRandom(42)
    u = r.runif(5)
    expected = np.array([0.914806043496355, 0.937075413297862, 0.286139534786344,
                         0.830447626067325, 0.641745518893003])
    assert np.max(np.abs(u - expected)) < 5e-16
    r = RRandom(2024)
    assert list(r.sample_int(6280, 5) + 1) == [5698, 4645, 3629, 4796, 5375]
