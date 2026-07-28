"""Parity tests for :mod:`pyspatialecotyper.stats` against R SpatialEcoTyper 1.0.4.

Every threshold is read out of ``data/manifest.yaml`` — the pre-registered gate
— rather than hard-coded here, and every metric comes from
``omicverse-rebuildr/engine/parity_metrics.py`` rather than being re-derived.

Generate the reference dump first::

    export R_LIBS=/scratch/users/steorra/Rlibs_set:\\
    /scratch/users/steorra/env/CMAP/lib/R/library:\\
    /scratch/users/steorra/env/setref/lib/R/library
    export TMPDIR=/scratch/users/steorra/tmp
    /scratch/users/steorra/env/CMAP/bin/Rscript tests/r_stats_driver.R \\
        data/Melanoma1_subset_counts.tsv.gz data/Melanoma1_subset_scmeta.tsv \\
        reference_out/stats all 4000

then::

    /scratch/users/steorra/env/omicdev/bin/python -m pytest tests/test_stats.py -q

Point ``PYSET_STATS_REF`` at a different directory to use another dump.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# The parity metrics live in the omicverse-rebuildr kit, not in this repo. On a
# fresh checkout (or in CI) the kit is absent and every R-parity test here would
# skip anyway for want of reference_out/stats, so make the import soft rather
# than letting collection fail.
_ENGINE = os.environ.get(
    "OMICVERSE_REBUILDR",
    os.path.join(os.path.dirname(ROOT), "omicverse-rebuildr"))
if os.path.isdir(_ENGINE):
    sys.path.insert(0, _ENGINE)

try:
    from engine.parity_metrics import compute_parity, is_pass   # noqa: E402
except ImportError:                                             # pragma: no cover
    pytest.skip("omicverse-rebuildr engine not on the path; set "
                "OMICVERSE_REBUILDR to the kit checkout to run the R-parity "
                "tests", allow_module_level=True)
from rio import read_dense, read_df, read_json, read_sparse_raw  # noqa: E402
from pyspatialecotyper import rrandom, stats                     # noqa: E402

REFDIR = os.environ.get("PYSET_STATS_REF",
                        os.path.join(ROOT, "reference_out", "stats"))

pytestmark = pytest.mark.skipif(
    not os.path.isdir(REFDIR) or
    not os.path.exists(os.path.join(REFDIR, "00_fixture.tsv")),
    reason=(f"R reference dump not found at {REFDIR}; run "
            "tests/r_stats_driver.R first (see this file's docstring)."),
)


# ---------------------------------------------------------------------------
# manifest — the pre-registered gate.  Read-only.
# ---------------------------------------------------------------------------

with open(os.path.join(ROOT, "data", "manifest.yaml")) as fh:
    MANIFEST = yaml.safe_load(fh)
GATES = {o["name"]: o for o in MANIFEST["outputs"]}


def gate(name):
    """``(algorithm_class, threshold)`` for a pre-registered output."""
    g = GATES[name]
    return g["metric"], float(g["threshold"])


def check(name, reference, candidate, **kwargs):
    """Compute the gated metric and assert it clears the registered threshold."""
    cls, thr = gate(name)
    value = compute_parity(reference, candidate, algorithm_class=cls, **kwargs)
    assert is_pass(value, cls, thr), (
        f"{name}: {cls} parity {value!r} did not clear threshold {thr!r}")
    return value


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    return read_json(REFDIR, "params")


@pytest.fixture(scope="module")
def meta():
    """The R driver's fixture, read back verbatim so no re-derivation is needed."""
    m = read_df(REFDIR, "00_fixture")
    for c in ("X", "Y", "Dist2Interface"):
        m[c] = m[c].astype(float)
    return m


@pytest.fixture(scope="module")
def counts():
    return read_dense(REFDIR, "00_counts")


def _named(name):
    """Read a ``dump_named`` JSON back as a Series."""
    j = read_json(REFDIR, name)
    vals = [np.nan if v is None else float(v) for v in j["value"]]
    return pd.Series(vals, index=[str(s) for s in j["name"]])


# ===========================================================================
# Tier 1 — deterministic / exact
# ===========================================================================

def test_partition_tissue(meta):
    """PartitionTissue: gated `classification` @ 1.0 (i.e. exact labels)."""
    ref = read_json(REFDIR, "partition_tissue")
    got = stats.partition_tissue(meta, nrow=3, ncol=5, x="X", y="Y")
    assert [str(v) for v in got.index] == [str(v) for v in ref["cell"]]
    value = check("partition_tissue",
                  np.array([str(v) for v in ref["partition"]], dtype=object),
                  np.array(got["Partition"].values, dtype=object))
    assert value == 1.0


def test_infer_ncells(counts):
    """InferNCells: gated `deterministic-strict` @ 1e-13; here bit-exact."""
    for tag, av in (("5", 5), ("1p5", 1.5), ("12", 12)):
        ref = np.asarray(read_json(REFDIR, f"infer_ncells_{tag}")["ncells"],
                         dtype=np.int64)
        got = stats.infer_ncells(counts.values, avg_number=av)
        assert got.dtype.kind == "i"
        check("infer_ncells", ref.astype(float), got.astype(float))
        assert np.array_equal(ref, got), f"avg_number={av} not exact"


@pytest.mark.parametrize("dump,kwargs", [
    ("metrics_f1", dict(cell_type=None, sample="Sample", metric="F1")),
    ("metrics_f2", dict(cell_type=None, sample="Sample", metric="F2")),
    ("metrics_recall", dict(cell_type=None, sample="Sample", metric="recall")),
    ("metrics_precision", dict(cell_type=None, sample="Sample",
                               metric="precision")),
    ("metrics_f1_ct", dict(cell_type="CellType", sample="Sample", metric="F1")),
    ("metrics_f1_nosample", dict(cell_type=None, sample=None, metric="F1")),
])
def test_compute_metrics(meta, dump, kwargs):
    """ComputeMetrics: gated `deterministic-standard` @ 1e-8."""
    ref = read_dense(REFDIR, dump)
    got = stats.compute_metrics(meta, se="SE", pred="cvPred", **kwargs)
    assert list(got.index) == list(ref.index), f"{dump}: row order differs"
    assert list(got.columns) == list(ref.columns), f"{dump}: col order differs"
    check("compute_metrics",
          np.nan_to_num(ref.values.astype(float)),
          np.nan_to_num(got.values.astype(float)))


def test_build_knn_weights(meta):
    """buildKNNWeights: the KNN kernel behind `se_abundance_by_sn`; exact 0/1."""
    coords = meta[["X", "Y"]]
    for dump, self_flag in (("knn_weights_self", True),
                            ("knn_weights_noself", False)):
        ref = read_sparse_raw(REFDIR, dump).toarray()
        got = stats.build_knn_weights(ref_coords=coords, k=20, radius=40,
                                      include_self=self_flag).toarray()
        check("se_abundance_by_sn", ref, got)
        assert np.array_equal(ref, got), f"{dump} not bit-identical"


def test_aggregate_by_weights(meta):
    """aggregateByWeights: gated with `se_abundance_by_sn` @ 1e-8."""
    ref = read_dense(REFDIR, "aggregate_weights")
    coords = meta[["X", "Y"]]
    w = stats.build_knn_weights(ref_coords=coords, k=20, radius=40,
                                include_self=True)
    se_levels = sorted(set(meta["SE"].astype(str)))
    cell2se = pd.DataFrame(
        (meta["SE"].astype(str).values[:, None] == np.array(se_levels)
         ).astype(float), index=[str(v) for v in meta.index], columns=se_levels)
    got = stats.aggregate_by_weights(cell2se, w, sum2one=True, min_cells=5)
    assert list(got.index) == list(ref.index)
    check("se_abundance_by_sn", ref.values.astype(float), got.values.astype(float))


def test_se_abundance_by_sn(meta):
    """ComputeSEAbundanceBySN: gated `deterministic-standard` @ 1e-8."""
    ref = read_df(REFDIR, "se_abundance_by_sn")
    got = stats.compute_se_abundance_by_sn(meta, spot_coords=None, radius=50,
                                           grid_size=50, x="X", y="Y", se="SE",
                                           min_cells=5)
    assert list(got.index) == list(ref.index), "SN order differs"
    assert list(got.columns) == list(ref.columns)
    check("se_abundance_by_sn", ref.values.astype(float), got.values.astype(float))


def test_smooth_se_abundances(meta):
    """SmoothSEAbundances: gated with `se_abundance_by_sn` @ 1e-8."""
    ref = read_df(REFDIR, "smooth_se_abundances")
    seab = stats.compute_se_abundance_by_sn(meta, spot_coords=None, radius=50,
                                            grid_size=50, x="X", y="Y", se="SE",
                                            min_cells=5)
    got = stats.smooth_se_abundances(seab.iloc[:, 2:], seab[["X", "Y"]], k=7,
                                     x="X", y="Y", include_self=True,
                                     min_neighbors=3)
    assert list(got.index) == list(ref.index)
    check("se_abundance_by_sn", ref.values.astype(float), got.values.astype(float))


def test_coassociation_index(meta):
    """Coassociation(test = FALSE): gated `deterministic-standard` @ 1e-8."""
    ref = read_dense(REFDIR, "coassociation_index")
    got = stats.coassociation(meta, sample="Sample", se="SEn",
                              cell_type="CellType", non_se="NonSE", test=False)
    assert list(got.index) == list(ref.index)
    check("coassociation_index", ref.values.astype(float),
          got.values.astype(float))


def test_colocalization_observed(meta):
    """.colocalization: the deterministic kernel under `colocalization_index`."""
    ref = read_dense(REFDIR, "colocalization_observed")
    mm = meta.copy()
    mm["CellState"] = (mm["SE"].astype(str) + "_" + mm["CellType"].astype(str))
    got = stats._colocalization(mm, coords=("X", "Y"), cell_state="CellState",
                                radius=50, k=100, min_cell=10)
    assert list(got.index) == list(ref.index)
    # Deterministic kernel: hold it to the tighter deterministic gate rather
    # than to `colocalization_index`'s permutation-level ordinal gate.
    check("coassociation_index", ref.values.astype(float),
          got.values.astype(float))


# ===========================================================================
# Tier 4 — permutation statistics
# ===========================================================================

def test_coassociation_pvals(params):
    """CoassociationTest: gated `inference` (Spearman >= 0.90 AND Jaccard >= 0.7).

    Run on R's OWN co-association matrix so the gate isolates the permutation
    path from any upstream drift.
    """
    mat = read_dense(REFDIR, "coassociation_index")
    ref_p = _named("coassociation_pvals")
    ref_z = _named("coassociation_zscore")

    rrandom.set_seed(int(params["seed"]))
    got = stats.coassociation_test(mat, nperm=int(params["nperm_coassoc"]))
    assert list(got.index) == list(ref_p.index)

    # `top_k` is left at the engine default (50); with only a handful of SEs
    # np.argsort truncates to the full set, so the Jaccard is over all of them.
    check("coassociation_pvals", ref_p.values, got.reindex(ref_p.index).values)
    # The Z-scores behind those p-values should be bit-close, since the RNG
    # stream is reproduced exactly.
    zerr = float(np.max(np.abs(ref_z.values -
                               got.attrs["Zscore"].reindex(ref_z.index).values)))
    assert zerr < 1e-8, f"CoassociationTest Z-scores differ by {zerr:.3e}"


def test_colocalization_index(meta, params):
    """Colocalization: gated `ordinal` @ 0.99 on the Z-score matrix.

    R's own permutation loop is only reproducible at ``ncores = 1`` (mclapply
    forks and reseeds per PID otherwise); the reference is generated at
    ``ncores = 1`` and this port reproduces that stream.
    """
    ref = read_dense(REFDIR, "colocalization_index")
    rrandom.set_seed(int(params["seed"]))
    got = stats.colocalization(meta, coords=("X", "Y"), se="SE",
                               cell_type="CellType", radius=50,
                               k=int(params["k_coloc"]), min_cell=10,
                               nperm=int(params["nperm_coloc"]), test=True,
                               ncores=1)
    assert list(got["ColocIndex"].index) == list(ref.index)
    check("colocalization_index", ref.values.astype(float),
          got["ColocIndex"].values.astype(float))

    ref_p = _named("colocalization_pvals")
    check("coassociation_pvals", ref_p.values,
          got["Pval"].reindex(ref_p.index).values)


def test_colocalization_meta(params):
    """ColocalizationMetaAnalysis: gated `ordinal` @ 0.99.

    Fed R's own per-sample ColocIndex/Pval so the gate isolates the
    meta-analysis arithmetic from the permutation null upstream of it.
    """
    results = []
    for i in (1, 2, 3):
        idx = read_dense(REFDIR, f"meta_input_index_S{i}")
        pval = _named(f"meta_input_pval_S{i}")
        pval.attrs["Zscore"] = _named(f"meta_input_zscore_S{i}")
        results.append({"ColocIndex": idx, "Pval": pval})

    for tag, kw in (("", dict(cap=5, min_samples=1)),
                    ("_cap2", dict(cap=2, min_samples=2))):
        ref_idx = read_dense(REFDIR, f"colocalization_meta_index{tag}")
        ref_p = _named(f"colocalization_meta_pvals{tag}")
        got = stats.colocalization_meta_analysis(results, **kw)
        assert list(got["MetaColocIndex"].index) == list(ref_idx.index)
        check("colocalization_meta", ref_idx.values.astype(float),
              got["MetaColocIndex"].values.astype(float))
        assert list(got["MetaPval"].index) == list(ref_p.index)
        check("colocalization_meta", ref_p.values,
              got["MetaPval"].reindex(ref_p.index).values)


def test_moran_observed(meta):
    """.moran: the spdep closed form, checked without the permutation null.

    Held to the deterministic 1e-8 gate rather than `normalized_moran_i`'s
    ordinal one, because no RNG is involved.
    """
    if not os.path.exists(os.path.join(REFDIR, "moran_observed.json")):
        pytest.skip("ComputeNormalizedMoranI reference absent (spdep missing)")
    ref = _named("moran_observed")
    lmeta = read_json(REFDIR, "moran_listw_meta")
    assert lmeta["S0"] == lmeta["n"], "style='W' should give S0 == n"

    # Rebuild the same style-'W' weights the R side handed to spdep::moran.
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    xy = meta[["X", "Y"]].values.astype(float)
    k = int(lmeta["k"])
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)
    nbr = np.array([[j for j in row if j != i][:k]
                    for i, row in enumerate(idx)], dtype=np.int64)
    rows = np.repeat(np.arange(len(xy)), k)
    w = sp.csr_matrix((np.full(nbr.size, 1.0 / k), (rows, nbr.ravel())),
                      shape=(len(xy), len(xy)))
    got = stats._moran(meta["SE"].astype(str).values, w, ncores=1)
    assert list(got.index) == list(ref.index)
    check("coassociation_index", ref.values, got.reindex(ref.index).values)


def test_normalized_moran_i(meta, params):
    """ComputeNormalizedMoranI: gated `ordinal` @ 0.95 on the Z-scores."""
    if not os.path.exists(os.path.join(REFDIR, "normalized_moran_i.json")):
        pytest.skip("ComputeNormalizedMoranI reference absent (spdep missing)")
    ref = _named("normalized_moran_i")
    rrandom.set_seed(int(params["seed"]))
    got = stats.compute_normalized_moran_i(
        meta, coords=("X", "Y"), se="SE", cell_type="CellType",
        nperm=int(params["nperm_moran"]), k=int(params["k_moran"]), ncores=1)
    assert list(got.index) == list(ref.index)
    check("normalized_moran_i", ref.values, got.reindex(ref.index).values)


# ===========================================================================
# Ungated diagnostics (no manifest entry) — reported, not gated.
# ===========================================================================

def test_create_pseudobulks(counts, meta, params):
    """CreatePseudobulks has no manifest gate; checked as a plain RNG diagnostic.

    Exercises rnorm + sample(replace = TRUE) off the ported R stream, so it is
    the sharpest available probe that the stream stays in phase.
    """
    if not os.path.exists(os.path.join(REFDIR, "pseudobulk_fracs.tsv.gz")):
        pytest.skip("CreatePseudobulks reference absent")
    ref_f = read_dense(REFDIR, "pseudobulk_fracs")
    ref_m = read_dense(REFDIR, "pseudobulk_mixtures")
    groups = pd.Series(meta["SE"].astype(str).values,
                       index=[str(v) for v in meta.index])
    rrandom.set_seed(int(params["seed"]))
    got = stats.create_pseudobulks(counts=counts, groups=groups,
                                   n_mixtures=int(params["n_mixtures"]))
    assert list(got["Fracs"].index) == list(ref_f.index)
    ferr = float(np.max(np.abs(ref_f.values - got["Fracs"].values)))
    merr = float(np.max(np.abs(ref_m.values - got["Mixtures"].values)))
    assert ferr < 1e-12, f"Fracs max abs error {ferr:.3e}"
    assert merr < 1e-8, f"Mixtures max abs error {merr:.3e}"
