"""Parity tests for :mod:`pyspatialecotyper._modularity` against Seurat 5.4.0.

Reference artefacts come from ``tests/r_reference_driver.R``:

* ``09_pca``       -- ``Embeddings(obj, "pca")``
* ``10_snn``       -- ``obj@graphs$RNA_snn`` from ``FindNeighbors(obj, dims = 1:10)``
* ``11_clusters``  -- ``as.character(obj$seurat_clusters)`` from
                      ``FindClusters(obj, resolution = 0.5)``

Run with::

    /scratch/users/steorra/env/omicdev/bin/python -m pytest tests/test_modularity.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from pyspatialecotyper._modularity import (
    JavaRandom,
    _to_int32,
    compute_snn,
    find_clusters,
    find_neighbors,
    group_singletons,
    run_modularity_clustering,
)

from . import rio

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = os.path.join(os.path.dirname(HERE), "reference_out")
CI_DIR = os.path.join(REF_ROOT, "ci")
FULL_DIR = os.path.join(REF_ROOT, "full")

# Seurat's own defaults for this fixture (r_reference_driver.R:185-187).
K_PARAM = 20
N_DIMS = 10
PRUNE_SNN = 1 / 15
RESOLUTION = 0.5


def _require(outdir):
    if not os.path.exists(os.path.join(outdir, "10_snn.mtx")):
        pytest.skip(f"R reference output missing: {outdir}")
    return outdir


def _load(outdir):
    _require(outdir)
    pca = rio.read_dense(outdir, "09_pca")
    snn_r = sp.csr_matrix(rio.read_sparse_raw(outdir, "10_snn"))
    rows = rio.read_lines(outdir, "10_snn.rows")
    clusters = rio.read_json(outdir, "11_clusters")
    assert list(pca.index) == list(rows)
    assert list(clusters["spot"]) == list(rows)
    ref_ids = np.array([int(x) for x in clusters["cluster"]], dtype=np.int64)
    return pca.values[:, :N_DIMS].astype(np.float64), snn_r, ref_ids


# ---------------------------------------------------------------------------
# JavaRandom
# ---------------------------------------------------------------------------


def test_java_random_matches_java_util_random():
    """``next(32)`` must equal ``new java.util.Random(seed).nextInt()``.

    These are the canonical published outputs of the JDK generator, so this
    pins the 48-bit LCG independently of Seurat's C++ transliteration.
    """
    assert [_to_int32(JavaRandom(0).next(32))] == [-1155484576]
    r = JavaRandom(0)
    assert [_to_int32(r.next(32)) for _ in range(3)] == [
        -1155484576,
        -723955400,
        1033096058,
    ]
    r = JavaRandom(42)
    assert [_to_int32(r.next(32)) for _ in range(3)] == [
        -1170105035,
        234785527,
        -1360544799,
    ]


def test_java_random_next_int_ranges():
    r = JavaRandom(0)
    # 16 takes the power-of-two branch, 10 the rejection branch.
    assert all(0 <= r.next_int(16) < 16 for _ in range(500))
    r = JavaRandom(0)
    assert all(0 <= r.next_int(10) < 10 for _ in range(500))
    with pytest.raises(ValueError):
        JavaRandom(0).next_int(0)


# ---------------------------------------------------------------------------
# 1. ComputeSNN
# ---------------------------------------------------------------------------


def _dense_snn_reference(nn_ranked, prune):
    """Independent dense re-derivation of snn.cpp:16-37, for cross-checking."""
    n, k = nn_ranked.shape
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in nn_ranked[i]:
            a[i, j - 1] += 1.0
    snn = a @ a.T
    snn = snn / (k + (k - snn))
    snn[snn < prune] = 0.0
    return snn


def test_compute_snn_elementwise_matches_dense_reference():
    """``compute_snn`` == the literal snn.cpp formula, to < 1e-10.

    Driven by the neighbour lists derived from R's own ``09_pca`` embeddings
    (first 10 PCs, k = 20), so this is the real fixture, not a toy.
    """
    emb, _, _ = _load(CI_DIR)
    _, idx = cKDTree(emb).query(emb, k=K_PARAM)
    nn_ranked = np.asarray(idx, dtype=np.int64) + 1

    got = compute_snn(nn_ranked, prune=PRUNE_SNN).toarray()
    want = _dense_snn_reference(nn_ranked, PRUNE_SNN)
    assert np.abs(got - want).max() < 1e-10
    # Structural invariants of Seurat's SNN.
    assert np.allclose(np.diag(got), 1.0)
    assert np.abs(got - got.T).max() < 1e-10
    assert got[got > 0].min() >= PRUNE_SNN


@pytest.mark.parametrize(
    "outdir,expected_identical_fraction,expected_max_abs_delta",
    [
        (CI_DIR, 0.905732, 0.25),
        (FULL_DIR, 0.968828, 0.14285714285714285),
    ],
)
def test_snn_vs_r_reference_graph(
    outdir, expected_identical_fraction, expected_max_abs_delta
):
    """Exact-kNN SNN vs R's ``RNA_snn``.

    # PARITY NOTE: this is the ONE place in the module that is not bit-exact
    # with R, and the cause is entirely the nearest-neighbour *search*, not
    # ``compute_snn``.  Seurat 5 defaults to ``nn.method = "annoy"`` (an
    # approximate index, ``n.trees = 50``); ``find_neighbors`` here uses an
    # exact cKDTree, because Annoy's tree construction is not reproducible
    # from Python.
    #
    # Measured, offline, against R 4.4.3 / Seurat 5.4.0 (see the agent report;
    # the R-side neighbour matrix is not committed to this repo):
    #   * feeding ``compute_snn`` R's OWN Annoy index matrix
    #     (``Seurat:::Indices(NNHelper(pca[, 1:10], k = 20, method = "annoy"))``)
    #     reproduces ``10_snn`` with max|delta| = 0.0 EXACTLY, identical nnz and
    #     0 differing stored entries -- 16323/16323 (ci), 142399/142399 (full).
    #     So the SNN arithmetic itself is bit-exact.
    #   * exact kNN vs Annoy neighbour SETS: identical for 264/297 = 88.89% of
    #     cells (ci) and 2036/2133 = 95.45% (full); shared (cell, neighbour)
    #     pairs 5888/5940 = 99.12% (ci) and 42528/42660 = 99.69% (full).
    #   * the exact kNN is never worse: the summed neighbour distance of the
    #     Annoy set is >= that of the exact set for 297/297 and 2133/2133
    #     cells, i.e. the residual is Annoy's approximation error, not ours.
    #   * ``Seurat:::NNHelper`` with Annoy was verified to be deterministic
    #     run-to-run, so this is a stable, not a stochastic, discrepancy.
    """
    emb, snn_r, _ = _load(outdir)
    _, snn_py = find_neighbors(emb, k_param=K_PARAM, prune_snn=PRUNE_SNN)

    delta = (snn_py - snn_r).tocsr()
    differing = (abs(delta) > 1e-10).nnz
    support = (
        _ones_like(snn_py) + _ones_like(snn_r)
    ).nnz  # size of the union of stored patterns
    identical_fraction = 1.0 - differing / support
    max_abs_delta = float(abs(delta).max()) if delta.nnz else 0.0

    assert identical_fraction == pytest.approx(expected_identical_fraction, abs=1e-6)
    assert max_abs_delta == pytest.approx(expected_max_abs_delta, abs=1e-9)


def _ones_like(m):
    out = sp.csr_matrix(m).copy()
    out.data = np.ones_like(out.data)
    return out


# ---------------------------------------------------------------------------
# 2. Louvain on R's own SNN graph -- the headline parity claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outdir", [CI_DIR, FULL_DIR])
def test_louvain_exact_labels_on_r_snn_graph(outdir):
    """``find_clusters`` on R's ``10_snn`` reproduces ``seurat_clusters`` exactly.

    Not merely ARI = 1.0 (equality up to permutation) -- the integer labels are
    identical element-for-element, because ``orderClustersByNNodes`` fixes the
    numbering by decreasing cluster size.
    """
    _, snn_r, ref_ids = _load(outdir)
    ids = find_clusters(
        snn_r,
        resolution=RESOLUTION,
        algorithm=1,
        n_start=10,
        n_iter=10,
        random_seed=0,
        group_singletons=True,
    )
    assert adjusted_rand_score(ref_ids, ids) == 1.0
    assert np.array_equal(ids, ref_ids)


def test_run_modularity_clustering_labels_ordered_by_size():
    """``orderClustersByNNodes``: label 0 is the largest cluster, and so on."""
    _, snn_r, _ = _load(CI_DIR)
    ids = run_modularity_clustering(snn_r, resolution=RESOLUTION, random_seed=0)
    _, counts = np.unique(ids, return_counts=True)
    assert list(counts) == sorted(counts, reverse=True)
    assert ids.min() == 0
    assert set(np.unique(ids).tolist()) == set(range(ids.max() + 1))


def test_random_seed_changes_nothing_but_is_honoured():
    """A different ``random_seed`` drives a different JavaRandom stream.

    Same seed -> byte-identical labels (determinism); the clustering itself may
    or may not change with the seed, so only determinism is asserted.
    """
    _, snn_r, _ = _load(CI_DIR)
    a = run_modularity_clustering(snn_r, resolution=RESOLUTION, random_seed=0)
    b = run_modularity_clustering(snn_r, resolution=RESOLUTION, random_seed=0)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. End to end: find_neighbors -> find_clusters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outdir,expected_ari",
    [
        (CI_DIR, 0.9416923768912725),
        (FULL_DIR, 0.9432103865000975),
    ],
)
def test_end_to_end_ari(outdir, expected_ari):
    """PCA -> find_neighbors -> find_clusters vs R's ``seurat_clusters``.

    # PARITY NOTE: this is < 1.0 for exactly one reason -- the exact-kNN vs
    # Annoy deviation documented in ``test_snn_vs_r_reference_graph``.  Given
    # R's own SNN graph the labels are element-wise identical
    # (``test_louvain_exact_labels_on_r_snn_graph``), so every point of ARI lost
    # here is lost in the neighbour search, not in the modularity optimiser.
    # Measured: ARI = 0.94169 (ci, 297 spots, 4 clusters in both) and
    # ARI = 0.94321 (full, 2133 spots, 8 clusters in both).
    """
    emb, _, ref_ids = _load(outdir)
    _, snn_py = find_neighbors(emb, k_param=K_PARAM, prune_snn=PRUNE_SNN)
    ids = find_clusters(
        snn_py,
        resolution=RESOLUTION,
        algorithm=1,
        n_start=10,
        n_iter=10,
        random_seed=0,
    )
    ari = adjusted_rand_score(ref_ids, ids)
    assert ari == pytest.approx(expected_ari, abs=1e-9)
    # Same number of communities as R, which is the qualitative claim that
    # matters downstream in SpatialEcoTyper.
    assert len(np.unique(ids)) == len(np.unique(ref_ids))


def test_find_neighbors_graph_shapes():
    emb, snn_r, _ = _load(CI_DIR)
    nn_graph, snn_graph = find_neighbors(emb, k_param=K_PARAM, prune_snn=PRUNE_SNN)
    n = emb.shape[0]
    assert nn_graph.shape == (n, n) == snn_graph.shape == snn_r.shape
    # Every row of the binary kNN graph holds exactly k.param ones, incl. self.
    assert np.array_equal(np.asarray(nn_graph.sum(axis=1)).ravel(),
                          np.full(n, K_PARAM, dtype=np.float64))
    assert np.allclose(nn_graph.diagonal(), 1.0)


# ---------------------------------------------------------------------------
# GroupSingletons
# ---------------------------------------------------------------------------


def test_group_singletons_absorbs_into_most_connected_cluster():
    """clustering.R:1356-1404 -- mean SNN weight decides the absorbing cluster."""
    # Nodes 0-2 = cluster 0, nodes 3-5 = cluster 1, node 6 = a singleton that is
    # more strongly tied to cluster 1.
    n = 7
    w = np.zeros((n, n))
    for a, b, v in [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
                    (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0),
                    (6, 0, 0.1), (6, 3, 0.9), (6, 4, 0.9)]:
        w[a, b] = w[b, a] = v
    np.fill_diagonal(w, 1.0)
    snn = sp.csr_matrix(w)
    ids = np.array([0, 0, 0, 1, 1, 1, 2])
    out = group_singletons(ids, snn, group_singletons=True)
    assert out[6] == 1
    assert np.array_equal(out[:6], ids[:6])

    # group.singletons = FALSE -> R writes the string "singleton"; we use -1.
    out2 = group_singletons(ids, snn, group_singletons=False)
    assert out2[6] == -1


def test_group_singletons_noop_without_singletons():
    _, snn_r, ref_ids = _load(CI_DIR)
    out = group_singletons(ref_ids, snn_r)
    assert np.array_equal(out, ref_ids)


# ---------------------------------------------------------------------------
# Argument validation, mirroring RModularityOptimizer.cpp:33-43
# ---------------------------------------------------------------------------


def test_argument_validation():
    _, snn_r, _ = _load(CI_DIR)
    with pytest.raises(ValueError, match="Modularity parameter"):
        run_modularity_clustering(snn_r, modularity=3)
    with pytest.raises(ValueError, match="Algorithm"):
        run_modularity_clustering(snn_r, algorithm=9)
    with pytest.raises(ValueError, match="at least one start"):
        run_modularity_clustering(snn_r, n_start=0)
    with pytest.raises(ValueError, match="at least one interation"):
        run_modularity_clustering(snn_r, n_iter=0)
    with pytest.raises(ValueError, match="resolution"):
        run_modularity_clustering(snn_r, modularity=2, resolution=1.5)
