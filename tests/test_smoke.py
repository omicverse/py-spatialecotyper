"""Fastest test: does the package import, expose its API, and run at all.

Needs neither R nor the reference dumps, so it is the first thing to check on a
fresh machine.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspatialecotyper as P  # noqa: E402


def test_version_and_exports():
    assert P.__version__ == "0.1.0"
    missing = [n for n in P.__all__ if not hasattr(P, n)]
    assert not missing, f"__all__ names with no attribute: {missing}"
    assert len(P.R_FUNCTION_MAP) >= 45
    unresolved = [r for r, py in P.R_FUNCTION_MAP.items()
                  if "." not in py and not hasattr(P, py)]
    assert not unresolved, f"R_FUNCTION_MAP targets missing from the API: {unresolved}"


def _toy(n_cells=600, n_genes=60, seed=0):
    """A small, seeded, spatially structured toy dataset.

    Four cell types on a 30 x 20 lattice, with two spatial zones that differ in
    which genes are expressed -- enough structure for the pipeline to find more
    than one ecotype.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(30) * 12.0, np.arange(20) * 12.0)
    xs, ys = xs.ravel()[:n_cells], ys.ravel()[:n_cells]
    ct = np.array(["A", "B", "C", "D"])[np.arange(n_cells) % 4]
    zone = (xs > 170).astype(int)
    counts = rng.poisson(2.0, size=(n_genes, n_cells)).astype(float)
    counts[:20, zone == 0] += 8
    counts[20:40, zone == 1] += 8
    for k, c in enumerate("ABCD"):
        counts[k::4, ct == c] += 4
    meta = pd.DataFrame({"X": xs, "Y": ys, "CellType": ct, "Zone": zone},
                        index=[f"cell{i}" for i in range(n_cells)])
    genes = [f"G{i}" for i in range(n_genes)]
    return sp.csc_matrix(counts), genes, meta


def test_pipeline_runs_end_to_end():
    counts, genes, meta = _toy()
    norm = P.normalize_data(counts)
    res = P.spatial_ecotyper(norm, meta, gene_names=genes, radius=25,
                             resolution=0.5, nfeatures=30, npcs=10,
                             min_cts_per_region=1, min_cells=2, min_features=3,
                             iterations=3, k=10, k_sn=15, verbose=False)
    assert res.fused.shape[0] == res.fused.shape[1] == len(res.spot_names)
    assert "SE" in res.spot_metadata.columns
    assert "SE" in res.metadata.columns
    assert res.spot_metadata["SE"].nunique() >= 2, "no structure recovered"
    assert res.metadata.shape[0] > 0


def test_class_api_matches_function_api():
    counts, genes, meta = _toy()
    norm = P.normalize_data(counts)
    kwargs = dict(radius=25, resolution=0.5, nfeatures=30, npcs=10,
                  min_cts_per_region=1, min_cells=2, min_features=3,
                  iterations=3, k=10, k_sn=15, verbose=False)
    a = P.spatial_ecotyper(norm, meta, gene_names=genes, **kwargs)
    b = P.SpatialEcoTyper(norm, meta, gene_names=genes, **kwargs).run()
    assert list(a.spot_metadata["SE"]) == list(b.result.spot_metadata["SE"])
    assert "spatial neighborhoods" in repr(b)


def test_r_name_aliases_are_the_same_objects():
    assert P.spatial_eco_typer is P.spatial_ecotyper
    assert P.integrate_spatial_eco_typer is P.integrate_spatial_ecotyper
    assert P.multi_spatial_eco_typer is P.multi_spatial_ecotyper
    assert P.infer_n_cells is P.infer_ncells
    assert P.nm_fpredict is P.nmf_predict
    assert P.compute_f_cs is P.compute_fcs


def test_sct_branch_is_refused_not_faked():
    with pytest.raises(NotImplementedError):
        P.integrate_spatial_ecotyper({}, {}, [], normalization_method="SCT")


def test_dominateset_reproduces_r_edge_cases():
    """R's `x[s$ix[1:(length(x) - KK)]]` has two documented edge cases."""
    m = sp.csr_matrix(np.arange(1, 26, dtype=float).reshape(5, 5))
    # n == KK: R's `1:0` is c(1, 0), index 0 is dropped -> zero the single
    # smallest element of each row, not none of them.
    out = np.asarray(P.dominateset(m, KK=5).todense())
    assert (out[:, 0] == 0).all() and (out[:, 1:] != 0).all()
    # n < KK: R aborts with a subscript error; so do we.
    with pytest.raises(ValueError, match="negative subscripts"):
        P.dominateset(m, KK=9)


def test_rrandom_streams_are_independent():
    a, b = P.RRandom(7), P.RRandom(7)
    assert np.array_equal(a.runif(5), b.runif(5))
    c = a.clone()
    assert np.array_equal(a.runif(3), c.runif(3))
