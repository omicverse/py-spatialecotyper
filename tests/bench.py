"""Benchmark harness for the Acceleration loop.

Times the single-sample pipeline on the canonical fixture and re-checks the
parity gate in the same process, so a rewrite can never be accepted on speed
alone.  Warmup run is discarded; 3 timed runs are reported as mean +/- stddev.

    python tests/bench.py reference_out/ci [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import rio  # noqa: E402
from pyspatialecotyper import core, network  # noqa: E402


def load(outdir):
    p = rio.read_json(outdir, "params")
    norm = rio.read_sparse_raw(outdir, "00_normdata").tocsc()
    genes = rio.read_lines(outdir, "00_normdata.rows")
    meta = rio.read_df(outdir, "00_scmeta")
    ref = rio.read_json(outdir, "12_se_spot")
    cts = rio.read_lines(outdir, "05_sn_celltypes")
    nets = [rio.read_sparse_raw(outdir, f"05_sn_{ct}").tocsr() for ct in cts]
    fused_ref = rio.read_sparse_raw(outdir, "06_snf_fused").tocsr()
    return p, norm, genes, meta, ref, nets, fused_ref


def run_pipeline(p, norm, genes, meta):
    return core.spatial_ecotyper(
        norm, meta, gene_names=genes, radius=p["radius"],
        resolution=p["resolution"], nfeatures=p["nfeatures"],
        min_cts_per_region=p["min.cts.per.region"], npcs=p["npcs"],
        min_cells=p["min.cells"], min_features=p["min.features"],
        iterations=p["iterations"], k=p["k"], k_sn=p["k.sn"], verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="reference_out/ci")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from sklearn.metrics import adjusted_rand_score
    p, norm, genes, meta, ref, nets, fused_ref = load(args.outdir)

    t0 = time.perf_counter()
    res = run_pipeline(p, norm, genes, meta)
    warmup = time.perf_counter() - t0

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        res = run_pipeline(p, norm, genes, meta)
        times.append(time.perf_counter() - t0)

    py = dict(zip(res.spot_metadata.index, res.spot_metadata["SE"]))
    common = [s for s in ref["spot"] if s in py]
    rmap = dict(zip(ref["spot"], ref["SE"]))
    ari = adjusted_rand_score([rmap[s] for s in common], [py[s] for s in common])

    # isolated SNF2 timing + its own exactness check, since SNF2 dominates
    t0 = time.perf_counter()
    fused = network.snf2(nets, t=p["iterations"], minibatch=p["minibatch"],
                         verbose=False)
    snf_t = time.perf_counter() - t0
    snf_err = float(np.max(np.abs(np.asarray((fused - fused_ref).todense()))))

    out = {
        "label": args.label,
        "warmup_run_s": round(warmup, 4),
        "wall_clock_runs_s": [round(t, 4) for t in times],
        "wall_clock_mean_s": round(float(np.mean(times)), 4),
        "wall_clock_stddev_s": round(float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 4),
        "snf2_only_s": round(snf_t, 4),
        "snf2_max_abs_err_vs_R": snf_err,
        "parity_metric": round(ari, 6),
        "parity_class": "clustering",
        "n_spots": len(common),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
