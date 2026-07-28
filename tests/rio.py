"""Readers for the artefacts written by the R reference drivers.

The drivers dump sparse matrices as MatrixMarket + two dimname files, dense
matrices as gzipped TSV + two dimname files, data.frames as TSV with a
``.rowname`` first column, and scalars/lists as JSON.  Everything here is a
test-side helper; nothing in ``pyspatialecotyper`` imports it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _names(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh]


def read_sparse(outdir: str, name: str) -> pd.DataFrame:
    """Read ``<name>.mtx`` + dimnames into a sparse-backed DataFrame."""
    m = scipy.io.mmread(os.path.join(outdir, f"{name}.mtx")).tocsr()
    rows = _names(os.path.join(outdir, f"{name}.rows.txt"))
    cols = _names(os.path.join(outdir, f"{name}.cols.txt"))
    return m, rows, cols


def read_sparse_raw(outdir: str, name: str):
    m = scipy.io.mmread(os.path.join(outdir, f"{name}.mtx")).tocsr()
    return m


def read_dense(outdir: str, name: str) -> pd.DataFrame:
    p = os.path.join(outdir, f"{name}.tsv.gz")
    df = pd.read_csv(p, sep="\t")
    rows = _names(os.path.join(outdir, f"{name}.rows.txt"))
    cols = _names(os.path.join(outdir, f"{name}.cols.txt"))
    if rows is not None and len(rows) == df.shape[0]:
        df.index = rows
    if cols is not None and len(cols) == df.shape[1]:
        df.columns = cols
    return df


def read_df(outdir: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(outdir, f"{name}.tsv"), sep="\t",
                     dtype={".rowname": str})
    if ".rowname" in df.columns:
        df = df.set_index(".rowname")
        df.index.name = None
    return df


def read_json(outdir: str, name: str):
    with open(os.path.join(outdir, f"{name}.json")) as fh:
        return json.load(fh)


def read_lines(outdir: str, name: str):
    return _names(os.path.join(outdir, f"{name}.txt"))


def align(ref_index, cand_index):
    """Return the positional mapping that puts ``cand`` in ``ref`` order."""
    pos = {k: i for i, k in enumerate(cand_index)}
    return np.array([pos[k] for k in ref_index], dtype=np.int64)
