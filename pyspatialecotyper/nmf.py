"""NMF layer: model training, prediction, consensus clustering, recovery.

Upstream calls the R ``NMF`` package with two *custom* ``NMFStrategy`` objects
that Lee-Seung-update only one factor:

* ``NMFGenerateW`` fixes ``H`` (a binary SE-membership matrix) and updates only
  ``W``.  With ``H`` fixed the KL divergence ``D(V || WH)`` is convex in ``W``
  (``WH`` is linear in ``W`` and ``D`` is convex in its second argument), so the
  multiplicative updates converge to the *global* optimum and the random
  initialisation only affects how far along that path 420 iterations get you.
* ``.nmf.predict`` fixes ``W`` and updates only ``H`` -- convex in ``H`` for the
  same reason.

``nmfClustering`` is the one place a full two-factor NMF runs, and there the
random restarts are the point.

The KL update kernels come from **`nmf-rs`** (omicverse/rust-NMF), which is
bit-equivalent to R's ``std.divergence.update.{w,h}`` within 1e-12.  A NumPy
fallback is kept so the package still works without the Rust wheel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rrandom import RRandom
from .seuratcompat import scale_data
from .utils import r_order

__all__ = [
    "posneg", "nmf_generate_w", "nmf_generate_w_list", "nmf_predict",
    "nmf_clustering", "aggregate_recover_models", "recover_se",
    "deconvolute_se", "loocv_predict", "cophcor", "connectivity",
    "consensus_matrix", "nmf_rnmf_init",
]

_EPS = np.finfo(np.float64).eps

try:                                    # ecosystem reuse -- see DISCOVERY.md
    from nmf_rs import update_h_brunet as _rs_update_h
    from nmf_rs import update_w_brunet as _rs_update_w
    _HAVE_RS = True
except Exception:                       # pragma: no cover
    _HAVE_RS = False


# ---------------------------------------------------------------------------
# KL multiplicative updates
# ---------------------------------------------------------------------------
def _np_update_w_brunet(V, W, H):
    """``NMF:::std.divergence.update.w`` in pure NumPy.

    ``w_ia <- w_ia * sum_j h_aj v_ij / (WH)_ij  /  sum_j h_aj``
    """
    wh = W @ H
    np.maximum(wh, np.finfo(np.float64).tiny, out=wh)
    return W * ((V / wh) @ H.T) / H.sum(axis=1)[None, :]


def _np_update_h_brunet(V, W, H):
    """``NMF:::std.divergence.update.h`` in pure NumPy.

    ``h_aj <- h_aj * sum_i w_ia v_ij / (WH)_ij  /  sum_i w_ia``
    """
    wh = W @ H
    np.maximum(wh, np.finfo(np.float64).tiny, out=wh)
    return H * (W.T @ (V / wh)) / W.sum(axis=0)[:, None]


def _update_w(V, W, H):
    return _rs_update_w(V, W, H) if _HAVE_RS else _np_update_w_brunet(V, W, H)


def _update_h(V, W, H):
    return _rs_update_h(V, W, H) if _HAVE_RS else _np_update_h_brunet(V, W, H)


# ---------------------------------------------------------------------------
# NMF package helpers
# ---------------------------------------------------------------------------
def posneg(x: np.ndarray) -> np.ndarray:
    """``NMF::posneg`` -- ``rbind(pmax(x, 0), pmax(-x, 0))``.

    Doubles the feature axis so a signed (z-scored) matrix can be fed to a
    non-negative factorisation.  The duplicated row names are what the
    ``__pos`` / ``__neg`` suffixing downstream keys off.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.vstack([np.maximum(x, 0.0), np.maximum(-x, 0.0)])


def nmf_rnmf_init(n: int, r: int, m: int | None = None,
                  max_value: float | None = None,
                  rng: RRandom | None = None):
    """``NMF::rnmf`` random initialisation, drawn from R's own RNG stream.

    ``rmatrix(x, y, dist = runif)`` fills **column-major**.  When a target
    matrix is supplied (the ``nmf(V, rank, seed = <int>)`` path) the draws are
    scaled by ``max(V)``; when only ``H`` is fixed (the ``NMFGenerateW`` path)
    they are plain ``U(0, 1)``.  Verified against R 4.4.3 + NMF 0.28.

    Returns ``W0`` (n x r), and ``H0`` (r x m) when ``m`` is given.
    """
    rng = rng or RRandom(0)
    scale = 1.0 if max_value is None else float(max_value)
    w0 = rng.runif(n * r).reshape((n, r), order="F") * scale
    if m is None:
        return w0
    h0 = rng.runif(r * m).reshape((r, m), order="F") * scale
    return w0, h0


def connectivity(H: np.ndarray) -> np.ndarray:
    """``NMF::connectivity`` -- ``outer(argmax_col(H), argmax_col(H), ==)``."""
    idx = np.argmax(H, axis=0)
    return (idx[:, None] == idx[None, :]).astype(np.float64)


def _stop_connectivity_runner(V, W, H, update, max_iter=2000,
                              stopconv=40, check_interval=10):
    """The ``nmf.stop.connectivity`` loop from ``NMFStrategyIterative-class.R``.

    Checks every ``check_interval`` iterations; a check that leaves the
    connectivity matrix unchanged increments a counter, any change resets it,
    and the run stops once the counter **exceeds** ``stopconv``.  The initial
    ``.consold`` is the zero matrix, so the first check always counts as a
    change -- which is why a run with a *frozen* ``H`` still takes
    ``check_interval * (stopconv + 2) = 420`` iterations rather than stopping
    at the first check.
    """
    consold = np.zeros((H.shape[1], H.shape[1]))
    inc = 0
    i = 0
    while i < max_iter:
        i += 1
        W, H = update(V, W, H, i)
        if i % check_interval != 0:
            continue
        cons = connectivity(H)
        if np.array_equal(cons, consold):
            inc += 1
        else:
            consold = cons
            inc = 0
        if inc > stopconv:
            break
    return W, H, i


def _pmax_inplace(x, eps=_EPS):
    """``NMF:::pmax.inplace(x, eps, <no fixed terms>)``."""
    return np.maximum(x, eps)


def consensus_matrix(H_list) -> np.ndarray:
    """Mean connectivity matrix over runs -- ``NMF::consensus``."""
    acc = None
    for H in H_list:
        c = connectivity(H)
        acc = c if acc is None else acc + c
    return acc / len(H_list)


def cophcor(cons: np.ndarray) -> float:
    """``NMF::cophcor`` -- cophenetic correlation of the consensus matrix.

    R: ``cophenetic(hclust(as.dist(1 - cons), method = "average"))`` correlated
    against the original ``1 - cons`` distances (Pearson).
    """
    from scipy.cluster.hierarchy import average, cophenet
    from scipy.spatial.distance import squareform
    d = squareform(1.0 - cons, checks=False)
    z = average(d)
    coph = cophenet(z)
    return float(np.corrcoef(d, coph)[0, 1])


# ---------------------------------------------------------------------------
# NMFGenerateW
# ---------------------------------------------------------------------------
def nmf_generate_w(Fracs, ExpMat, feature_names=None, sample_names=None,
                   se_names=None, scale: bool = True,
                   nfeature: int = 300, nfeature_per_se: int = 50,
                   method: str = "brunet", rng: RRandom | None = None,
                   max_iter: int = 2000):
    """``NMFGenerateW(Fracs, ExpMat, scale, nfeature, nfeature.per.se, method)``.

    Trains the SE-deconvolution basis matrix ``W``: rows are (signed) features,
    columns are spatial ecotypes.

    Parameters
    ----------
    Fracs
        samples x SE fraction matrix (R transposes it internally).
    ExpMat
        genes x samples expression matrix.
    scale
        z-score each gene before factorising (R default ``TRUE``).
    nfeature
        Cap on the number of top-variance genes carried into the NMF.
    nfeature_per_se
        Cap on the number of features retained per SE after the ``delta``
        specificity filter.

    Returns
    -------
    (W, rownames, colnames)

    Feature selection (R lines 239-252) uses ``delta = W + apply(W, 1, function(x) sort(-x)[2])``,
    i.e. each entry minus the **second largest** value in its row -- positive
    only for the SE where that feature is most specific.
    """
    ExpMat = np.asarray(ExpMat, dtype=np.float64)
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(ExpMat.shape[0])]
    feature_names = list(map(str, feature_names))

    if ExpMat.min() >= 0 and ExpMat.max() > 80:
        ExpMat = np.log2(ExpMat + 1)

    if ExpMat.shape[0] > nfeature:
        v = ExpMat.var(axis=1, ddof=1)
        keep = r_order(-v)[:min(nfeature, ExpMat.shape[0])]
        keep = np.sort(keep)
        ExpMat = ExpMat[keep]
        feature_names = [feature_names[i] for i in keep]

    if scale:
        ExpMat = scale_data(ExpMat)
    to_predict = np.nan_to_num(ExpMat, nan=0.0)
    rn = list(feature_names)
    if (to_predict < 0).sum() > 0:
        to_predict = posneg(to_predict)
        rn = list(feature_names) + list(feature_names)
    to_predict = np.nan_to_num(to_predict, nan=0.0)

    keep = to_predict.var(axis=1, ddof=1) > 0
    to_predict = to_predict[keep]
    rn = [f for f, k in zip(rn, keep) if k]

    FracsF = np.asarray(Fracs, dtype=np.float64).T      # t(Fracs): SE x samples
    if se_names is None:
        se_names = [f"SE{i + 1}" for i in range(FracsF.shape[0])]
    rank = FracsF.shape[0]

    # dummy <- NMF::rnmf(nrow(to_predict), H = FracsF): W random U(0,1), H fixed
    W0 = nmf_rnmf_init(to_predict.shape[0], rank, rng=rng)
    W0 = np.nan_to_num(W0, nan=0.0)
    H0 = np.nan_to_num(FracsF, nan=0.0) + _EPS          # dummyHF + .Machine$double.eps

    def _upd(V, W, H, i):
        W = _update_w(V, W, H)
        if i % 10 == 0:
            W = _pmax_inplace(W)
        return W, H                                      # H is FIXED

    W, _, _ = _stop_connectivity_runner(to_predict, W0, H0, _upd, max_iter=max_iter)
    W = np.nan_to_num(W, nan=0.0)

    # duplicated rownames -> __pos / __neg
    seen, names = set(), []
    for f in rn:
        if f in seen:
            names.append(f + "__neg")
        else:
            seen.add(f)
            names.append(f + "__pos")
    if len(set(rn)) == len(rn):
        names = list(rn)                                 # no duplication happened

    # Feature selection
    second = np.sort(-W, axis=1)[:, 1] if W.shape[1] > 1 else np.zeros(W.shape[0])
    delta = W + second[:, None]
    genes = set()
    for j in range(delta.shape[1]):
        order = r_order(-delta[:, j])
        pos = [i for i in order if delta[i, j] > 0]
        pos = pos[:nfeature_per_se]
        genes.update(names[i].split("__")[0] for i in pos)
    keep = np.array([n.split("__")[0] in genes for n in names])
    W = W[keep]
    names = [n for n, k in zip(names, keep) if k]
    order = r_order(-W.sum(axis=1))
    return W[order], [names[i] for i in order], list(se_names)


def nmf_generate_w_list(scdata, scmeta: pd.DataFrame, gene_names=None,
                        CellType: str = "CellType", SE: str = "SE",
                        scale: bool = True, Sample: str | None = None,
                        balance_sample: bool = True, nfeature: int = 300,
                        nfeature_per_se: int = 50, min_cells: int = 20,
                        downsample: int = 2500, ncores: int = 8,
                        seed: int = 2024, verbose: bool = True):
    """``NMFGenerateWList(scdata, scmeta, CellType, SE, ...)``.

    One ``W`` per cell type.  Cell types are kept only when they have more than
    one SE with more than ``min_cells`` cells (``count(CellType, SE) |>
    filter(n > min.cells) |> count(CellType) |> filter(n > 1)``).

    All ``sample()`` calls run on R's Mersenne-Twister via
    :mod:`pyspatialecotyper.rrandom`, so the balancing and down-sampling pick
    the same cells R would.
    """
    scdata = np.asarray(scdata, dtype=np.float64)
    scmeta = pd.DataFrame(scmeta).copy()
    scmeta["CellType"] = scmeta[CellType].to_numpy()
    scmeta["SE"] = scmeta[SE].to_numpy()
    if gene_names is None:
        gene_names = [f"F{i}" for i in range(scdata.shape[0])]
    if scmeta.shape[0] != scdata.shape[1]:
        raise ValueError("The number of rows in scmeta must equal the number of "
                         "columns in scdata.")
    if scdata.min() >= 0 and scdata.max() > 80:
        scdata = np.log2(scdata + 1)

    counts = scmeta.groupby(["CellType", "SE"], sort=True).size()
    counts = counts[counts > min_cells].reset_index().groupby("CellType").size()
    cts = sorted(counts[counts > 1].index)

    Ws = {}
    for ct in cts:
        if verbose:
            print(f"Training on {ct} cells...")
        sel = (scmeta["CellType"] == ct).to_numpy()
        tmpmeta = scmeta.loc[sel].copy()
        tmpdat = scdata[:, sel]
        names = list(gene_names)

        if Sample is None:
            if tmpdat.shape[0] > nfeature:
                v = tmpdat.var(axis=1, ddof=1)
                keep = np.sort(r_order(-v)[:min(nfeature, tmpdat.shape[0])])
                tmpdat, names = tmpdat[keep], [names[i] for i in keep]
            if scale:
                tmpdat = scale_data(tmpdat)
        else:
            tmpmeta["Sample"] = tmpmeta[Sample].to_numpy()
            sc = tmpmeta["Sample"].value_counts()
            samples = [s for s in sorted(sc.index) if sc[s] > min_cells]
            if tmpdat.shape[0] > nfeature:
                ranks = None
                for s in samples:
                    m = (tmpmeta["Sample"] == s).to_numpy()
                    from .utils import r_rank
                    rk = r_rank(tmpdat[:, m].var(axis=1, ddof=1))
                    ranks = rk if ranks is None else ranks + rk
                keep = np.sort(r_order(-ranks)[:nfeature])
                tmpdat, names = tmpdat[keep], [names[i] for i in keep]
            if scale:
                blocks, order = [], []
                for s in samples:
                    m = np.flatnonzero((tmpmeta["Sample"] == s).to_numpy())
                    blocks.append(scale_data(tmpdat[:, m]))
                    order.append(m)
                tmpdat = np.hstack(blocks)
                order = np.concatenate(order)
                tmpmeta = tmpmeta.iloc[order]
            if balance_sample:
                rng = RRandom(seed)
                sizes = tmpmeta["Sample"].value_counts()
                balancesize = max(int(np.floor(np.median(sizes.to_numpy()))), min_cells)
                blocks, order = [], []
                for s in samples:
                    idx = np.flatnonzero((tmpmeta["Sample"] == s).to_numpy())
                    if len(idx) > balancesize:
                        idx = idx[rng.sample_int(len(idx), balancesize)]
                    order.append(idx)
                order = np.concatenate(order)
                tmpdat = tmpdat[:, order]
                tmpmeta = tmpmeta.iloc[order]

        if tmpmeta.shape[0] > downsample:
            rng = RRandom(seed)
            idx = rng.sample_int(tmpmeta.shape[0], downsample)
            tmpmeta = tmpmeta.iloc[idx]
            tmpdat = tmpdat[:, idx]

        ses = list(pd.unique(tmpmeta["SE"]))
        H = np.zeros((len(ses), tmpmeta.shape[0]))
        pos = {s: i for i, s in enumerate(ses)}
        H[[pos[s] for s in tmpmeta["SE"]], np.arange(tmpmeta.shape[0])] = 1.0
        keep = H.sum(axis=1) > 2
        H, ses = H[keep], [s for s, k in zip(ses, keep) if k]
        if H.shape[0] < 2:
            continue
        Ws[ct] = nmf_generate_w(H.T, tmpdat, feature_names=names, se_names=ses,
                                scale=False, nfeature=nfeature,
                                nfeature_per_se=nfeature_per_se)
    return Ws


# ---------------------------------------------------------------------------
# NMFpredict
# ---------------------------------------------------------------------------
def _nmf_predict(W, w_names, se_names, testdat, test_genes, test_cells,
                 scale: bool = False, normalize: bool = False,
                 max_iter: int = 2000):
    """``.nmf.predict(W, testdat, scale, normalize)`` (``R/NMFpredict.R``).

    ``set.seed(39)`` at the top of the R function makes the random ``H``
    initialisation reproducible, so this path is deterministic end to end.
    """
    rng = RRandom(39)                                   # set.seed(39)
    testdat = np.asarray(testdat, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    signed = "__" in w_names[0]

    base = [n.split("__")[0] for n in w_names]
    train_set = [g for g in pd.unique(np.asarray(base, dtype=object)) if g in set(test_genes)] if signed \
        else [g for g in w_names if g in set(test_genes)]
    if len(train_set) < 5:
        return np.full((len(se_names), len(test_cells)), np.nan), list(se_names)

    gi = {g: i for i, g in enumerate(test_genes)}
    testdat = testdat[[gi[g] for g in train_set]]
    if scale:
        testdat = scale_data(testdat)
    testdat = np.nan_to_num(testdat, nan=0.0)

    if signed:
        to_predict = posneg(testdat)
        names = [g + "__pos" for g in train_set] + [g + "__neg" for g in train_set]
    else:
        to_predict = testdat
        names = list(train_set)

    keep = to_predict.var(axis=1, ddof=1) > 0
    to_predict, names = to_predict[keep], [n for n, k in zip(names, keep) if k]

    common = [f for f in names if f in set(w_names)]
    wi = {n: i for i, n in enumerate(w_names)}
    ti = {n: i for i, n in enumerate(names)}
    Wm = np.nan_to_num(W[[wi[f] for f in common]], nan=0.0)
    Vm = np.nan_to_num(to_predict[[ti[f] for f in common]], nan=0.0)

    rank = Wm.shape[1]
    # dummy <- NMF::rnmf(ncol(ws), to_predict): both factors drawn, scaled by max(V)
    _, H0 = nmf_rnmf_init(Vm.shape[0], rank, Vm.shape[1],
                          max_value=Vm.max(), rng=rng)

    def _upd(V, Wc, Hc, i):
        Hc = _update_h(V, Wc, Hc)
        if i % 10 == 0:
            Hc = _pmax_inplace(Hc)
        return Wc, Hc                                    # W is FIXED

    _, H, _ = _stop_connectivity_runner(Vm, Wm, H0, _upd, max_iter=max_iter)
    if normalize:
        H = H / H.sum(axis=0)[None, :]
    return H, list(se_names)


def nmf_predict(W, w_names, se_names, testdat, test_genes, test_cells,
                scale: bool = False, ncell_per_run: int = 500,
                sum2one: bool = True, ncores: int = 1):
    """``NMFpredict(W, testdat, scale, ncell.per.run, sum2one, ncores)``.

    Splits the cells into ``ceil(n / ncell.per.run)`` contiguous chunks whose
    sizes differ by at most 1 (the R code builds them explicitly with
    ``fold_sizes``), predicts each chunk independently, and returns a
    **cells x SE** matrix (R transposes ``H`` on the way out).
    """
    testdat = np.asarray(testdat, dtype=np.float64)
    if scale:
        ngenes = (testdat > 1e-16).sum(axis=0)
        keep = ngenes >= 3
        testdat = testdat[:, keep]
        test_cells = [c for c, k in zip(test_cells, keep) if k]
        testdat = scale_data(testdat)

    n = testdat.shape[1]
    if n > ncell_per_run:
        nfold = int(np.ceil(n / ncell_per_run))
        base, rem = divmod(n, nfold)
        sizes = np.full(nfold, base)
        sizes[:rem] += 1
        ends = np.cumsum(sizes)
        starts = ends - sizes
        blocks = []
        for a, b in zip(starts, ends):
            h, _ = _nmf_predict(W, w_names, se_names, testdat[:, a:b],
                                test_genes, test_cells[a:b],
                                scale=False, normalize=sum2one)
            blocks.append(h)
        H = np.hstack(blocks)
    else:
        H, _ = _nmf_predict(W, w_names, se_names, testdat, test_genes,
                            test_cells, scale=False, normalize=sum2one)
    return H.T, list(se_names), list(test_cells)


# ---------------------------------------------------------------------------
# nmfClustering
# ---------------------------------------------------------------------------
def nmf_clustering(mat, row_names=None, col_names=None, ranks=10,
                   nrun_per_rank: int = 30, min_coph: float = 0.95,
                   nmf_method: str = "brunet", ncores: int = 8,
                   seed: int = 2024, max_iter: int = 2000, verbose: bool = False):
    """``nmfClustering(mat, ranks, nrun.per.rank, min.coph, nmf.method, seed)``.

    Consensus NMF over ``nrun_per_rank`` restarts per rank.  The restart seeds
    are ``sample(1:6280, nrun.per.rank, replace = FALSE)`` after ``set.seed(seed)``,
    reproduced exactly through :mod:`pyspatialecotyper.rrandom`.

    ``bestK`` is the **largest** rank whose cophenetic coefficient exceeds
    ``min_coph``; if none does, the rank with the maximum coefficient.

    Returns ``dict`` with ``bestK``, ``cophenetic`` (DataFrame), ``labels``
    (consensus membership at ``bestK``), and ``fits``.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if row_names is None:
        row_names = [f"F{i}" for i in range(mat.shape[0])]
    if col_names is None:
        col_names = [f"S{i}" for i in range(mat.shape[1])]
    row_names = list(map(str, row_names))

    if mat.min() < 0:
        mat = posneg(mat)
        row_names = row_names + row_names
    mat = np.nan_to_num(mat, nan=0.0)

    seen, names = set(), []
    for f in row_names:
        if f in seen:
            names.append(f + "__neg")
        else:
            seen.add(f)
            names.append(f + "__pos")
    if len(set(row_names)) == len(row_names):
        names = list(row_names)
    keep = mat.var(axis=1, ddof=1) > 0
    mat, names = mat[keep], [n for n, k in zip(names, keep) if k]

    rng = RRandom(seed)
    seeds = rng.sample_int(6280, nrun_per_rank) + 1     # sample(1:6280, n)

    ranks = sorted(np.atleast_1d(ranks).tolist())
    fits, cophs = {}, []
    for k in ranks:
        Hs, Ws, devs = [], [], []
        for i in range(nrun_per_rank):
            r = RRandom(int(seeds[i]))                  # nmf(..., seed = seeds[i])
            W0, H0 = nmf_rnmf_init(mat.shape[0], k, mat.shape[1],
                                   max_value=mat.max(), rng=r)

            def _upd(V, W, H, it):
                H = _update_h(V, W, H)
                W = _update_w(V, W, H)
                if it % 10 == 0:
                    H = _pmax_inplace(H)
                    W = _pmax_inplace(W)
                return W, H

            W, H, _ = _stop_connectivity_runner(mat, W0, H0, _upd, max_iter=max_iter)
            Hs.append(H)
            Ws.append(W)
            devs.append(kl_deviance(mat, W, H))
        cons = consensus_matrix(Hs)
        fits[f"K.{k}"] = {"consensus": cons, "H": Hs, "W": Ws,
                          "deviance": np.asarray(devs),
                          "minfit": int(np.argmin(devs))}
        cophs.append(cophcor(cons))
        if verbose:
            print(f"  rank {k}: cophenetic {cophs[-1]:.4f}")

    cophs = pd.DataFrame({"K": ranks, "Cophenetic": cophs})
    best = cophs["K"][cophs["Cophenetic"] > min_coph]
    bestK = int(best.max()) if len(best) else int(cophs["K"][cophs["Cophenetic"].idxmax()])

    # `NMF::predict(<NMFfitX>)` defaults to what = "columns", which delegates to
    # `predict(minfit(object))` -- the argmax over the columns of the *best*
    # single run's H (lowest KL deviance), NOT a cut of the consensus tree.
    # See NMF/R/NMFSet-class.R.  Getting this wrong costs ~0.6 ARI even when
    # the consensus matrix itself is bit-identical to R's.
    fit = fits[f"K.{bestK}"]
    labels = np.argmax(fit["H"][fit["minfit"]], axis=0) + 1
    consensus_labels = predict_consensus(fit["consensus"], bestK)
    return {"bestK": bestK, "cophenetic": cophs, "labels": labels,
            "consensus_labels": consensus_labels, "fits": fits,
            "col_names": list(col_names)}


def kl_deviance(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """KL deviance ``sum(V log(V/WH) - V + WH)`` -- NMF's ``brunet`` objective.

    Used to pick ``minfit`` among the restarts, exactly as ``NMF::nmf`` does
    when it merges ``nrun`` fits into an ``NMFfitX``.
    """
    wh = W @ H
    tiny = np.finfo(np.float64).tiny
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(V > 0, V * np.log(np.maximum(V, tiny) / np.maximum(wh, tiny)), 0.0)
    return float(np.sum(term - V + wh))


def predict_consensus(cons: np.ndarray, k: int) -> np.ndarray:
    """``NMF::predict(<NMFfitX>, what = "consensus")``.

    ``cutree(hclust(as.dist(1 - consensus), method = "average"), k)``.
    """
    from scipy.cluster.hierarchy import average, fcluster
    from scipy.spatial.distance import squareform
    z = average(squareform(1.0 - cons, checks=False))
    return fcluster(z, t=k, criterion="maxclust")


# ---------------------------------------------------------------------------
# AggregateRecoverModels / RecoverSE / DeconvoluteSE / LoocvPredict
# ---------------------------------------------------------------------------
def aggregate_recover_models(model_list, delta_threshold: float = 0.01,
                             min_model_fraction: float = 0.5):
    """``AggregateRecoverModels(model_list, delta.threshold, min.model.fraction)``.

    Averages several per-cell-type ``W`` matrices, keeping only the features
    that were selected as SE-specific in more than ``min_model_fraction`` of the
    models.  The average ignores NA (``sum(x, na.rm) / sum(!is.na(x))``), so a
    feature missing from one model does not drag its mean toward zero.
    """
    out = {}
    cts = list(next(iter(model_list.values())).keys()) if isinstance(model_list, dict) \
        else list(model_list[0].keys())
    models = list(model_list.values()) if isinstance(model_list, dict) else list(model_list)

    for ct in cts:
        Ws = [m[ct] for m in models]
        rows = []
        for W, names, ses in Ws:
            second = np.sort(-W, axis=1)[:, 1] if W.shape[1] > 1 else np.zeros(W.shape[0])
            delta = W + second[:, None]
            for j, se in enumerate(ses):
                order = r_order(-delta[:, j])
                for i in order:
                    if delta[i, j] <= delta_threshold:
                        break
                    if "__pos" in names[i]:
                        rows.append((se, names[i].split("_")[0]))
        gene_df = pd.DataFrame(rows, columns=["SE", "Gene"])
        cnt = gene_df.value_counts().reset_index(name="n")
        cnt["frac"] = cnt["n"] / len(models)
        cnt = cnt[cnt["frac"] > min_model_fraction]
        features = list(pd.unique([f"{g}__pos" for g in cnt["Gene"]]
                                  + [f"{g}__neg" for g in cnt["Gene"]]))
        ses = list(pd.unique(np.concatenate([np.asarray(s, dtype=object)
                                             for _, _, s in Ws])))
        num = np.zeros((len(features), len(ses)))
        den = np.zeros((len(features), len(ses)))
        for W, names, s in Ws:
            ni = {n: i for i, n in enumerate(names)}
            si = {x: i for i, x in enumerate(s)}
            for a, f in enumerate(features):
                if f not in ni:
                    continue
                for b, se in enumerate(ses):
                    if se not in si:
                        continue
                    num[a, b] += W[ni[f], si[se]]
                    den[a, b] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ct] = (num / den, features, ses)
    return out


def recover_se(dat, gene_names, cell_names, celltypes, Ws,
               scale: bool = True, ncell_per_run: int = 500,
               min_score: float = 0.6, ncores: int = 8, verbose: bool = True):
    """``RecoverSE(dat, celltypes, scale, ncell.per.run, Ws, min.score, ncores)``.

    Predicts an SE cell state per cell using the per-cell-type NMF models, then
    hard-assigns each cell to its argmax SE and demotes low-confidence calls to
    ``"NonSE"``.  Only the ``Ws = <custom>`` branch is ported -- the default
    branch loads the MERSCOPE models shipped in the R package's ``inst/extdata``
    and applies the published ``SE01..SE11 -> SE1..SE9`` relabelling, which
    requires those ``.rds`` files.
    """
    dat = np.asarray(dat, dtype=np.float64)
    celltypes = np.asarray(celltypes, dtype=object)
    model_genes = set()
    for _, names, _ in Ws.values():
        model_genes.update(n.split("_")[0] for n in names)
    keep = np.array([g in model_genes for g in gene_names])
    dat, gene_names = dat[keep], [g for g, k in zip(gene_names, keep) if k]

    if dat.shape[1] != len(celltypes):
        raise ValueError("The length of 'celltypes' does not match the number "
                         "of columns in 'dat'.")
    ses = sorted({s for _, _, se in Ws.values() for s in se})
    cts = [c for c in pd.unique(np.asarray(celltypes, dtype=object)) if c in Ws]
    if not cts:
        raise ValueError("At least one of the following cell types must be "
                         "present for SE recovery: " + ", ".join(Ws))
    if dat.min() >= 0 and dat.max() > 80:
        dat = np.log2(dat + 1)

    rows, idx_all = [], []
    for ct in cts:
        sel = np.flatnonzero(celltypes == ct)
        if len(sel) < 20:
            continue
        if verbose:
            print(f"Recover {ct} cell states...")
        W, names, se = Ws[ct]
        H, _, cells = nmf_predict(W, names, se, dat[:, sel], gene_names,
                                  [cell_names[i] for i in sel],
                                  scale=scale, ncell_per_run=ncell_per_run,
                                  sum2one=True, ncores=ncores)
        block = np.zeros((H.shape[0], len(ses)))
        for j, s in enumerate(se):
            block[:, ses.index(s)] = H[:, j]
        rows.append(block)
        idx_all.append(sel)

    res = np.vstack(rows) if rows else np.zeros((0, len(ses)))
    order = np.concatenate(idx_all) if idx_all else np.array([], dtype=int)
    full = np.zeros((len(cell_names), len(ses)))
    full[order] = np.nan_to_num(res, nan=0.0)

    best = np.argmax(full, axis=1)
    score = full.max(axis=1)
    se_lab = np.array([ses[b] for b in best], dtype=object)
    se_lab[score < min_score] = "NonSE"
    se_lab[score == 0] = "NonSE"
    return pd.DataFrame({"CID": list(cell_names), "CellType": celltypes,
                         "InitSE": [ses[b] for b in best], "SE": se_lab,
                         "PredScore": score})


def deconvolute_se(dat, gene_names, sample_names, W, w_names, se_names,
                   scale: bool = True, nsample_per_run: int = 500,
                   sum2one: bool = True, ncores: int = 8):
    """``DeconvoluteSE(dat, scale, W, nsample.per.run, sum2one, ncores)``.

    Bulk deconvolution: returns a samples x SE fraction matrix.
    """
    dat = np.asarray(dat, dtype=np.float64)
    base = {n.split("_")[0] for n in w_names}
    keep = np.array([g in base for g in gene_names])
    dat, gene_names = dat[keep], [g for g, k in zip(gene_names, keep) if k]
    if dat.min() >= 0 and dat.max() > 80:
        dat = np.log2(dat + 1)
    if scale:
        dat = scale_data(dat)
    return nmf_predict(W, w_names, se_names, dat, gene_names, sample_names,
                       scale=False, ncell_per_run=nsample_per_run,
                       sum2one=sum2one, ncores=ncores)


def loocv_predict(scdata, scmeta, gene_names, Sample: str = "Sample",
                  CellType: str = "CellType", SE: str = "SE",
                  repeats: int = 30, ncores: int = 4, scale: bool = True,
                  verbose: bool = True, seed: int | None = None, **kwargs):
    """``LoocvPredict(scdata, scmeta, Sample, CellType, SE, repeats, ...)``.

    Leave-one-sample-out cross-validation of the SE recovery models.  With a
    single sample R instead splits each cell type in half at random.

    Note R's ``LoocvPredict`` calls ``sample(1:10000, repeats)`` **without**
    setting a seed first, so the repeat seeds inherit whatever RNG state the
    caller left behind; pass ``seed=`` here to make it reproducible.
    """
    scmeta = pd.DataFrame(scmeta).copy()
    rng = RRandom(seed) if seed is not None else RRandom(0)
    seeds = rng.sample_int(10000, repeats) + 1

    preds = []
    for ii in range(repeats):
        r = RRandom(int(seeds[ii]))
        if scmeta[Sample].nunique() < 2:
            split = np.empty(scmeta.shape[0], dtype=object)
            for ct in sorted(scmeta[CellType].unique()):
                idx = np.flatnonzero((scmeta[CellType] == ct).to_numpy())
                lab = np.array(["train"] * (len(idx) // 2)
                               + ["test"] * (len(idx) - len(idx) // 2), dtype=object)
                split[idx] = r.sample(lab, len(lab))
            scmeta = scmeta.assign(Split=split)
        else:
            scmeta = scmeta.assign(Split=scmeta[Sample].to_numpy())

        for ss in pd.unique(scmeta["Split"]):
            tr = (scmeta["Split"] == ss).to_numpy()
            Ws = nmf_generate_w_list(scdata[:, tr], scmeta.loc[tr],
                                     gene_names=gene_names, CellType=CellType,
                                     SE=SE, scale=scale, Sample="Split",
                                     seed=int(seeds[ii]), verbose=verbose, **kwargs)
            te = ~tr
            p = recover_se(scdata[:, te], gene_names,
                           list(scmeta.index[te]),
                           scmeta.loc[te, CellType].to_numpy(),
                           Ws, scale=scale, ncell_per_run=5000,
                           min_score=0.0, verbose=verbose)
            preds.append(p)

    allp = pd.concat(preds)
    top = (allp.value_counts(["CID", "SE"]).reset_index(name="n")
           .sort_values("n", ascending=False, kind="stable")
           .drop_duplicates("CID"))
    m = dict(zip(top["CID"], top["SE"]))
    scmeta = scmeta.copy()
    scmeta["cvPred"] = [m.get(c) for c in scmeta.index]
    return scmeta
