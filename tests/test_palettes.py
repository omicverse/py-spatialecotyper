"""Byte-comparison of :mod:`pyspatialecotyper.palettes` against real ``pals``.

``reference_out/palettes.json`` holds the literal output of the R
``getColors`` from ``SpatialEcoTyper-ref/R/util.R`` backed by ``pals`` 1.10.
Regenerate it with the R interpreter that has ``pals`` installed::

    Rscript -e '
      suppressMessages(library(pals)); suppressMessages(library(jsonlite))
      source("SpatialEcoTyper-ref/R/util.R")
      cases <- list()
      add <- function(n, palette, categoric, exclude = NULL){
        val <- getColors(n, palette = palette, categoric = categoric, exclude = exclude)
        cases[[length(cases)+1]] <<- list(n = n, palette = palette,
          categoric = categoric,
          exclude = if (is.null(exclude)) list() else as.list(exclude),
          expected = as.list(unname(val)))
      }
      # ... the grid described in test_palette_case_matches_r ...
      write_json(cases, "reference_out/palettes.json", auto_unbox = TRUE)
    '

The tests skip cleanly when the JSON is absent so the suite still runs on a
machine without R.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from pyspatialecotyper.palettes import (
    brewer_dark2,
    brewer_set1,
    brewer_set2,
    color_ramp_palette,
    get_colors,
    kelly,
    viridis,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_REF = _REPO / "reference_out" / "palettes.json"
_TMP = pathlib.Path("/scratch/users/steorra/tmp")

os.environ.setdefault("MPLCONFIGDIR", "/scratch/users/steorra/.cache/matplotlib")


def _load_reference():
    if not _REF.exists():
        return None
    with _REF.open() as fh:
        return json.load(fh)


_CASES = _load_reference()


def _as_list(value):
    """``write_json(auto_unbox = TRUE)`` collapses length-1 vectors."""
    if value is None:
        return [None]
    return value if isinstance(value, list) else [value]


def _case_id(case) -> str:
    kind = "cat" if case["categoric"] else "cont"
    ex = len(_as_list(case["exclude"])) if case["exclude"] else 0
    return "%s-p%s-n%s-ex%s" % (kind, case["palette"], case["n"], ex)


# ---------------------------------------------------------------------------
# getColors vs pals
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_CASES is None, reason="reference_out/palettes.json not generated")
@pytest.mark.parametrize("case", _CASES or [], ids=[_case_id(c) for c in (_CASES or [])])
def test_palette_case_matches_r(case):
    """Every colour string must be identical to what ``pals`` produced.

    The grid covers categorical palettes 1..7 (plus the ``else`` arm) at sizes
    below, at and above the palette length -- so the
    ``set.seed(N); sample(..., replace = TRUE)`` top-up branch of
    ``util.R`` lines 87-89 is exercised -- six ``exclude`` sets, and all 14
    continuous palettes at n = 1, 2, 3, 5, 11, 25, 100, 256.
    """
    exclude = _as_list(case["exclude"]) if case["exclude"] else None
    if exclude is not None and len(exclude) == 0:
        exclude = None
    got = get_colors(case["n"], palette=case["palette"],
                     categoric=case["categoric"], exclude=exclude)
    assert got == _as_list(case["expected"])


@pytest.mark.skipif(_CASES is None, reason="reference_out/palettes.json not generated")
def test_reference_covers_every_palette():
    """Guard against a reference file that quietly lost coverage."""
    cat = {c["palette"] for c in _CASES if c["categoric"]}
    cont = {c["palette"] for c in _CASES if not c["categoric"]}
    assert set(range(1, 8)).issubset(cat)
    assert set(range(1, 15)) == cont
    topped_up = [c for c in _CASES
                 if c["categoric"] and len(_as_list(c["expected"])) == c["n"]
                 and c["n"] > 36]
    assert topped_up, "no case exercises the sample() top-up branch"


# ---------------------------------------------------------------------------
# palette internals
# ---------------------------------------------------------------------------

def test_color_ramp_palette_truncates_like_r():
    """``rgb(maxColorValue = 255)`` truncates rather than rounds.

    R 4.4.3: ``colorRampPalette(c("#000000", "#FFFFFF"))(7)`` -> the second
    entry is 255/6 = 42.5 which becomes 42 (0x2A), and the fifth is 212.5
    which becomes 212 (0xD4).  Rounding would give 0x2B / 0xD5.
    """
    assert color_ramp_palette(["#000000", "#FFFFFF"])(7) == [
        "#000000", "#2A2A2A", "#555555", "#7F7F7F", "#AAAAAA", "#D4D4D4",
        "#FFFFFF"]
    assert color_ramp_palette(["#000000", "#FFFFFF"])(1) == ["#000000"]


def test_brewer_ramps_past_its_table():
    assert len(brewer_set1(8)) == 8
    assert len(brewer_set2(8)) == 8
    assert brewer_dark2(8) != brewer_dark2(12)[:8]   # 12 goes through colorRampPalette
    assert len(brewer_dark2(12)) == 12


def test_kelly_is_capped_at_22():
    assert len(kelly(50)) == 22


def test_continuous_lengths():
    for n in (1, 2, 7, 64):
        assert len(viridis(n)) == n
        assert len(get_colors(n, palette=5, categoric=False)) == n


def test_bad_continuous_palette_raises():
    with pytest.raises(ValueError):
        get_colors(5, palette=99, categoric=False)


# ---------------------------------------------------------------------------
# plotting smoke tests
# ---------------------------------------------------------------------------

_TMP.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    d = _TMP / "pyspatialecotyper_plot_tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path):
    fig.savefig(path, dpi=100)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path.stat().st_size


def _toy_meta(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0, 100, n),
        "Y": rng.uniform(0, 100, n),
        "CellType": rng.choice(["CD4T", "CD8T", "B", "Mac", "Fib"], n),
        "Score": rng.normal(size=n),
    }, index=["Cell%d" % i for i in range(n)])


def _toy_matrix(nr=12, nc=12, seed=1):
    rng = np.random.default_rng(seed)
    labels = ["SE%d_%s" % (1 + i // 4, ["CD4T", "CD8T", "B", "Mac"][i % 4])
              for i in range(nr)]
    return pd.DataFrame(rng.uniform(0, 1.2, (nr, nc)),
                        index=labels, columns=labels[:nc])


def test_spatial_view_smoke(outdir):
    meta = _toy_meta()
    fig, ax = spatial_view_import()(meta, by="CellType", x="X", y="Y",
                                    coord_fix=True, pt_size=1.0)
    assert _save(fig, outdir / "spatial_view.png") > 5_000


def test_spatial_view_numeric_and_highlight(outdir):
    meta = _toy_meta()
    sv = spatial_view_import()
    fig, _ = sv(meta, by="Score", x="X", y="Y")
    assert _save(fig, outdir / "spatial_view_numeric.png") > 5_000
    fig, _ = sv(meta, by="CellType", x="X", y="Y", jitter=True,
                highlight_cells=list(meta.index[:120]), bg_downsample=50)
    assert _save(fig, outdir / "spatial_view_highlight.png") > 5_000


def test_heatmap_view_smoke(outdir):
    from pyspatialecotyper.plotting import heatmap_view
    mat = _toy_matrix()
    rowann = pd.DataFrame({"Group": ["a"] * 6 + ["b"] * 6,
                           "index": np.arange(12)}, index=mat.index)
    fig = heatmap_view(mat, left_ann=rowann, top_ann=rowann,
                       cluster_rows=True, cluster_cols=True,
                       column_names_rot=90)
    assert fig.heatmap_view_info["name"] == "hmap"
    assert sum(len(s) for s in fig.heatmap_view_info["row_order"]) == 12
    assert _save(fig, outdir / "heatmap_view.png") > 5_000


def test_heatmap_view_split(outdir):
    from pyspatialecotyper.plotting import heatmap_view
    mat = _toy_matrix()
    split = ["a"] * 6 + ["b"] * 6
    fig = heatmap_view(mat, row_split=split, column_split=split,
                       breaks=(0, 0.6, 1.2))
    assert len(fig.heatmap_view_info["row_order"]) == 2
    assert _save(fig, outdir / "heatmap_view_split.png") > 5_000


def test_draw_rectangle_annotation_smoke(outdir):
    from pyspatialecotyper.plotting import draw_rectangle_annotation, heatmap_view
    mat = _toy_matrix()
    labels = [v.split("_")[0] for v in mat.index]
    fig = heatmap_view(mat, show_row_names=True, show_column_names=True)
    out = draw_rectangle_annotation(fig, rows=labels, columns=labels, col="black")
    assert out is fig
    assert _save(fig, outdir / "draw_rectangle_annotation.png") > 5_000


def test_cooccurrence_heatmap_view_smoke(outdir):
    from pyspatialecotyper.plotting import cooccurrence_heatmap_view
    mat = _toy_matrix()
    mat = (mat + mat.T) / 2
    pvals = {"SE1": 0.01, "SE2": 0.2, "SE3": 0.0005}
    fig = cooccurrence_heatmap_view(mat, pval=pvals)
    assert _save(fig, outdir / "cooccurrence_heatmap.png") > 5_000
    fig = cooccurrence_heatmap_view(mat)
    assert _save(fig, outdir / "cooccurrence_heatmap_nopval.png") > 5_000


@pytest.mark.parametrize("breaks,colors,values,expected", [
    # circlize 0.4.16 under R 4.4.3:
    #   f = colorRamp2(c(0, 0.6, 1.2), c("#ffffd9", "#edf8b1", "#225ea8"))
    #   f(c(-1, 0, 0.15, 0.3, 0.6, 0.75, 0.9, 1.2, 5))
    ([0, 0.6, 1.2], ["#ffffd9", "#edf8b1", "#225ea8"],
     [-1, 0, 0.15, 0.3, 0.6, 0.75, 0.9, 1.2, 5],
     ["#FFFFD9", "#FFFFD9", "#FBFDCF", "#F6FCC5", "#EDF8B1", "#C3CFB1",
      "#98A7AF", "#225EA8", "#225EA8"]),
    # ...and the blue/white/red ramp heatmap_annotation builds for numeric
    # annotation columns (HeatmapView.R line 195).
    ([1, 5, 9], ["blue", "white", "red"],
     [0, 1, 2.5, 5, 7, 9, 12],
     ["#0000FF", "#0000FF", "#9B6FFF", "#FFFFFF", "#FF9E81", "#FF0000",
      "#FF0000"]),
])
def test_color_ramp2_matches_circlize(breaks, colors, values, expected):
    """LAB interpolation plus clamping, byte-compared to ``circlize``."""
    from pyspatialecotyper.plotting import color_ramp2
    ramp = color_ramp2(breaks, colors)
    assert list(ramp.to_hex(np.asarray(values, dtype=float))) == expected


def spatial_view_import():
    from pyspatialecotyper.plotting import spatial_view
    return spatial_view
