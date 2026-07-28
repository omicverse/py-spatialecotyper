"""Colour palettes, ported from ``SpatialEcoTyper-ref/R/util.R`` (``getColors``).

``getColors`` is a thin dispatcher over the R package **pals**, so this module
first reproduces the small slice of ``pals`` 1.10 that ``getColors`` touches:

* the six categorical palettes that make up ``allcolors``
  (``kelly``, ``cols25``, ``polychrome``, ``glasbey``, ``alphabet2``, ``alphabet``),
* the three ColorBrewer palettes used by ``palette = 4..7``, and
* the twelve base vectors behind the continuous palettes, which ``pals`` all
  expose as ``colorRampPalette(syspals$<name>)(n)``.

Every hex string below was dumped out of a live ``pals`` 1.10 install rather
than transcribed by hand.  The two non-obvious pieces of R semantics that have
to be reproduced exactly for the output to byte-match are:

* ``setdiff(x, y)`` — de-duplicates ``x`` *and* drops names, keeping the order
  of first appearance (``util.R`` lines 86, 89);
* the ``set.seed(N); sample(..., replace = TRUE)`` top-up branch taken when the
  caller asks for more colours than the palette holds (``util.R`` lines 87-90).
  That draw goes through :mod:`pyspatialecotyper.rrandom`, which reproduces R's
  Mersenne-Twister stream bit-for-bit, so the topped-up colours are identical
  and not merely plausible.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .rrandom import RRandom

__all__ = [
    "get_colors",
    "color_ramp_palette",
    "kelly",
    "cols25",
    "polychrome",
    "glasbey",
    "alphabet2",
    "alphabet",
    "brewer_set1",
    "brewer_set2",
    "brewer_dark2",
    "viridis",
    "parula",
    "magma",
    "coolwarm",
    "warmcool",
    "inferno",
    "plasma",
    "CATEGORIC_PALETTES",
    "CONTINUOUS_PALETTES",
]


# ---------------------------------------------------------------------------
# pals 1.10 palette data (dumped from R, not hand-typed)
# ---------------------------------------------------------------------------

_KELLY = [
    "#F2F3F4", "#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032",
    "#C2B280", "#848482", "#008856", "#E68FAC", "#0067A5", "#F99379", "#604E97",
    "#F6A600", "#B3446C", "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822",
    "#2B3D26"
]

_COLS25 = [
    "#1F78C8", "#ff0000", "#33a02c", "#6A33C2", "#ff7f00", "#565656", "#FFD700",
    "#a6cee3", "#FB6496", "#b2df8a", "#CAB2D6", "#FDBF6F", "#999999", "#EEE685",
    "#C8308C", "#FF83FA", "#C814FA", "#0000FF", "#36648B", "#00E2E5", "#00FF00",
    "#778B00", "#BEBE00", "#8B3B00", "#A52A3C"
]

_POLYCHROME = [
    "#5A5156", "#E4E1E3", "#F6222E", "#FE00FA", "#16FF32", "#3283FE", "#FEAF16",
    "#B00068", "#1CFFCE", "#90AD1C", "#2ED9FF", "#DEA0FD", "#AA0DFE", "#F8A19F",
    "#325A9B", "#C4451C", "#1C8356", "#85660D", "#B10DA1", "#FBE426", "#1CBE4F",
    "#FA0087", "#FC1CBF", "#F7E1A0", "#C075A6", "#782AB6", "#AAF400", "#BDCDFF",
    "#822E1C", "#B5EFB5", "#7ED7D1", "#1C7F93", "#D85FF7", "#683B79", "#66B0FF",
    "#3B00FB"
]

_GLASBEY = [
    "#0000FF", "#FF0000", "#00FF00", "#000033", "#FF00B6", "#005300", "#FFD300",
    "#009FFF", "#9A4D42", "#00FFBE", "#783FC1", "#1F9698", "#FFACFD", "#B1CC71",
    "#F1085C", "#FE8F42", "#DD00FF", "#201A01", "#720055", "#766C95", "#02AD24",
    "#C8FF00", "#886C00", "#FFB79F", "#858567", "#A10300", "#14F9FF", "#00479E",
    "#DC5E93", "#93D4FF", "#004CFF", "#F2F318"
]

_ALPHABET2 = [
    "#AA0DFE", "#3283FE", "#85660D", "#782AB6", "#565656", "#1C8356", "#16FF32",
    "#F7E1A0", "#E2E2E2", "#1CBE4F", "#C4451C", "#DEA0FD", "#FE00FA", "#325A9B",
    "#FEAF16", "#F8A19F", "#90AD1C", "#F6222E", "#1CFFCE", "#2ED9FF", "#B10DA1",
    "#C075A6", "#FC1CBF", "#B00068", "#FBE426", "#FA0087"
]

_ALPHABET = [
    "#F0A0FF", "#0075DC", "#993F00", "#4C005C", "#191919", "#005C31", "#2BCE48",
    "#FFCC99", "#808080", "#94FFB5", "#8F7C00", "#9DCC00", "#C20088", "#003380",
    "#FFA405", "#FFA8BB", "#426600", "#FF0010", "#5EF1F2", "#00998F", "#E0FF66",
    "#740AFF", "#990000", "#FFFF80", "#FFE100", "#FF5005"
]

_BREWER_SET1 = {
    3: ["#E41A1C", "#377EB8", "#4DAF4A"],
    4: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"],
    5: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"],
    6: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33"],
    7: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628"],
    8: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF"],
    9: ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"],
}

_BREWER_SET2 = {
    3: ["#66C2A5", "#FC8D62", "#8DA0CB"],
    4: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"],
    5: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"],
    6: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"],
    7: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494"],
    8: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
}

_BREWER_DARK2 = {
    3: ["#1B9E77", "#D95F02", "#7570B3"],
    4: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"],
    5: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"],
    6: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"],
    7: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D"],
    8: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
}

_CONT_BASE = {
    "viridis": [
        "#440154", "#482173", "#423E85", "#38588C", "#2D6F8E", "#25848E", "#1E9B89",
        "#2AB07E", "#50C469", "#85D449", "#C1DF23", "#FDE725"
    ],
    "parula": [
        "#352A87", "#343EB1", "#1558D9", "#036CE0", "#107AD9", "#1387D3", "#0997D1",
        "#06A4C9", "#0CADBB", "#23B4A9", "#43BB97", "#6ABE83", "#8FBF73", "#AEBD66",
        "#CBBB5B", "#E6B94F", "#FDBE3C", "#FBCF2C", "#F5E21E", "#F9FB0E"
    ],
    "magma": [
        "#000004", "#0C0926", "#221151", "#410F74", "#5E177F", "#7B2382", "#982D80",
        "#B63679", "#D2426E", "#EA5660", "#F8765C", "#FD9869", "#FEBA80", "#FDDB9D",
        "#FCFDBF"
    ],
    "coolwarm": [
        "#3B4CC0", "#4F6BD9", "#6788EE", "#80A3FA", "#99BAFF", "#B2CBFB", "#C9D7EF",
        "#DDDDDD", "#EDD1C1", "#F6BEA5", "#F7A788", "#F08B6D", "#E26953", "#CD433A",
        "#B40426"
    ],
    "inferno": [
        "#000004", "#0A0723", "#200C49", "#3C0964", "#560F6D", "#70196E", "#89226A",
        "#A22B61", "#BB3654", "#D04544", "#E35932", "#F1711E", "#F98C09", "#FCAA0F",
        "#F9C932", "#F2E865", "#FCFFA4"
    ],
    "plasma": [
        "#0D0887", "#4B02A1", "#7D03A8", "#A92295", "#CB4678", "#E56B5C", "#F89440",
        "#FDC327", "#F0F921"
    ],
    "kovesi.linear_bgy_10_95_c74": [
        "#000C7D", "#00108B", "#001599", "#001BA6", "#0022B0", "#002BB9", "#0034BC",
        "#003EBB", "#0048B7", "#0052AF", "#005CA6", "#00659D", "#006E92", "#007788",
        "#00817A", "#118A6B", "#219259", "#2B9B43", "#30A430", "#31AC21", "#32B51A",
        "#33BD1A", "#34C519", "#36CE19", "#39D71A", "#55DD1A", "#76E31B", "#96E81C",
        "#B5EB1D", "#D3EE1F", "#F1F021", "#FFF123"
    ],
    "kovesi.linear_bgyw_15_100_c67": [
        "#1B0084", "#1B099A", "#1C14AE", "#1C20C0", "#1F2DCE", "#233DD6", "#2950CE",
        "#2E68AB", "#377989", "#3F876A", "#46954D", "#53A036", "#65AB26", "#7BB41A",
        "#95BE16", "#AFC61C", "#C6CE26", "#DAD636", "#EAE04E", "#F1EC74", "#F8F7AF",
        "#FFFFFF"
    ],
    "kovesi.linear_blue_95_50_c20": [
        "#F1F1F1", "#C0D2EB", "#93B5DC", "#7097C1", "#3B7CB2"
    ],
    "kovesi.linear_bmw_5_95_c86": [
        "#00024B", "#040687", "#0B0BC6", "#451AF4", "#971DFE", "#D625FE", "#F957FE",
        "#FE90FD", "#FEC0FE", "#FEEBFE"
    ],
    "kovesi.linear_bmy_10_95_c78": [
        "#000C7D", "#00149C", "#2415AA", "#710B99", "#9B028E", "#C10085", "#E2037B",
        "#FC2C6E", "#FF555F", "#FF7E4B", "#FFA031", "#FFBC20", "#FFD71D", "#FFF123"
    ],
    "kovesi.linear_kry_5_95_c72": [
        "#111111", "#450D09", "#6D0203", "#900302", "#B40703", "#D81304", "#F73207",
        "#FF650E", "#FD9116", "#F5B925", "#EFDB2C", "#F7F909"
    ],
}


# ---------------------------------------------------------------------------
# R primitives that the palettes are built on
# ---------------------------------------------------------------------------

def _hex_to_rgb255(col: str) -> tuple[int, int, int]:
    """``col2rgb`` for the ``"#RRGGBB"`` / ``"#RRGGBBAA"`` forms pals emits."""
    s = col.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb255_to_hex(r: float, g: float, b: float) -> str:
    """``rgb(r, g, b, maxColorValue = 255)``.

    R coerces the doubles to integer for ``maxColorValue == 255``, i.e. it
    *truncates* rather than rounds: ``rgb(127.9, 0, 0, maxColorValue = 255)``
    is ``"#7F0000"``.  Verified against R 4.4.3.
    """
    return "#%02X%02X%02X" % (int(r), int(g), int(b))


def _seq_len_out(n: int) -> list[float]:
    """``seq.int(0, 1, length.out = n)``.

    R fills the interior with ``i * by`` and pins both ends to the exact
    endpoints (``src/main/seq.c``).  ``numpy.linspace`` computes
    ``(i * delta) / div`` instead, which differs in the last bit and can flip
    the truncation in :func:`_rgb255_to_hex`, so the R form is used verbatim.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    if n == 2:
        return [0.0, 1.0]
    by = 1.0 / (n - 1)
    out = [i * by for i in range(n)]
    out[0] = 0.0
    out[-1] = 1.0
    return out


def _approx1(v: float, x: Sequence[float], y: Sequence[float]) -> float:
    """``stats::approxfun(x, y)(v)`` for ``method = "linear"``.

    Bisection plus the exact arithmetic of ``approx1`` in
    ``src/library/stats/src/approx.c``; the operand order matters because the
    result is later truncated to an integer.
    """
    n = len(x)
    if v < x[0]:
        return y[0]
    if v > x[n - 1]:
        return y[n - 1]
    i, j = 0, n - 1
    while i < j - 1:
        ij = (i + j) // 2
        if v < x[ij]:
            j = ij
        else:
            i = ij
    if v == x[j]:
        return y[j]
    if v == x[i]:
        return y[i]
    return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))


def color_ramp_palette(colors: Sequence[str]):
    """``grDevices::colorRampPalette(colors)`` with the default arguments.

    ``bias = 1``, ``space = "rgb"``, ``interpolate = "linear"``, ``alpha = FALSE``:
    channels are scaled to ``[0, 1]``, linearly interpolated over
    ``seq(0, 1, length.out = length(colors))``, clamped, rescaled to 255 and
    truncated.
    """
    rgb = [_hex_to_rgb255(c) for c in colors]
    nc = len(rgb)
    anchors = _seq_len_out(nc)
    chans = [[c[k] / 255.0 for c in rgb] for k in range(3)]

    def ramp(n: int) -> list[str]:
        n = int(n)
        if n < 1:
            return []
        if nc == 1:
            return [_rgb255_to_hex(*(c * 255.0 for c in (chans[0][0], chans[1][0], chans[2][0])))] * n
        out = []
        for v in _seq_len_out(n):
            vals = []
            for k in range(3):
                val = _approx1(v, anchors, chans[k])
                val = max(min(val, 1.0), 0.0)   # colorRamp's roundcolor()
                vals.append(val * 255.0)
            out.append(_rgb255_to_hex(*vals))
        return out

    return ramp


def _setdiff(x: Iterable[str], y: Iterable[str] | None) -> list[str]:
    """``setdiff(x, y)`` — unique elements of ``x`` absent from ``y``.

    R's ``setdiff`` is ``unique(x[!x %in% y])``, so it silently de-duplicates
    ``x`` and drops names.  ``exclude = NULL`` still de-duplicates.
    """
    drop = set() if y is None else set(y)
    seen: set[str] = set()
    out: list[str] = []
    for v in x:
        if v in drop or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# pals palette accessors
# ---------------------------------------------------------------------------

def _capped(pal: Sequence[str], n: int, limit: int) -> list[str]:
    if n > limit:
        n = limit
    return list(pal[:n])


def kelly(n: int = 22) -> list[str]:
    return _capped(_KELLY, n, 22)


def cols25(n: int = 25) -> list[str]:
    return _capped(_COLS25, n, 25)


def polychrome(n: int = 36) -> list[str]:
    return _capped(_POLYCHROME, n, 36)


def glasbey(n: int = 32) -> list[str]:
    return _capped(_GLASBEY, n, 32)


def alphabet2(n: int = 26) -> list[str]:
    return _capped(_ALPHABET2, n, 26)


def alphabet(n: int = 26) -> list[str]:
    return _capped(_ALPHABET, n, 26)


def _get_brewer_pal(bpal: dict[int, list[str]], n: int) -> list[str]:
    """``pals:::get.brewer.pal`` — index the table, or ramp past its maximum."""
    maxn = max(bpal)
    if n <= maxn:
        return list(bpal[n])
    return color_ramp_palette(bpal[maxn])(n)


def brewer_set1(n: int = 9) -> list[str]:
    return _get_brewer_pal(_BREWER_SET1, n)


def brewer_set2(n: int = 8) -> list[str]:
    return _get_brewer_pal(_BREWER_SET2, n)


def brewer_dark2(n: int = 8) -> list[str]:
    return _get_brewer_pal(_BREWER_DARK2, n)


def _cont(name: str):
    ramp = color_ramp_palette(_CONT_BASE[name])
    return lambda n: ramp(n)


viridis = _cont("viridis")
parula = _cont("parula")
magma = _cont("magma")
coolwarm = _cont("coolwarm")
inferno = _cont("inferno")
plasma = _cont("plasma")


def warmcool(n: int) -> list[str]:
    """``pals::warmcool`` is literally ``rev(coolwarm(n))``."""
    return list(reversed(coolwarm(n)))


#: ``getColors(categoric = FALSE)`` dispatch table, ``util.R`` lines 139-169.
#: Note that ``palette = 9`` and ``palette = 14`` are the same palette in the
#: R source -- that duplication is reproduced, not fixed.
CONTINUOUS_PALETTES = {
    1: viridis,
    2: parula,
    3: magma,
    4: coolwarm,
    5: warmcool,
    6: inferno,
    7: plasma,
    8: _cont("kovesi.linear_bgy_10_95_c74"),
    9: _cont("kovesi.linear_bgyw_15_100_c67"),
    10: _cont("kovesi.linear_blue_95_50_c20"),
    11: _cont("kovesi.linear_bmw_5_95_c86"),
    12: _cont("kovesi.linear_bmy_10_95_c78"),
    13: _cont("kovesi.linear_kry_5_95_c72"),
    14: _cont("kovesi.linear_bgyw_15_100_c67"),
}


# ``kelly()[c(3:6, 2, 7:22, 1)]`` -- util.R line 83/85, 1-based R indices.
_KELLY_ORDER = [3, 4, 5, 6, 2] + list(range(7, 23)) + [1]


def _kelly_reordered() -> list[str]:
    return [_KELLY[i - 1] for i in _KELLY_ORDER]


def _all_colors() -> list[str]:
    """``allcolors`` from ``util.R`` line 83 (167 entries, duplicates kept)."""
    return (_kelly_reordered() + list(_COLS25) + list(_POLYCHROME)
            + list(_GLASBEY) + list(_ALPHABET2) + list(_ALPHABET))


#: ``getColors(categoric = TRUE)`` dispatch table, ``util.R`` lines 84-132.
#: The value is ``(base palette, seed used by the sample() top-up)``.
CATEGORIC_PALETTES = {
    1: (_kelly_reordered, 1),
    2: (lambda: list(_COLS25), 2),
    3: (lambda: list(_POLYCHROME), 3),
    4: (lambda: brewer_set1(8), 4),
    5: (lambda: brewer_set2(8), 5),
    6: (lambda: brewer_dark2(8), 6),
    7: (lambda: brewer_dark2(12), 7),
}


# ---------------------------------------------------------------------------
# getColors
# ---------------------------------------------------------------------------

def get_colors(n, palette: int = 1, categoric: bool = True,
               exclude: Sequence[str] | None = None) -> list[str]:
    """Generate a list of colours -- R ``getColors`` (``util.R`` line 80).

    Parameters
    ----------
    n
        Number of colours required.
    palette
        Palette index: ``1..7`` for categorical, ``1..14`` for continuous.
        Any other value in the categorical branch falls through to R's
        ``else`` arm and returns the concatenated ``allcolors`` vector.
    categoric
        ``TRUE`` selects the categorical palettes, ``FALSE`` the continuous ones.
    exclude
        Colours to drop from the categorical palette before slicing.  Ignored
        by the continuous branch, exactly as in R.

    Returns
    -------
    list of str
        ``n`` hex colours.  In the categorical ``else`` arm a request for more
        than ``len(allcolors)`` colours yields trailing ``None`` entries, which
        is how R's ``colors[1:n]`` surfaces its ``NA``s (``util.R`` line 137).
    """
    n = int(n)
    if categoric:
        allcolors = _all_colors()
        entry = CATEGORIC_PALETTES.get(int(palette))
        if entry is not None:
            base, seed = entry
            colors = _setdiff(base(), exclude)          # util.R line 86
            if n > len(colors):
                # util.R lines 87-89: top up by sampling, with replacement,
                # from everything left over -- under a palette-specific seed.
                pool = _setdiff(allcolors, list(colors) + list(exclude or []))
                extra = RRandom(seed).sample(pool, n - len(colors), replace=True)
                colors = list(colors) + [str(c) for c in extra]
            else:
                colors = colors[:n]
        else:
            colors = allcolors                          # util.R line 134
        # names(colors) <- NULL; colors[1:n]  -- util.R lines 136-137
        out: list[str] = list(colors[:n])
        if len(out) < n:
            out += [None] * (n - len(out))              # R's NA padding
        return out

    fn = CONTINUOUS_PALETTES.get(int(palette))
    if fn is None:
        # util.R line 168
        raise ValueError("Palette not found. Please provide a valid palette name.")
    return list(fn(n))
