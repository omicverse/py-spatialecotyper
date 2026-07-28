"""R's random-number stream, ported bit-for-bit.

Why this module exists
----------------------
SpatialEcoTyper's numerics are seeded, not seed-free: ``NMF::rnmf`` draws its
initial basis from ``runif``; ``nmfClustering`` picks its per-run seeds with
``sample(1:6280, nrun.per.rank)``; ``NMFGenerateWList`` down-samples cells with
``sample``; and ``Colocalization`` / ``CoassociationTest`` /
``ComputeNormalizedMoranI`` build their null distributions from ``sample``
permutations.  If Python draws from NumPy's stream instead, every one of those
outputs degrades from "reproducible" to "distributionally similar".

Porting R's stream turns them back into deterministic, diffable outputs.

What is reproduced
------------------
* ``set.seed(k)`` — R's initial scrambling + ``RNG_Init`` LCG seeding of the
  625-word Mersenne-Twister state (``src/main/RNG.c``).
* ``unif_rand()`` — ``MT_genrand`` with R's ``fixup()`` clamp.
* ``R_unif_index(dn)`` — the post-R-3.6 ``"Rejection"`` sampler
  (``sample.kind = "Rejection"``, the default since R 3.6.0).
* ``sample()`` / ``sample.int()`` — with and without replacement, using the
  in-place swap-with-last loop from ``do_sample``.
* ``rnorm()`` — the default ``INVERSION`` normal, including the two-uniform
  ``BIG = 2^27`` precision trick.

Verified in ``tests/test_rrandom.py`` against R 4.4.3 for
``runif``/``rnorm``/``sample`` over a grid of seeds.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RRandom", "set_seed", "runif", "rnorm", "sample_int", "sample"]

_UINT32 = 0xFFFFFFFF

# Mersenne-Twister constants, verbatim from R's src/extra/mt19937ar / RNG.c
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000

_I2_32M1 = 2.328306437080797e-10   # 1 / (2^32 - 1)
_TWO_M32 = 2.3283064365386963e-10  # 1 / 2^32
_BIG = 134217728.0                 # 2^27, the INVERSION precision multiplier


class RRandom:
    """A standalone instance of R's Mersenne-Twister ``.Random.seed``.

    Using an object rather than module-level state means permutation loops can
    be parallelised without the workers fighting over one global stream — each
    worker owns a clone advanced to its own position.
    """

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int | None = None):
        self._mt = np.zeros(_N, dtype=np.uint32)
        self._mti = _N + 1
        if seed is not None:
            self.set_seed(seed)

    # -- seeding ---------------------------------------------------------
    def set_seed(self, seed: int) -> "RRandom":
        """Reproduce ``set.seed(seed)`` for ``kind = "Mersenne-Twister"``.

        R applies the ``69069`` LCG scrambler twice: 50 rounds in ``do_setseed``
        before handing the seed to ``RNG_Init``, then once more per state word
        inside ``RNG_Init``.  ``n_seed`` for MT is 625 — word 0 is the ``mti``
        cursor, which ``FixupSeeds(initial = TRUE)`` immediately overwrites
        with ``N``, so only words 1..624 survive as the actual MT state.
        """
        s = int(seed) & _UINT32
        for _ in range(50):                      # do_setseed initial scrambling
            s = (69069 * s + 1) & _UINT32
        # RNG_Init: fill all 625 words, then discard word 0 (the mti cursor).
        s = (69069 * s + 1) & _UINT32            # i_seed[0] -> mti, overwritten
        state = np.empty(_N, dtype=np.uint32)
        for j in range(_N):
            s = (69069 * s + 1) & _UINT32
            state[j] = s
        self._mt = state
        self._mti = _N                           # FixupSeeds: I624 = 624
        return self

    def clone(self) -> "RRandom":
        out = RRandom.__new__(RRandom)
        out._mt = self._mt.copy()
        out._mti = self._mti
        return out

    # -- core generator --------------------------------------------------
    def _genrand(self) -> float:
        if self._mti >= _N:
            mt = self._mt.astype(np.int64)       # int64 avoids uint32 overflow
            mag01 = (0, _MATRIX_A)
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self._mt = (mt & _UINT32).astype(np.uint32)
            self._mti = 0

        y = int(self._mt[self._mti])
        self._mti += 1
        y ^= (y >> 11)
        y = (y ^ ((y << 7) & _TEMPERING_MASK_B)) & _UINT32
        y = (y ^ ((y << 15) & _TEMPERING_MASK_C)) & _UINT32
        y ^= (y >> 18)
        return float(y) * _TWO_M32

    def unif_rand(self) -> float:
        """``unif_rand()`` including R's ``fixup()`` open-interval clamp."""
        v = self._genrand()
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if (1.0 - v) <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def runif(self, n: int = 1, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
        out = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            out[i] = self.unif_rand()
        return out * (hi - lo) + lo if (lo, hi) != (0.0, 1.0) else out

    # -- normal (INVERSION, R's default normal.kind) ----------------------
    def norm_rand(self) -> float:
        from scipy.special import ndtri
        u1 = self.unif_rand()
        u1 = int(_BIG * u1) + self.unif_rand()
        return float(ndtri(u1 / _BIG))

    def rnorm(self, n: int = 1, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        return np.array([self.norm_rand() for _ in range(int(n))]) * sd + mean

    # -- discrete uniform (R >= 3.6 "Rejection" sampler) -------------------
    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(np.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: float) -> float:
        """``R_unif_index(dn)`` — uniform on ``{0, ..., dn-1}`` by rejection."""
        if dn <= 0:
            return 0.0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return float(dv)

    def sample_int(self, n: int, size: int | None = None,
                   replace: bool = False) -> np.ndarray:
        """``sample.int(n, size, replace)`` → **0-based** indices.

        R returns 1-based; the caller converts.  The without-replacement branch
        reproduces ``do_sample``'s swap-with-last loop exactly, which is what
        makes the resulting permutation identical rather than merely uniform.
        """
        n = int(n)
        k = n if size is None else int(size)
        if replace:
            return np.array([int(self.unif_index(n)) for _ in range(k)],
                            dtype=np.int64)
        if k > n:
            raise ValueError("cannot take a sample larger than the population "
                             "when 'replace = FALSE'")
        x = np.arange(n, dtype=np.int64)
        out = np.empty(k, dtype=np.int64)
        nn = n
        for i in range(k):
            j = int(self.unif_index(nn))
            out[i] = x[j]
            nn -= 1
            x[j] = x[nn]
        return out

    def sample(self, x, size: int | None = None, replace: bool = False):
        """``sample(x, size, replace)`` for a vector ``x``."""
        x = np.asarray(x)
        idx = self.sample_int(len(x), size, replace)
        return x[idx]


# ------------------------------------------------------------------------
# Module-level convenience mirroring R's single global stream.
# ------------------------------------------------------------------------
_GLOBAL = RRandom(0)


def set_seed(seed: int) -> RRandom:
    """``set.seed(seed)`` on the module-global stream."""
    _GLOBAL.set_seed(seed)
    return _GLOBAL


def runif(n: int = 1, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return _GLOBAL.runif(n, lo, hi)


def rnorm(n: int = 1, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    return _GLOBAL.rnorm(n, mean, sd)


def sample_int(n: int, size: int | None = None, replace: bool = False) -> np.ndarray:
    return _GLOBAL.sample_int(n, size, replace)


def sample(x, size: int | None = None, replace: bool = False):
    return _GLOBAL.sample(x, size, replace)
