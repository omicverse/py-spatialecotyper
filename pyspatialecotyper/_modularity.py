"""Seurat's graph-clustering stack, ported for bit-level reproducibility.

Why this module exists
----------------------
``SpatialEcoTyper`` clusters its spatial neighbourhoods with Seurat's
``FindNeighbors`` + ``FindClusters``.  Seurat does **not** use igraph/leidenalg
for the default path: it ships its own C++ transliteration of Ludo Waltman and
Nees Jan van Eck's Java ``ModularityOptimizer``, right down to a re-implemented
``java.util.Random``.  That is what makes ``random.seed = 0`` mean the same
thing in every Seurat run on every platform.  Swapping in
``networkx``/``leidenalg`` here would turn a deterministic, diffable label
vector into a merely "distributionally similar" one, so the Java LCG, the
``std::stable_sort``, and the exact floating-point accumulation orders are all
reproduced verbatim below.

Sources ported (Seurat 5.4.0):

* ``src/snn.cpp``                  -> :func:`compute_snn`
* ``src/ModularityOptimizer.cpp``  -> :class:`JavaRandom`, :class:`Clustering`,
                                      :class:`Network`,
                                      :class:`VOSClusteringTechnique`
* ``src/RModularityOptimizer.cpp`` -> :func:`run_modularity_clustering`
* ``R/clustering.R``               -> :func:`find_neighbors`,
                                      :func:`find_clusters`,
                                      :func:`group_singletons`

Deliberate deviation
--------------------
:func:`find_neighbors` computes *exact* Euclidean k-NN with
``scipy.spatial.cKDTree``; Seurat 5 defaults to ``nn.method = "annoy"``, an
approximate index (``n.trees = 50``).  See that function's docstring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

__all__ = [
    "JavaRandom",
    "compute_snn",
    "find_neighbors",
    "run_modularity_clustering",
    "find_clusters",
    "group_singletons",
]


# ----------------------------------------------------------------------------
# java.util.Random  (ModularityOptimizer.cpp:34-63)
# ----------------------------------------------------------------------------

_MULT = 0x5DEECE66D
_ADDEND = 0xB
_MASK48 = (1 << 48) - 1


def _to_int32(x: int) -> int:
    """Wrap to a signed 32-bit int, so C++ ``int`` overflow is reproduced."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


class JavaRandom:
    """``java.util.Random``'s 48-bit LCG, as re-implemented in Seurat's C++.

    Only :meth:`next_int` is used by the optimizer, and only through
    :func:`_generate_random_permutation`.  Reproducing this generator exactly
    is the single thing that makes ``run_modularity_clustering`` return the
    same labels as R for a given ``random_seed``.
    """

    __slots__ = ("seed",)

    def __init__(self, seed: int = 0):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        # ModularityOptimizer.cpp:38-41
        self.seed = (int(seed) ^ _MULT) & _MASK48

    def next(self, bits: int) -> int:
        # ModularityOptimizer.cpp:43-47.  Only 31 bits are ever requested, so
        # the C++ static_cast<int> can never go negative here.
        self.seed = (self.seed * _MULT + _ADDEND) & _MASK48
        return self.seed >> (48 - bits)

    def next_int(self, n: int) -> int:
        """``nextInt(n)`` -- uniform on ``{0, ..., n-1}``."""
        # ModularityOptimizer.cpp:50-63
        if n <= 0:
            raise ValueError("n must be positive")
        if (n & -n) == n:  # n is a power of 2
            return (n * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % n
            # The C++ loop condition relies on signed 32-bit overflow to detect
            # the (astronomically rare) non-uniform tail; reproduce the wrap.
            if _to_int32(bits - val + (n - 1)) >= 0:
                return val


def _generate_random_permutation(n_elements: int, random: JavaRandom) -> list:
    """``Arrays2::generateRandomPermutation`` (ModularityOptimizer.cpp:66-79).

    Note this is *not* a Fisher-Yates shuffle: ``j`` is drawn from the full
    range on every step, exactly as in the Java original.  Do not "fix" it.
    """
    permutation = list(range(n_elements))
    for i in range(n_elements):
        j = random.next_int(n_elements)
        k = permutation[i]
        permutation[i] = permutation[j]
        permutation[j] = k
    return permutation


# ----------------------------------------------------------------------------
# ComputeSNN  (snn.cpp:16-37)
# ----------------------------------------------------------------------------


def compute_snn(nn_ranked: np.ndarray, prune: float = 1 / 15) -> sp.csr_matrix:
    """Shared-nearest-neighbour graph; mirrors Seurat's ``ComputeSNN``.

    Parameters
    ----------
    nn_ranked
        ``(n, k)`` array of **1-based** neighbour indices, self included (and,
        for an exact k-NN, first).  This is ``Indices(NNHelper(...))`` in
        ``FindNeighbors.default`` (clustering.R:618-636).
    prune
        ``prune.SNN``; edges with a Jaccard index below this are dropped.
        Seurat's default is ``1/15`` (clustering.R:574).

    Returns
    -------
    ``(n, n)`` CSR matrix of Jaccard indices, diagonal ``1.0``.

    Notes
    -----
    ``A`` is the binary k-NN membership matrix, ``SNN = A %*% t(A)`` counts
    shared neighbours, and ``v / (k + (k - v))`` converts that count into the
    Jaccard index ``|N(i) & N(j)| / |N(i) | N(j)|``.  The comparison against
    ``prune`` is made on the *converted* value (snn.cpp:30-32).
    """
    nn_ranked = np.asarray(nn_ranked)
    n_rows, k = nn_ranked.shape
    # snn.cpp:20-26 -- triplets (i, nn_ranked(i,j)-1, 1); Eigen's
    # setFromTriplets sums duplicates, which is what coo_matrix does too.
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    cols = np.asarray(nn_ranked, dtype=np.int64).ravel() - 1
    if cols.min() < 0 or cols.max() >= n_rows:
        raise ValueError("nn_ranked must hold 1-based indices into [1, n]")
    a = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(n_rows, n_rows),
    ).tocsr()
    snn = (a @ a.T).tocsr()
    # snn.cpp:27-34 -- Jaccard conversion, then prune.  Written elementwise on
    # `.data` because the stored pattern is exactly Eigen's stored pattern.
    snn.data = snn.data / (k + (k - snn.data))
    snn.data[snn.data < prune] = 0.0
    snn.eliminate_zeros()  # Eigen's SNN.prune(0.0)
    snn.sort_indices()
    return snn


# ----------------------------------------------------------------------------
# FindNeighbors.default  (clustering.R:567-673)
# ----------------------------------------------------------------------------


def find_neighbors(
    embeddings: np.ndarray,
    k_param: int = 20,
    prune_snn: float = 1 / 15,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """``FindNeighbors.default`` -- returns ``(nn_graph, snn_graph)``.

    Parameters
    ----------
    embeddings
        ``(n_cells, n_dims)`` cell embeddings, e.g. ``Embeddings(obj, "pca")``
        subset to ``dims``.
    k_param
        ``k.param``, default 20 (clustering.R:571).
    prune_snn
        ``prune.SNN``, default ``1/15`` (clustering.R:574).

    Returns
    -------
    ``nn_graph``
        Binary ``(n, n)`` k-NN membership matrix, ``nn.matrix`` in R
        (clustering.R:650-653).
    ``snn_graph``
        The pruned Jaccard graph from :func:`compute_snn`.

    Deviation from Seurat
    ---------------------
    Seurat 5 defaults to ``nn.method = "annoy"`` -- an *approximate* nearest
    neighbour index (RcppAnnoy, ``n.trees = 50``, angular/euclidean trees with
    random split hyperplanes).  Annoy's result depends on its own RNG and on
    the compiled index, neither of which is reproducible from Python.  We
    therefore compute the *exact* Euclidean k-NN with ``scipy.spatial.cKDTree``.

    Consequences, measured on the reference fixture (297 spots x 10 PCs,
    k = 20): the exact neighbour sets differ from Annoy's for a small fraction
    of (cell, neighbour) pairs; because ``compute_snn`` then intersects
    neighbour sets and prunes at 1/15, the induced SNN differences are smaller
    still.  ``tests/test_modularity.py`` measures and pins both.  If you need
    literal Annoy parity, feed :func:`compute_snn` the ``nn.idx`` matrix
    produced by R.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n_cells = embeddings.shape[0]
    if n_cells < k_param:
        # clustering.R:597-603
        k_param = n_cells - 1
    tree = cKDTree(embeddings)
    # Exact k-NN; cKDTree returns neighbours sorted by increasing distance, so
    # column 0 is the point itself (distance 0), matching Annoy's convention.
    _, idx = tree.query(embeddings, k=k_param, workers=-1)
    nn_ranked = np.asarray(idx, dtype=np.int64).reshape(n_cells, k_param) + 1

    # clustering.R:650-653 -- nn.ranked -> binary Graph.
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k_param)
    cols = nn_ranked.ravel() - 1
    nn_graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()

    snn_graph = compute_snn(nn_ranked, prune=prune_snn)
    return nn_graph, snn_graph


# ----------------------------------------------------------------------------
# Clustering  (ModularityOptimizer.cpp:82-162)
# ----------------------------------------------------------------------------


class Clustering:
    """Port of ``ModularityOptimizer::Clustering``."""

    __slots__ = ("n_nodes", "n_clusters", "cluster")

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.n_clusters = 1
        self.cluster = [0] * n_nodes

    def get_n_nodes_per_cluster(self) -> list:
        # ModularityOptimizer.cpp:96-102
        counts = [0] * self.n_clusters
        for c in self.cluster:
            counts[c] += 1
        return counts

    def get_nodes_per_cluster(self) -> list:
        # ModularityOptimizer.cpp:104-116
        nodes = [[] for _ in range(self.n_clusters)]
        for i in range(self.n_nodes):
            nodes[self.cluster[i]].append(i)
        return nodes

    def init_singleton_clusters(self) -> None:
        # ModularityOptimizer.cpp:123-128
        self.cluster = list(range(self.n_nodes))
        self.n_clusters = self.n_nodes

    def order_clusters_by_n_nodes(self) -> None:
        """Relabel clusters 0..n-1 by *decreasing* size.

        ModularityOptimizer.cpp:130-156.  The C++ uses ``stable_sort`` with
        ``b.first < a.first`` so equal-sized clusters keep their original index
        order; Python's ``sorted`` is stable, so ``key=-size`` matches.  This
        is where ``seurat_clusters``' "0 is the biggest cluster" convention
        comes from.
        """
        n_nodes_per_cluster = self.get_n_nodes_per_cluster()
        order = sorted(range(self.n_clusters), key=lambda c: -n_nodes_per_cluster[c])
        new_cluster = [0] * self.n_clusters
        i = 0
        while True:  # C++ do/while: index 0 is always assigned
            new_cluster[order[i]] = i
            i += 1
            if not (i < self.n_clusters and n_nodes_per_cluster[order[i]] > 0):
                break
        self.n_clusters = i
        self.cluster = [new_cluster[c] for c in self.cluster]

    def merge_clusters(self, clustering: "Clustering") -> None:
        # ModularityOptimizer.cpp:158-162
        self.cluster = [clustering.cluster[c] for c in self.cluster]
        self.n_clusters = clustering.n_clusters


# ----------------------------------------------------------------------------
# Network  (ModularityOptimizer.cpp:165-445)
# ----------------------------------------------------------------------------


def _accumulate(values, start: int, stop: int) -> float:
    """Sequential left-to-right sum, i.e. ``std::accumulate``.

    ``numpy.sum`` uses pairwise summation and would differ in the last bits.
    Those bits matter: they feed ``resolution2``, the node weights and the
    quality function, all of which are compared with ``>`` and ``==``.
    """
    total = 0.0
    for i in range(start, stop):
        total += values[i]
    return total


class Network:
    """Port of ``ModularityOptimizer::Network`` (adjacency in CSR-like form)."""

    __slots__ = (
        "n_nodes",
        "n_edges",
        "node_weight",
        "first_neighbor_index",
        "neighbor",
        "edge_weight",
        "total_edge_weight_self_links",
    )

    def __init__(
        self,
        n_nodes: int = 0,
        first_neighbor_index: Optional[list] = None,
        neighbor: Optional[list] = None,
        edge_weight: Optional[list] = None,
        node_weight: Optional[list] = None,
    ):
        # ModularityOptimizer.cpp:167-185
        self.n_nodes = n_nodes
        self.first_neighbor_index = first_neighbor_index or []
        self.neighbor = neighbor or []
        self.n_edges = len(self.neighbor)
        self.edge_weight = (
            list(edge_weight) if edge_weight is not None else [1.0] * self.n_edges
        )
        self.total_edge_weight_self_links = 0.0
        if node_weight is not None:
            self.node_weight = list(node_weight)
        else:
            self.node_weight = self.get_total_edge_weight_per_node()

    def get_total_edge_weight(self) -> float:
        # ModularityOptimizer.cpp:266-268
        return _accumulate(self.edge_weight, 0, self.n_edges) / 2.0

    def get_total_edge_weight_node(self, node: int) -> float:
        # ModularityOptimizer.cpp:270-274
        return _accumulate(
            self.edge_weight,
            self.first_neighbor_index[node],
            self.first_neighbor_index[node + 1],
        )

    def get_total_edge_weight_per_node(self) -> list:
        # ModularityOptimizer.cpp:276-282
        return [self.get_total_edge_weight_node(i) for i in range(self.n_nodes)]

    def create_reduced_network(self, clustering: Clustering) -> "Network":
        """``Network::createReducedNetwork`` (ModularityOptimizer.cpp:320-371).

        Kept as an explicit loop rather than a sparse matrix product: the order
        in which ``reducedNetworkEdgeWeight2`` accumulates, and the order the
        reduced neighbours end up in, both feed later floating-point sums.
        """
        reduced = Network.__new__(Network)
        n_clusters = clustering.n_clusters
        reduced.n_nodes = n_clusters
        reduced.n_edges = 0
        reduced.node_weight = [0.0] * n_clusters
        reduced.first_neighbor_index = [0] * (n_clusters + 1)
        reduced.total_edge_weight_self_links = self.total_edge_weight_self_links

        reduced_neighbor1 = [0] * self.n_edges
        reduced_edge_weight1 = [0.0] * self.n_edges
        reduced_neighbor2 = [0] * max(n_clusters - 1, 0)
        reduced_edge_weight2 = [0.0] * n_clusters

        node_per_cluster = clustering.get_nodes_per_cluster()
        first = self.first_neighbor_index
        neighbor = self.neighbor
        edge_weight = self.edge_weight
        cl = clustering.cluster
        node_weight = self.node_weight

        for i in range(n_clusters):
            j = 0
            for l in node_per_cluster[i]:
                reduced.node_weight[i] += node_weight[l]
                for m in range(first[l], first[l + 1]):
                    n = cl[neighbor[m]]
                    if n != i:
                        if reduced_edge_weight2[n] == 0:
                            reduced_neighbor2[j] = n
                            j += 1
                        reduced_edge_weight2[n] += edge_weight[m]
                    else:
                        reduced.total_edge_weight_self_links += edge_weight[m]
            base = reduced.n_edges
            for k in range(j):
                nb = reduced_neighbor2[k]
                reduced_neighbor1[base + k] = nb
                reduced_edge_weight1[base + k] = reduced_edge_weight2[nb]
                reduced_edge_weight2[nb] = 0.0
            reduced.n_edges += j
            reduced.first_neighbor_index[i + 1] = reduced.n_edges

        reduced.neighbor = reduced_neighbor1[: reduced.n_edges]
        reduced.edge_weight = reduced_edge_weight1[: reduced.n_edges]
        return reduced


def matrix_to_network(
    node1: list,
    node2: list,
    edge_weight1: list,
    modularity_function: int,
    n_nodes: int,
) -> Network:
    """``ModularityOptimizer::matrixToNetwork`` (ModularityOptimizer.cpp:759-803).

    Only pairs with ``node1 < node2`` contribute, and each contributes twice
    (once per direction).  ``modularityFunction == 1`` leaves ``nodeWeight``
    NULL so the Network constructor sets it to the total edge weight per node;
    ``== 2`` sets every node weight to 1.
    """
    n_neighbors = [0] * n_nodes
    m = len(node1)
    for i in range(m):
        if node1[i] < node2[i]:
            n_neighbors[node1[i]] += 1
            n_neighbors[node2[i]] += 1

    first_neighbor_index = [0] * (n_nodes + 1)
    n_edges = 0
    for i in range(n_nodes):
        first_neighbor_index[i] = n_edges
        n_edges += n_neighbors[i]
    first_neighbor_index[n_nodes] = n_edges

    neighbor = [0] * n_edges
    edge_weight2 = [0.0] * n_edges
    n_neighbors = [0] * n_nodes
    for i in range(m):
        a = node1[i]
        b = node2[i]
        if a < b:
            j = first_neighbor_index[a] + n_neighbors[a]
            neighbor[j] = b
            edge_weight2[j] = edge_weight1[i]
            n_neighbors[a] += 1
            j = first_neighbor_index[b] + n_neighbors[b]
            neighbor[j] = a
            edge_weight2[j] = edge_weight1[i]
            n_neighbors[b] += 1

    if modularity_function == 1:
        return Network(n_nodes, first_neighbor_index, neighbor, edge_weight2)
    return Network(
        n_nodes,
        first_neighbor_index,
        neighbor,
        edge_weight2,
        node_weight=[1.0] * n_nodes,
    )


# ----------------------------------------------------------------------------
# VOSClusteringTechnique  (ModularityOptimizer.cpp:447-695)
# ----------------------------------------------------------------------------


class VOSClusteringTechnique:
    """Port of ``ModularityOptimizer::VOSClusteringTechnique``."""

    __slots__ = ("network", "clustering", "resolution")

    def __init__(self, network: Network, resolution: float,
                 clustering: Optional[Clustering] = None):
        self.network = network
        self.resolution = resolution
        if clustering is None:
            # ModularityOptimizer.cpp:447-453
            clustering = Clustering(network.n_nodes)
            clustering.init_singleton_clusters()
        self.clustering = clustering

    def calc_quality_function(self) -> float:
        """``calcQualityFunction`` (ModularityOptimizer.cpp:460-480).

        The accumulation order is load-bearing: ``RunModularityClusteringCpp``
        compares this against ``maxModularity`` with ``>`` to pick the winning
        random start, so a last-bit difference can select a different restart.
        """
        network = self.network
        clustering = self.clustering
        cl = clustering.cluster
        first = network.first_neighbor_index
        neighbor = network.neighbor
        edge_weight = network.edge_weight

        quality_function = 0.0
        for i in range(network.n_nodes):
            j = cl[i]
            for k in range(first[i], first[i + 1]):
                if cl[neighbor[k]] == j:
                    quality_function += edge_weight[k]
        quality_function += network.total_edge_weight_self_links

        cluster_weight = [0.0] * clustering.n_clusters
        for i in range(network.n_nodes):
            cluster_weight[cl[i]] += network.node_weight[i]
        for i in range(clustering.n_clusters):
            quality_function -= cluster_weight[i] * cluster_weight[i] * self.resolution

        quality_function /= (
            2 * network.get_total_edge_weight() + network.total_edge_weight_self_links
        )
        return quality_function

    def run_local_moving_algorithm(self, random: JavaRandom) -> bool:
        """``runLocalMovingAlgorithm`` (ModularityOptimizer.cpp:482-581).

        Deliberately a scalar Python loop.  ``edgeWeightPerCluster[l] +=
        edgeWeight[k]`` must accumulate in neighbour order, and the move
        decision uses ``>`` / ``==`` on those sums, so vectorising with
        ``np.add.at`` (pairwise/unordered) can flip a decision and then diverge
        chaotically.  Plain Python float arithmetic is IEEE-754 double, i.e.
        bit-identical to the C++ here.
        """
        network = self.network
        clustering = self.clustering
        resolution = self.resolution
        n_nodes = network.n_nodes
        if n_nodes == 1:
            return False

        update = False
        cluster = clustering.cluster
        node_weight = network.node_weight
        first = network.first_neighbor_index
        neighbor = network.neighbor
        edge_weight = network.edge_weight

        cluster_weight = [0.0] * n_nodes
        n_nodes_per_cluster = [0] * n_nodes
        for i in range(n_nodes):
            c = cluster[i]
            cluster_weight[c] += node_weight[i]
            n_nodes_per_cluster[c] += 1

        n_unused_clusters = 0
        unused_cluster = [0] * n_nodes
        for i in range(n_nodes):
            if n_nodes_per_cluster[i] == 0:
                unused_cluster[n_unused_clusters] = i
                n_unused_clusters += 1

        node_permutation = _generate_random_permutation(n_nodes, random)
        edge_weight_per_cluster = [0.0] * n_nodes
        neighboring_cluster = [0] * (n_nodes - 1)
        n_stable_nodes = 0
        i = 0
        while True:  # C++ do/while (nStableNodes < nNodes)
            j = node_permutation[i]
            n_neighboring_clusters = 0
            for k in range(first[j], first[j + 1]):
                l = cluster[neighbor[k]]
                if edge_weight_per_cluster[l] == 0:
                    neighboring_cluster[n_neighboring_clusters] = l
                    n_neighboring_clusters += 1
                edge_weight_per_cluster[l] += edge_weight[k]

            cj = cluster[j]
            cluster_weight[cj] -= node_weight[j]
            n_nodes_per_cluster[cj] -= 1
            if n_nodes_per_cluster[cj] == 0:
                unused_cluster[n_unused_clusters] = cj
                n_unused_clusters += 1

            best_cluster = -1
            max_quality_function = 0.0
            nwj = node_weight[j]
            for k in range(n_neighboring_clusters):
                l = neighboring_cluster[k]
                quality_function = (
                    edge_weight_per_cluster[l] - nwj * cluster_weight[l] * resolution
                )
                # ModularityOptimizer.cpp:541 -- note the tie-break compares
                # against bestCluster, which starts at -1, so the first strictly
                # positive candidate always wins its own tie.
                if quality_function > max_quality_function or (
                    quality_function == max_quality_function and l < best_cluster
                ):
                    best_cluster = l
                    max_quality_function = quality_function
                edge_weight_per_cluster[l] = 0.0
            if max_quality_function == 0:
                # Move j into a free cluster label.  n_unused_clusters is >= 1
                # here because j itself may just have emptied its own label.
                best_cluster = unused_cluster[n_unused_clusters - 1]
                n_unused_clusters -= 1

            cluster_weight[best_cluster] += nwj
            n_nodes_per_cluster[best_cluster] += 1
            if best_cluster == cluster[j]:
                n_stable_nodes += 1
            else:
                cluster[j] = best_cluster
                n_stable_nodes = 1
                update = True

            i = i + 1 if i < n_nodes - 1 else 0
            if n_stable_nodes >= n_nodes:
                break

        # ModularityOptimizer.cpp:569-578 -- compact the used labels.
        new_cluster = [0] * n_nodes
        clustering.n_clusters = 0
        for i in range(n_nodes):
            if n_nodes_per_cluster[i] > 0:
                new_cluster[i] = clustering.n_clusters
                clustering.n_clusters += 1
        for i in range(n_nodes):
            cluster[i] = new_cluster[cluster[i]]
        return update

    def run_louvain_algorithm(self, random: JavaRandom) -> bool:
        # ModularityOptimizer.cpp:583-602
        if self.network.n_nodes == 1:
            return False
        update = self.run_local_moving_algorithm(random)
        if self.clustering.n_clusters < self.network.n_nodes:
            sub = VOSClusteringTechnique(
                self.network.create_reduced_network(self.clustering), self.resolution
            )
            update2 = sub.run_louvain_algorithm(random)
            if update2:
                update = True
                self.clustering.merge_clusters(sub.clustering)
        return update

    def run_louvain_algorithm_with_multilevel_refinement(
        self, random: JavaRandom
    ) -> bool:
        # ModularityOptimizer.cpp:616-635  (algorithm = 2)
        if self.network.n_nodes == 1:
            return False
        update = self.run_local_moving_algorithm(random)
        if self.clustering.n_clusters < self.network.n_nodes:
            sub = VOSClusteringTechnique(
                self.network.create_reduced_network(self.clustering), self.resolution
            )
            update2 = sub.run_louvain_algorithm_with_multilevel_refinement(random)
            if update2:
                update = True
                self.clustering.merge_clusters(sub.clustering)
                self.run_local_moving_algorithm(random)
        return update

    def run_smart_local_moving_algorithm(self, random: JavaRandom) -> bool:
        # ModularityOptimizer.cpp:649-688  (algorithm = 3)
        network = self.network
        clustering = self.clustering
        if network.n_nodes == 1:
            return False
        update = self.run_local_moving_algorithm(random)
        if clustering.n_clusters < network.n_nodes:
            subnetworks = _create_subnetworks(network, clustering)
            node_per_cluster = clustering.get_nodes_per_cluster()
            clustering.n_clusters = 0
            n_nodes_per_cluster_reduced = [0] * len(subnetworks)
            for i, sub_net in enumerate(subnetworks):
                sub_vos = VOSClusteringTechnique(sub_net, self.resolution)
                sub_vos.run_local_moving_algorithm(random)
                for j in range(sub_net.n_nodes):
                    clustering.cluster[node_per_cluster[i][j]] = (
                        clustering.n_clusters + sub_vos.clustering.cluster[j]
                    )
                clustering.n_clusters += sub_vos.clustering.n_clusters
                n_nodes_per_cluster_reduced[i] = sub_vos.clustering.n_clusters

            vos2 = VOSClusteringTechnique(
                network.create_reduced_network(clustering), self.resolution
            )
            i = 0
            for j, cnt in enumerate(n_nodes_per_cluster_reduced):
                for _ in range(cnt):
                    vos2.clustering.cluster[i] = j
                    i += 1
            vos2.clustering.n_clusters = len(n_nodes_per_cluster_reduced)
            update |= vos2.run_smart_local_moving_algorithm(random)
            clustering.merge_clusters(vos2.clustering)
        return update


def _create_subnetworks(network: Network, clustering: Clustering) -> list:
    """``Network::createSubnetworks`` / ``createSubnetwork``
    (ModularityOptimizer.cpp:308-317, 404-445).  Only used by ``algorithm=3``."""
    node_per_cluster = clustering.get_nodes_per_cluster()
    subnetwork_node = [0] * network.n_nodes
    out = []
    first = network.first_neighbor_index
    neighbor = network.neighbor
    edge_weight = network.edge_weight
    for c in range(clustering.n_clusters):
        node = node_per_cluster[c]
        sub = Network.__new__(Network)
        sub.n_nodes = len(node)
        sub.total_edge_weight_self_links = 0.0
        if sub.n_nodes == 1:
            sub.n_edges = 0
            sub.node_weight = [network.node_weight[node[0]]]
            sub.first_neighbor_index = [0, 0]
            sub.neighbor = []
            sub.edge_weight = []
        else:
            for i, nd in enumerate(node):
                subnetwork_node[nd] = i
            sub.n_edges = 0
            sub.node_weight = [0.0] * sub.n_nodes
            sub.first_neighbor_index = [0] * (sub.n_nodes + 1)
            sub_neighbor = []
            sub_edge_weight = []
            for i in range(sub.n_nodes):
                j = node[i]
                sub.node_weight[i] = network.node_weight[j]
                for k in range(first[j], first[j + 1]):
                    if clustering.cluster[neighbor[k]] == c:
                        sub_neighbor.append(subnetwork_node[neighbor[k]])
                        sub_edge_weight.append(edge_weight[k])
                        sub.n_edges += 1
                sub.first_neighbor_index[i + 1] = sub.n_edges
            sub.neighbor = sub_neighbor
            sub.edge_weight = sub_edge_weight
        out.append(sub)
    return out


# ----------------------------------------------------------------------------
# RunModularityClusteringCpp  (RModularityOptimizer.cpp:23-179)
# ----------------------------------------------------------------------------


def _snn_to_edge_list(snn) -> Tuple[list, list, list, int]:
    """Extract the strict lower triangle in Eigen's column-major stored order.

    ``RModularityOptimizer.cpp:70-80`` walks the Eigen (column-major) sparse
    matrix and keeps ``it.col() < it.row()``, emitting ``node1 = col``,
    ``node2 = row``.  So the edge list is ordered by (column asc, row asc),
    which is exactly CSC traversal with sorted inner indices.
    """
    snn = sp.csc_matrix(snn)
    snn.sort_indices()
    n_nodes = max(snn.shape[0], snn.shape[1])  # RModularityOptimizer.cpp:82
    rows = snn.indices
    data = snn.data
    cols = np.repeat(np.arange(snn.shape[1], dtype=np.int64), np.diff(snn.indptr))
    mask = rows > cols
    return (
        cols[mask].tolist(),
        rows[mask].astype(np.int64).tolist(),
        data[mask].tolist(),
        n_nodes,
    )


def run_modularity_clustering(
    snn,
    resolution: float = 0.8,
    algorithm: int = 1,
    n_start: int = 10,
    n_iter: int = 10,
    random_seed: int = 0,
    modularity: int = 1,
) -> np.ndarray:
    """``RunModularityClustering`` / ``RunModularityClusteringCpp``.

    Argument names and defaults follow ``RunModularityClustering``
    (clustering.R:1864-1891); the argument *order* into the C++ entry point is
    ``(SNN, modularity, resolution, algorithm, n.start, n.iter, random.seed,
    print.output, edge.file.name)``.

    Parameters
    ----------
    snn
        Symmetric SNN graph (scipy sparse).  Only its strict lower triangle is
        read, as in the C++.
    resolution
        ``resolution``, default 0.8.
    algorithm
        1 = Louvain, 2 = Louvain with multilevel refinement, 3 = SLM.
    n_start, n_iter
        ``n.start`` / ``n.iter``, both default 10.
    random_seed
        ``random.seed``, default 0 -- seeds a single :class:`JavaRandom` that
        is shared across all restarts (RModularityOptimizer.cpp:103).
    modularity
        ``modularity.fxn``; 1 = standard, 2 = alternative.

    Returns
    -------
    ``(n,)`` int array of 0-based labels, already ordered by decreasing
    cluster size (``clustering->orderClustersByNNodes()``).
    """
    # RModularityOptimizer.cpp:33-43 -- argument validation, verbatim.
    if modularity not in (1, 2):
        raise ValueError("Modularity parameter must be equal to 1 or 2.")
    if algorithm not in (1, 2, 3):
        raise ValueError(
            "Algorithm for modularity optimization must be 1, 2, or 3 "
            "(4 = Leiden is not part of this port)"
        )
    if n_start < 1:
        raise ValueError("Have to have at least one start")
    if n_iter < 1:
        raise ValueError("Need at least one interation")
    if modularity == 2 and resolution > 1.0:
        raise ValueError("error: resolution<1 for alternative modularity")

    node1, node2, edge_weights, n_nodes = _snn_to_edge_list(snn)
    if len(node1) == 0:
        raise ValueError("Matrix contained no network data.  Check format.")
    network = matrix_to_network(node1, node2, edge_weights, modularity, n_nodes)

    # RModularityOptimizer.cpp:97
    if modularity == 1:
        resolution2 = resolution / (
            2 * network.get_total_edge_weight() + network.total_edge_weight_self_links
        )
    else:
        resolution2 = resolution

    clustering = None
    max_modularity = -np.inf
    random = JavaRandom(random_seed)  # one stream for *all* restarts

    for _ in range(n_start):
        vos = VOSClusteringTechnique(network, resolution2)
        j = 0
        update = True
        while True:  # C++ do/while ((j < nIterations) && update)
            if algorithm == 1:
                update = vos.run_louvain_algorithm(random)
            elif algorithm == 2:
                update = vos.run_louvain_algorithm_with_multilevel_refinement(random)
            else:
                vos.run_smart_local_moving_algorithm(random)
            j += 1
            modularity_value = vos.calc_quality_function()
            if not (j < n_iter and update):
                break
        if modularity_value > max_modularity:
            clustering = vos.clustering
            max_modularity = modularity_value

    if clustering is None:
        raise RuntimeError("Clustering step failed.")
    clustering.order_clusters_by_n_nodes()  # RModularityOptimizer.cpp:167
    return np.asarray(clustering.cluster, dtype=np.int64)


# ----------------------------------------------------------------------------
# GroupSingletons  (clustering.R:1356-1404)
# ----------------------------------------------------------------------------


def _sample_one(candidates: list) -> str:
    """``set.seed(1); sample(x, 1)`` on a character vector.

    ``GroupSingletons`` re-seeds R's global stream to 1 immediately before the
    draw (clustering.R:1379), so the pick is deterministic.  With a single
    candidate R returns it unchanged, which is the overwhelmingly common case;
    for genuine ties we defer to the ported R RNG in
    :mod:`pyspatialecotyper.rrandom` when it is importable.
    """
    if len(candidates) == 1:
        return candidates[0]
    try:
        from .rrandom import RRandom

        idx = int(RRandom(1).sample_int(len(candidates), 1)[0])
        return candidates[idx]
    except Exception:  # pragma: no cover - fallback keeps the module standalone
        return candidates[0]


def group_singletons(ids, snn, group_singletons: bool = True):
    """``GroupSingletons`` (clustering.R:1356-1404).

    Clusters holding exactly one cell are folded into whichever surviving
    cluster they are most connected to, connectivity being the mean SNN weight
    between the singleton and that cluster's cells.

    ``ids`` is a 0-based integer label vector.  Singletons are visited in order
    of first appearance in ``ids`` (R's ``intersect(unique(ids), singletons)``
    keeps the order of its first argument), and ``ids`` is updated in place as
    we go, so later connectivities see earlier merges -- as in R.

    With ``group_singletons=False`` R overwrites the labels with the string
    ``"singleton"``; here those cells get ``-1``.
    """
    ids = np.asarray(ids, dtype=np.int64).copy()
    snn = sp.csr_matrix(snn)

    values, counts = np.unique(ids, return_counts=True)
    singleton_set = set(values[counts == 1].tolist())
    if not singleton_set:
        return ids
    # order of first appearance, matching intersect(unique(ids), singletons)
    seen = []
    seen_set = set()
    for v in ids.tolist():
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    singletons = [v for v in seen if v in singleton_set]

    if not group_singletons:
        ids[np.isin(ids, list(singleton_set))] = -1
        return ids

    cluster_names = [v for v in seen if v not in singleton_set]
    if not cluster_names:
        return ids

    for s in singletons:
        i_cells = np.flatnonzero(ids == s)
        connectivity = np.empty(len(cluster_names), dtype=np.float64)
        for jj, c in enumerate(cluster_names):
            j_cells = np.flatnonzero(ids == c)
            sub = snn[i_cells][:, j_cells]
            connectivity[jj] = sub.sum() / (sub.shape[0] * sub.shape[1])
        m = np.nanmax(connectivity)
        tied = [cluster_names[t] for t in np.flatnonzero(connectivity == m)]
        ids[i_cells] = _sample_one(tied)
    return ids


# ``find_clusters`` takes a *boolean* parameter of the same name (to match R's
# ``group.singletons``), which would shadow the function; keep a private alias.
_group_singletons = group_singletons


# ----------------------------------------------------------------------------
# FindClusters.default  (clustering.R:307-424)
# ----------------------------------------------------------------------------


def find_clusters(
    snn,
    resolution: float = 0.8,
    algorithm: int = 1,
    n_start: int = 10,
    n_iter: int = 10,
    random_seed: int = 0,
    group_singletons: bool = True,
    modularity_fxn: int = 1,
) -> np.ndarray:
    """``FindClusters`` on an SNN graph -- 0-based integer labels.

    Equivalent to the single-resolution branch of ``FindClusters.default``
    (clustering.R:386-421): modularity clustering, then :func:`group_singletons`,
    then ``factor()``.

    Labelling
    ---------
    ``run_modularity_clustering`` already renumbers clusters by *decreasing*
    size (``orderClustersByNNodes``), which is where ``seurat_clusters``' level
    order comes from.  ``factor(ids)`` in R only sorts the levels; it does not
    renumber them, so if singleton grouping empties a label the remaining
    labels keep their numbering and the sequence has a gap -- exactly as
    ``levels(seurat_clusters)`` does.  The integers returned here are therefore
    the same values as ``as.character(obj$seurat_clusters)``.
    """
    ids = run_modularity_clustering(
        snn,
        resolution=resolution,
        algorithm=algorithm,
        n_start=n_start,
        n_iter=n_iter,
        random_seed=random_seed,
        modularity=modularity_fxn,
    )
    ids = _group_singletons(ids, snn, group_singletons=group_singletons)
    return ids
