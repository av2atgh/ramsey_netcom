import os
import json
import random
import numpy as np
import pandas as pd
import graph_tool.all as gt
import igraph as ig
import networkx as nx
from infomap import Infomap
import itertools
from ramsey_netcom.hgc import hgc
from scipy.signal import argrelmax


def split_symmetric(neighbours, i, j):
    l = 1 if degree_j == 2 else np.random.randint(1, len(neighbours[j]) - 1)
    neighbours_ = list(np.random.permutation(neighbours[j]))
    new_neighbours = neighbours_[:l]
    neighbours[j] = neighbours_[l:]
    for k in new_neighbours:
        neighbours[k][neighbours[k].index(j)] = i
    neighbours.append(new_neighbours)
    return neighbours


def generator_ring(n, params, seed):
    mod = params["mod"]
    neighbours = [[i + 1] for i in range(n - 1)] + [[0]]
    for l in range(n // mod):
        i = mod * l
        neighbours[i].append(i + 2 if i < n - 2 else 0)
    return neighbours


def generator_gnk_random_graph(n, params, seed):
    K = params["K"]
    G = nx.gnm_random_graph(n, (K * n) // 2, seed=seed)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator_gnk_random_graph_kln(n, params, seed):
    K = params["K"] * np.log(n)
    G = nx.gnm_random_graph(n, (K * n) // 2, seed=seed)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator_regular(n, params, seed):
    K = params["K"]
    G = nx.random_regular_graph(K, n, seed=seed)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator_triadic_closure(n, params, seed):
    p = params["p"]
    np.random.seed(seed)
    neighbours = [[1], [0]] + [[] for i in range(2, n)]
    bound = np.arange(2, n)
    first = np.random.randint(bound)
    second = np.random.randint(bound)
    while (second == first).any():
        second[second == first] = np.random.randint(bound[second == first])
    local = np.random.random(n - 2) < p
    for i in range(2, n):
        j = first[i - 2]
        k = np.random.choice(neighbours[j]) if local[i - 2] else second[i - 2]
        neighbours[i].append(j)
        neighbours[j].append(i)
        neighbours[i].append(k)
        neighbours[k].append(i)
    return neighbours


def generator_bubbles(n, params, seed):
    L = params["L"]
    p = params["p"] if "p" in params else 1
    n_bubbles_min = (n - 2) // L
    n_bubbles = n_bubbles_min + 1 if np.fmod(n - 2, L) else n_bubbles_min
    n = 2 + n_bubbles * L
    neighbours = [[] for i in range(n)]
    for i in range(L + 2):
        neighbours[i] += [i - 1 if i > 0 else L + 1, i + 1 if i < L + 1 else 0]
    edges = [(i, i + 1 if i < L + 1 else 0) for i in range(L + 2)]
    i = L + 2
    np.random.seed(seed)
    while i < n:
        j, k = (
            edges[np.random.randint(len(edges))]
            if p == 1 or np.random.random() < p
            else tuple(np.random.choice(np.arange(i), size=2, replace=False))
        )
        new_neighbours = [j] + list(range(i, i + L)) + [k]
        for l in range(1, L + 2):
            j = new_neighbours[l - 1]
            k = new_neighbours[l]
            edges.append((j, k))
            if j < n and k < n:
                neighbours[j].append(k)
                neighbours[k].append(j)
        i += L
    return neighbours


def generator_bubbles_L1(n, params, seed):
    L = params["L"]
    p = params["p"] if "p" in params else 1
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    np.random.seed(seed)
    for i in range(2, n):
        j, k = (
            edges[np.random.randint(len(edges))]
            if p == 1 or np.random.random() < p
            else tuple(np.random.choice(np.arange(i), size=2, replace=False))
        )
        neighbours[i] = [j, k]
        neighbours[j].append(i)
        neighbours[k].append(i)
        edges.append((i, j))
        edges.append((i, k))
    return neighbours


def generator_bubbles_L1_split(n, params, seed):
    np.random.seed(seed)
    s = params["s"]
    neighbours = [[1], [0]] + [[] for i in range(2, n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    split = np.random.random(n - 2) < s
    for i in range(2, n):
        e = np.random.randint(len(edges))
        j, k = edges[e]
        if split[i - 2]:
            neighbours[j][neighbours[j].index(k)] = i
            neighbours[k][neighbours[k].index(j)] = i
            edges[e] = (j, i)
        else:
            neighbours[j].append(i)
            neighbours[k].append(i)
            edges.append((j, i))
        neighbours[i] = [j, k]
        edges.append((k, i))
    return neighbours


def generator_bubbles_L1_ba(n, params, seed):
    L = params["L"]
    p = params["p"] if "p" in params else 1
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    edge_ends = [0, 1]
    np.random.seed(seed)
    for i in range(2, n):
        j, k = (
            edges[np.random.randint(len(edges))]
            if p == 1 or np.random.random() < p
            else tuple(np.random.choice(edge_ends, size=2, replace=False))
        )
        neighbours[i] = [j, k]
        neighbours[j].append(i)
        neighbours[k].append(i)
        edges.append((i, j))
        edges.append((i, k))
        edge_ends = edge_ends + [i, i, j, k]
    return neighbours


def generator_bubbles_house(n, params, seed):
    p = params["p"]
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    np.random.seed(seed)
    i = 2
    while i < n:
        j, k = (
            edges[np.random.randint(len(edges))]
            if p == 1 or np.random.random() < p
            else tuple(np.random.choice(np.arange(i), size=2, replace=False))
        )
        neighbours[i] = (
            [j, i + 1, i + 2] if i + 2 < n else [j, i + 1] if i + 1 < n else [j]
        )
        neighbours[j].append(i)
        edges.append((j, i))
        if i + 1 < n:
            neighbours[i + 1] = [i, i + 2] if i + 2 < n else [i]
            edges.append((i, i + 1))
        if i + 2 < n:
            neighbours[i + 2] = [i, i + 1, k]
            neighbours[k].append(i + 2)
            edges.append((i, i + 2))
            edges.append((i + 1, i + 2))
            edges.append((k, i + 2))
        i += 3 if i + 2 < n else 2 if i + 1 < n else 1
    return neighbours


def generator_big_bubbles(n, params, seed):
    """A chain of L nodes is attached to two distinct nodes, any distance appart
    - conjecture: this model breaks locality and does not has the emergent communities property (based on simulations)
    """
    L = params["L"]
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    i = 2
    while i < n:
        j = np.random.randint(i)
        k = np.random.randint(i)
        while k == j:
            k = np.random.randint(i)
        new_neighbours = [j] + list(range(i, i + L)) + [k]
        for l in range(1, L + 2):
            j = new_neighbours[l - 1]
            k = new_neighbours[l]
            if j < n and k < n:
                neighbours[j].append(k)
                neighbours[k].append(j)
        i += L
    return neighbours


def generator_bubbles_2(n, params, seed):
    """A chain of L nodes is attached to two distinct nodes at most W steps appart.
    - for W = 1 reduces to the bubble model
    - very slow, calculates A^W to identify nodes at most W steps appart
    - conjecture: has the mergent community property if and only if W in [1, 2] (based on simulations up to n=1280)
    """
    L = params["L"]
    W = params["W"]
    neighbours = [[] for i in range(n)]
    A = np.zeros((n, n), dtype=np.int8)
    neighbours[0].append(1)
    neighbours[1].append(0)
    A[0, 1] = 1
    A[1, 0] = 1
    i = 2
    while i < n:
        Ai = A[:i, :i]
        An = np.copy(Ai)
        AW = np.copy(Ai)
        for l in range(W):
            An = np.dot(An, Ai)
            AW = np.maximum(An, AW)
        np.fill_diagonal(AW, 0)
        paths = np.nonzero(AW)
        l = np.random.randint(len(paths[0]))
        j = paths[0][l]
        k = paths[1][l]
        new_neighbours = [j] + list(range(i, i + L)) + [k]
        for l in range(1, L + 2):
            j = new_neighbours[l - 1]
            k = new_neighbours[l]
            if j < n and k < n:
                A[j, k] = 1
                A[k, j] = 1
                neighbours[j].append(k)
                neighbours[k].append(j)
        i += L
    return neighbours


def generator_bubbles_2_L1_W2(n, params, seed):
    """A chain of L=1 nodes is attached to two distinct nodes at most W=2 steps appart.
    - conjecture: has the emergent community property
    """
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    pairs = [(0, 1)]
    i = 2
    while i < n:
        j1, j2 = pairs[np.random.randint(len(pairs))]
        n2_j = np.unique([j1, j2] + neighbours[j1] + neighbours[j2])
        pairs = pairs + [(k, i) for k in n2_j]
        neighbours[j1].append(i)
        neighbours[j2].append(i)
        neighbours[i].append(j1)
        neighbours[i].append(j2)
        i += 1
    return neighbours


def generator_bubbles_2_L1_W3(n, params, seed):
    """A chain of L nodes is attached to two distinct nodes at most W=3 steps appart.
    - conjecture: does bot has the mergent community property
    """
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    pairs = set([frozenset([0, 1])])
    i = 2
    while i < n:
        j1, j2 = tuple(np.random.choice(list(pairs)))
        n2_j = set([j1, j2] + neighbours[j1] + neighbours[j2])
        for k in neighbours[j1]:
            n2_j = n2_j.union(set(neighbours[k]))
        for k in neighbours[j2]:
            n2_j = n2_j.union(set(neighbours[k]))
        pairs = pairs.union(set([frozenset([k, i]) for k in n2_j]))
        pairs = pairs.union(
            set([frozenset([j1, k]) for k in neighbours[j2] if k != j1])
        )
        pairs = pairs.union(
            set([frozenset([j2, k]) for k in neighbours[j1] if k != j2])
        )
        neighbours[j1].append(i)
        neighbours[j2].append(i)
        neighbours[i].append(j1)
        neighbours[i].append(j2)
        i += 1
    return neighbours


def generator_bubbles_deterministic(n_generations, params, seed):
    """Deterministic Dorogovtsev-Goltsev-Mendes network."""
    G = nx.dorogovtsev_goltsev_mendes_graph(n_generations)
    return [list(G.neighbors(i)) for i in G.nodes]


def watts_strogatz(n, params, seed):
    """An extension of the Watts-Strogatz model.
    Input
        params: {'K': list, 'p': float}
    Description:
        n nodes are arranged in a ring.
        Each node is connected to len(K) neighbors to the right,
        their distances specified by K = [x_1, x_2, ...].
        Examples:
            Canonical Watts-Strogatz (K, p):  {'K': [1,2,..,k], 'p': p}, where k = K/2
            Watts-Strogatz with skip (s, k, p): {'K': [s, s+1, ...,s+k], 'p': p}
        Each of the len(K) links is reqired with probability p.
    """
    K = np.array(params["K"])
    p = params["p"]
    neighbours = [[] for i in range(n)]
    np.random.seed(seed=seed)
    for i in range(n):
        for j in i + K:
            if np.random.random() < p:
                j_ = np.random.randint(n)
                while j_ in neighbours[i] and j_ == i:
                    j_ = np.random.randint(n)
            else:
                j_ = j - n if j > n - 1 else j
            neighbours[i].append(j_)
            neighbours[j_].append(i)
    return neighbours


def generator_barabasi_albert(n, params, seed):
    """Barabasi-Albert model with m new links per node and baseline attractiveness a."""
    m = params["m"]
    a = params["a"] if "a" in params else m
    founders = set(range(m + 1))
    neighbours = [list(founders - {i}) for i in range(m + 1)]
    replicas = [i for i in range(m + 1) for _ in range(a)]
    np.random.seed(seed=seed)
    for i in range(m + 1, n):
        n_replicas = len(replicas)
        new_neighbours = []
        for u in range(m):
            r = np.random.randint(n_replicas)
            j = replicas[r]
            while i in neighbours[j]:
                r = np.random.randint(n_replicas)
                j = replicas[r]
            new_neighbours.append(j)
            neighbours[j].append(i)
            replicas += [j]
        replicas += [i] * a
        neighbours.append(new_neighbours)
    return neighbours


def generator_local_search(n, params, seed):
    """Local-Search model with depth d."""
    d = params["d"]
    neighbours = [[1], [0]]
    np.random.seed(seed=seed)
    for i in range(2, n):
        visited = [np.random.randint(i)]
        for step in range(d):
            j = visited[step]
            l = np.random.randint(len(neighbours[j]))
            k = neighbours[j][l]
            visited.append(k)
        visited_unique = list(np.unique(visited))
        for j in visited_unique:
            neighbours[j].append(i)
        neighbours.append(visited_unique)
    return neighbours


def generator_dup_split(n, params, seed):
    """Duplication-Split model with duplication rate q."""
    np.random.seed(seed=seed)
    p = params["p"] if "p" in params else 0
    q = params["q"]
    neighbours = neighbours = [[3, 1], [0, 2], [1, 3], [2, 0]]
    is_duplicate = np.random.random(n) < q
    for i in range(4, n):
        j = np.random.randint(i)
        degree_j = len(neighbours[j])
        if is_duplicate[i]:
            for k in neighbours[j]:
                neighbours[k].append(i)
            new_neighbours = [k for k in neighbours[j]]
            if p > 0 and np.random.random() < p:
                neighbours[j].append(i)
                new_neighbours.append(j)
            neighbours.append(new_neighbours)
        else:
            degree_j = len(neighbours[j])
            l = np.random.randint(degree_j)
            k = neighbours[j][l]
            neighbours[j][l] = i
            neighbours[k][neighbours[k].index(j)] = i
            neighbours.append([j, k])
    return neighbours


def generator_dup_split_directed(n, params, seed):
    """Directed Duplication-Split model with duplication rate q.

    Each step adds one node by acting on a uniformly random existing node j:

    Duplication (prob q) -- new node i copies j's arcs in both directions:
        for every arc  m -> j:  add  m -> i
        for every arc  j -> k:  add  i -> k

    Split (prob 1 - q) -- two rules, selected by params["split"]:
        "node" (default): new node i takes over ALL of j's out-arcs
            for every arc  j -> k:  replace with  i -> k   (j keeps in-arcs)
            add arc  j -> i
        "link": subdivide a single random out-arc j -> k of j
            replace  j -> k  with  j -> i -> k
        (if j has no out-arc, "link" falls back to "node".)

    With duplicate_ends=False (default) the two endpoints of the seed path,
    node 0 (source) and node 2 (sink), are never selected for duplication or
    split, so they remain the unique source and sink. With duplicate_ends=True
    every node is selectable.

    Returns the out-adjacency: out[v] is the list of successors of v.
    """
    np.random.seed(seed=seed)
    q = params["q"]
    duplicate_ends = params.get("duplicate_ends", False)
    link_split = params.get("split", "node") == "link"
    # directed seed: a path 0 -> 1 -> 2
    out = [[1], [2], []]
    inn = [[], [0], [1]]
    is_duplicate = np.random.random(n) < q
    for i in range(3, n):
        if duplicate_ends:
            j = np.random.randint(i)
        else:
            # select uniformly among all nodes except the ends {0, 2}
            j = np.random.randint(i)
            while j == 0 or j == 2:
                j = np.random.randint(i)
        if is_duplicate[i]:
            for m in inn[j]:  # predecessors: m -> i
                out[m].append(i)
            for k in out[j]:  # successors: i -> k
                inn[k].append(i)
            inn.append(list(inn[j]))
            out.append(list(out[j]))
        elif link_split and out[j]:
            # link split: subdivide one random out-arc j -> k into j -> i -> k
            l = np.random.randint(len(out[j]))
            k = out[j][l]
            out[j][l] = i  # j -> i
            inn[k][inn[k].index(j)] = i  # k's predecessor j -> i
            out.append([k])  # i -> k
            inn.append([j])  # i's predecessor is j
        else:
            # node split: new node i takes over all of j's out-arcs
            old_out = list(out[j])
            for k in old_out:  # redirect j -> k into i -> k
                inn[k][inn[k].index(j)] = i
            out.append(old_out)  # out[i] = j's old out-arcs
            inn.append([j])  # i's only predecessor is j
            out[j] = [i]  # j now points only to i
    return out


def generator_dup_divergence(n, params, seed):
    """Duplication-Split model with duplication rate q."""
    np.random.seed(seed=seed)
    p = params["p"] if "p" in params else 0
    q = params["q"]
    neighbours = [[1], [0]] + [[] for i in range(2, n)]
    for i in range(2, n):
        j = np.random.randint(i)
        new_neighbours = np.random.permutation(neighbours[j])
        m = np.random.binomial(len(neighbours[j]), 1 - q)
        while m == 0:
            m = np.random.binomial(len(neighbours[j]), 1 - q)
        for k in new_neighbours[:m]:
            neighbours[i].append(k)
            neighbours[k].append(i)
        if np.random.random() < p:
            neighbours[i].append(j)
            neighbours[j].append(i)
    return neighbours


def generator_dup_divergence_symmetric(n, params, seed):
    """Duplication-Split model with duplication rate q."""
    np.random.seed(seed=seed)
    p = params["p"] if "p" in params else 0
    q = params["q"]
    neighbours = [[1], [0]] + [[] for i in range(2, n)]
    for i in range(2, n):
        j = np.random.randint(i)
        for k in neighbours[j]:
            if np.random.random() < q:
                if np.random.random() < 0.5:
                    neighbours[k].remove(j)
                    neighbours[j].remove(k)
                    neighbours[i].append(k)
                    neighbours[k].append(i)
            else:
                neighbours[i].append(k)
                neighbours[k].append(i)
        if np.random.random() < p:
            neighbours[i].append(j)
            neighbours[j].append(i)
    return neighbours


def nearest_neighbor(n, params, seed=0):
    u = params["u"]

    def index2tuple(k):
        i = int(n - 2 - np.floor(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
        j = k + i + 1 - (n * (n - 1)) // 2 + ((n - i) * ((n - i) - 1)) // 2
        return (i, j)

    def tuple2index(i, j):
        return (n * (n - 1)) // 2 - ((n - i) * ((n - i) - 1)) // 2 + j - i - 1

    neighbors = [[1], [0]] + [[] for i in range(2, n)]
    potential_edges = []
    np.random.seed(seed)
    i = 2
    while i < n:
        if np.random.random() < 1 - u:
            j = np.random.randint(i)
            potential_edges = potential_edges + [
                tuple2index(k, i) for k in neighbors[j]
            ]
            neighbors[j].append(i)
            neighbors[i].append(j)
            i += 1
        elif potential_edges:
            l = np.random.randint(len(potential_edges))
            j, k = index2tuple(potential_edges.pop(l))
            new_potential_edges = {
                tuple2index(min(l, j), max(l, j))
                for l in neighbors[k]
                if l not in neighbors[j]
            }
            potential_edges = set(potential_edges).union(new_potential_edges)
            new_potential_edges = {
                tuple2index(min(l, k), max(l, k))
                for l in neighbors[j]
                if l not in neighbors[k]
            }
            potential_edges = list(potential_edges.union(new_potential_edges))
            neighbors[j].append(k)
            neighbors[k].append(j)
    return neighbors


def generator_dorogovtsev_goltsev_mendes(n, params, seed):
    G = nx.dorogovtsev_goltsev_mendes_graph(n)
    return [list(G.neighbors(i)) for i in G.nodes]


try:
    # Cython-accelerated drop-in replacements (build: python3 setup_fast.py build_ext --inplace).
    # NOTE: uses a C++ RNG, so a given seed produces different graphs than the pure-Python versions.
    from ramsey_netcom.generators_fast import (
        generator_local_search,
        generator_bubbles,
        generator_dup_split,
        generator_bubbles_2,
        generator_bubbles_2_L1_W3,
        generator_dup_split_directed,
        nearest_neighbor,
        average_shortest_path_multiplicity_csr as _fast_multiplicity,
        directed_average_shortest_path_multiplicity_csr as _fast_multiplicity_directed,
    )
except ImportError:
    # use the pure-Python versions defined above
    _fast_multiplicity = None
    _fast_multiplicity_directed = None


def generator(n, params, seed):
    """a graph generator calling different models"""

    if params["model-"] == "ring":
        neighbours = generator_ring(n, params, seed)
    elif params["model-"] == "gnk":
        neighbours = generator_gnk_random_graph(n, params, seed)
    elif params["model-"] == "gnk_kln":
        neighbours = generator_gnk_random_graph_kln(n, params, seed)
    elif params["model-"] == "ws":
        neighbours = watts_strogatz(n, params, seed)
    elif params["model-"] == "dd":
        neighbours = generator_dup_divergence(n, params, seed)
    elif params["model-"] == "dds":
        neighbours = generator_dup_divergence_symmetric(n, params, seed)
    elif params["model-"] == "ds":
        neighbours = generator_dup_split(n, params, seed)
    elif params["model-"] == "dsd":
        neighbours = generator_dup_split_directed(n, params, seed)
    elif params["model-"] == "ba":
        neighbours = generator_barabasi_albert(n, params, seed)
    elif params["model-"] == "ls":
        neighbours = generator_local_search(n, params, seed)
    elif params["model-"] == "tc":
        neighbours = generator_triadic_closure(n, params, seed)
    elif params["model-"] == "nn":
        neighbours = nearest_neighbor(n, params, seed)
    elif params["model-"] == "bb":
        if params["L"] == 1:
            if "W" in params:
                if params["W"] == 2:
                    neighbours = generator_bubbles_2_L1_W2(n, params, seed)
                elif params["W"] == 3:
                    neighbours = generator_bubbles_2_L1_W3(n, params, seed)
                elif params["W"] == "any":
                    neighbours = generator_big_bubbles(n, params, seed)
                else:
                    neighbours = generator_bubbles_2(n, params, seed)
            else:
                neighbours = generator_bubbles_L1(n, params, seed)
        else:
            if "W" in params:
                if params["W"] == "any":
                    neighbours = generator_big_bubbles(n, params, seed)
                else:
                    neighbours = generator_bubbles_2(n, params, seed)
            else:
                neighbours = generator_bubbles(n, params, seed)
    elif params["model-"] == "bbs":
        if params["L"] == 1:
            neighbours = generator_bubbles_L1_split(n, params, seed)
    elif params["model-"] == "bbh":
        neighbours = generator_bubbles_house(n, params, seed)
    elif params["model-"] == "bbd":
        neighbours = generator_bubbles_deterministic(n, params, seed)
    elif params["model-"] == "regular":
        neighbours = generator_regular(n, params, seed)
    elif params["model-"] == "dgm":
        neighbours = generator_dorogovtsev_goltsev_mendes(n, params, seed)
    return neighbours


def get_model_name(params):
    return "_".join(
        [
            f"{k}{'_'.join([str(e) for e in v]) if type(v) == list else v}"
            for (k, v) in params.items()
        ]
    )


def graphtool_from_neighbours(neighbours_input, directed=False):
    """Converts a list of adjecencies into a graph-tool undirected graph."""
    neighbours = neighbours_input.copy()
    n = len(neighbours)
    for i in range(n):
        neighbours_ = np.array(neighbours[i])
        neighbours[i] = list(neighbours_[neighbours_ > i])
    return gt.Graph(dict(zip(range(n), neighbours)), directed=directed)


def graphtool_directed_from_out_neighbours(out_neighbours):
    """Build a directed graph-tool graph from an out-adjacency (list of successors)."""
    n = len(out_neighbours)
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    g.add_edge_list([(i, k) for i in range(n) for k in out_neighbours[i]])
    return g


def igraph_from_neighbours(neighbours, directed=False):
    n = len(neighbours)
    edges = [(i, j) for i in range(n) for j in neighbours[i] if i < j]
    return ig.Graph(n=n, edges=edges, directed=directed)


def get_n_communities(g, method="blocks", neighbours=None, line_graph=False):
    if line_graph:
        g, vmap = gt.line_graph(g)
    if method == "blocks":
        state = gt.minimize_blockmodel_dl(g)
        labels = state.get_blocks()
    elif method == "blocks-nodeg":
        state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=False))
        labels = state.get_blocks()
    elif method == "modularity":
        gx = nx.from_edgelist([(s, t) for s, t, i in g.iter_edges([g.edge_index])])
        communities = nx.community.louvain_communities(gx)
        labels = np.empty(g.num_vertices(), dtype=int)
        for c in range(len(communities)):
            labels[np.array(list(communities[c]))] = c
        state = None
    elif method == "potts":
        g_ = ig.Graph.from_graph_tool(g)
        state = g_.community_spinglass(gamma=0.1, update_rule="config")
        labels = state.membership
    elif method == "infomap":
        im = Infomap(silent=True, regularized=True)
        im.add_links([(s, t) for s, t, i in g.iter_edges([g.edge_index])])
        im.run()
        state = None
        labels = [node.module_id for node in im.nodes]
    elif method == "heatcapacity":
        # calculates the number of maxima in the heat capacity plot vs 1/tau
        tau, s, c = heat_capacity(neighbours)
        maxima = np.array(argrelmax(c)[0])
        return g, None, [], len(maxima)
    elif method == "heatcapacity_rw":
        # calculates the number of maxima in the heat capacity plot vs 1/tau
        tau, s, c = heat_capacity(neighbours, random_walk=True)
        maxima = np.array(argrelmax(c)[0])
        return g, None, [], len(maxima)
    elif method == "hgc":
        edge2nodes = neighbours
        node2edges = neighbours
        n_instances = 1
        n_groups_max = 10
        p, labels = hgc(edge2nodes, node2edges, n_groups_max, n_instances)
        state = None
    return g, state, labels, len(np.unique(labels))


def get_n_communities_from_neighbours(
    neighbours, method="blocks", directed=False, line_graph=False
):
    g = graphtool_from_neighbours(neighbours, directed=directed)
    return get_n_communities(g, line_graph=line_graph, method=method)


def n_instances_with_cummunities(n, params, nr, method, rewire, line_graph):
    c = np.empty(nr, dtype=int)
    for r in range(nr):
        neighbours = generator(n, params, seed=r)
        g = graphtool_from_neighbours(neighbours, directed=False)
        if rewire > 0:
            rejection_count = gt.random_rewire(g, n_iter=rewire)
        g, state, labels, n_communities = get_n_communities(
            g, method=method, neighbours=neighbours, line_graph=line_graph
        )
        c[r] = n_communities
    return c


def count_wc(n, params, nr, method, rewire, kappa, line_graph):
    c = n_instances_with_cummunities(n, params, nr, method, rewire, line_graph)
    return np.sum(c >= kappa)


def ramsey_community_number(
    params, epsilon, nr, n0=10, method="blocks", rewire=0, kappa=2, line_graph=False
):
    """Estimates the Ramsey community number of a graph."""
    nr_95 = int((1 - epsilon) * nr)
    # find upper bound, some n satisfying nr_c >= nr_95
    n = n0
    nr_c = count_wc(n, params, nr, method, rewire, kappa, line_graph)
    if nr_c > nr_95:
        # n0 is an upper bound, find lower bound
        while nr_c > nr_95:
            n_right = n
            n = n // 2
            nr_c = count_wc(n, params, nr, method, rewire, kappa, line_graph)
            print(f"[{n}, {n_right}], fraction with communities: {nr_c/nr} ...")
        n_left = n
    elif nr_c < nr_95:
        # n0 is an lower bound, find upper bound
        while nr_c < nr_95:
            n_left = n
            n *= 2
            nr_c = count_wc(n, params, nr, method, rewire, kappa, line_graph)
            print(f"[{n_left}, {n}], fraction with communities: {nr_c/nr} ...")
        n_right = n
    else:
        # nothing to do
        n_left = nr_c
        n_right = nr_c
    if n_left != n_right:
        # binary search, find min n satisfying nr_c = nr_95
        while abs(n_left - n_right) > 1:
            print(
                f"binary search: [{n_left}, {n_right}], n: {n}, fraction with communities: {nr_c/nr} ..."
            )
            n = (n_left + n_right) // 2
            nr_c_previous = nr_c
            nr_c = count_wc(n, params, nr, method, rewire, kappa, line_graph)
            if nr_c >= nr_95:
                n_right = n
            else:
                n_left = n
        if nr_c < nr_95:
            n += 1
            nr_c = count_wc(n, params, nr, method, rewire, kappa, line_graph)
    print(
        f"binary search: [{n_left}, {n_right}], r_c: {n}, fraction with communities: {nr_c/nr}"
    )
    return n


def draw_instance(n, params, seed, path, rewire=0, extension="pdf"):
    """Auxiliary method to draw a graph instance."""
    neighbours = generator(n, params, seed)
    g = graphtool_from_neighbours(neighbours)
    model_name = get_model_name(params)
    filename = os.path.join(path, f"{model_name}_n{n}_seed{seed}.pdf")
    state = gt.minimize_blockmodel_dl(g)
    state.draw(output=filename)
    if rewire > 0:
        rejection_count = gt.random_rewire(g, n_iter=rewire)
        filename = os.path.join(
            path, f"{model_name}_n{n}_seed{seed}_rewire{rewire}.{extension}"
        )
        state = gt.minimize_blockmodel_dl(g)
        state.draw(output=filename)


def find_instances_with_communities(
    n,
    params,
    n_instances,
    path,
    method="blocks",
    has_communities=True,
    line_graph=False,
    extension="pdf",
):
    """Auxiliary method to find and draw instances with 2 or more communities."""
    model_name = get_model_name(params)
    seed = 0
    count = 0
    while count < n_instances:
        neighbours = generator(n, params, seed)
        g, state, labels, n_communities = get_n_communities_from_neighbours(
            neighbours,
            method=method,
        )
        if (has_communities and n_communities > 1) or (
            not has_communities and n_communities == 1
        ):
            if has_communities:
                filename = os.path.join(
                    path, f"{model_name}_n{n}_seed{seed}_{method}.{extension}"
                )
            else:
                filename = os.path.join(
                    path, f"{model_name}_n{n}_seed{seed}_{method}_no_com.{extension}"
                )
            if method == "blocks":
                state.draw(output=filename)
            elif method == "potts":
                labels = list(labels)
                print(labels)
                # Normalised RGB color.
                # 0->Red, 1->Blue
                #                red_blue_map = dict((i, (1,1,0)) for i in range(10))
                red_blue_map = {
                    0: (0, 0, 0),
                    1: (1, 0, 0),
                    2: (0, 1, 0),
                    3: (1, 1, 0),
                    4: (0, 0, 1),
                    5: (1, 0, 1),
                    6: (0, 1, 1),
                    7: (1, 1, 1),
                }
                # Create new vertex property
                plot_color = g.new_vertex_property("vector<double>")
                # add that property to graph
                g.vertex_properties["plot_color"] = plot_color
                # assign a value to that property for each node of that graph
                for v in g.vertices():
                    plot_color[v] = red_blue_map[labels[int(v)]]
                gt.graph_draw(
                    g,
                    vertex_fill_color=g.vertex_properties["plot_color"],
                    output=filename,
                )
            #                gt.graph_draw(g, vertex_fill_color=dict(zip(range(len(labels)), labels)), output=filename)
            count += 1
        seed += 1


def total_energy(g):
    a = gt.adjacency(g)
    # adjacency is symmetric -> eigvalsh is ~6x faster than the general eigvals
    lambda_ = np.linalg.eigvalsh(a.toarray())
    return np.sum(np.maximum(0, lambda_))


def heat_capacity(
    neighbours, inv_tau_min=0.0001, inv_tau_max=1000, n_points=1000, random_walk=False
):
    n = len(neighbours)
    L = np.zeros((n, n))
    if random_walk:
        nodes_with_links = [i for i in range(n) if neighbours[i]]
        for i in nodes_with_links:
            L[i, np.array(neighbours[i])] = -1 / len(neighbours[i])
            L[i, i] = 1
    else:
        nodes_with_links = [i for i in range(n) if neighbours[i]]
        for i in nodes_with_links:
            L[i, np.array(neighbours[i])] = -1
            L[i, i] = len(neighbours[i])
    # the random-walk Laplacian is not symmetric (row-normalized); the standard
    # Laplacian is, so use the faster symmetric solver in that case.
    lambda_ = np.real(np.linalg.eigvals(L)) if random_walk else np.linalg.eigvalsh(L)
    lambda_[lambda_ < 0] = 0
    tau = 1 / np.exp(np.linspace(np.log(inv_tau_min), np.log(inv_tau_max), n_points))
    lambda_v = np.repeat(lambda_[:, np.newaxis], n_points, axis=1)
    tau_v = np.repeat(tau[np.newaxis, :], n, axis=0)
    e = np.exp(-lambda_v * tau_v)
    sum_0 = np.sum(e, axis=0)
    sum_1 = np.sum(lambda_v * e, axis=0)
    mu = e / sum_0
    mu_log_mu = mu * (1 + np.log(mu + 1e-10))
    s = -np.sum(mu * np.log(mu + 1e-10), axis=0) / np.log(n)
    c = tau * np.sum((sum_1 / sum_0 - lambda_v) * mu_log_mu, axis=0) / np.log(n)
    return tau, s, c


def _out_csr(g):
    """Out-edge adjacency of g in CSR form (int32 indptr/indices)."""
    n = g.num_vertices()
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(g.get_out_degrees(np.arange(n)))
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for v in range(n):
        indices[indptr[v] : indptr[v + 1]] = g.get_out_neighbors(v)
    return indptr, indices


def average_shortest_path_multiplicity_directed_py(g):
    """Mean number of shortest *directed* paths over all reachable (source, target)
    ordered pairs. Reference Python implementation (the Cython version mirrors it).

    For each source s, a forward BFS over out-edges yields sigma[t] = number of
    shortest directed paths s -> t. The average is taken only over pairs where t
    is reachable from s (unreachable pairs are excluded, not counted as zero).
    """
    n = g.num_vertices()
    out = [np.asarray(g.get_out_neighbors(v)) for v in range(n)]
    total = 0.0
    reachable = 0
    for s in range(n):
        dist = np.full(n, -1, dtype=np.int64)
        sigma = np.zeros(n)
        dist[s] = 0
        sigma[s] = 1.0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            for w in out[u]:
                if dist[w] == -1:
                    dist[w] = du + 1
                    sigma[w] = sigma[u]
                    queue.append(w)
                elif dist[w] == du + 1:
                    sigma[w] += sigma[u]
        for t in queue[1:]:  # reachable targets (s excluded)
            total += sigma[t]
        reachable += len(queue) - 1
    if reachable == 0:
        return float("nan")
    return total / reachable


def average_shortest_path_multiplicity(g, directed=False):
    if directed:
        if _fast_multiplicity_directed is not None:
            indptr, indices = _out_csr(g)
            return _fast_multiplicity_directed(indptr, indices, g.num_vertices())
        return average_shortest_path_multiplicity_directed_py(g)
    if _fast_multiplicity is not None:
        A = gt.adjacency(g).tocsr()
        return _fast_multiplicity(
            np.ascontiguousarray(A.indptr, dtype=np.int32),
            np.ascontiguousarray(A.indices, dtype=np.int32),
            g.num_vertices(),
        )
    v = g.vertices()
    return np.mean(
        [gt.count_shortest_paths(g, i, j) for i, j in itertools.combinations(v, r=2)]
    )


def connected_pairs_fraction(g):
    """Mean over all ordered node pairs of (shortest-path multiplicity > 0), i.e.
    the fraction of node pairs that are connected.

    Computed from the connected-component sizes: the reachable ordered pairs are
    sum_c s_c (s_c - 1), normalized by n (n - 1) (same normalization as the
    undirected average_shortest_path_multiplicity).
    """
    n = g.num_vertices()
    if n < 2:
        return float("nan")
    _, hist = gt.label_components(g, directed=False)
    sizes = np.asarray(hist, dtype=float)
    return float(np.sum(sizes * (sizes - 1)) / (n * (n - 1)))


def shortest_path_multiplicity_random_removal(g, n_realizations=100):
    """Average shortest-path multiplicity vs network size under random node removal.

    Target sizes are n_nodes_array = [n, 2**l, 2**(l-1), ..., 2**3] with
    l = floor(log2(n)) (a leading 2**l == n is dropped). For each realization the
    graph is copied and nodes are removed at random progressively down through the
    target sizes, measuring at each size the average shortest-path multiplicity
    (mean over node pairs) and the fraction of connected pairs
    (mean over node pairs of multiplicity > 0). Results are averaged over
    n_realizations.

    Returns (n_nodes_array, multiplicity_array, connected_pairs_array).
    """
    n = g.num_vertices()
    l = int(np.log(n) / np.log(2))
    n_nodes_array = [n] + [2 ** k for k in range(l, 2, -1) if 2 ** k < n]
    mult = np.zeros(len(n_nodes_array))
    conn = np.zeros(len(n_nodes_array))
    # full graph (i == 0) has no removal -> deterministic, compute once
    mult[0] = average_shortest_path_multiplicity(g)
    conn[0] = connected_pairs_fraction(g)
    for _ in range(n_realizations):
        g_r = g.copy()
        for i in range(1, len(n_nodes_array)):
            n_remove = n_nodes_array[i - 1] - n_nodes_array[i]
            victims = np.random.choice(
                g_r.num_vertices(), size=n_remove, replace=False
            )
            g_r.remove_vertex(victims, fast=True)
            mult[i] += average_shortest_path_multiplicity(g_r)
            conn[i] += connected_pairs_fraction(g_r)
    mult[1:] /= n_realizations
    conn[1:] /= n_realizations
    return n_nodes_array, mult, conn


def load_network(name):
    """Load a real network from the Netzschleuder repository (networks.skewed.de)
    as a graph-tool graph, e.g. load_network("celegansneural").

    Browse the catalogue with ``list(gt.collection.ns)`` and inspect metadata via
    ``gt.collection.ns_info[name]``. The graph is downloaded on first use and
    cached locally by graph-tool.
    """
    return gt.collection.ns[name]


def network_shortest_path_multiplicity(name, directed=None):
    """Average shortest-path multiplicity of a named Netzschleuder network.

    ``directed`` defaults to the graph's own orientation (``g.is_directed()``);
    pass True/False to override. For the size-vs-multiplicity removal curve use
    ``shortest_path_multiplicity_random_removal(load_network(name))``.
    """
    g = load_network(name)
    if directed is None:
        directed = g.is_directed()
    return average_shortest_path_multiplicity(g, directed=directed)


def get_degrees(g):
    k = g.get_total_degrees(g.get_vertices())
    mean_k = k.mean()
    mean_k2 = (k**2).mean()
    avg_excess_degree = (mean_k2 - mean_k) / mean_k
    return mean_k, avg_excess_degree


def n_communities_vs_n(
    params,
    savepath,
    nr=1000,
    n_min=10,
    n_max=200,
    dn=10,
    log=False,
    method="blocks",
    append=False,
    full=False,
    model=None,
    directed=False,
):
    """wrapper to investigate community properties at different network sizes"""

    if model is None:
        model = get_model_name(params)
    log_label = "_log" if log else ""
    method_label = "" if method == "blocks" else f"_{method}_regularized"
    filename = f"{savepath}/{model}_nr{nr}_vs_n{log_label}{method_label}.csv"
    print(filename)

    if append:
        df = pd.read_csv(filename)
        data = dict()
        for c in df.columns:
            data[c] = df[c].to_list()
    else:
        data = dict(
            n=[],
            unique_mean=[],
            unique_p=[],
            energy=[],
            multipath=[],
            mean_k=[],
            mean_kk=[],
            rnd_unique_mean=[],
            rnd_unique_p=[],
            rnd_energy=[],
            multipath_rnd=[],
        )
        if full:
            data["c"] = []
            data["c_rnd"] = []
            data["diameter"] = []
            data["diameter_rnd"] = []

    n = n_min
    while n < n_max:
        print(n)
        kappa = []
        e = []
        kappa_rnd = []
        e_rnd = []
        c = []
        c_rnd = []
        diameter = []
        diameter_rnd = []
        multipath = []
        mean_k = []
        mean_kk = []
        multipath_rnd = []
        for r in range(nr):
            neighbours = generator(n, params, r)
            g, state, labels, n_communities = get_n_communities_from_neighbours(
                neighbours, method=method, directed=directed,
            )
            kappa.append(n_communities)
            e.append(total_energy(g))
            if full:
                c.append(gt.vertex_average(g, gt.local_clustering(g))[0])
                d = np.mean(gt.shortest_distance(g))
                diameter.append(d)
            multipath.append(average_shortest_path_multiplicity(g, directed=directed))
            mean_k_, mean_kk_ = get_degrees(g)
            mean_k.append(mean_k_)
            mean_kk.append(mean_kk_)
            rejection_count = gt.random_rewire(g)
            g, state, labels, n_communities_rnd = get_n_communities(g, method=method)
            kappa_rnd.append(n_communities_rnd)
            e_rnd.append(total_energy(g))
            if full:
                c_rnd.append(gt.vertex_average(g, gt.local_clustering(g))[0])
                d_rnd = np.mean(gt.shortest_distance(g))
                diameter_rnd.append(d_rnd)
            multipath_rnd.append(average_shortest_path_multiplicity(g, directed=directed))
        data["n"].append(n)
        x = np.array(kappa)
        data["unique_mean"].append(x.mean())
        data["unique_p"].append(np.mean(x > 1))
        data["energy"].append(np.mean(e))
        data["multipath"].append(np.array(multipath).mean())
        data["mean_k"].append(np.array(mean_k).mean())
        data["mean_kk"].append(np.array(mean_kk).mean())
        x = np.array(kappa_rnd)
        data["rnd_unique_mean"].append(x.mean())
        data["rnd_unique_p"].append(np.mean(x > 1))
        data["rnd_energy"].append(np.mean(e_rnd))
        if full:
            data["c"].append(np.array(c).mean())
            data["c_rnd"].append(np.array(c_rnd).mean())
            data["diameter"].append(np.array(diameter).mean())
            data["diameter_rnd"].append(np.array(diameter_rnd).mean())
        data["multipath_rnd"].append(np.array(multipath_rnd).mean())

        if log:
            n *= 2
        else:
            n += dn
        pd.DataFrame(data).to_csv(filename, index=False)
    pd.DataFrame(data).to_csv(filename, index=False)
