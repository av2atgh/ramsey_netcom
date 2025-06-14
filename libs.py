import os
import json
import numpy as np
import graph_tool.all as gt
import igraph as ig
import networkx as nx
from infomap import Infomap
from ramsey_netcom.hgc import hgc


def generator_gnk_random_graph(n, params, seed):
    K = params["K"]
    G = nx.gnm_random_graph(n, K * n, seed=seed)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator_regular(n, params, seed):
    K = params["K"]
    G = nx.random_regular_graph(K, n, seed=seed)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator_bubbles(n, params, seed):
    L = params["L"]
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    i = 2
    while i < n:
        e = np.random.randint(len(edges))
        j = edges[e][0]
        k = edges[e][1]
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
    - conjecture: has the mergent community property
    """
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    pairs = set([frozenset([0, 1])])
    i = 2
    while i < n:
        j1, j2 = tuple(np.random.choice(list(pairs)))
        n2_j = set([j1, j2] + neighbours[j1] + neighbours[j2])
        pairs = pairs.union(set([frozenset([i, k]) for k in n2_j]))
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
        pairs = pairs.union(set([frozenset([i, k]) for k in n2_j]))
        pairs = pairs.union(
            set([frozenset([j1, k]) for k in neighbours[j2] if j1 != k])
        )
        pairs = pairs.union(
            set([frozenset([j2, k]) for k in neighbours[j1] if j2 != k])
        )
        neighbours[j1].append(i)
        neighbours[j2].append(i)
        neighbours[i].append(j1)
        neighbours[i].append(j2)
        i += 1
    print(len(pairs) / (n * (n - 1) / 2))
    return neighbours


def generator_bubbles_deterministic(n_generations, params, seed):
    """A chain of L nodes is attached to the nodes at the end of every link in the network.
    - number of nodes increases as n = (3^n_generations + 1) / 2
    """
    L = params["L"]
    n = (2 + L + L * (L + 2) ** n_generations) // (L + 1)
    neighbours = [[] for i in range(n)]
    neighbours[0].append(1)
    neighbours[1].append(0)
    edges = [(0, 1)]
    i = 2
    g = 0
    while g < n_generations:
        edges_current = edges.copy()
        for e in edges_current:
            j = e[0]
            k = e[1]
            new_neighbours = [j] + list(range(i, i + L)) + [k]
            for l in range(1, L + 2):
                j = new_neighbours[l - 1]
                k = new_neighbours[l]
                edges.append((j, k))
                if j < n and k < n:
                    neighbours[j].append(k)
                    neighbours[k].append(j)
            i += L
        g += 1
    return neighbours


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
    neighbours = [[1], [0]]
    is_duplicate = np.random.random(n) < q
    for i in range(2, n):
        j = np.random.randint(i)
        if is_duplicate[i]:
            for k in neighbours[j]:
                neighbours[k].append(i)
            new_neighbours = [k for k in neighbours[j]]
            if p > 0 and np.random.random() < p:
                neighbours[j].append(i)
                new_neighbours.append(j)
        else:
            l = np.random.randint(len(neighbours[j]))
            k = neighbours[j][l]
            m = neighbours[k].index(j)
            neighbours[j][l] = i
            neighbours[k][m] = i
            new_neighbours = [j, k]
        neighbours.append(new_neighbours)
    return neighbours


def generator_dup_divergence(n, params, seed):
    """Duplication-Split model with duplication rate q."""
    np.random.seed(seed=seed)
    p = params["p"] if "p" in params else 0
    q = params["q"]
    neighbours = [[1], [0]]
    connect_duplicates = np.random.random(size=n) < p
    for i in range(2, n):
        j = np.random.randint(i)
        selected_neighbours = np.random.permutation(neighbours[j])
        n_neighbours = len(selected_neighbours)
        m = np.random.binomial(n_neighbours, 1 - q)
        while m == 0:
            m = np.random.binomial(n_neighbours, 1 - q)
        new_neighbours = list(selected_neighbours[:m])
        for k in new_neighbours:
            neighbours[k].append(i)
        if connect_duplicates[i]:
            neighbours[j].append(i)
            new_neighbours.append(j)
        neighbours.append(list(new_neighbours))
    return neighbours


def generator_dorogovtsev_goltsev_mendes(n, params, seed):
    G = nx.dorogovtsev_goltsev_mendes_graph(n)
    return [list(G.neighbors(i)) for i in G.nodes]


def generator(n, params, seed):
    if params["model-"] == "gnk":
        neighbours = generator_gnk_random_graph(n, params, seed)
    elif params["model-"] == "ws":
        neighbours = watts_strogatz(n, params, seed)
    elif params["model-"] == "dd":
        neighbours = generator_dup_divergence(n, params, seed)
    elif params["model-"] == "ds":
        neighbours = generator_dup_split(n, params, seed)
    elif params["model-"] == "ba":
        neighbours = generator_barabasi_albert(n, params, seed)
    elif params["model-"] == "ls":
        neighbours = generator_local_search(n, params, seed)
    elif params["model-"] == "bb":
        neighbours = generator_bubbles(n, params, seed)
    elif params["model-"] == "bb2":
        if params["L"] == 1 and params["W"] == 2:
            neighbours = generator_bubbles_2_L1_W2(n, params, seed)
        elif params["L"] == 1 and params["W"] == 3:
            neighbours = generator_bubbles_2_L1_W3(n, params, seed)
        else:
            neighbours = generator_bubbles_2(n, params, seed)
    elif params["model-"] == "bbb":
        neighbours = generator_big_bubbles(n, params, seed)
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


def igraph_from_neighbours(neighbours, directed=False):
    n = len(neighbours)
    edges = [(i, j) for i in range(n) for j in neighbours[i] if i < j]
    return ig.Graph(n=n, edges=edges, directed=directed)


def get_n_communities(g, method="blocks", neighbours=None):
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
    elif method == "hgc":
        edge2nodes = neighbours
        node2edges = neighbours
        n_instances = 1
        n_groups_max = 10
        p, labels = hgc(edge2nodes, node2edges, n_groups_max, n_instances)
        state = None
    return g, state, labels, len(np.unique(labels))


def get_n_communities_from_neighbours(neighbours, method="blocks", directed=False):
    g = graphtool_from_neighbours(neighbours, directed=directed)
    return get_n_communities(g, method=method)


def n_instances_with_cummunities(n, params, nr, method, rewire):
    c = np.empty(nr, dtype=int)
    for r in range(nr):
        neighbours = generator(n, params, seed=r)
        g = graphtool_from_neighbours(neighbours, directed=False)
        if rewire > 0:
            rejection_count = gt.random_rewire(g, n_iter=rewire)
        g, state, labels, n_communities = get_n_communities(
            g, method=method, neighbours=neighbours
        )
        c[r] = n_communities
    return c


def count_wc(n, params, nr, method, rewire, kappa):
    c = n_instances_with_cummunities(n, params, nr, method, rewire)
    return np.sum(c >= kappa)


def ramsey_community_number(
    params, epsilon, nr, n0=10, method="blocks", rewire=0, kappa=2
):
    """Estimates the Ramsey community number of a graph."""
    nr_95 = int((1 - epsilon) * nr)
    # find upper bound, some n satisfying nr_c >= nr_95
    n = n0
    nr_c = count_wc(n, params, nr, method, rewire, kappa)
    if nr_c > nr_95:
        # n0 is an upper bound, find lower bound
        while nr_c > nr_95:
            n_right = n
            n = n // 2
            nr_c = count_wc(n, params, nr, method, rewire, kappa)
            print(f"[{n}, {n_right}], fraction with communities: {nr_c/nr} ...")
        n_left = n
    elif nr_c < nr_95:
        # n0 is an lower bound, find upper bound
        while nr_c < nr_95:
            n_left = n
            n *= 2
            nr_c = count_wc(n, params, nr, method, rewire, kappa)
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
            nr_c = count_wc(n, params, nr, method, rewire, kappa)
            if nr_c >= nr_95:
                n_right = n
            else:
                n_left = n
        if nr_c < nr_95:
            n += 1
            nr_c = count_wc(n, params, nr, method, rewire, kappa)
    print(
        f"binary search: [{n_left}, {n_right}], r_c: {n}, fraction with communities: {nr_c/nr}"
    )
    return n


def draw_instance(n, params, seed, path, rewire=0):
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
            path, f"{model_name}_n{n}_seed{seed}_rewire{rewire}.pdf"
        )
        state = gt.minimize_blockmodel_dl(g)
        state.draw(output=filename)


def find_instances_with_communities(
    n, params, n_instances, path, method="blocks", has_communities=True
):
    """Auxiliary method to find and draw instances with 2 or more communities."""
    model_name = get_model_name(params)
    seed = 0
    count = 0
    while count < n_instances:
        neighbours = generator(n, params, seed)
        g, state, labels, n_communities = get_n_communities_from_neighbours(
            neighbours, method=method
        )
        if (has_communities and n_communities > 1) or (
            not has_communities and n_communities == 1
        ):
            if has_communities:
                filename = os.path.join(
                    path, f"{model_name}_n{n}_seed{seed}_{method}.pdf"
                )
            else:
                filename = os.path.join(
                    path, f"{model_name}_n{n}_seed{seed}_{method}_no_com.pdf"
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
