import os
import json
import numpy as np
import graph_tool.all as gt
import igraph as ig


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


def random_graph(n, params, seed):
    """Erdos-Renyi random graph model with M edges."""
    np.random.seed(seed=seed)
    M = params["M"]
    neighbours = [[] for i in range(n)]
    for l in range(M):
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i in neighbours[j]:
            i = np.random.randint(n)
            j = np.random.randint(n)
        neighbours[i].append(j)
        neighbours[j].append(i)
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
                j = np.random.randint(n)
                while j in neighbours[i]:
                    j = np.random.randint(n)
            else:
                j = j - n if j > n - 1 else j
            neighbours[i].append(j)
            neighbours[j].append(i)
    return neighbours


def watts_strogatz_vec(n, params, seed):
    K = np.array(params["K"])
    p = params["p"]
    x = np.random.randint(n, size=(n, K.size))
    y = np.repeat((np.arange(n))[:, np.newaxis], K.size, axis=1) + np.repeat(
        K[np.newaxis, :], n, axis=0
    )
    y = np.where(y < n, y, y - n)
    z = np.random.binomial(1, p, size=(n, K.size))
    return list(np.where(z == 1, x, y))


def generator_barabasi_albert(n, params, seed):
    """Barabasi-Albert model with m new links per node."""
    m = params["m"]
    neighbours = [[1], [0]]
    replicas = [0, 1]
    np.random.seed(seed=seed)
    for i in range(2, n):
        n_replicas = len(replicas)
        new_neighbours = []
        for u in range(min(m, i)):
            v = np.random.randint(n_replicas)
            j = replicas[v]
            while i in neighbours[j]:
                v = np.random.randint(n_replicas)
                j = replicas[v]
            new_neighbours.append(j)
            neighbours[j].append(i)
            replicas += [i, j]
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
    is_duplicate = np.random.random(n) < q
    for i in range(2, n):
        j = np.random.randint(i)
        node = np.array([i, j])
        neighbours_ = np.array(neighbours[j])
        m = np.random.binomial(neighbours_.size, q)
        if m > 0:
            choice = node[np.random.randint(2, size=m)]
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


def generator(n, params, seed):
    if params["model-"] == "er":
        neighbours = random_graph(n, params, seed)
    elif params["model-"] == "ws":
        neighbours = watts_strogatz(n, params, seed)
    elif params["model-"] == "wsv":
        neighbours = watts_strogatz_vec(n, params, seed)
    elif params["model-"] == "ds":
        neighbours = generator_dup_split(n, params, seed)
    elif params["model-"] == "ba":
        neighbours = generator_barabasi_albert(n, params, seed)
    elif params["model-"] == "ls":
        neighbours = generator_local_search(n, params, seed)
    elif params["model-"] == "bb":
        neighbours = generator_bubbles(n, params, seed)
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


def get_n_communities(g, method="blocks"):
    if method == "blocks":
        state = gt.minimize_blockmodel_dl(g)
        labels = state.get_blocks()
    elif method == "modularity":
        state = gt.ModularityState(g)
        for beta in [1, 10, 100, 1000]:
            state.gibbs_sweep(beta=beta, niter=10)
        labels = state.b.fa
    elif method == "potts":
        g_ = ig.Graph.from_graph_tool(g)
        state = g_.community_spinglass(gamma=1, update_rule="config")
        labels = state.membership
    return g, state, labels, len(np.unique(labels))


def get_n_communities_from_neighbours(neighbours, method="blocks", directed=False):
    g = graphtool_from_neighbours(neighbours, directed=directed)
    return get_n_communities(g, method=method)


def n_instances_with_cummunities(n, params, nr, method="blocks"):
    c = np.empty(nr, dtype=int)
    for r in range(nr):
        neighbours = generator(n, params, seed=r)
        g, state, labels, n_communities = get_n_communities_from_neighbours(
            neighbours, method=method
        )
        c[r] = n_communities
    return c


def ramsey_community_number(params, epsilon, nr, n0=10, method="blocks"):
    """Estimates the Ramsey community number of a graph."""
    nr_95 = int((1 - epsilon) * nr)
    # find upper bound, some n satisfying nr_c >= nr_95
    n = n0
    nr_c = np.sum(n_instances_with_cummunities(n, params, nr, method=method) > 1)
    if nr_c > nr_95:
        # n0 is an upper bound, find lower bound
        while nr_c > nr_95:
            n_right = n
            n = n // 2
            nr_c = np.sum(
                n_instances_with_cummunities(n, params, nr, method=method) > 1
            )
            print(f"[{n}, {n_right}], fraction with communities: {nr_c/nr} ...")
        n_left = n
    elif nr_c < nr_95:
        # n0 is an lower bound, find upper bound
        while nr_c < nr_95:
            n_left = n
            n *= 2
            nr_c = np.sum(
                n_instances_with_cummunities(n, params, nr, method=method) > 1
            )
            print(f"[{n_left}, {n}], fraction with communities: {nr_c/nr} ...")
        n_right = n
    else:
        # nothng to do
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
            nr_c = np.sum(
                n_instances_with_cummunities(n, params, nr, method=method) > 1
            )
            if nr_c >= nr_95:
                n_right = n
            else:
                n_left = n
        if nr_c < nr_95:
            n += 1
            nr_c = np.sum(
                n_instances_with_cummunities(n, params, nr, method=method) > 1
            )
    print(
        f"binary search: [{n_left}, {n_right}], r_c: {n}, fraction with communities: {nr_c/nr}"
    )
    return n


def draw_instance(n, params, seed, path, rewire=False):
    """Auxiliary method to draw a graph instance."""
    neighbours = generator(n, params, seed)
    g = graphtool_from_neighbours(neighbours)
    model_name = get_model_name(params)
    filename = os.path.join(path, f"{model_name}_n{n}_seed{seed}.pdf")
    state = gt.minimize_blockmodel_dl(g)
    state.draw(output=filename)
    if rewire:
        rejection_count = gt.random_rewire(g, n_iter=10)
        filename = os.path.join(path, f"{model_name}_n{n}_seed{seed}_rewire.pdf")
        state = gt.minimize_blockmodel_dl(g)
        state.draw(output=filename)


def find_instances_with_communities(n, params, n_instances, path, method="blocks"):
    """Auxiliary method to find and draw instances with 2 or more communities."""
    model_name = get_model_name(params)
    seed = 0
    count = 0
    while count < n_instances:
        neighbours = generator(n, params, seed)
        g, state, labels, n_communities = get_n_communities(neighbours, method=method)
        if n_communities > 1:
            filename = os.path.join(path, f"{model_name}_n{n}_seed{seed}_{method}.pdf")
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
