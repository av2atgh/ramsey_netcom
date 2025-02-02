import os
import json
import numpy as np
import graph_tool.all as gt


def neighbours_to_graph(neighbours):
    """Converts a list of adjecencies into a graph-tool undirected graph."""
    n = len(neighbours)
    for i in range(n):
        neighbours_ = np.array(neighbours[i])
        neighbours[i] = list(neighbours_[neighbours_ > i])
    return gt.Graph(dict(zip(range(n), neighbours)), directed=False), neighbours


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
    """Watts-Strogatz model with K neighbors and reqiring rate p."""
    np.random.seed(seed=seed)
    K = np.array(params["K"])
    p = params["p"]
    neighbours = [[] for i in range(n)]
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


def generator_barabasi_albert(n, params, seed):
    """Barabasi-Albert model with m new links per node."""
    np.random.seed(seed=seed)
    m = params["m"]
    neighbours = [[1], [0]]
    replicas = [0, 1]
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
    np.random.seed(seed=seed)
    d = params["d"]
    neighbours = [[1], [0]]
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


def generator(n, params, seed):
    if params["model-"] == "er":
        neighbours = random_graph(n, params, seed)
    elif params["model-"] == "ws":
        neighbours = watts_strogatz(n, params, seed)
    elif params["model-"] == "ds":
        neighbours = generator_dup_split(n, params, seed)
    elif params["model-"] == "ba":
        neighbours = generator_barabasi_albert(n, params, seed)
    elif params["model-"] == "ls":
        neighbours = generator_local_search(n, params, seed)
    return neighbours


def get_model_name(params):
    return "_".join(
        [
            f"{k}{'_'.join([str(e) for e in v]) if type(v) == list else v}"
            for (k, v) in params.items()
        ]
    )


def draw_instance(n, params, seed, path):
    """Auxiliary method to draw a graph instance."""
    neighbours = generator(n, params, seed)
    g = neighbours_to_graph(neighbours)
    model_name = get_model_name(params)
    filename = os.path.join(path, f"{model_name}_seed{seed}.pdf")
    export_draw(g, filename)


def find_instances_with_communities(n, params, n_instances, path):
    """Auxiliary method to find and draw instances with 2 or more communities."""
    model_name = get_model_name(params)
    seed = 0
    count = 0
    while count < n_instances:
        neighbours = generator(n, params, seed)
        g = neighbours_to_graph(neighbours)
        state, n_communities = get_n_communities(g)
        if n_communities["unique"] > 1:
            filename = os.path.join(path, f"{model_name}_n{n}_seed{seed}.pdf")
            export_draw(state, filename, is_state=True)
            count += 1
        seed += 1


def n_instances_with_cummunities(n, params, nr):
    c = np.empty(nr, dtype=int)
    for r in range(nr):
        neighbours = generator(n, params, seed=r)
        g = neighbours_to_graph(neighbours)
        state = gt.minimize_blockmodel_dl(g)
        c[r] = np.unique(np.array(state.get_blocks())).size
    return c


def ramsey_community_number(params, epsilon, nr, n0=10):
    """Estimates the Ramsey community number of a graph."""
    nr_95 = int((1 - epsilon) * nr)
    # find upper bound, some n satisfying nr_c >= nr_95
    n = n0
    nr_c = np.sum(n_instances_with_cummunities(n, params, nr) > 1)
    while nr_c < nr_95:
        n *= 2
        nr_c = np.sum(n_instances_with_cummunities(n, params, nr) > 1)
        print(f"upper bound: {n}, fraction with communities: {nr_c/nr} ...")
    if nr_c > nr_95:
        # binary search, find min n satisfying nr_c = nr_95
        n_left = n // 2
        n_right = n
        while abs(n_left - n_right) > 1:
            print(
                f"binary search: [{n_left}, {n_right}], n: {n}, fraction with communities: {nr_c/nr} ..."
            )
            n = (n_left + n_right) // 2
            nr_c_previous = nr_c
            nr_c = np.sum(n_instances_with_cummunities(n, params, nr) > 1)
            if nr_c >= nr_95:
                n_right = n
            else:
                n_left = n
        if nr_c < nr_95:
            n += 1
            nr_c = np.sum(n_instances_with_cummunities(n, params, nr) > 1)
    else:
        n_left = nr_c
        n_right = nr_c
    print(
        f"binary search: [{n_left}, {n_right}], r_c: {n}, fraction with communities: {nr_c/nr}"
    )
    return n


def get_n_communities(g):
    """Computes the number of communities in a graph using the graph-tool block model."""
    state = gt.minimize_blockmodel_dl(g)
    return state, dict(
        effective=state.get_Be(),
        unique=np.unique(np.array(state.get_blocks())).size,
    )


def export_draw(g, filename, is_state=False):
    """Draws a network highlighting its communities."""
    if is_state:
        g.draw(output=filename)
    else:
        state = gt.minimize_blockmodel_dl(g)
        state.draw(output=filename)
