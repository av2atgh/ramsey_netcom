# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython ports of the hot graph generators from libs.py.

Drop-in replacements that return the same list-of-lists `neighbours` structure.
NOTE: uses a C++ mt19937 RNG, so a given `seed` produces DIFFERENT graphs than
the pure-Python versions in libs.py (by design — chosen for speed).
"""
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libc.math cimport sqrt, floor


# ---------------------------------------------------------------------------
def average_shortest_path_multiplicity_csr(int[::1] indptr, int[::1] indices, int n):
    """Mean number of shortest paths over all vertex pairs, from a CSR adjacency.

    One BFS per source (Brandes forward pass) computes the shortest-path count
    sigma to every target at once -> O(n*(n+m)) instead of O(n^2*(n+m)).
    Unreachable pairs contribute 0 (matching gt.count_shortest_paths).
    """
    if n < 2:
        return float("nan")
    cdef vector[int] dist
    cdef vector[double] sigma
    cdef vector[int] queue
    cdef double total = 0.0
    cdef int s, u, w, e, head, tail
    dist.resize(n)
    sigma.resize(n)
    queue.resize(n)
    with nogil:
        for s in range(n):
            for u in range(n):
                dist[u] = -1
                sigma[u] = 0.0
            dist[s] = 0
            sigma[s] = 1.0
            queue[0] = s
            head = 0
            tail = 1
            while head < tail:
                u = queue[head]
                head += 1
                for e in range(indptr[u], indptr[u + 1]):
                    w = indices[e]
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        sigma[w] = sigma[u]
                        queue[tail] = w
                        tail += 1
                    elif dist[w] == dist[u] + 1:
                        sigma[w] += sigma[u]
            for u in range(n):
                if u != s:
                    total += sigma[u]
    return total / (<double>n * (<double>n - 1.0))


# ---------------------------------------------------------------------------
def directed_average_shortest_path_multiplicity_csr(int[::1] indptr, int[::1] indices, int n):
    """Mean number of shortest *directed* paths over all reachable (source, target)
    pairs, from an out-edge CSR adjacency.

    Forward BFS per source over out-edges gives sigma[t] = number of shortest
    directed paths s -> t. Only reachable pairs are averaged (the denominator is
    the count of reachable ordered pairs, not n*(n-1)).
    """
    if n < 2:
        return float("nan")
    cdef vector[int] dist
    cdef vector[double] sigma
    cdef vector[int] queue
    cdef double total = 0.0
    cdef long reachable = 0
    cdef int s, u, w, e, head, tail
    dist.resize(n)
    sigma.resize(n)
    queue.resize(n)
    with nogil:
        for s in range(n):
            for u in range(n):
                dist[u] = -1
                sigma[u] = 0.0
            dist[s] = 0
            sigma[s] = 1.0
            queue[0] = s
            head = 0
            tail = 1
            while head < tail:
                u = queue[head]
                head += 1
                for e in range(indptr[u], indptr[u + 1]):
                    w = indices[e]
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        sigma[w] = sigma[u]
                        queue[tail] = w
                        tail += 1
                    elif dist[w] == dist[u] + 1:
                        sigma[w] += sigma[u]
            # queue[1 .. tail-1] are exactly the reachable targets (s excluded)
            for e in range(1, tail):
                total += sigma[queue[e]]
            reachable += tail - 1
    if reachable == 0:
        return float("nan")
    return total / <double>reachable


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)
        unsigned int operator()() nogil


cdef inline int randint(mt19937& g, int n) noexcept nogil:
    # uniform in [0, n); modulo bias negligible for the n used here
    return <int>(g() % <unsigned int>n)


cdef inline double randdouble(mt19937& g) noexcept nogil:
    return g() / 4294967296.0  # 2**32


cdef inline long randlong(mt19937& g, long n) noexcept nogil:
    # 64-bit draw for ranges that can exceed 2**32 (within-W pair counts)
    cdef unsigned long long x = (<unsigned long long>g() << 32) | <unsigned long long>g()
    return <long>(x % <unsigned long long>n)


# Bijection between an unordered pair (a < b) and a single index in [0, n(n-1)/2),
# mirroring tuple2index / index2tuple in libs.py.
cdef inline long t2i(int a, int b, int n) noexcept nogil:
    return (<long>n * (n - 1)) // 2 - (<long>(n - a) * ((n - a) - 1)) // 2 + b - a - 1


cdef inline pair[int, int] i2t(long code, int n) noexcept nogil:
    cdef double disc = <double>(-8 * code + 4 * <long>n * (n - 1) - 7)
    cdef int a = <int>(n - 2 - floor(sqrt(disc) / 2.0 - 0.5))
    cdef long b = code + a + 1 - (<long>n * (n - 1)) // 2 + (<long>(n - a) * ((n - a) - 1)) // 2
    cdef pair[int, int] r
    r.first = a
    r.second = <int>b
    return r


cdef inline long pair_code(int a, int b, int n) noexcept nogil:
    # encode an unordered pair given in either order
    return t2i(a, b, n) if a < b else t2i(b, a, n)


# ---------------------------------------------------------------------------
def generator_local_search(int n, params, int seed):
    """Local-Search model with depth d."""
    cdef int d = params["d"]
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[int] visited, vunique
    cdef int i, step, j, l, k, m, idx
    cdef bint dup

    nb.resize(2)
    nb[0].push_back(1)
    nb[1].push_back(0)

    for i in range(2, n):
        visited.clear()
        visited.push_back(randint(g, i))
        for step in range(d):
            j = visited[step]
            l = randint(g, <int>nb[j].size())
            visited.push_back(nb[j][l])
        # dedup `visited` (replaces np.unique; order within an adjacency list is irrelevant)
        vunique.clear()
        for m in range(<int>visited.size()):
            k = visited[m]
            dup = False
            for idx in range(<int>vunique.size()):
                if vunique[idx] == k:
                    dup = True
                    break
            if not dup:
                vunique.push_back(k)
        for idx in range(<int>vunique.size()):
            nb[vunique[idx]].push_back(i)
        nb.push_back(vunique)
    return nb


# ---------------------------------------------------------------------------
def generator_bubbles(int n, params, int seed):
    cdef int L = params["L"]
    cdef double p = params["p"] if "p" in params else 1.0
    cdef int n_bubbles_min = (n - 2) // L
    cdef int n_bubbles = n_bubbles_min + 1 if (n - 2) % L else n_bubbles_min
    cdef int nn = 2 + n_bubbles * L
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[pair[int, int]] edges
    cdef vector[int] newnb
    cdef int i, l, j, k, e
    cdef pair[int, int] pr

    nb.resize(nn)
    for i in range(L + 2):
        nb[i].push_back(i - 1 if i > 0 else L + 1)
        nb[i].push_back(i + 1 if i < L + 1 else 0)
        edges.push_back(pair[int, int](i, i + 1 if i < L + 1 else 0))

    i = L + 2
    while i < nn:
        if p == 1.0 or randdouble(g) < p:
            e = randint(g, <int>edges.size())
            j = edges[e].first
            k = edges[e].second
        else:
            j = randint(g, i)
            k = randint(g, i)
            while k == j:
                k = randint(g, i)
        # new_neighbours = [j] + range(i, i+L) + [k]
        newnb.clear()
        newnb.push_back(j)
        for l in range(L):
            newnb.push_back(i + l)
        newnb.push_back(k)
        for l in range(1, L + 2):
            j = newnb[l - 1]
            k = newnb[l]
            edges.push_back(pair[int, int](j, k))
            if j < nn and k < nn:
                nb[j].push_back(k)
                nb[k].push_back(j)
        i += L
    return nb


# ---------------------------------------------------------------------------
def generator_dup_split(int n, params, int seed):
    """Duplication-Split model with duplication rate q."""
    cdef double p = params["p"] if "p" in params else 0.0
    cdef double q = params["q"]
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[int] newnb
    cdef int i, j, k, l, degree_j, idx, m

    nb.resize(4)
    nb[0].push_back(3); nb[0].push_back(1)
    nb[1].push_back(0); nb[1].push_back(2)
    nb[2].push_back(1); nb[2].push_back(3)
    nb[3].push_back(2); nb[3].push_back(0)

    for i in range(4, n):
        j = randint(g, i)
        if randdouble(g) < q:
            degree_j = <int>nb[j].size()
            newnb.clear()
            for m in range(degree_j):
                k = nb[j][m]
                nb[k].push_back(i)
                newnb.push_back(k)
            if p > 0.0 and randdouble(g) < p:
                nb[j].push_back(i)
                newnb.push_back(j)
            nb.push_back(newnb)
        else:
            degree_j = <int>nb[j].size()
            l = randint(g, degree_j)
            k = nb[j][l]
            nb[j][l] = i
            for idx in range(<int>nb[k].size()):
                if nb[k][idx] == j:
                    nb[k][idx] = i
                    break
            newnb.clear()
            newnb.push_back(j)
            newnb.push_back(k)
            nb.push_back(newnb)
    return nb


# ---------------------------------------------------------------------------
def generator_dup_split_directed(int n, params, int seed):
    """Directed Duplication-Split model with duplication rate q.

    Duplication (prob q): new node i copies j's in- and out-arcs.
        m -> j  =>  m -> i      and      j -> k  =>  i -> k
    Split (prob 1 - q): new node i takes over j's out-arcs.
        j -> k  =>  i -> k  (for all k),  then add  j -> i

    Returns the out-adjacency (list of successor lists).
    """
    cdef double q = params["q"]
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] out
    cdef vector[vector[int]] inn
    cdef vector[int] tmp
    cdef int i, j, k, m, idx

    out.resize(2)
    inn.resize(2)
    out[0].push_back(1)   # directed seed: a single arc 0 -> 1
    inn[1].push_back(0)

    for i in range(2, n):
        j = randint(g, i)
        if randdouble(g) < q:
            # duplication: copy predecessors then successors of j to new node i
            for idx in range(<int>inn[j].size()):
                out[inn[j][idx]].push_back(i)
            for idx in range(<int>out[j].size()):
                inn[out[j][idx]].push_back(i)
            tmp = inn[j]            # copy before the outer vectors grow (avoid aliasing)
            inn.push_back(tmp)
            tmp = out[j]
            out.push_back(tmp)
        else:
            # split: redirect every j -> k into i -> k, then add j -> i
            for idx in range(<int>out[j].size()):
                k = out[j][idx]
                for m in range(<int>inn[k].size()):
                    if inn[k][m] == j:
                        inn[k][m] = i
                        break
            tmp = out[j]           # out[i] = j's old out-arcs
            out.push_back(tmp)
            tmp.clear()
            tmp.push_back(j)       # inn[i] = [j]
            inn.push_back(tmp)
            out[j].clear()         # j now points only to i
            out[j].push_back(i)
    return out


# ---------------------------------------------------------------------------
def generator_bubbles_2(int n, params, int seed):
    """A chain of L nodes attached to two nodes at most W steps apart.

    Replaces the O(n^4) dense-A^W version with a BFS-to-depth-W sweep: each step
    enumerates all ordered pairs within distance W and samples one uniformly
    (two passes: count, then select), so it stays O(n*(n+m)) per step.
    """
    cdef int L = params["L"]
    cdef int W = params["W"]
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[int] dist, queue
    cdef int cur, s, head, tail, u, e, w, du, j, k, l, prev, node
    cdef long total, r

    nb.resize(n)
    nb[0].push_back(1)
    nb[1].push_back(0)
    dist.assign(n, -1)
    queue.resize(n)
    cur = 2
    while cur < n:
        # ---- pass 1: count ordered pairs (s, t) with 1 <= dist(s,t) <= W ----
        total = 0
        for s in range(cur):
            dist[s] = 0
            queue[0] = s
            head = 0
            tail = 1
            while head < tail:
                u = queue[head]
                head += 1
                du = dist[u]
                if du == W:
                    continue
                for e in range(<int>nb[u].size()):
                    w = nb[u][e]
                    if dist[w] == -1:
                        dist[w] = du + 1
                        queue[tail] = w
                        tail += 1
            total += tail - 1
            for e in range(tail):
                dist[queue[e]] = -1
        # ---- pass 2: select the r-th pair in the same BFS order ----
        r = randlong(g, total)
        j = -1
        k = -1
        for s in range(cur):
            dist[s] = 0
            queue[0] = s
            head = 0
            tail = 1
            while head < tail:
                u = queue[head]
                head += 1
                du = dist[u]
                if du == W:
                    continue
                for e in range(<int>nb[u].size()):
                    w = nb[u][e]
                    if dist[w] == -1:
                        dist[w] = du + 1
                        queue[tail] = w
                        tail += 1
            if r < tail - 1:
                j = s
                k = queue[1 + r]
                for e in range(tail):
                    dist[queue[e]] = -1
                break
            r -= tail - 1
            for e in range(tail):
                dist[queue[e]] = -1
        # ---- attach the chain [j] - cur..cur+L-1 - [k] ----
        prev = j
        for l in range(L):
            node = cur + l
            if node < n:
                nb[prev].push_back(node)
                nb[node].push_back(prev)
                prev = node
        nb[prev].push_back(k)
        nb[k].push_back(prev)
        cur += L
    return nb


# ---------------------------------------------------------------------------
def generator_bubbles_2_L1_W3(int n, params, int seed):
    """L=1, W=3 special case: attach a node to a random pair within distance 3,
    maintaining the candidate-pair set incrementally."""
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[long] pv          # unique candidate pairs (encoded), index-addressable
    cdef unordered_set[long] ps   # membership for dedup
    cdef unordered_set[int] n2
    cdef pair[int, int] pr
    cdef int i, j1, j2, k, idx, a, b
    cdef long code

    nb.resize(n)
    nb[0].push_back(1)
    nb[1].push_back(0)
    code = t2i(0, 1, n)
    pv.push_back(code)
    ps.insert(code)
    i = 2
    while i < n:
        pr = i2t(pv[randint(g, <int>pv.size())], n)
        j1 = pr.first
        j2 = pr.second
        # n2 = nodes within distance 2 of {j1, j2}
        n2.clear()
        n2.insert(j1)
        n2.insert(j2)
        for idx in range(<int>nb[j1].size()):
            n2.insert(nb[j1][idx])
        for idx in range(<int>nb[j2].size()):
            n2.insert(nb[j2][idx])
        for idx in range(<int>nb[j1].size()):
            a = nb[j1][idx]
            for b in range(<int>nb[a].size()):
                n2.insert(nb[a][b])
        for idx in range(<int>nb[j2].size()):
            a = nb[j2][idx]
            for b in range(<int>nb[a].size()):
                n2.insert(nb[a][b])
        # new candidate pairs
        for k in n2:
            code = t2i(k, i, n)  # k < i (i is the newest node)
            if ps.find(code) == ps.end():
                ps.insert(code)
                pv.push_back(code)
        for idx in range(<int>nb[j2].size()):
            k = nb[j2][idx]
            if k != j1:
                code = pair_code(j1, k, n)
                if ps.find(code) == ps.end():
                    ps.insert(code)
                    pv.push_back(code)
        for idx in range(<int>nb[j1].size()):
            k = nb[j1][idx]
            if k != j2:
                code = pair_code(j2, k, n)
                if ps.find(code) == ps.end():
                    ps.insert(code)
                    pv.push_back(code)
        nb[j1].push_back(i)
        nb[j2].push_back(i)
        nb[i].push_back(j1)
        nb[i].push_back(j2)
        i += 1
    return nb


# ---------------------------------------------------------------------------
def nearest_neighbor(int n, params, int seed=0):
    """Connecting-nearest-neighbor model with conversion probability u."""
    cdef double u = params["u"]
    cdef mt19937 g = mt19937(<unsigned int>seed)
    cdef vector[vector[int]] nb
    cdef vector[long] pe          # potential edges (encoded); may carry duplicates
    cdef unordered_set[long] s
    cdef pair[int, int] pr
    cdef int i, j, k, idx, a, b, l
    cdef long code
    cdef bint present

    nb.resize(n)
    nb[0].push_back(1)
    nb[1].push_back(0)
    i = 2
    while i < n:
        if randdouble(g) < 1.0 - u:
            j = randint(g, i)
            for idx in range(<int>nb[j].size()):
                k = nb[j][idx]          # k < i
                pe.push_back(t2i(k, i, n))
            nb[j].push_back(i)
            nb[i].push_back(j)
            i += 1
        elif not pe.empty():
            l = randint(g, <int>pe.size())
            code = pe[l]
            pe[l] = pe[pe.size() - 1]   # swap-pop (sampling order is irrelevant)
            pe.pop_back()
            pr = i2t(code, n)
            j = pr.first
            k = pr.second
            # dedup the remaining potential edges, then add the new ones
            s.clear()
            for idx in range(<int>pe.size()):
                s.insert(pe[idx])
            for idx in range(<int>nb[k].size()):
                a = nb[k][idx]
                present = False
                for b in range(<int>nb[j].size()):
                    if nb[j][b] == a:
                        present = True
                        break
                if not present:
                    s.insert(pair_code(a, j, n))
            for idx in range(<int>nb[j].size()):
                a = nb[j][idx]
                present = False
                for b in range(<int>nb[k].size()):
                    if nb[k][b] == a:
                        present = True
                        break
                if not present:
                    s.insert(pair_code(a, k, n))
            pe.clear()
            for code in s:
                pe.push_back(code)
            nb[j].push_back(k)
            nb[k].push_back(j)
    return nb
