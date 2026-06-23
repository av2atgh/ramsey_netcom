# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython ports of the hot graph generators from libs.py.

Drop-in replacements that return the same list-of-lists `neighbours` structure.
NOTE: uses a C++ mt19937 RNG, so a given `seed` produces DIFFERENT graphs than
the pure-Python versions in libs.py (by design — chosen for speed).
"""
from libcpp.vector cimport vector
from libcpp.pair cimport pair


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


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)
        unsigned int operator()() nogil


cdef inline int randint(mt19937& g, int n) nogil:
    # uniform in [0, n); modulo bias negligible for the n used here
    return <int>(g() % <unsigned int>n)


cdef inline double randdouble(mt19937& g) nogil:
    return g() / 4294967296.0  # 2**32


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
