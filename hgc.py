import numpy as np


def gammaln0(x):
    z = np.arange(1, 101)
    return -np.euler_gamma * x - np.log(x) + np.sum(x / z - np.log(1 + x / z))

gammaln = np.vectorize(gammaln0)

def digamma0(x, y):
    z = np.arange(1, 101)
    return np.sum((x - y) / ((x + z - 1) * (y + z - 1)))

digamma = np.vectorize(digamma0)


def hgc1(edge2nodes, node2edges, n_groups_max, seed):
    np.random.seed(seed=seed)
    n_nodes = len(node2edges)
    n_edges = len(edge2nodes)
    p = np.random.dirichlet([10.0] * n_groups_max, size=n_nodes)
    if isinstance(edge2nodes[0], list):
        edge2nodes = [np.array(e) for e in edge2nodes]
    if isinstance(node2edges[0], list):
        node2edges = [np.array(v) for v in node2edges]

    # Set algorithm parameters

    rerror = 1e-12  # Relative error stopping threshold
    tilde_alpha = (
        1e-6  # prior exponent (non-informative limit, or Jayne's prior, tilde_alpha->0)
    )
    tilde_beta = (
        1e-6  # prior exponent (non-informative limit, or Jayne's prior, tilde_beta->0)
    )
    tilde_gamma = (
        1e-6  # prior exponent (non-informative limit, or Jayne's prior, tilde_gamma->0)
    )
    tilde_alpha_plus_beta = tilde_alpha + tilde_beta

    F0 = 1
    F = 2 * F0

    sgamma = n_groups_max * tilde_gamma + n_nodes

    while 2 * np.abs(F - F0) > (np.abs(F) + np.abs(F0)) * rerror:

        # Update gamma and <log pi>

        gamma = tilde_gamma + np.sum(p, axis=0)
        logpi = digamma(gamma, sgamma)

        # Update alpha, beta, <log theta> and <log (1-theta)>

        alpha = np.zeros((n_groups_max, n_edges))
        for j in np.arange(n_edges):
            alpha[:, j] = tilde_alpha + np.sum(p[edge2nodes[j], :], axis=0)
        alpha_plus_beta = tilde_alpha_plus_beta + gamma - tilde_gamma
        alpha_plus_beta = np.repeat(alpha_plus_beta[:, np.newaxis], n_edges, axis=1)
        logtheta = digamma(alpha, alpha_plus_beta)
        log1_theta = digamma(alpha_plus_beta - alpha, alpha_plus_beta)

        # Update p

        for i in range(n_nodes):
            h = (
                logpi
                + np.sum(log1_theta, axis=1)
                - np.sum(
                    logtheta[:, node2edges[i]] - log1_theta[:, node2edges[i]], axis=1
                )
            )
            p[i, :] = np.exp(h - h.max())
            p[i, :] /= np.sum(p[i, :])

        # Update optimization objetive

        F0 = F
        F = np.sum(p * np.log(p))
        F += np.sum(
            gammaln(alpha_plus_beta) - gammaln(alpha) - gammaln(alpha_plus_beta - alpha)
        )
        F += gammaln(sgamma) - np.sum(gammaln(gamma))

    return p, F


def hgc(edge2nodes, node2edges, n_groups_max, n_instances):
    """
    Author: Alexei Vazquez, avazque1@protonmail.com
    Cite:
        Vazquez A Finding hypergraph communities:
        Bayesian approach and variational solution.
        J. Stat. Mech. (2009) P07006
    Input:
        edge2nodes, list of edges nodes set
        node2edges, list of nodes edge set
        ng, maximum number of clusters
        nr, number of sampling initial conditions
    Output
        p, two-dimensional array, probability p(i,k) that vertex i belongs to group k
        state, one-dimensional array assigning samples to clusters (after removal of empty clusters)
    """

    # Recursive clustering algorithm, sampling NR initial conditions. Selects the solution with lowest F.

    Fmin = 0
    for r in np.arange(n_instances):
        p1, F1 = hgc1(edge2nodes, node2edges, n_groups_max, r)
        if r == 0 or F1 < Fmin:
            Fmin = F1
            p = p1

    # Assign best group

    best = np.argmax(p, axis=1)
    groups = np.unique(best)
    blocks = np.argmin(
        np.abs(
            np.repeat(best[:, np.newaxis], len(groups), axis=1)
            - np.repeat(groups[np.newaxis, :], len(best), axis=0)
        ),
        axis=1,
    )

    return p, blocks 
