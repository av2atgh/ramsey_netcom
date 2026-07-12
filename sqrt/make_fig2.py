#!/usr/bin/env python3
r"""Figure 1: the two model limits that bracket community growth.

Panel (a): connected caveman graph CC(6,4) -- the small-world / modular limit,
one community per clique, B = n/k proportional to n.
Panel (b): pseudofractal web (Dorogovtsev-Goltsev-Mendes) at t=3, n=42 -- the
self-similar / fractal limit, B growing as sqrt(n).  Communities are colour
coded.  The pseudofractal build and layout follow av2atgh/pseudofractalweb.

Produces fig2.pdf.
"""

import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CMAP = plt.get_cmap("tab10")

# ---- caveman (panel a) ---------------------------------------------------
L, K = 6, 4


def caveman_layout(l, k, ring_r=1.0, clique_r=0.30):
    pos = {}
    for c in range(l):
        theta = 2 * np.pi * c / l + np.pi / 2
        cx, cy = ring_r * np.cos(theta), ring_r * np.sin(theta)
        for j in range(k):
            phi = 2 * np.pi * j / k - theta
            pos[c * k + j] = (cx + clique_r * np.cos(phi), cy + clique_r * np.sin(phi))
    return pos


def draw_caveman(ax):
    G = nx.connected_caveman_graph(L, K)
    pos = caveman_layout(L, K)
    colors = [CMAP((v // K) % 10) for v in G.nodes()]
    intra = [(u, v) for u, v in G.edges() if u // K == v // K]
    inter = [(u, v) for u, v in G.edges() if u // K != v // K]
    nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax, edge_color="0.55", width=1.1)
    nx.draw_networkx_edges(G, pos, edgelist=inter, ax=ax, edge_color="0.2",
                           width=1.6, style="dashed")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color=colors,
                           edgecolors="white", linewidths=0.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title(r"(a) small-world limit: $B\propto n$", fontsize=11)


# ---- pseudofractal web (panel b) -----------------------------------------
# build, layout, three-community partition and colours follow Fig. 4 of
# av2atgh/pseudofractalweb (partition_3way): community b = descent branch b plus
# the corresponding seed hub.
PF_COLORS = {0: "#2f6db0", 1: "#e08a1e", 2: "#3a9a54"}


def build_dgm(T):
    """Pseudofractal web after T generations; also the branch label of each node
    (0/1/2 for the three descent branches, -1 for the seed hubs)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    branch = {0: -1, 1: -1, 2: -1}
    eb = {(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 1, (0, 2): 2, (2, 0): 2}
    nxt = 3
    for _ in range(T):
        for (u, v) in list(G.edges()):
            b = eb[(u, v)]
            w = nxt; nxt += 1
            G.add_edge(w, u); G.add_edge(w, v)
            branch[w] = b
            for e in ((w, u), (u, w), (w, v), (v, w)):
                eb[e] = b
    return G, branch


def dgm_layout(G):
    """Pin the three seed hubs at the corners of a triangle, spring-relax rest."""
    ang = {0: 90, 1: 210, 2: 330}
    init = {h: (np.cos(np.radians(a)), np.sin(np.radians(a))) for h, a in ang.items()}
    return nx.spring_layout(G, pos=init, fixed=[0, 1, 2],
                            k=1.6 / np.sqrt(len(G)), iterations=400, seed=7)


def draw_pseudofractal(ax):
    G, branch = build_dgm(3)                           # n = 42
    pos = dgm_layout(G)
    # symmetric three-community partition: block b = branch b + seed hub b
    part = {v: (branch[v] if branch[v] >= 0 else v) for v in G.nodes()}
    colors = [PF_COLORS[part[v]] for v in G.nodes()]
    deg = dict(G.degree())
    sizes = [40 + 45 * np.sqrt(deg[v]) for v in G.nodes()]
    within = [(u, v) for u, v in G.edges() if part[u] == part[v]]
    cross = [(u, v) for u, v in G.edges() if part[u] != part[v]]
    nx.draw_networkx_edges(G, pos, edgelist=within, ax=ax, edge_color="#b9b9b9",
                           width=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=cross, ax=ax, edge_color="#111111",
                           width=1.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="white", linewidths=0.7)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"(b) fractal limit: $B\propto\sqrt{n}$", fontsize=11)
    print(f"pseudofractal t=3: n={G.number_of_nodes()}, "
          f"3-community partition (one per branch)")


def main():
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 4.2))
    draw_caveman(ax[0])
    draw_pseudofractal(ax[1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2.pdf"), bbox_inches="tight")
    print("wrote fig2.pdf")


if __name__ == "__main__":
    main()
