#!/usr/bin/env python3
"""Draw the (b,s)=(2,2) diamond hierarchical lattice for t=0,1,2.

Produces diamond_lattice.pdf, the construction figure used in the
"Diamond lattice" section of multiplath.tex.  Each edge of G_{t-1} is
replaced by b=2 parallel paths of s=2 edges (one interior hub each),
placed symmetrically about the edge to give the self-similar diamond
embedding.  The two poles A (top) and B (bottom) are highlighted.
"""

import math
import matplotlib.pyplot as plt


def build(t, off_frac=0.32):
    """Recursive geometric construction of G_t for b=s=2.

    Returns (edges, pos) with unique integer node ids; poles are 0 (A)
    and 1 (B).  Each edge -> two hubs offset +/- perpendicular to it.
    """
    pos = {0: (0.0, 1.0), 1: (0.0, -1.0)}
    edges = [(0, 1)]
    nxt = [2]

    def subdivide(edges, level):
        if level == 0:
            return edges
        new = []
        for (u, v) in edges:
            xu, yu = pos[u]
            xv, yv = pos[v]
            mx, my = (xu + xv) / 2.0, (yu + yv) / 2.0
            dx, dy = xv - xu, yv - yu
            L = math.hypot(dx, dy)
            px, py = -dy / L, dx / L            # perpendicular unit
            off = off_frac * L
            ml, mr = nxt[0], nxt[0] + 1
            nxt[0] += 2
            pos[ml] = (mx + px * off, my + py * off)
            pos[mr] = (mx - px * off, my - py * off)
            new += [(u, ml), (ml, v), (u, mr), (mr, v)]
        return subdivide(new, level - 1)

    return subdivide(edges, t), pos


def n_t(t):
    return (2 * 4 ** t + 4) // 3


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, t in zip(axes, (0, 1, 2)):
        edges, pos = build(t)
        for (u, v) in edges:
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], "-", color="0.35", lw=1.1, zorder=1)
        xs = [pos[k][0] for k in pos]
        ys = [pos[k][1] for k in pos]
        # interior nodes
        inner = [k for k in pos if k not in (0, 1)]
        ax.scatter([pos[k][0] for k in inner], [pos[k][1] for k in inner],
                   s=26, color="#4477aa", zorder=2, edgecolors="white",
                   linewidths=0.4)
        # poles
        ax.scatter([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]],
                   s=55, color="#cc3311", zorder=3, edgecolors="white",
                   linewidths=0.5)
        ax.annotate("A", pos[0], textcoords="offset points", xytext=(6, 4),
                    fontsize=10, fontweight="bold")
        ax.annotate("B", pos[1], textcoords="offset points", xytext=(6, -12),
                    fontsize=10, fontweight="bold")
        ax.set_title(r"$t=%d$   ($n_t=%d$)" % (t, n_t(t)), fontsize=10)
        m = max(1.05, max(abs(min(xs)), abs(max(xs)), abs(min(ys)),
                          abs(max(ys))) + 0.1)
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("diamond_lattice.pdf", bbox_inches="tight")
    print("wrote diamond_lattice.pdf; n_t =", [n_t(t) for t in (0, 1, 2)])


if __name__ == "__main__":
    main()
