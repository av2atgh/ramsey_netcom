"""Average shortest-path multiplicity vs. number of remaining nodes under random
node removal, for a real network from the Netzschleuder repository.

Example: the yeast protein interactome. Produces a linear-x / log-y plot.

Run from anywhere:
    python examples/random_removal_multiplicity.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make `ramsey_netcom` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramsey_netcom import libs as L

NETWORK = "interactome_yeast"
N_REALIZATIONS = 100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{NETWORK}_multiplicity.png")


def main():
    g = L.load_network(NETWORK)
    g.set_directed(False)
    n, e = g.num_vertices(), g.num_edges()

    sizes, mult = L.shortest_path_multiplicity_random_removal(g, n_realizations=N_REALIZATIONS)
    sizes = np.array(sizes)
    for s, m in zip(sizes, mult):
        print(f"  n_remaining={s:5d}  multiplicity={m:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(sizes, mult, "o-", color="#2a6fdb", lw=2, ms=7, mfc="white", mec="#2a6fdb", mew=2)
    ax.set_yscale("log")  # linear x, log y
    ax.set_xlabel("remaining nodes (random removal)", fontsize=12)
    ax.set_ylabel("average shortest-path multiplicity", fontsize=12)
    ax.set_title(f"{NETWORK}  (n={n}, E={e})  ·  {N_REALIZATIONS} realizations", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
