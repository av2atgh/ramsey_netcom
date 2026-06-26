"""Average shortest-path multiplicity vs. number of remaining nodes under random
node removal, for several real networks from the Netzschleuder repository.

Loops over the configured networks and writes one linear-x / log-y PNG per
network (``<name>_multiplicity.png``) next to this script.

Run from anywhere:
    python examples/real_networks_random_removal.py
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

HERE = os.path.dirname(os.path.abspath(__file__))

# (Netzschleuder name, n_realizations, color). Fewer realizations for the large
# network keeps the run practical (its multiplicity is far more expensive).
NETWORKS = [
    ("interactome_yeast", 100, "#2a6fdb"),
    ("netscience", 100, "#d1495b"),
    ("internet_as", 10, "#2a9d5b"),
]


def plot_network(name, n_realizations, color):
    g = L.load_network(name)
    g.set_directed(False)
    n, e = g.num_vertices(), g.num_edges()

    sizes, mult, conn = L.shortest_path_multiplicity_random_removal(g, n_realizations=n_realizations)
    sizes = np.array(sizes)

    # export the data
    csv_path = os.path.join(HERE, f"{name}_multiplicity.csv")
    np.savetxt(
        csv_path,
        np.column_stack([sizes, mult, conn]),
        delimiter=",",
        header="n_remaining,multiplicity,connected_pairs",
        comments="",
        fmt=["%d", "%.8g", "%.8g"],
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(sizes, mult, "o-", color=color, lw=2, ms=7, mfc="white", mec=color, mew=2)
    ax.set_yscale("log")  # linear x, log y
    ax.set_xlabel("remaining nodes (random removal)", fontsize=12)
    ax.set_ylabel("average shortest-path multiplicity", fontsize=12)
    ax.set_title(f"{name}  (n={n}, E={e})  ·  {n_realizations} realizations", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(HERE, f"{name}_multiplicity.png")
    fig.savefig(png_path)
    plt.close(fig)
    return png_path, csv_path


def main():
    for name, n_realizations, color in NETWORKS:
        print(f"[{name}] {n_realizations} realizations ...")
        png_path, csv_path = plot_network(name, n_realizations, color)
        print(f"  saved {png_path}")
        print(f"  saved {csv_path}")


if __name__ == "__main__":
    main()
