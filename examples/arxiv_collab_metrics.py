"""Metrics for the arXiv co-authorship networks (Netzschleuder 'arxiv_collab').

The arxiv_collab entry bundles several co-authorship snapshots by category and
year (cond-mat 1999/2003/2005, astro-ph 1999, hep-th 1999). For each sub-network
this computes the number of nodes, the average shortest-path multiplicity (mean
over node pairs) and the fraction of connected node pairs, writing one row per
sub-network to examples/arxiv_collab_metrics.csv.

Run from anywhere:
    python examples/arxiv_collab_metrics.py
"""
import os
import sys

import graph_tool.all as gt

# make `ramsey_netcom` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramsey_netcom import libs as L

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTION = "arxiv_collab"
OUT_CSV = os.path.join(HERE, "arxiv_collab_metrics.csv")


def main():
    subnets = gt.collection.ns_info[COLLECTION]["nets"]  # discovered dynamically
    rows = []
    for i, sub in enumerate(subnets, 1):
        g = L.load_network(f"{COLLECTION}/{sub}")
        g.set_directed(False)
        gt.remove_parallel_edges(g)  # simple graph for shortest-path multiplicity
        gt.remove_self_loops(g)
        n, e = g.num_vertices(), g.num_edges()
        category, year = sub.rsplit("-", 1)
        mean_k, excess_k = L.get_degrees(g)  # mean degree, <k(k-1)>/<k>
        mult = L.average_shortest_path_multiplicity(g)
        conn = L.connected_pairs_fraction(g)
        rows.append((sub, category, year, n, e, mean_k, excess_k, mult, conn))
        print(
            f"[{i}/{len(subnets)}] {sub}: n={n} E={e} <k>={mean_k:.2f} "
            f"excess={excess_k:.2f} mult={mult:.4f} conn={conn:.4f}"
        )

    with open(OUT_CSV, "w") as f:
        f.write(
            "network,category,year,n_nodes,n_edges,mean_degree,mean_excess_degree,"
            "multiplicity,connected_pairs\n"
        )
        for sub, category, year, n, e, mean_k, excess_k, mult, conn in rows:
            f.write(
                f"{sub},{category},{year},{n},{e},{mean_k:.8g},{excess_k:.8g},"
                f"{mult:.8g},{conn:.8g}\n"
            )
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
