"""Temporal metrics for the SNAP Autonomous Systems AS-733 dataset.

Downloads as-733.tar.gz from SNAP (once, cached under examples/data/), then for
each daily snapshot (Nov 1997 - Jan 2000) computes:
    - number of nodes
    - average shortest-path multiplicity (mean over node pairs)
    - fraction of connected node pairs (mean over node pairs of multiplicity > 0)
and writes one row per snapshot to examples/as733_metrics.csv.

Set AS733_LIMIT=<k> to process only the first k snapshots (quick test).

Run from anywhere:
    python examples/as733_metrics.py
"""
import os
import re
import sys
import tarfile
import urllib.request

import graph_tool.all as gt

# make `ramsey_netcom` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramsey_netcom import libs as L

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://snap.stanford.edu/data/as-733.tar.gz"
TARBALL = os.path.join(DATA_DIR, "as-733.tar.gz")
EXTRACT_DIR = os.path.join(DATA_DIR, "as-733")
OUT_CSV = os.path.join(HERE, "as733_metrics.csv")


def ensure_data():
    """Download and extract the AS-733 tarball if not already present."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if any(f.endswith(".txt") for f in os.listdir(EXTRACT_DIR)):
        return
    if not os.path.exists(TARBALL):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, TARBALL)
    print("extracting ...")
    with tarfile.open(TARBALL) as tf:
        tf.extractall(EXTRACT_DIR, filter="data")


def load_snapshot(path):
    """Parse a SNAP AS edge-list file into a simple undirected graph-tool graph."""
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    g = gt.Graph(directed=False)
    g.add_edge_list(edges, hashed=True, hash_type="int64_t")  # ASNs are arbitrary ints
    gt.remove_parallel_edges(g)  # collapse reciprocal listings -> simple graph
    gt.remove_self_loops(g)
    return g


def date_of(filename):
    m = re.search(r"(\d{8})", filename)
    return m.group(1) if m else filename


def main():
    ensure_data()
    files = sorted(f for f in os.listdir(EXTRACT_DIR) if f.endswith(".txt"))
    limit = int(os.environ.get("AS733_LIMIT", "0"))
    if limit > 0:
        files = files[:limit]

    rows = []
    for i, fn in enumerate(files, 1):
        g = load_snapshot(os.path.join(EXTRACT_DIR, fn))
        date = date_of(fn)
        n, e = g.num_vertices(), g.num_edges()
        mean_k, excess_k = L.get_degrees(g)  # mean degree, <k(k-1)>/<k>
        mult = L.average_shortest_path_multiplicity(g)
        conn = L.connected_pairs_fraction(g)
        rows.append((date, n, e, mean_k, excess_k, mult, conn))
        print(
            f"[{i}/{len(files)}] {date}: n={n} E={e} <k>={mean_k:.2f} "
            f"excess={excess_k:.2f} mult={mult:.4f} conn={conn:.4f}"
        )

    with open(OUT_CSV, "w") as f:
        f.write("date,n_nodes,n_edges,mean_degree,mean_excess_degree,multiplicity,connected_pairs\n")
        for date, n, e, mean_k, excess_k, mult, conn in rows:
            f.write(f"{date},{n},{e},{mean_k:.8g},{excess_k:.8g},{mult:.8g},{conn:.8g}\n")
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
