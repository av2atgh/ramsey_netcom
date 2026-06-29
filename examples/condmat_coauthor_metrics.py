"""Metrics for the locally-built cumulative cond-mat co-authorship networks.

Reads the per-year graphs produced by build_condmat_coauthor.py
(examples/data/condmat_coauthor_upto_<year>.gt.gz) and computes, per yearly
snapshot, the number of nodes, mean degree, mean excess degree (<k(k-1)>/<k>),
average shortest-path multiplicity and the fraction of connected node pairs.
One row per year -> examples/condmat_coauthor_metrics.csv.

WARNING: average shortest-path multiplicity is O(n*(n+m)). These cumulative
networks grow to hundreds of thousands of nodes, so the later years are
impractical to compute exactly. Cap the range with:
    ARXIV_MAX_YEAR=<year>    process only snapshots up to this year
    ARXIV_MAX_NODES=<k>      skip any snapshot larger than k nodes

Run from anywhere:
    ARXIV_MAX_YEAR=2005 python examples/condmat_coauthor_metrics.py
"""
import glob
import os
import re
import sys

import graph_tool.all as gt

# make `ramsey_netcom` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramsey_netcom import libs as L

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PATTERN = os.path.join(DATA_DIR, "condmat_coauthor_upto_*.gt.gz")
OUT_CSV = os.path.join(HERE, "condmat_coauthor_metrics.csv")

_YEAR_RE = re.compile(r"upto_(\d{4})\.gt\.gz$")


def year_of(path):
    return int(_YEAR_RE.search(path).group(1))


def main():
    files = sorted(glob.glob(PATTERN), key=year_of)
    if not files:
        sys.exit(f"no networks found at {PATTERN}; run build_condmat_coauthor.py first")
    max_year = int(os.environ.get("ARXIV_MAX_YEAR", "0")) or None
    max_nodes = int(os.environ.get("ARXIV_MAX_NODES", "0")) or None

    # Write+flush each row as its year finishes, so the CSV always holds every
    # completed year even if a later (large) year hangs or the run is killed.
    with open(OUT_CSV, "w") as out:
        out.write("year,n_nodes,n_edges,mean_degree,mean_excess_degree,multiplicity,connected_pairs\n")
        out.flush()
        for path in files:
            year = year_of(path)
            if max_year and year > max_year:
                continue
            g = gt.load_graph(path)
            g.set_directed(False)
            n, e = g.num_vertices(), g.num_edges()
            if max_nodes and n > max_nodes:
                print(f"{year}: n={n} exceeds ARXIV_MAX_NODES={max_nodes}, skipping")
                continue
            mean_k, excess_k = L.get_degrees(g)  # mean degree, <k(k-1)>/<k>
            mult = L.average_shortest_path_multiplicity(g)
            conn = L.connected_pairs_fraction(g)
            out.write(f"{year},{n},{e},{mean_k:.8g},{excess_k:.8g},{mult:.8g},{conn:.8g}\n")
            out.flush()
            os.fsync(out.fileno())
            print(
                f"{year}: n={n} E={e} <k>={mean_k:.2f} excess={excess_k:.2f} "
                f"mult={mult:.4f} conn={conn:.4f}  (saved)"
            )
    print("done ->", OUT_CSV)


if __name__ == "__main__":
    main()
