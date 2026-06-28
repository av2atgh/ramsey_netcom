"""Metrics for BioGRID protein-interaction networks (yeast, fly, worm, human).

Downloads BioGRID's combined per-organism archive (BIOGRID-ORGANISM-LATEST.tab3,
cached under examples/data/), then for each target organism builds the undirected
*physical* interaction graph (Experimental System Type == "physical") and reports
the number of nodes, the average shortest-path multiplicity (mean over node pairs)
and the fraction of connected node pairs. One row per organism -> CSV.

Run from anywhere (note: large one-time download, and human is slow):
    python examples/biogrid_interactome_metrics.py
"""
import os
import shutil
import sys
import zipfile
import urllib.request

import graph_tool.all as gt

# make `ramsey_netcom` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramsey_netcom import libs as L

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip"
ZIP_PATH = os.path.join(DATA_DIR, "BIOGRID-ORGANISM-LATEST.tab3.zip")
OUT_CSV = os.path.join(HERE, "biogrid_interactome_metrics.csv")

# (label, BioGRID organism file name fragment)
ORGANISMS = [
    ("yeast", "Saccharomyces_cerevisiae_S288c"),
    ("fly", "Drosophila_melanogaster"),
    ("worm", "Caenorhabditis_elegans"),
    ("human", "Homo_sapiens"),
]

# tab3 column indices (0-based)
COL_BIOGRID_A, COL_BIOGRID_B, COL_SYS_TYPE = 3, 4, 12


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"downloading {URL} ...")
        # BioGRID blocks the default urllib User-Agent, so set a browser-like one
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"  saved {os.path.getsize(ZIP_PATH) / 1e6:.0f} MB")


def build_physical_graph(zf, fragment):
    """Undirected physical-interaction graph for one organism file in the zip."""
    member = next(
        m for m in zf.namelist() if fragment in m and m.endswith(".tab3.txt")
    )
    edges = []
    with zf.open(member) as fh:
        next(fh)  # header
        for raw in fh:
            row = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(row) <= COL_SYS_TYPE or row[COL_SYS_TYPE] != "physical":
                continue
            try:
                edges.append((int(row[COL_BIOGRID_A]), int(row[COL_BIOGRID_B])))
            except ValueError:
                continue
    g = gt.Graph(directed=False)
    g.add_edge_list(edges, hashed=True, hash_type="int64_t")  # BioGRID IDs
    gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)
    return g, member


def main():
    ensure_data()
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for label, fragment in ORGANISMS:
            g, member = build_physical_graph(zf, fragment)
            n, e = g.num_vertices(), g.num_edges()
            mult = L.average_shortest_path_multiplicity(g)
            conn = L.connected_pairs_fraction(g)
            rows.append((label, n, e, mult, conn))
            print(f"{label:6s} n={n:6d} E={e:7d} mult={mult:.4f} conn={conn:.4f}  [{member}]")

    with open(OUT_CSV, "w") as f:
        f.write("organism,n_nodes,n_edges,multiplicity,connected_pairs\n")
        for label, n, e, mult, conn in rows:
            f.write(f"{label},{n},{e},{mult:.8g},{conn:.8g}\n")
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
