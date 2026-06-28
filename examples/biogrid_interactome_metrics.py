"""Metrics for every BioGRID per-organism protein-interaction network.

Downloads BioGRID's combined per-organism archive (BIOGRID-ORGANISM-LATEST.tab3,
cached under examples/data/), then for each organism in the archive builds the
undirected *physical* interaction graph (Experimental System Type == "physical")
and reports the number of nodes, the average shortest-path multiplicity (mean
over node pairs) and the fraction of connected node pairs. One row per organism
-> examples/biogrid_interactome_metrics.csv.

Note: large one-time download, and the big organisms (e.g. human) are slow.
Organisms with no physical interactions report n=0 and nan metrics.

Run from anywhere:
    python examples/biogrid_interactome_metrics.py
"""
import os
import re
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

# tab3 column indices (0-based)
COL_BIOGRID_A, COL_BIOGRID_B, COL_SYS_TYPE = 3, 4, 12

_MEMBER_RE = re.compile(r"BIOGRID-ORGANISM-(.+)-\d+\.\d+\.\d+\.tab3\.txt$")


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"downloading {URL} ...")
        # BioGRID blocks the default urllib User-Agent, so set a browser-like one
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"  saved {os.path.getsize(ZIP_PATH) / 1e6:.0f} MB")


def organism_of(member):
    m = _MEMBER_RE.search(os.path.basename(member))
    return m.group(1) if m else os.path.basename(member)


def build_physical_graph(zf, member):
    """Undirected physical-interaction graph for one organism file in the zip."""
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
    if edges:
        g.add_edge_list(edges, hashed=True, hash_type="int64_t")  # BioGRID IDs
        gt.remove_parallel_edges(g)
        gt.remove_self_loops(g)
    return g


def main():
    ensure_data()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(
            m for m in zf.namelist()
            if "BIOGRID-ORGANISM-" in m and m.endswith(".tab3.txt")
        )
        rows = []
        for i, member in enumerate(members, 1):
            org = organism_of(member)
            g = build_physical_graph(zf, member)
            n, e = g.num_vertices(), g.num_edges()
            mult = L.average_shortest_path_multiplicity(g)
            conn = L.connected_pairs_fraction(g)
            rows.append((org, n, e, mult, conn))
            print(f"[{i}/{len(members)}] {org}: n={n} E={e} mult={mult:.4f} conn={conn:.4f}")

    with open(OUT_CSV, "w") as f:
        f.write("organism,n_nodes,n_edges,multiplicity,connected_pairs\n")
        for org, n, e, mult, conn in rows:
            f.write(f"{org},{n},{e},{mult:.8g},{conn:.8g}\n")
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
