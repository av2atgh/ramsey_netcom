#!/usr/bin/env python3
r"""Community count vs. size for three specific real systems (longitudinal).

Complements the cross-network Netzschleuder analysis with within-system growth,
using the data already downloaded under ../examples/data:
  * as733     -- SNAP Autonomous Systems, daily internet snapshots (subsampled);
  * biogrid   -- BioGRID physical interactomes, one per organism;
  * condmat   -- cumulative arXiv cond-mat co-authorship networks, one per year.

For each network it takes the largest connected component and counts communities
with the degree-corrected SBM (minimize_blockmodel_dl), over several independent
realizations, reporting B_mean and B_std.  Results are appended to
examples_communities.csv as each network finishes (resumable).
"""

import os
import re
import csv
import glob
import json
import zipfile

import numpy as np
import pandas as pd
import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
DATA = os.path.join(EX, "data")
CSV = os.path.join(HERE, "examples_communities.csv")
COLS = ["system", "label", "n", "E", "R", "B_mean", "B_std", "B_runs"]

AS733_SUBSAMPLE = 35          # evenly-spaced daily snapshots
BIOGRID_ZIP = os.path.join(DATA, "BIOGRID-ORGANISM-LATEST.tab3.zip")
COL_A, COL_B, COL_SYS = 3, 4, 12


def realizations(n):
    return 3 if n < 30000 else (2 if n < 150000 else 1)


def lcc_simple(g):
    g = gt.Graph(g, directed=False, prune=True)
    gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    return gt.Graph(g, prune=True)


def count_B(g):
    n = g.num_vertices()
    R = realizations(n)
    Bs = [gt.minimize_blockmodel_dl(g).get_nonempty_B() for _ in range(R)]
    return R, Bs


# ---- loaders -------------------------------------------------------------
def as733_graph(path):
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                edges.append((int(p[0]), int(p[1])))
    g = gt.Graph(directed=False)
    g.add_edge_list(edges, hashed=True, hash_type="int64_t")
    return g


def biogrid_graph(zf, member):
    edges = []
    with zf.open(member) as fh:
        next(fh)
        for raw in fh:
            row = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(row) <= COL_SYS or row[COL_SYS] != "physical":
                continue
            try:
                edges.append((int(row[COL_A]), int(row[COL_B])))
            except ValueError:
                continue
    g = gt.Graph(directed=False)
    if edges:
        g.add_edge_list(edges, hashed=True, hash_type="int64_t")
    return g


# ---- work list -----------------------------------------------------------
def work_items():
    """Yield (system, label, graph-thunk) for every network, small systems first."""
    # AS733 (subsampled, evenly spaced in time)
    files = sorted(glob.glob(os.path.join(DATA, "as-733", "*.txt")))
    if files:
        idx = np.linspace(0, len(files) - 1, AS733_SUBSAMPLE).round().astype(int)
        for i in sorted(set(idx)):
            fn = files[i]
            date = re.search(r"(\d{8})", fn).group(1)
            yield ("as733", date, lambda p=fn: as733_graph(p))
    # BioGRID interactomes (one per organism)
    if os.path.exists(BIOGRID_ZIP):
        zf = zipfile.ZipFile(BIOGRID_ZIP)
        members = sorted(m for m in zf.namelist()
                         if "BIOGRID-ORGANISM-" in m and m.endswith(".tab3.txt"))
        for m in members:
            org = re.search(r"ORGANISM-(.+)-\d", os.path.basename(m)).group(1)
            yield ("biogrid", org, lambda zf=zf, m=m: biogrid_graph(zf, m))
    # cond-mat co-authorship (cumulative, one per year), ascending
    years = sorted(glob.glob(os.path.join(DATA, "condmat_coauthor_upto_*.gt.gz")))
    for path in years:
        yr = re.search(r"upto_(\d{4})", path).group(1)
        yield ("condmat", yr, lambda p=path: gt.load_graph(p))


def done_keys():
    if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
        return set()
    d = pd.read_csv(CSV)
    return set(zip(d["system"], d["label"].astype(str)))


def main():
    gt.seed_rng(42)
    done = done_keys()
    fresh = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if fresh:
        w.writerow(COLS)
        f.flush()
    for system, label, thunk in work_items():
        if (system, label) in done:
            continue
        try:
            g = lcc_simple(thunk())
            n = g.num_vertices()
            if n < 30:                       # too small to be meaningful
                continue
            R, Bs = count_B(g)
            w.writerow([system, label, n, g.num_edges(), R,
                        float(np.mean(Bs)), float(np.std(Bs)), json.dumps(Bs)])
            f.flush()
            os.fsync(f.fileno())
            print(f"{system:8s} {label:28s} n={n:7d}  B={np.mean(Bs):6.1f} "
                  f"+/-{np.std(Bs):5.1f}", flush=True)
        except Exception as e:
            print(f"{system} {label}: FAILED {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
    f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
