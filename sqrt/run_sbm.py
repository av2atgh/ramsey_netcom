#!/usr/bin/env python3
r"""Long-running SBM community count for real undirected networks (incremental).

Launch this yourself, e.g.

    python run_sbm.py

It processes every feasible undirected network from network_list.json in
ascending size order (n = 34 up to ~4,000,000; only the >10M-node giants are
excluded).  Each network's result is APPENDED to communities_vs_n.csv the moment
its inference finishes, so the file is always usable while the run continues, and
plot_from_csv.py can be run at any time to draw the current picture.

The run is resumable: networks already present in the CSV are skipped, so you can
stop it (Ctrl-C) and relaunch without losing or repeating work.  Failures on
individual networks (download/memory/time) are caught and skipped; those
networks are simply retried on the next launch.
"""

import os
import csv
import json
import time

import numpy as np
import pandas as pd
import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(HERE, "network_list.json")
CSV = os.path.join(HERE, "communities_vs_n.csv")
COLS = ["network", "n", "E", "R", "B_mean", "B_std", "B_runs"]


def realizations(n):
    if n < 30000:
        return 3
    if n < 150000:
        return 2
    return 1


def lcc_simple(g):
    g = gt.Graph(g, directed=False, prune=True)
    gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    return gt.Graph(g, prune=True)


def already_done():
    if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
        return set()
    try:
        return set(pd.read_csv(CSV)["network"].astype(str))
    except Exception:
        return set()


def main():
    gt.seed_rng(42)
    nets = json.load(open(LIST))              # [[path, n_meta, E_meta], ...] ascending
    done = already_done()
    fresh = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if fresh:
        w.writerow(COLS)
        f.flush()

    print(f"{len(nets)} networks total; {len(done)} already done", flush=True)
    for path, n_meta, _ in nets:
        if path in done:
            continue
        t0 = time.time()
        try:
            g0 = gt.collection.ns[path]
            if g0.is_directed():
                print(f"{path}: directed; skip", flush=True)
                continue
            g = lcc_simple(g0)
            n = g.num_vertices()
            R = realizations(n)
            Bs = [gt.minimize_blockmodel_dl(g).get_nonempty_B() for _ in range(R)]
            w.writerow([path, n, g.num_edges(), R,
                        float(np.mean(Bs)), float(np.std(Bs)), json.dumps(Bs)])
            f.flush()
            os.fsync(f.fileno())
            print(f"{path:46s} n={n:8d}  B={np.mean(Bs):7.1f} +/-{np.std(Bs):5.1f}"
                  f"  (R={R}, {time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"{path}: FAILED {type(ex).__name__}: {str(ex)[:120]}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            continue
    f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
