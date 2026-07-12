#!/usr/bin/env python3
r"""Community count vs. size for two generative network models.

Tests whether simple growth models reproduce the empirical community scaling:
  * local_search (d=1)  -- a social-network model (cf. co-authorship / cond-mat);
  * dup_split   (q=0.3) -- a duplication model of protein networks (cf. BioGRID).

For each model and each target size it generates several independent realizations,
counts communities with one degree-corrected SBM run per realization
(minimize_blockmodel_dl, for speed), and appends one row per realization to
models_communities.csv (resumable).
"""

import os
import csv
import sys

import numpy as np
import pandas as pd
import graph_tool.all as gt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ramsey_netcom.libs import generator, graphtool_from_neighbours  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "models_communities.csv")
COLS = ["model", "n_target", "seed", "n", "E", "B"]

MODELS = [
    ("ls", {"model-": "ls", "d": 1}),        # social  -> cond-mat
    ("ds", {"model-": "ds", "q": 0.3}),      # protein -> BioGRID
]
SIZES = [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000]
N_REAL = 10


def lcc_simple(g):
    gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    return gt.Graph(g, prune=True)


def done_keys():
    if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
        return set()
    d = pd.read_csv(CSV)
    return set(zip(d["model"], d["n_target"], d["seed"]))


def main():
    gt.seed_rng(1)
    done = done_keys()
    fresh = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if fresh:
        w.writerow(COLS)
        f.flush()
    # smaller sizes first so points fill in fast
    for nt in SIZES:
        for model, params in MODELS:
            for seed in range(N_REAL):
                if (model, nt, seed) in done:
                    continue
                nb = generator(nt, params, seed=seed)
                g = lcc_simple(graphtool_from_neighbours(nb))
                B = gt.minimize_blockmodel_dl(g).get_nonempty_B()
                w.writerow([model, nt, seed, g.num_vertices(), g.num_edges(), B])
                f.flush()
                os.fsync(f.fileno())
            sub = pd.read_csv(CSV)
            sub = sub[(sub.model == model) & (sub.n_target == nt)]
            if len(sub):
                print(f"{model}  n={nt:6d}  B={sub.B.mean():6.1f} +/-{sub.B.std():5.1f}"
                      f"  ({len(sub)} realizations)", flush=True)
    f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
