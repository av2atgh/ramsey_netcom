#!/usr/bin/env python3
r"""Test whether geographic (spatially embedded) networks differ from the rest.

Reads communities_vs_n.csv, fits the power law B = c n^beta, and classifies each
network as geographic (roads, streets, power grids, transport, spatial adjacency,
molecular lattice) or non-geographic by name.  Tests whether geographic networks
sit below the fit line (lower residual) and whether they are sparser (lower mean
degree 2E/n) than non-geographic ones, by one-sided Mann-Whitney tests.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "communities_vs_n.csv")

GEO_KEYS = ["road", "street", "power", "euroroad", "contiguous_usa",
            "london_transport", "airline", "airport", "flight", "faa",
            "openflights", "fullerene", "us_roads", "roadnet", "transport",
            "metro", "urban"]


def is_geo(name):
    s = name.lower()
    return any(k in s for k in GEO_KEYS)


def main():
    df = pd.read_csv(CSV).drop_duplicates("network")
    df = df[df["B_mean"] > 0].copy()
    df["geo"] = df["network"].apply(is_geo)
    n = df["n"].to_numpy(float)
    B = df["B_mean"].to_numpy(float)
    k = 2 * df["E"].to_numpy(float) / n

    ln_n, ln_B = np.log(n), np.log(B)
    beta, c = np.polyfit(ln_n, ln_B, 1)
    resid = ln_B - (beta * ln_n + c)

    g = df["geo"].to_numpy(bool)
    print(f"{len(df)} networks; fit exponent beta = {beta:.3f}")
    print(f"geographic ({g.sum()}): "
          + ", ".join(df.loc[g, "network"].str.split("/").str[0]))
    print()
    for label, x in [("fit residual", resid), ("mean degree <k>", k)]:
        xg, xn = x[g], x[~g]
        # one-sided: geographic LESS than non-geographic
        U, p = stats.mannwhitneyu(xg, xn, alternative="less")
        rb = 2 * U / (len(xg) * len(xn)) - 1
        print(f"{label:16s}: geo median = {np.median(xg):7.2f}  "
              f"non-geo median = {np.median(xn):7.2f}  "
              f"Mann-Whitney(geo<non-geo) p = {p:.2e}  (rank-biserial {rb:.2f})")


if __name__ == "__main__":
    main()
