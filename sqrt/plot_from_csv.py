#!/usr/bin/env python3
r"""Draw the real-network figure from communities_vs_n.csv.

Reads the incremental output of run_sbm.py, plots the number of SBM communities B
versus network size n (log-log) with per-network error bars, a power-law fit and
a bootstrap 95% CI over networks.  Points are coloured by mean degree, so that
denser networks (which sit above the line) and sparse ones (below) are visible.
Safe to run at any time while run_sbm.py is still going.  Produces fig1.pdf.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "communities_vs_n.csv")
N_BOOT = 4000
SEED = 42

# geographic / spatially embedded networks are drawn as squares
GEO_KEYS = ["road", "street", "power", "euroroad", "contiguous_usa",
            "london_transport", "airline", "airport", "flight", "faa",
            "openflights", "fullerene", "us_roads", "roadnet", "transport",
            "metro", "urban"]


def is_geo(name):
    s = name.lower()
    return any(k in s for k in GEO_KEYS)


def main():
    df = pd.read_csv(CSV).drop_duplicates("network").sort_values("n")
    df = df[df["B_mean"] > 0].reset_index(drop=True)
    n = df["n"].to_numpy(float)
    B = df["B_mean"].to_numpy(float)
    Berr = df["B_std"].to_numpy(float)
    kmean = 2 * df["E"].to_numpy(float) / n          # mean degree
    geo = df["network"].apply(is_geo).to_numpy(bool)  # geographic networks

    ln_n, ln_B = np.log(n), np.log(B)
    beta, logc = np.polyfit(ln_n, ln_B, 1)
    # sqrt(n) prefactor from a log-scale fit with the exponent fixed at 0.5
    a = np.exp(np.mean(ln_B - 0.5 * ln_n))

    rng = np.random.default_rng(SEED)
    m = len(n)
    betas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, m, m)
        betas[i] = np.polyfit(ln_n[idx], ln_B[idx], 1)[0]
    lo, hi = np.percentile(betas, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(6.1, 4.3))
    ng = np.logspace(np.log10(n.min()), np.log10(n.max()), 100)
    ax.plot(ng, np.exp(logc) * ng ** beta, "-", color="#333333", lw=1.5, zorder=2,
            label=rf"$B \propto n^{{{beta:.2f}}}$  (95% CI [{lo:.2f}, {hi:.2f}])")
    ax.errorbar(n, B, yerr=Berr, fmt="none", ecolor="0.6", elinewidth=0.8,
                capsize=2.0, zorder=2)
    norm = LogNorm(vmin=kmean.min(), vmax=kmean.max())
    ax.scatter(n[~geo], B[~geo], c=kmean[~geo], cmap="coolwarm", norm=norm,
               marker="o", s=42, edgecolors="white", linewidths=0.5, zorder=3)
    sc = ax.scatter(n[geo], B[geo], c=kmean[geo], cmap="coolwarm", norm=norm,
                    marker="s", s=48, edgecolors="white", linewidths=0.5, zorder=4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"mean degree $\langle k\rangle$", fontsize=10)
    # marker-shape legend (circle = non-geographic, square = geographic)
    from matplotlib.lines import Line2D
    shape = [Line2D([], [], marker="o", ls="", mfc="0.6", mec="white", ms=7,
                    label="non-geographic"),
             Line2D([], [], marker="s", ls="", mfc="0.6", mec="white", ms=7,
                    label="geographic")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of nodes, $n$", fontsize=11)
    ax.set_ylabel(r"number of communities, $B$", fontsize=11)
    leg1 = ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=shape, frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1.pdf"), bbox_inches="tight")

    print(f"{len(df)} networks, n = {int(n.min())}..{int(n.max())}; "
          f"beta = {beta:.3f} (95% CI [{lo:.3f}, {hi:.3f}])")


if __name__ == "__main__":
    main()
