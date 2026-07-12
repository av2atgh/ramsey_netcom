#!/usr/bin/env python3
r"""Plot community count vs. size for the three specific systems.

Reads examples_communities.csv (written by examples_communities.py) and draws one
panel per system -- AS733 internet, BioGRID interactomes, cond-mat co-authorship
-- each with per-network error bars, a power-law fit (exponent + bootstrap 95%
CI), and the sqrt(n) reference.  Safe to run at any time while the counting is
still going.

Produces fig_examples.pdf.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "examples_communities.csv")
N_BOOT = 4000
SEED = 42

SYSTEMS = [
    ("as733", "AS internet (daily)", "#ee7733"),
    ("biogrid", "BioGRID interactomes", "#009988"),
    ("condmat", "cond-mat co-authorship", "#cc3311"),
]


def fit(n, B):
    ln_n, ln_B = np.log(n), np.log(B)
    beta, logc = np.polyfit(ln_n, ln_B, 1)
    rng = np.random.default_rng(SEED)
    m = len(n)
    betas = np.array([np.polyfit(ln_n[i], ln_B[i], 1)[0]
                      for i in (rng.integers(0, m, m) for _ in range(N_BOOT))])
    lo, hi = np.percentile(betas, [2.5, 97.5])
    a = np.exp(np.mean(ln_B - 0.5 * ln_n))          # sqrt(n) prefactor, log-scale
    return beta, np.exp(logc), lo, hi, a


def main():
    df = pd.read_csv(CSV).drop_duplicates(["system", "label"])
    df = df[df["B_mean"] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for i, (ax, (sys_key, title, color)) in enumerate(zip(axes, SYSTEMS)):
        letter = "abcdefg"[i]
        sub = df[df.system == sys_key].sort_values("n")
        if len(sub) < 2:
            ax.set_title(f"({letter}) {title}\n(waiting: {len(sub)} pts)")
            continue
        n = sub["n"].to_numpy(float)
        B = sub["B_mean"].to_numpy(float)
        Berr = sub["B_std"].to_numpy(float)
        beta, c, lo, hi, a = fit(n, B)
        ng = np.logspace(np.log10(n.min()), np.log10(n.max()), 100)
        ax.plot(ng, c * ng ** beta, "-", color="#333333", lw=1.5,
                label=rf"$B\propto n^{{{beta:.2f}}}$  (95% CI [{lo:.2f}, {hi:.2f}])")
        ax.errorbar(n, B, yerr=Berr, fmt="o", color=color, ms=5.5,
                    ecolor=color, elinewidth=0.9, capsize=2,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"number of nodes, $n$", fontsize=11)
        ax.set_ylabel(r"number of communities, $B$", fontsize=11)
        ax.set_title(f"({letter}) {title}  ($N={len(sub)}$)", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(True, which="both", alpha=0.2)
        print(f"{sys_key:8s} N={len(sub):3d}  n=[{int(n.min())},{int(n.max())}]  "
              f"beta={beta:.3f}  CI95=[{lo:.3f},{hi:.3f}]")

    fig.tight_layout()
    out = os.path.join(HERE, "fig_examples.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
