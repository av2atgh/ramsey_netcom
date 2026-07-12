#!/usr/bin/env python3
r"""Plot community count vs. size for the two generative models.

Reads models_communities.csv, aggregates B (mean +/- std) per size, and for each
model fits a power law B = c n^beta with a bootstrap 95% confidence interval on
the exponent.  Produces fig_models.pdf.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "models_communities.csv")
N_BOOT = 4000
SEED = 42

MODELS = [
    ("ls", "triadic closure  (social)", "#3366aa"),
    ("ds", "duplication–split  (protein)", "#009988"),
]


def fits(n, B):
    ln_n, ln_B = np.log(n), np.log(B)
    beta, logc = np.polyfit(ln_n, ln_B, 1)
    rng = np.random.default_rng(SEED)
    m = len(n)
    betas = np.array([np.polyfit(ln_n[i], ln_B[i], 1)[0]
                      for i in (rng.integers(0, m, m) for _ in range(N_BOOT))])
    lo, hi = np.percentile(betas, [2.5, 97.5])
    return dict(beta=beta, c=np.exp(logc), lo=lo, hi=hi)


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (ax, (mk, title, color)) in enumerate(zip(axes, MODELS)):
        letter = "abcdefg"[i]
        s = df[df.model == mk].groupby("n_target").agg(
            n=("n", "mean"), B=("B", "mean"), Bs=("B", "std"), k=("B", "size")).reset_index()
        s = s[s.B > 0].sort_values("n")
        if len(s) < 3:
            ax.set_title(f"({letter}) {title}\n(waiting: {len(s)} sizes)")
            continue
        n = s["n"].to_numpy(float); B = s["B"].to_numpy(float); Bs = s["Bs"].to_numpy(float)
        r = fits(n, B)
        ng = np.logspace(np.log10(n.min()), np.log10(n.max()), 100)
        ax.plot(ng, r["c"] * ng ** r["beta"], "-", color="#333333", lw=1.5,
                label=rf"$B\propto n^{{{r['beta']:.2f}}}$  (95% CI [{r['lo']:.2f}, {r['hi']:.2f}])")
        ax.errorbar(n, B, yerr=Bs, fmt="o", color=color, ms=5.5,
                    ecolor=color, elinewidth=0.9, capsize=2,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"number of nodes, $n$", fontsize=11)
        ax.set_ylabel(r"number of communities, $B$", fontsize=11)
        ax.set_title(rf"({letter}) {title}", fontsize=11)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        ax.grid(True, which="both", alpha=0.2)
        print(f"{mk} ({letter}): N={len(s)} n=[{int(n.min())},{int(n.max())}]  "
              f"beta={r['beta']:.3f}  CI95=[{r['lo']:.3f},{r['hi']:.3f}]")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_models.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
