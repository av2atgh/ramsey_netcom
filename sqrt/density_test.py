#!/usr/bin/env python3
r"""Test whether networks above the power-law fit are denser than those below.

Reads communities_vs_n.csv, fits the power law B = c n^beta (least squares on
log B), and classifies each network by the sign of its residual (above/below the
fit line).  Tests whether above-line networks have a higher mean degree
2E/n than below-line ones (one-sided Mann-Whitney), and whether the residual
correlates with mean degree (Spearman).  Supports the density statement in the
Results.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "communities_vs_n.csv")


def main():
    df = pd.read_csv(CSV).drop_duplicates("network")
    df = df[df["B_mean"] > 0].copy()
    n = df["n"].to_numpy(float)
    B = df["B_mean"].to_numpy(float)
    k = 2 * df["E"].to_numpy(float) / n                 # mean degree

    ln_n, ln_B = np.log(n), np.log(B)
    beta, c = np.polyfit(ln_n, ln_B, 1)
    resid = ln_B - (beta * ln_n + c)                    # > 0: above the line
    above, below = resid > 0, resid < 0
    ka, kb = k[above], k[below]

    U, p_mw = stats.mannwhitneyu(ka, kb, alternative="greater")
    rho, p_sp = stats.spearmanr(resid, k)
    rank_biserial = 2 * U / (len(ka) * len(kb)) - 1

    print(f"{len(df)} networks; fit exponent beta = {beta:.3f}")
    print(f"  above the line: {above.sum():3d}  median <k> = {np.median(ka):5.1f}"
          f"  mean <k> = {ka.mean():5.1f}")
    print(f"  below the line: {below.sum():3d}  median <k> = {np.median(kb):5.1f}"
          f"  mean <k> = {kb.mean():5.1f}")
    print(f"Mann-Whitney (above > below): U = {U:.0f}, p = {p_mw:.2e}, "
          f"rank-biserial = {rank_biserial:.3f}")
    print(f"Spearman (residual vs <k>):   rho = {rho:.3f}, p = {p_sp:.2e}")


if __name__ == "__main__":
    main()
