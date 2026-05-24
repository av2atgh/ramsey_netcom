"""
wrapper for paper: Local network evolution rules drive shortest path multiplicity
"""

import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from ramsey_netcom.libs import n_communities_vs_n


DATAPATH = "data"

GENERATE_DATA = False

NR = 100


if GENERATE_DATA:
    """
    WARNING: it will take days to finish!
    Reduce NR for faster yet less accurate results.
    """

    if not os.path.isdir(DATAPATH):
        os.makedirs(DATAPATH)

    n_communities_vs_n(
        {"model-": "ls", "d": 1}, DATAPATH, log=True, n_min=8, n_max=1025, nr=NR
    )
    n_communities_vs_n(
        {"model-": "ds", "q": 0.3}, DATAPATH, log=True, n_min=8, n_max=1025, nr=NR
    )
    for L in range(4):
        n_communities_vs_n(
            {"model-": "bb", "L": L}, DATAPATH, log=True, n_min=8, n_max=1025, nr=NR
        )


def get_fit(x, y, fit, num=1000, report=False):
    if fit == "linear":
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x1 = np.linspace(x.min(), x.max(), num=num)
        y1 = p(x1)
    elif fit == "log-linear" or fit == "log-quad" or fit == "log-log":
        x_ = np.log(x)
        y_ = np.log(y) if fit == "log-log" else y
        z = np.polyfit(x_, y_, 2 if fit == "log-quad" else 1)
        p = np.poly1d(z)
        x1_ = np.linspace(x_.min(), x_.max(), num=num)
        x1 = np.exp(x1_)
        y1 = p(x1_)
        if fit == "log-log":
            y1 = np.exp(y1)
    elif fit == "exp":
        y_ = np.log(y)
        z = np.polyfit(x, y_, 1)
        p = np.poly1d(z)
        x1 = np.linspace(x.min(), x.max(), num=num)
        y1 = np.exp(p(x1))
    if report:
        print(p)
    return x1, y1


def line_type(fit):
    return "-" if fit == "linear" else "--" if fit == "exp" else "-."


# figure 1

plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots()

symbol = iter(["o", "s", "^", "X", "D"])

for model, label in [
    (f"model-ls_d1_nr{NR}_vs_n_log.csv", r"$LS(n, 1)$"),
    (f"model-ds_q0.3_nr{NR}_vs_n_log.csv", r"DS$(n, 0.3)$"),
    (f"model-bb_l1_nr{NR}_vs_n_log.csv", r"$BB(n, 1)$"),
]:
    df = pd.read_csv(f"{DATAPATH}/{model}")
    ax.scatter(df.n, df.multipath_rnd, marker=next(symbol), label=label)
    x = np.log(df.n)
    z = np.polyfit(x, df.multipath_rnd, 1)
    p = np.poly1d(z)
    ax.plot(df.n, p(x))

ax.legend(loc="lower right", frameon=0)
ax.set(xlabel=r"$n$", ylabel=r"$\left<\mu\right>$", xscale="log")
plt.savefig(
    "fig1.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)

# figure 2


def plot1(ax, model, fit, xscale, yscale, fit1, xscale1, yscale1, label):
    df = pd.read_csv(f"{DATAPATH}/{model}")
    m = model.split("-")[1]
    m = m.split("_")
    ax[0].scatter(df.n, df.multipath, label=label, s=100)
    x1, y1 = get_fit(df.n, df.multipath, fit)
    ax[0].plot(x1, y1, line_type(fit), label=fit)
    ax[0].set(xlabel=r"$n$", ylabel=r"$\left<\mu\right>$", xscale=xscale, yscale=yscale)
    ax[0].legend(loc="upper left", frameon=0)
    ax[1].scatter(df.unique_mean, df.multipath, label=label, s=100)
    df1 = df.loc[df.unique_mean > 1]
    x1, y1 = get_fit(df1.unique_mean, df1.multipath, fit1)
    ax[1].plot(x1, y1, line_type(fit1), label=fit1)
    ax[1].set(
        xlabel=r"$\left<\kappa\right>$",
        ylabel=r"$\left<\mu\right>$",
        xscale=xscale1,
        yscale=yscale1,
    )
    ax[1].legend(loc="upper left", frameon=0)


models = [
    (
        f"model-ls_d1_nr{NR}_vs_n_log.csv",
        "log-quad",
        "log",
        "linear",
        "linear",
        "linear",
        "linear",
        r"$LS(n, 1)$",
    ),
    (
        f"model-ds_q0.3_nr{NR}_vs_n_log.csv",
        "exp",
        "linear",
        "log",
        "exp",
        "linear",
        "log",
        r"$DS(n, 0.3)$",
    ),
]

plt.rcParams.update({"font.size": 24})

fig, ax = plt.subplots(2, 2)
i = 0
for model, fit, xscale, yscale, fit1, xscale1, yscale1, label in models:
    ax[i][0].set_title("a)" if i == 0 else "b)", x=-0.1, y=1.05)
    ax[i][1].set_title("c)" if i == 0 else "d)", x=-0.1, y=1.05)
    plot1(ax[i], model, fit, xscale, yscale, fit1, xscale1, yscale1, label)
    i += 1

plt.subplots_adjust(bottom=0, right=2, wspace=0.25, top=2, hspace=0.35)
plt.savefig(
    "fig2.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)

# figure 3


def plot2(ax, L, fit, xscale, yscale, fit1, xscale1, yscale1):
    df = pd.read_csv(f"{DATAPATH}/model-bb_L{L}_nr{NR}_vs_n_log.csv")
    ax[0].scatter(df.n, df.multipath, label=f"$L=${L}", s=100)
    x1, y1 = get_fit(df.n, df.multipath, fit)
    ax[0].plot(x1, y1, line_type(fit))
    ax[0].set(xlabel=r"$n$", ylabel=r"$\left<\mu\right>$", xscale=xscale, yscale=yscale)
    ax[0].legend(loc="upper left", frameon=0)
    ax[1].scatter(df.unique_mean, df.multipath, label=f"$L=${L}", s=100)
    df1 = df.loc[df.unique_mean > 1].reset_index(drop=True)
    if len(df1) > 1:
        x1, y1 = get_fit(df1.unique_mean, df1.multipath, fit1)
        ax[1].plot(x1, y1, line_type(fit1))
    ax[1].set(
        xlabel=r"$\left<\kappa\right>$",
        ylabel=r"$\left<\mu\right>$",
        xscale=xscale1,
        yscale=yscale1,
    )
    ax[1].legend(loc="lower right", frameon=0)


plt.rcParams.update({"font.size": 24})

fig, ax = plt.subplots(2, 2)

for i in [0, 1]:
    ax[i][0].set_title(
        r"a)" if i == 0 else r"b)", x=-0.1, y=1.05, horizontalalignment="left"
    )
    ax[i][1].set_title(r"c)" if i == 0 else r"d)", x=-0.1, y=1.05)

for L in [1, 3]:
    plot2(ax[0], L, "log-quad", "log", "linear", "linear", "linear", "linear")

for L in [2, 4]:
    plot2(ax[1], L, "log-quad", "log", "linear", "linear", "linear", "linear")

plt.subplots_adjust(bottom=0, right=2, wspace=0.2, top=2, hspace=0.3)
plt.savefig(
    "fig3.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)


# figure 4

plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots()

df = pd.read_csv(f"{DATAPATH}/model-bb_deterministic_vs_n_log.csv")
df["multipath"] = df.M / (df.n * (df.n - 1) / 2)

ax.scatter(df.n, df.multipath)
x1, y1 = get_fit(df.n, df.multipath, "log-linear")
ax.plot(x1, y1)
ax.set(xlabel=r"$n$", ylabel=r"$\mu$", xscale="log")

plt.savefig(
    "fig4.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)
