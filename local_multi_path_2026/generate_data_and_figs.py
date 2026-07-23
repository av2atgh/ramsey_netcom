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

    data_n_communities_vs_n(
        {"model-": "ba", "m": 2, "a": 2}, log=True, n_min=8, n_max=1025, nr=100
    )
    n_communities_vs_n(
        {"model-": "ls", "d": 1}, DATAPATH, log=True, n_min=8, n_max=1025, nr=NR
    )
    n_communities_vs_n(
        {"model-": "ds", "q": 1 / 3},
        DATAPATH,
        log=True,
        n_min=8,
        n_max=1025,
        nr=NR,
        model="model-ds_q1_3",
    )
    n_communities_vs_n(
        {"model-": "ds", "q": 2 / 5},
        DATAPATH,
        log=True,
        n_min=8,
        n_max=1025,
        nr=NR,
        model="model-ds_q2_5",
    )
    for L in range(4):
        n_communities_vs_n(
            {"model-": "bb", "L": L}, DATAPATH, log=True, n_min=8, n_max=1025, nr=NR
        )
    data_n_communities_vs_n({"model-": "bbd"}, n_min=1, n_max=7, dn=1, nr=1)


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

plt.rcParams.update({"font.size": 18})

fig, ax = plt.subplots(3, 2)

symbol = iter(["o", "s", "^", "X", "D"])
color = iter(plt.cm.Set1.colors[:5])

s = 70

for model, gamma, label, fit, ax_, title in [
    ("model-ls_d1_nr100_vs_n_log.csv", 5, r"$LS(n, 1)^*$", "log-linear", ax[0, 0], "a"),
    (
        "model-ds_q1_3_nr100_vs_n_log.csv",
        3,
        r"DS$(n, 1/3)^*$",
        "log-linear",
        ax[0, 1],
        "b",
    ),
    ("model-bb_l1_nr100_vs_n_log.csv", 3, r"$BB(n, 1)^*$", "log-linear", ax[1, 0], "c"),
    (
        "model-ba_m2_a2_nr100_vs_n_log.csv",
        3,
        r"$BA(n, 2)$",
        "log-linear",
        ax[1, 1],
        "d",
    ),
    (
        "model-ds_q2_5_nr100_vs_n_log.csv",
        5 / 2,
        r"DS$(n, 2/5)^*$",
        "log-linear",
        ax[2, 0],
        "e",
    ),
]:
    df = pd.read_csv(f"data/{model}")
    if "ds_q2_5" in model:
        q = 2 / 5
        x = np.log(df.n)
        y = np.log(df.multipath_rnd)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        slope = 1 / 5
        intercept = np.mean(y - slope * x)
        ax_.plot(df.n, np.exp(intercept + slope * x), "k--", label=r"$b n^{1/5}$")
        yscale = "log"
    else:
        x = np.log(df.n)
        y = df.multipath if "ba" in model else df.multipath_rnd
        z = np.polyfit(x[:-3], y[:-3], 1) if "ls" in model else np.polyfit(x, y, 1)
        p = np.poly1d(z)
        if "ls" in model:
            ax_.plot(df.n, p(x), "k-.", label=r"$a+b\log n$")
        else:
            slope = 1 / np.e
            intercept = np.mean(y - slope * x)
            ax_.plot(df.n, intercept + slope * x, "k-.", label=r"$a +(1/e)\log n$")
        print(model, p)
        yscale = "linear"
    y = df.multipath if "ba" in model else df.multipath_rnd
    ax_.scatter(df.n, y, marker=next(symbol), label=label, s=s, c=next(color))

    ax_.legend(loc="upper left", frameon=0)
    ax_.set(xlabel=r"$n$", ylabel=r"$\left<\mu\right>$", xscale="log", yscale=yscale)
    ax_.set_title(title, x=-0.1, y=1.05)

ax_ = ax[2, 1]

minkk = np.inf
maxkk = -np.inf

symbol = iter(["o", "s", "^", "X", "D", ">", "+"])
color = iter(plt.cm.Set1.colors[:5])

for model, label in [
    ("model-ls_d1_nr100_vs_n_log.csv", r"$LS(n, 1)^*$"),
    ("model-ds_q1_3_nr100_vs_n_log.csv", r"DS$(n, 1/3)^*$"),
    ("model-bb_l1_nr100_vs_n_log.csv", r"$BB(n, 1)^*$"),
    ("model-ba_m2_a2_nr100_vs_n_log.csv", r"$BA(n, 2)$"),
    ("model-ds_q2_5_nr100_vs_n_log.csv", r"DS$(n, 2/5)^*$"),
]:
    df = pd.read_csv(f"data/{model}")
    x = df.mean_kk**2 / df.mean_k
    x = df.mean_kk
    y = df.multipath if "BA" in label else df.multipath_rnd
    ax_.scatter(x, y, marker=next(symbol), s=s, c=next(color))
    minkk = min(minkk, x.min())
    maxkk = max(maxkk, x.max())

maxkk = 10
x = np.array([minkk, maxkk])
ax_.plot(x, x / np.e, "k-", label="Mean-field")

ax_.legend(loc="upper left", frameon=0)
ax_.set(xlabel=r"$c_2$", ylabel=r"$\left<\mu\right>$")
ax_.set_title("f", x=-0.1, y=1.05)

plt.subplots_adjust(bottom=0, right=1.5, wspace=0.25, top=2, hspace=0.35)

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
        f"model-ds_q1_3_nr{NR}_vs_n_log.csv",
        "exp",
        "linear",
        "log",
        "exp",
        "linear",
        "log",
        r"$DS(n, 1/3)$",
    ),
]

plt.rcParams.update({"font.size": 24})

fig, ax = plt.subplots(2, 2)
i = 0
for model, fit, xscale, yscale, fit1, xscale1, yscale1, label in models:
    ax[i][0].set_title("a)" if i == 0 else "c)", x=-0.1, y=1.05)
    ax[i][1].set_title("b)" if i == 0 else "d)", x=-0.1, y=1.05)
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
    ax[1].legend(loc="upper left", frameon=0)


plt.rcParams.update({"font.size": 24})

fig, ax = plt.subplots(2, 2)

for i in [0, 1]:
    ax[i][0].set_title(
        r"a)" if i == 0 else r"c)", x=-0.1, y=1.05, horizontalalignment="left"
    )
    ax[i][1].set_title(r"b)" if i == 0 else r"d)", x=-0.1, y=1.05)

for L in [1, 3]:
    plot2(ax[0], L, "log-quad", "log", "linear", "linear", "linear", "linear")

for L in [2, 4]:
    plot2(ax[1], L, "log-quad", "log", "linear", "linear", "linear", "linear")

plt.subplots_adjust(bottom=0, right=2, wspace=0.2, top=2, hspace=0.3)
plt.savefig(
    "fig3.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)


# figure 4

plt.rcParams.update({"font.size": 16})

fig, ax = plt.subplots(3,1)

ax_ = ax[0]
df = pd.read_csv('data/as733_metrics.csv')
df['multiplicity1'] = df.multiplicity / df.connected_pairs
df['multiplicity_rnd'] = df.mean_excess_degree/np.e
ax_.scatter(df.n_nodes, df.multiplicity1, label='Data')
ax_.scatter(df.n_nodes, df.multiplicity_rnd, label='MF')
fit = 'log-log'
x1, y1 = get_fit(df.n_nodes, df.multiplicity1, fit)
ax_.plot(x1, y1, ls='--', label=fit)
x1, y1 = get_fit(df.n_nodes, df.multiplicity_rnd, fit)
ax_.plot(x1, y1, ls='-.', label=fit)
ax_.set(xscale='log', yscale='log', xlabel=r'$n$', ylabel=r'$\langle\mu\rangle_c$')
ax_.legend(loc='upper left', frameon=False)
ax_.set_title(r'a) Internet Autonomous System (AS) level', x=-0.1, y=1.05, horizontalalignment='left')

ax_ = ax[1]
df = pd.read_csv('data/biogrid_interactome_metrics.csv')
df = df.loc[(df.n_nodes > 999) & ~df.organism.str.contains('irus')].reset_index(drop=True)
df['multiplicity1'] = df.multiplicity / df.connected_pairs
df['multiplicity_rnd'] = df.mean_excess_degree/np.e
ax_.scatter(df.n_nodes, df.multiplicity1, label='Data')
ax_.scatter(df.n_nodes, df.multiplicity_rnd, label='MF')
fit = 'exp'
x1, y1 = get_fit(df.n_nodes, df.multiplicity1, fit)
ax_.plot(x1, y1, ls='--', label=fit)
x1, y1 = get_fit(df.n_nodes, df.multiplicity_rnd, fit)
ax_.plot(x1, y1, ls='-.', label=fit)
ax_.set(xscale='linear', yscale='log', xlabel=r'$n$', ylabel=r'$\langle\mu\rangle_c$')
ax_.legend(loc='lower right', frameon=False)
ax_.set_title(r'b) Protein Interaction Networks', x=-0.1, y=1.05, horizontalalignment='left')

ax_ = ax[2]
df = pd.read_csv('data/condmat_coauthor_metrics.csv')
#df = df.iloc[1:].reset_index(drop=True)
df['multiplicity1'] = df.multiplicity / df.connected_pairs
df['multiplicity_rnd'] = df.mean_excess_degree/np.e
ax_.scatter(df.n_nodes, df.multiplicity1, label='Data')
ax_.scatter(df.n_nodes, df.multiplicity_rnd, label='MF')
fit = 'log-log'
x1, y1 = get_fit(df.n_nodes, df.multiplicity1, fit)
ax_.plot(x1, y1, ls='--', label=fit)
x1, y1 = get_fit(df.n_nodes, df.multiplicity_rnd, fit)
ax_.plot(x1, y1, ls='-.', label=fit)
ax_.set(xscale='log', yscale='log', xlabel=r'$n$', ylabel=r'$\langle\mu\rangle_c$')
ax_.legend(loc='upper left', frameon=False)
ax_.set_title(r'c) arxiv/cond-mat co-authorship', x=-0.1, y=1.05, horizontalalignment='left')

plt.subplots_adjust(bottom=0,  hspace=0.35, top=3)

plt.savefig(
    "fig4.pdf", bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300
)

