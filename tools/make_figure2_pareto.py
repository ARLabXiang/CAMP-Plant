"""Replacement for Figure 2: two-panel Pareto scatter (SSIM vs Pixel MAE,
SSIM vs POI-MAE) on the Komatsuna test set.

Design choices for legibility under heavy point overlap:
- No inline text labels; every method gets a unique (color, marker shape) pair.
- Pareto-optimal methods drawn larger with a bold black edge ring.
- Single legend on the right grouped by family (CAMP / recurrent / non-recurrent).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "work_dirs" / "komatsuna_ep150_results.csv"
OUT_DIR = ROOT / "figures" / "figure2_replacement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordered groups → legend ordering. (display_name, color, marker)
GROUPS = {
    "CAMP family": [
        ("CAMP",        "camp",        "#d62728", "o"),
        ("CAMP_no_cls", "camp_no_cls", "#ff7f0e", "^"),
        ("CAMP_full",   "camp_full",   "#8c564b", "s"),
    ],
    "Recurrent baselines": [
        ("ConvLSTM",    "convlstm",    "#1f77b4", "o"),
        ("PredRNN",     "predrnn",     "#17becf", "D"),
        ("PhyDNet",     "phydnet",     "#9467bd", "v"),
        ("MIM",         "mim",         "#2ca02c", "P"),
        ("MIM_full",    "mim_full",    "#98df8a", "X"),
    ],
    "Non-recurrent baselines": [
        ("SimVP",       "simvp",       "#555555", "o"),
        ("SimVP_full",  "simvp_full",  "#aaaaaa", "X"),
        ("TAU",         "tau",         "#bcbd22", "*"),
        ("TAU_no_cls",  "tau_no_cls",  "#dbdb8d", "p"),
    ],
}

# Flatten to lookup tables for plotting.
STYLE = {}      # method_key → (color, marker)
DISPLAY = {}    # method_key → display name
for entries in GROUPS.values():
    for disp, key, color, marker in entries:
        STYLE[key] = (color, marker)
        DISPLAY[key] = disp


def pareto_front(ssim, err):
    """Higher SSIM, lower err is better. Returns mask of non-dominated points."""
    n = len(ssim)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ssim[j] >= ssim[i] and err[j] <= err[i] and (ssim[j] > ssim[i] or err[j] < err[i]):
                dominated[i] = True
                break
    return ~dominated


def panel(ax, df, ycol, ylabel, title):
    x = df["ssim"].to_numpy()
    y = df[ycol].to_numpy()
    keys = df["method"].tolist()

    front = pareto_front(x, y)
    order = np.argsort(x[front])
    ax.plot(x[front][order], y[front][order],
            color="0.55", lw=1.0, ls="--", zorder=1)

    # Plot star markers (TAU) slightly larger so they read at the same visual weight.
    for i, key in enumerate(keys):
        color, marker = STYLE[key]
        size = 110 if front[i] else 70
        if marker == "*":
            size = int(size * 1.7)
        ax.scatter(x[i], y[i],
                   s=size,
                   c=color,
                   marker=marker,
                   edgecolor="black" if front[i] else "0.3",
                   linewidth=1.4 if front[i] else 0.6,
                   zorder=3)

    ax.set_xlabel("SSIM ↑", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, loc="left", pad=6)
    ax.tick_params(labelsize=9, length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, color="0.85", zorder=0)

    pad_x = (x.max() - x.min()) * 0.07
    pad_y = (y.max() - y.min()) * 0.10
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)


def make_legend(fig, df):
    """Bottom legend, flat (no group headers), arranged as 3 rows × 5 columns
    so each plot gets the full horizontal space."""
    method_keys = set(df["method"].tolist())
    handles = []
    labels = []

    # Methods in family order so colors cluster nicely in the legend grid.
    for entries in GROUPS.values():
        for disp, key, color, marker in entries:
            if key not in method_keys:
                continue
            handles.append(Line2D([0], [0],
                                  marker=marker, color="w",
                                  markerfacecolor=color,
                                  markeredgecolor="0.3",
                                  markersize=8, lw=0))
            labels.append(disp)

    # Status block at the end of the same legend.
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="0.7", markeredgecolor="black",
                          markersize=9, markeredgewidth=1.4, lw=0))
    labels.append("Pareto-optimal")
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="0.7", markeredgecolor="0.4",
                          markersize=7, lw=0))
    labels.append("Dominated")
    handles.append(Line2D([0], [0], color="0.55", ls="--", lw=1.0))
    labels.append("Pareto frontier")

    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=8.5,
               handletextpad=0.5, columnspacing=1.4, labelspacing=0.6)


def main():
    df = pd.read_csv(CSV)
    df = df[~df["method"].isin(["camp_base", "tau_full"])].reset_index(drop=True)
    df["display"] = df["method"].map(DISPLAY)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 4.6))

    panel(ax_a, df, "mae",
          ylabel="Pixel MAE ↓ (scene-level)",
          title="(a) Scene-level: SSIM vs Pixel MAE")
    panel(ax_b, df, "poi_mae",
          ylabel="POI-MAE ↓ (plant-region)",
          title="(b) Plant-level: SSIM vs POI-MAE")

    make_legend(fig, df)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.28, wspace=0.28)

    pdf_path = OUT_DIR / "figure2_pareto.pdf"
    png_path = OUT_DIR / "figure2_pareto.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
