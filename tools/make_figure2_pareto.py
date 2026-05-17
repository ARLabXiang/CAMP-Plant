"""Replacement for Figure 2: two-panel Pareto scatter (SSIM vs MAE, SSIM vs POI-MAE)
on the Komatsuna test set. Drop-in replacement that matches the caption and
surrounding text in CAMP_CEA_paper_AZ_v31_revised.docx.

Outputs vector PDF + 300 dpi PNG to figures/figure2_replacement/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "work_dirs" / "komatsuna_ep150_results.csv"
OUT_DIR = ROOT / "figures" / "figure2_replacement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "convlstm":    "ConvLSTM",
    "predrnn":     "PredRNN",
    "phydnet":     "PhyDNet",
    "simvp":       "SimVP",
    "mim":         "MIM",
    "tau":         "TAU",
    "camp":        "CAMP",
    "camp_no_cls": "CAMP_no_cls",
    "camp_full":   "CAMP_full",
    "mim_full":    "MIM_full",
    "tau_no_cls":  "TAU_no_cls",
    "simvp_full":  "SimVP_full",
}

# Color by family: CAMP = red family, recurrent baselines = blues, non-recurrent = greens.
FAMILY_COLOR = {
    "CAMP":        "#d62728",
    "CAMP_no_cls": "#ff7f0e",
    "CAMP_full":   "#8c564b",
    "ConvLSTM":    "#1f77b4",
    "PredRNN":     "#17becf",
    "PhyDNet":     "#9467bd",
    "MIM":         "#2ca02c",
    "MIM_full":    "#98df8a",
    "SimVP":       "#7f7f7f",
    "SimVP_full":  "#c7c7c7",
    "TAU":         "#bcbd22",
    "TAU_no_cls":  "#dbdb8d",
}


def pareto_front(ssim, err):
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
    names = df["display"].tolist()

    front = pareto_front(x, y)
    order = np.argsort(x[front])
    ax.plot(x[front][order], y[front][order],
            color="0.55", lw=1.0, ls="--", zorder=1)

    for i, name in enumerate(names):
        on_front = front[i]
        ax.scatter(x[i], y[i],
                   s=80 if on_front else 46,
                   facecolor=FAMILY_COLOR[name],
                   edgecolor="black" if on_front else "0.35",
                   linewidth=1.2 if on_front else 0.6,
                   zorder=3)

    texts = []
    for i, name in enumerate(names):
        texts.append(ax.text(x[i], y[i], name,
                             fontsize=7.8,
                             fontweight="bold" if front[i] else "normal",
                             color="black" if front[i] else "0.2"))
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        expand=(1.2, 1.5),
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
    )

    ax.set_xlabel("SSIM ↑", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, loc="left", pad=6)
    ax.tick_params(labelsize=8.5, length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    pad_x = (x.max() - x.min()) * 0.08
    pad_y = (y.max() - y.min()) * 0.10
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)


def main():
    df = pd.read_csv(CSV)
    # camp_base ≡ predrnn numerically (same MAE/MSE/SSIM/PSNR/POI); tau_full ≡ tau_no_cls.
    # Drop the duplicates so markers don't stack.
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

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    panel(axes[0], df, "mae",
          ylabel="Pixel MAE ↓ (scene-level)",
          title="(a) Scene-level: SSIM vs Pixel MAE")
    panel(axes[1], df, "poi_mae",
          ylabel="POI-MAE ↓ (plant-region)",
          title="(b) Plant-level: SSIM vs POI-MAE")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
               markeredgecolor="black", markersize=9, label="Pareto-optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
               markeredgecolor="0.4", markersize=7, label="Dominated"),
        Line2D([0], [0], color="0.55", ls="--", lw=1.0, label="Pareto frontier"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    pdf_path = OUT_DIR / "figure2_pareto.pdf"
    png_path = OUT_DIR / "figure2_pareto.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
