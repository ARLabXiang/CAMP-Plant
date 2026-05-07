"""Render a 2-panel heatmap (p-values, Cohen's d) for paired-sample significance tests.

Source CSVs come from `tools/significance_test.py` and live in
`work_dirs/<dataset>_ep150_analysis/significance/`.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def _annotate(ax, mat: np.ndarray, fmt):
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7,
                    color="black")


def render(metric: str, sig_dir: str, out_path: str, dataset: str):
    p_df = _load(os.path.join(sig_dir, f"pvalues_{metric}.csv"))
    d_df = _load(os.path.join(sig_dir, f"cohens_d_{metric}.csv"))

    # Reindex to common method order
    methods = list(p_df.index)
    p_df = p_df.reindex(index=methods, columns=methods)
    d_df = d_df.reindex(index=methods, columns=methods)

    p_mat = p_df.to_numpy(dtype=float)
    d_mat = d_df.to_numpy(dtype=float)
    np.fill_diagonal(p_mat, np.nan)
    np.fill_diagonal(d_mat, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(methods) * 0.85), 6.5))

    # --- p-values: log scale, blue → white at α = 0.05 ---
    p_plot = np.where(p_mat <= 0, 1e-12, p_mat)  # avoid log(0)
    norm_p = LogNorm(vmin=1e-6, vmax=1.0)
    im0 = axes[0].imshow(p_plot, cmap="Blues_r", norm=norm_p)
    axes[0].set_title(f"Paired t-test p-values\n(metric = {metric.upper()})")
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(methods, fontsize=8)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.85)
    cb0.set_label("p-value (log scale)")
    _annotate(axes[0], p_mat,
              fmt=lambda v: ("<1e-6" if v < 1e-6 else f"{v:.2g}"))
    # mark α = 0.05 contour by hatching p > 0.05
    sig = p_mat <= 0.05
    for i in range(p_mat.shape[0]):
        for j in range(p_mat.shape[1]):
            if not np.isnan(p_mat[i, j]) and not sig[i, j]:
                axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False, hatch="///",
                                                edgecolor="gray",
                                                linewidth=0.0, alpha=0.5))

    # --- Cohen's d: diverging, centred at 0 ---
    vmax = max(0.5, np.nanmax(np.abs(d_mat)))
    norm_d = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im1 = axes[1].imshow(d_mat, cmap="RdBu_r", norm=norm_d)
    axes[1].set_title(f"Cohen's d (row vs column)\n(metric = {metric.upper()})")
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(methods, fontsize=8)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.85)
    cb1.set_label("Cohen's d")
    _annotate(axes[1], d_mat, fmt=lambda v: f"{v:+.2f}")

    fig.suptitle(
        f"{dataset.capitalize()} — pairwise significance ({metric.upper()})",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[significance_heatmap] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", required=True,
                    help="e.g. work_dirs/komatsuna_ep150_analysis")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["poi_mae", "mae", "ssim"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sig_dir = os.path.join(args.analysis_root, "significance")

    for m in args.metrics:
        out = os.path.join(args.out_dir,
                           f"fig_significance_{m}_{args.dataset}.png")
        render(m, sig_dir, out, args.dataset)


if __name__ == "__main__":
    main()
