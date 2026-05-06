"""
error_distribution.py
---------------------
Violin + box plots of per-sample error for every method, per metric.
Addresses Reviewer 1's comments on experimental rigor and over-reliance
on qualitative analysis: shows the full distribution, not just the mean.

For each method we compute per-sample MAE, MSE, SSIM, PSNR, POI_MAE and
render them as violin plots so the reader can see skew, outliers, and
overlap between methods.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/error_distribution/
    per_sample_metrics.npz         one key per method, each value is a
                                   structured ndarray with fields
                                   {mae, mse, ssim, psnr, poi_mae}
    violin_{metric}.png            violin plot across methods

Usage
-----
    python tools/error_distribution.py --dataset komatsuna --epoch 150
"""

import argparse
import os.path as osp
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir,
    per_sample_mae, per_sample_mse, per_sample_ssim, per_sample_psnr,
    per_sample_poi_mae, METHOD_DISPLAY,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "error_distribution")

    per_method = {}
    for method, saved in models.items():
        preds, trues = load_preds_trues(saved)
        run_name = osp.basename(osp.dirname(saved))
        d = {
            "mae":  per_sample_mae(preds, trues),
            "mse":  per_sample_mse(preds, trues),
            "ssim": per_sample_ssim(preds, trues),
            "psnr": per_sample_psnr(preds, trues),
        }
        poi = per_sample_poi_mae(saved, run_name)
        if poi is not None:
            d["poi_mae"] = poi
        per_method[method] = d
        print(f"Loaded {method}: N={len(d['mae'])}")

    # Save raw arrays for downstream reuse.
    flat = {}
    for method, d in per_method.items():
        for metric, arr in d.items():
            flat[f"{method}__{metric}"] = arr
    np.savez(osp.join(out_dir, "per_sample_metrics.npz"), **flat)

    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    methods = list(per_method.keys())
    x = np.arange(len(methods))
    metric_names = ["mae", "mse", "ssim", "psnr", "poi_mae"]

    for metric in metric_names:
        data = []
        for method in methods:
            if metric in per_method[method]:
                data.append(per_method[method][metric])
            else:
                data.append(np.array([np.nan]))

        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(methods)), 4.5))
        parts = ax.violinplot(data, positions=x, showmeans=False,
                              showmedians=False, showextrema=False,
                              widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.45)
            pc.set_edgecolor("steelblue")

        # Overlay a classical boxplot with median + quartiles.
        box = ax.boxplot(data, positions=x, widths=0.25,
                         patch_artist=True, showfliers=True,
                         boxprops=dict(facecolor="white",
                                       edgecolor="#2e4a6b"),
                         medianprops=dict(color="#d95f02"),
                         whiskerprops=dict(color="#2e4a6b"),
                         capprops=dict(color="#2e4a6b"),
                         flierprops=dict(marker="o", markersize=3,
                                         markerfacecolor="gray",
                                         markeredgecolor="gray", alpha=0.6))

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods],
                           rotation=30, ha="right")
        ax.set_ylabel(metric.upper())
        direction = "↓" if metric in {"mae", "mse", "poi_mae"} else "↑"
        ax.set_title(f"Per-sample {metric.upper()} ({direction}) — "
                     f"{args.dataset}, {args.epoch} ep")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, f"violin_{metric}.png"), dpi=150)
        plt.close(fig)

    print(f"\nOutputs → {out_dir}")


if __name__ == "__main__":
    main()
