"""
metric_correlation.py
---------------------
Spearman rank correlation between the POI_MAE metric and the standard
image-quality metrics {MAE, MSE, SSIM, PSNR, LPIPS}. Addresses Reviewer
1's comment: "No comparison with existing physically informed or
perceptual metrics."

Two perspectives are reported:
  1. Within-method correlation: how strongly POI_MAE tracks each other
     metric across the 30 test samples for one method. One row per method.
  2. Across-method correlation: how strongly the method-level ranking by
     POI_MAE agrees with the ranking by each other metric (one scalar per
     metric, computed over all ready methods).

If LPIPS is unavailable (lpips or torch not installed), it is silently
skipped.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/metric_correlation/
    within_method_spearman.csv       method × {mae, mse, ssim, psnr, lpips}
    across_method_spearman.csv       one row: metric, rho, p
    heatmap_within.png               within-method correlation heatmap

Usage
-----
    python tools/metric_correlation.py --dataset komatsuna --epoch 150
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir,
    per_sample_mae, per_sample_mse, per_sample_ssim, per_sample_psnr,
    per_sample_poi_mae, METHOD_DISPLAY,
)


def try_per_sample_lpips(preds, trues):
    """LPIPS per sample (mean over frames). None if lpips unavailable."""
    try:
        import torch
        import lpips  # type: ignore
    except Exception:
        return None
    net = lpips.LPIPS(net="alex", verbose=False)
    net.eval()
    N, T, C, H, W = preds.shape
    out = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            # LPIPS expects tensors in [-1, 1].
            p = torch.from_numpy(preds[i] * 2 - 1).float()
            t = torch.from_numpy(trues[i] * 2 - 1).float()
            d = net(p, t).squeeze().cpu().numpy()
            out[i] = float(np.mean(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--skip_lpips", action="store_true",
                    help="skip LPIPS even if lpips is installed")
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    # Compute per-sample metrics for each method.
    per_method = {}
    for method, saved in models.items():
        run_name = osp.basename(osp.dirname(saved))
        poi = per_sample_poi_mae(saved, run_name)
        if poi is None:
            print(f"Skipping {method}: POI data missing "
                  f"(run tools/eval_poi.py first).")
            continue
        preds, trues = load_preds_trues(saved)
        d = {
            "poi_mae": poi,
            "mae":  per_sample_mae(preds, trues),
            "mse":  per_sample_mse(preds, trues),
            "ssim": per_sample_ssim(preds, trues),
            "psnr": per_sample_psnr(preds, trues),
        }
        if not args.skip_lpips:
            lp = try_per_sample_lpips(preds, trues)
            if lp is not None:
                d["lpips"] = lp
        per_method[method] = d
        print(f"Loaded {method}: N={len(poi)}")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "metric_correlation")

    other_metrics = ["mae", "mse", "ssim", "psnr", "lpips"]

    # ---- Within-method Spearman ---------------------------------------- #
    with open(osp.join(out_dir, "within_method_spearman.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + other_metrics)
        heat = []
        row_labels = []
        for method, d in per_method.items():
            row = [method]
            hrow = []
            for m in other_metrics:
                if m not in d:
                    row.append("")
                    hrow.append(np.nan)
                    continue
                rho, _ = spearmanr(d["poi_mae"], d[m])
                row.append(f"{rho:.4f}")
                hrow.append(float(rho))
            w.writerow(row)
            heat.append(hrow)
            row_labels.append(method)
    heat = np.array(heat)

    # Heatmap of within-method correlations.
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(row_labels))))
    im = ax.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(other_metrics)))
    ax.set_xticklabels([m.upper() for m in other_metrics])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in row_labels])
    ax.set_title(f"Spearman ρ(POI_MAE, metric) within method — "
                 f"{args.dataset}")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(heat[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "heatmap_within.png"), dpi=150)
    plt.close(fig)

    # ---- Across-method Spearman --------------------------------------- #
    # Aggregate each method → one scalar per metric (mean over samples),
    # then compute Spearman ρ between the method-ranking by POI_MAE and by
    # each other metric.
    method_order = list(per_method.keys())
    poi_rank = np.array([per_method[m]["poi_mae"].mean() for m in method_order])
    with open(osp.join(out_dir, "across_method_spearman.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["other_metric", "spearman_rho", "p_value", "n_methods"])
        for m in other_metrics:
            vals = []
            for method in method_order:
                if m in per_method[method]:
                    vals.append(per_method[method][m].mean())
                else:
                    vals.append(np.nan)
            vals = np.array(vals)
            valid = np.isfinite(vals)
            if valid.sum() < 3:
                w.writerow([m, "", "", valid.sum()])
                continue
            rho, p = spearmanr(poi_rank[valid], vals[valid])
            w.writerow([m, f"{rho:.4f}", f"{p:.4g}", int(valid.sum())])

    print(f"\nOutputs → {out_dir}")


if __name__ == "__main__":
    main()
