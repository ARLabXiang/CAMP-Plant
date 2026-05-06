"""
per_frame_error.py
------------------
Break down each method's errors by prediction horizon (frame 1 → frame 5).
Addresses Reviewer 1's comment: "Insufficient experimental rigor."

Autoregressive and Markov-style models typically degrade at longer
horizons; feed-forward models can flatline. Splitting MAE / MSE / SSIM /
PSNR / POI_MAE by frame index makes that dynamic explicit.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/per_frame/
    per_frame_metrics.csv       rows = (method, frame), cols = metrics
    per_frame_{metric}.png      line plot, one line per method

Usage
-----
    python tools/per_frame_error.py --dataset komatsuna --epoch 150
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir,
    compute_exgi, METHOD_DISPLAY,
)


def per_frame_metrics(preds: np.ndarray, trues: np.ndarray,
                      tau: float = 0.1):
    """
    Compute per-frame MAE, MSE, SSIM, PSNR, and POI count error.

    Returns a dict metric → np.ndarray of shape (T,).
    """
    from skimage.metrics import structural_similarity as ssim
    N, T, C, H, W = preds.shape
    mae = np.abs(preds - trues).reshape(N, T, -1).sum(axis=2).mean(axis=0)
    mse = ((preds - trues) ** 2).reshape(N, T, -1).sum(axis=2).mean(axis=0)

    # SSIM per (sample, frame, channel); average over channels and samples.
    ssim_vals = np.empty((N, T), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            ch_vals = [ssim(trues[i, t, c], preds[i, t, c], data_range=1.0)
                       for c in range(C)]
            ssim_vals[i, t] = float(np.mean(ch_vals))
    ssim_out = ssim_vals.mean(axis=0)

    # PSNR: per-frame MSE in pixel space.
    mse_pix = ((preds - trues) ** 2).reshape(N, T, -1).mean(axis=2)
    mse_pix = np.maximum(mse_pix, 1e-12)
    psnr = (10.0 * np.log10(1.0 / mse_pix)).mean(axis=0)

    # POI count error per frame.
    exgi_p = compute_exgi(preds)
    exgi_t = compute_exgi(trues)
    count_p = (exgi_p > tau).sum(axis=(-1, -2))
    count_t = (exgi_t > tau).sum(axis=(-1, -2))
    poi_mae = np.abs(count_p - count_t).mean(axis=0)

    return {"mae": mae, "mse": mse, "ssim": ssim_out, "psnr": psnr,
            "poi_mae": poi_mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="ExGI threshold for POI count")
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "per_frame")

    all_rows = []
    per_method = {}
    for method, saved in models.items():
        preds, trues = load_preds_trues(saved)
        metrics = per_frame_metrics(preds, trues, args.threshold)
        per_method[method] = metrics
        T = len(metrics["mae"])
        for t in range(T):
            all_rows.append({
                "method": method, "frame": t + 1,
                "mae":     f"{metrics['mae'][t]:.4f}",
                "mse":     f"{metrics['mse'][t]:.4f}",
                "ssim":    f"{metrics['ssim'][t]:.4f}",
                "psnr":    f"{metrics['psnr'][t]:.4f}",
                "poi_mae": f"{metrics['poi_mae'][t]:.4f}",
            })
        print(f"{method:15s}  MAE t1={metrics['mae'][0]:.1f} → "
              f"t{T}={metrics['mae'][-1]:.1f}")

    csv_path = osp.join(out_dir, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "frame", "mae", "mse",
                                          "ssim", "psnr", "poi_mae"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Plots.
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    for metric_name in ["mae", "mse", "ssim", "psnr", "poi_mae"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, metrics in per_method.items():
            T = len(metrics[metric_name])
            ax.plot(np.arange(1, T + 1), metrics[metric_name], marker="o",
                    label=METHOD_DISPLAY.get(method, method))
        ax.set_xlabel("Prediction frame index (1 = next frame)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} vs prediction horizon — "
                     f"{args.dataset}, {args.epoch} ep")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, f"per_frame_{metric_name}.png"), dpi=150)
        plt.close(fig)

    print(f"\nCSV → {csv_path}")
    print(f"Plots → {out_dir}/per_frame_*.png")


if __name__ == "__main__":
    main()
