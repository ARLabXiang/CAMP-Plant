"""
bootstrap_ci.py
---------------
Bootstrap 95% confidence intervals for every headline metric on every
trained method. Addresses Reviewer 1's comment: "Absence of confidence
intervals or variance analysis."

Resamples the test set with replacement B times (default 10,000) and
reports the 2.5th / 97.5th percentile of the mean metric.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/bootstrap_ci/
    bootstrap_ci.csv            method × metric → mean, ci_low, ci_high
    bootstrap_ci_{metric}.png   forest plot per metric

Usage
-----
    python tools/bootstrap_ci.py --dataset komatsuna --epoch 150 -B 10000
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
    per_sample_mae, per_sample_mse, per_sample_ssim, per_sample_psnr,
    per_sample_poi_mae, METHOD_DISPLAY,
)

METRICS = ["mae", "mse", "ssim", "psnr", "poi_mae"]


def bootstrap_mean_ci(values: np.ndarray, B: int, rng: np.random.Generator,
                     alpha: float = 0.05):
    """Return (mean, ci_low, ci_high) for the bootstrap distribution of the mean."""
    N = len(values)
    idx = rng.integers(0, N, size=(B, N))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(low), float(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("-B", "--n_bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    rng = np.random.default_rng(args.seed)

    # Compute per-sample values for every (method, metric) once.
    per_sample = {m: {} for m in models}
    for method, saved in models.items():
        preds, trues = load_preds_trues(saved)
        per_sample[method]["mae"]  = per_sample_mae(preds, trues)
        per_sample[method]["mse"]  = per_sample_mse(preds, trues)
        per_sample[method]["ssim"] = per_sample_ssim(preds, trues)
        per_sample[method]["psnr"] = per_sample_psnr(preds, trues)
        run_name = osp.basename(osp.dirname(saved))
        poi = per_sample_poi_mae(saved, run_name)
        if poi is not None:
            per_sample[method]["poi_mae"] = poi
        print(f"Loaded {method}: N={len(per_sample[method]['mae'])}")

    # Run bootstrap for all (method, metric) pairs.
    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "bootstrap_ci")
    rows = []
    for method in per_sample:
        for metric in METRICS:
            if metric not in per_sample[method]:
                continue
            m_val, lo, hi = bootstrap_mean_ci(per_sample[method][metric],
                                              args.n_bootstrap, rng, args.alpha)
            rows.append({"method": method, "metric": metric,
                         "mean": m_val, "ci_low": lo, "ci_high": hi,
                         "ci_width": hi - lo})

    csv_path = osp.join(out_dir, "bootstrap_ci.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "mean",
                                          "ci_low", "ci_high", "ci_width"])
        w.writeheader()
        for r in rows:
            r = {**r, "mean": f"{r['mean']:.6g}", "ci_low": f"{r['ci_low']:.6g}",
                 "ci_high": f"{r['ci_high']:.6g}", "ci_width": f"{r['ci_width']:.6g}"}
            w.writerow(r)
    print(f"\nCSV → {csv_path}")

    # Forest plot per metric.
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    for metric in METRICS:
        metric_rows = [r for r in rows if r["metric"] == metric]
        if not metric_rows:
            continue
        # Sort by mean (best first) — lower-is-better for mae/mse/poi_mae.
        lower_better = metric in {"mae", "mse", "poi_mae"}
        metric_rows.sort(key=lambda r: r["mean"], reverse=not lower_better)

        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(metric_rows))))
        y = np.arange(len(metric_rows))
        means = np.array([r["mean"] for r in metric_rows])
        lows  = np.array([r["ci_low"] for r in metric_rows])
        highs = np.array([r["ci_high"] for r in metric_rows])
        err   = np.vstack([means - lows, highs - means])

        ax.errorbar(means, y, xerr=err, fmt="o", color="steelblue",
                    ecolor="gray", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_DISPLAY.get(r["method"], r["method"])
                           for r in metric_rows])
        ax.invert_yaxis()
        # Display-friendly names
        metric_display = {
            "mae": "MAE", "mse": "MSE", "ssim": "SSIM",
            "psnr": "PSNR", "poi_mae": "POI-MAE",
        }.get(metric, metric.upper())
        dataset_display = args.dataset.capitalize()
        ax.set_xlabel(metric_display)
        ax.set_title(f"{metric_display} on {dataset_display} "
                     f"(95% confidence intervals)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        png = osp.join(out_dir, f"bootstrap_ci_{metric}.png")
        pdf = osp.join(out_dir, f"bootstrap_ci_{metric}.pdf")
        fig.savefig(png, dpi=100)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Plot → {png}")
        print(f"Plot (PDF) → {pdf}")


if __name__ == "__main__":
    main()
