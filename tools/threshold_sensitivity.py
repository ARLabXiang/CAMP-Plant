"""
threshold_sensitivity.py
------------------------
Sweep the ExGI threshold τ used by the POI metric and measure whether the
model ranking by POI_MAE is stable. Addresses Reviewer 1's comment:
"Lack of theoretical grounding and reproducibility details [for POI]."

For each τ ∈ {0.05, 0.1, 0.15, 0.2, 0.25} we recompute the POI mask from
the saved preds.npy / trues.npy and report:
  - POI_MAE per method at each τ
  - Spearman ρ and Kendall τ of the method ranking versus the ranking at
    the default threshold (τ = 0.1).

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/threshold_sensitivity/
    poi_mae_by_threshold.csv       rows = methods, cols = τ values
    rank_stability.csv             rank correlation versus τ = 0.1
    poi_mae_by_threshold.png       line plot, one line per method

Usage
-----
    python tools/threshold_sensitivity.py --dataset komatsuna --epoch 150
    python tools/threshold_sensitivity.py --thresholds 0.05 0.1 0.15 0.2 0.25
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir,
    compute_exgi, METHOD_DISPLAY,
)


def poi_mae_at_threshold(preds: np.ndarray, trues: np.ndarray, tau: float) -> float:
    """
    Mean absolute error between predicted and ground-truth POI counts,
    averaged over frames then over samples. Matches eval_poi.evaluate_poi.
    """
    # ExGI maps: [N, T, H, W]
    exgi_p = compute_exgi(preds)
    exgi_t = compute_exgi(trues)
    # POI counts per (sample, frame): [N, T]
    count_p = (exgi_p > tau).sum(axis=(-1, -2))
    count_t = (exgi_t > tau).sum(axis=(-1, -2))
    # Per-sample mean absolute count error, then mean over samples.
    return float(np.abs(count_p - count_t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25])
    ap.add_argument("--reference", type=float, default=0.10,
                    help="threshold used as the rank-stability reference")
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")
    if args.reference not in args.thresholds:
        sys.exit(f"--reference {args.reference} must be in --thresholds "
                 f"{args.thresholds}")

    # Pre-load preds/trues once per method.
    print(f"Loading {len(models)} methods…")
    cached = {}
    for method, saved in models.items():
        preds, trues = load_preds_trues(saved)
        cached[method] = (preds, trues)

    # Matrix: rows = methods, cols = τ values.
    table = {}
    for method, (preds, trues) in cached.items():
        table[method] = []
        for tau in args.thresholds:
            table[method].append(poi_mae_at_threshold(preds, trues, tau))
        print(f"  {method}: {dict(zip(args.thresholds, [round(x, 2) for x in table[method]]))}")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "threshold_sensitivity")

    # Write CSV.
    csv_path = osp.join(out_dir, "poi_mae_by_threshold.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"tau_{t}" for t in args.thresholds])
        for method, vals in table.items():
            w.writerow([method] + [f"{v:.4f}" for v in vals])

    # Rank stability.
    methods = list(table.keys())
    ref_idx = args.thresholds.index(args.reference)
    ref_values = np.array([table[m][ref_idx] for m in methods])
    stability_path = osp.join(out_dir, "rank_stability.csv")
    with open(stability_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "spearman_vs_ref", "kendall_vs_ref"])
        for k, tau in enumerate(args.thresholds):
            vals = np.array([table[m][k] for m in methods])
            rho, _ = spearmanr(ref_values, vals)
            kt, _  = kendalltau(ref_values, vals)
            w.writerow([tau, f"{rho:.4f}", f"{kt:.4f}"])

    # Plot.
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times New Roman", "DejaVu Serif"],
        "font.size":        15,
        "axes.titlesize":   17,
        "axes.labelsize":   16,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  13,
    })

    # Skip predrnn in the plot — camp_base is the same model shown instead.
    SKIP = {"predrnn"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, vals in table.items():
        if method in SKIP:
            continue
        label = METHOD_DISPLAY.get(method, method)
        # Use "CAMP_base" label consistently (suppress "PredRNN")
        if label == "PredRNN":
            label = "CAMP_base"
        ax.plot(args.thresholds, vals, marker="o", label=label)
    ax.axvline(args.reference, color="gray", linestyle="--", alpha=0.4,
               label=f"τ = {args.reference} (ref)")
    ax.set_xlabel("ExGI threshold τ")
    ax.set_ylabel("POI_MAE (↓)")
    ax.set_title(f"POI_MAE vs τ — {args.dataset}, {args.epoch} ep")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    png = osp.join(out_dir, "poi_mae_by_threshold.png")
    pdf = osp.join(out_dir, "poi_mae_by_threshold.pdf")
    fig.savefig(png, dpi=100)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"\nCSV → {csv_path}")
    print(f"Rank stability → {stability_path}")
    print(f"Plot → {png}")
    print(f"Plot (PDF) → {pdf}")


if __name__ == "__main__":
    main()
