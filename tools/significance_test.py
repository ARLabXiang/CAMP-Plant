"""
significance_test.py
--------------------
Pairwise Wilcoxon signed-rank test + Cohen's d effect size on per-sample
metrics. Addresses Reviewer 1's comment: "no statistical significance
testing is provided."

For every pair of trained methods in work_dirs, we compute a per-sample
metric (default: POI_MAE) and run:
  - Wilcoxon signed-rank test (two-sided, paired by sample index).
  - Cohen's d effect size (standardised mean difference).

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/significance/
    pvalues_{metric}.csv           symmetric matrix of p-values
    cohens_d_{metric}.csv          anti-symmetric matrix, +ve = row < col
    significance_summary_{metric}.md  human-readable winners per row

Usage
-----
    python tools/significance_test.py --dataset komatsuna --epoch 150
    python tools/significance_test.py --dataset komatsuna --epoch 150 --metric ssim
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir,
    filter_to_majority_N,
    per_sample_mae, per_sample_mse, per_sample_ssim, per_sample_psnr,
    per_sample_poi_mae, METHOD_DISPLAY,
)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d = mean(x - y) / pooled std. Sign: positive → x > y."""
    diff = x - y
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def compute_metric(metric: str, saved_dir: str, method: str):
    """Return a per-sample vector for the requested metric."""
    if metric == "poi_mae":
        v = per_sample_poi_mae(saved_dir, f"{osp.basename(osp.dirname(saved_dir))}")
        if v is None:
            raise FileNotFoundError(
                f"POI data missing for {method}. Run tools/eval_poi.py first.")
        return v
    preds, trues = load_preds_trues(saved_dir)
    return {
        "mae":  per_sample_mae,
        "mse":  per_sample_mse,
        "ssim": per_sample_ssim,
        "psnr": per_sample_psnr,
    }[metric](preds, trues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--metric", default="poi_mae",
                    choices=["mae", "mse", "ssim", "psnr", "poi_mae"])
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance threshold for the summary")
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    models = filter_to_majority_N(models)
    if len(models) < 2:
        sys.exit(f"Need >=2 ready models in {args.work_dir}; found {list(models)}")

    print(f"Computing {args.metric} per sample for {len(models)} methods…")
    per_sample = {}
    for method, saved in models.items():
        per_sample[method] = compute_metric(args.metric, saved, method)
        print(f"  {method}: N={len(per_sample[method])}, "
              f"mean={per_sample[method].mean():.4f}")

    # After filter_to_majority_N, all kept methods share the same N.

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "significance")

    # Lower-is-better metrics for the "winner" summary.
    lower_better = args.metric in {"mae", "mse", "poi_mae"}

    method_list = list(per_sample.keys())
    n = len(method_list)
    p_matrix = np.full((n, n), np.nan)
    d_matrix = np.full((n, n), np.nan)

    for i, m_i in enumerate(method_list):
        for j, m_j in enumerate(method_list):
            if i == j:
                p_matrix[i, j] = 1.0
                d_matrix[i, j] = 0.0
                continue
            x = per_sample[m_i]
            y = per_sample[m_j]
            try:
                _, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p = 1.0  # happens when all differences are zero
            p_matrix[i, j] = float(p)
            d_matrix[i, j] = cohens_d(x, y)

    # ---- Save matrices --------------------------------------------------- #
    def write_matrix(path, mat):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + method_list)
            for i, row in enumerate(mat):
                w.writerow([method_list[i]] + [f"{v:.6g}" for v in row])

    p_path = osp.join(out_dir, f"pvalues_{args.metric}.csv")
    d_path = osp.join(out_dir, f"cohens_d_{args.metric}.csv")
    write_matrix(p_path, p_matrix)
    write_matrix(d_path, d_matrix)

    # ---- Human-readable summary ----------------------------------------- #
    summary_path = osp.join(out_dir, f"significance_summary_{args.metric}.md")
    with open(summary_path, "w") as f:
        f.write(f"# Significance tests ({args.metric}) — "
                f"{args.dataset}, {args.epoch} epochs\n\n")
        f.write(f"Paired Wilcoxon signed-rank, two-sided, α={args.alpha}. "
                f"Cohen's d: sign convention = positive means row > column.\n\n")
        f.write("For each method, how many peers it significantly "
                f"{'beats (lower' if lower_better else 'beats (higher'}) "
                "than):\n\n")
        f.write("| Method | Mean | Wins | Losses | Ties | Peers |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for i, m_i in enumerate(method_list):
            mean_v = per_sample[m_i].mean()
            wins = losses = ties = 0
            for j, m_j in enumerate(method_list):
                if i == j:
                    continue
                p = p_matrix[i, j]
                d = d_matrix[i, j]
                if p >= args.alpha:
                    ties += 1
                    continue
                better = (d < 0) if lower_better else (d > 0)
                if better:
                    wins += 1
                else:
                    losses += 1
            f.write(f"| {METHOD_DISPLAY.get(m_i, m_i)} | "
                    f"{mean_v:.4f} | {wins} | {losses} | {ties} | "
                    f"{n - 1} |\n")

    print()
    print(f"p-values     → {p_path}")
    print(f"Cohen's d    → {d_path}")
    print(f"Summary (md) → {summary_path}")


if __name__ == "__main__":
    main()
