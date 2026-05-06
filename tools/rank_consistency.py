"""
rank_consistency.py
-------------------
For each (method, metric) pair, compute the method's rank (1 = best) on
each metric, then render a heatmap so you can see at a glance which
methods are consistently good.

Useful for spotting metric-specific wins (e.g., CAMP_full best on POI_MAE
but only middle on SSIM) vs. universal winners.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/rank_consistency/
    rank_table.csv                method × metric ranks
    rank_heatmap.png              visual

Usage
-----
    python tools/rank_consistency.py --dataset komatsuna --epoch 150
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
from _analysis_utils import ensure_analysis_dir, METHOD_DISPLAY, METHODS


LOWER_BETTER = {"mae", "mse", "poi_mae", "lpips"}


def read_bootstrap_ci(path):
    """Return {(method, metric): mean} from bootstrap_ci.csv."""
    out = {}
    if not osp.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["method"], row["metric"])] = float(row["mean"])
    return out


def read_lpips_csv(path):
    """Return {method: lpips_mean} from lpips_per_method.csv."""
    out = {}
    if not osp.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["method"]] = float(row["lpips_mean"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--metrics", nargs="+",
                    default=["mae", "mse", "ssim", "psnr", "poi_mae", "lpips"])
    args = ap.parse_args()

    analysis_root = osp.join(args.work_dir,
                             f"{args.dataset}_ep{args.epoch}_analysis")
    ci = read_bootstrap_ci(osp.join(analysis_root, "bootstrap_ci",
                                    "bootstrap_ci.csv"))
    lpips = read_lpips_csv(osp.join(analysis_root, "lpips",
                                    "lpips_per_method.csv"))

    # Fold LPIPS into the same {(method, metric): mean} structure.
    for m, v in lpips.items():
        ci[(m, "lpips")] = v

    if not ci:
        sys.exit(f"No bootstrap_ci.csv under {analysis_root} — "
                 f"run tools/bootstrap_ci.py first.")

    # Discover methods that appear in the loaded means.
    methods_seen = sorted({m for (m, _) in ci.keys()},
                          key=lambda x: METHODS.index(x)
                          if x in METHODS else 99)

    # Build rank table: rows = methods, cols = metrics.
    rank_table = np.full((len(methods_seen), len(args.metrics)), np.nan)
    value_table = np.full_like(rank_table, np.nan)
    for col, metric in enumerate(args.metrics):
        vals = [(ci.get((m, metric), np.nan), m) for m in methods_seen]
        vals_valid = [v for v in vals if np.isfinite(v[0])]
        lower_better = metric in LOWER_BETTER
        vals_valid.sort(key=lambda x: x[0], reverse=not lower_better)
        rank_map = {m: i + 1 for i, (_, m) in enumerate(vals_valid)}
        for row, m in enumerate(methods_seen):
            value_table[row, col] = ci.get((m, metric), np.nan)
            if m in rank_map:
                rank_table[row, col] = rank_map[m]

    # CSV output.
    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "rank_consistency")
    csv_path = osp.join(out_dir, "rank_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + args.metrics + ["mean_rank"])
        for row, m in enumerate(methods_seen):
            ranks = rank_table[row].tolist()
            mean_rank = np.nanmean(ranks)
            w.writerow([m] + [f"{r:.0f}" if np.isfinite(r) else ""
                              for r in ranks] + [f"{mean_rank:.2f}"])

    # Heatmap.
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(args.metrics)),
                                    max(3, 0.4 * len(methods_seen))))
    # Use a reversed viridis so rank 1 = yellow (best) and worst rank = dark.
    im = ax.imshow(rank_table, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(args.metrics)))
    ax.set_xticklabels([m.upper() for m in args.metrics])
    ax.set_yticks(range(len(methods_seen)))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in methods_seen])
    ax.set_title(f"Method ranks across metrics — "
                 f"{args.dataset}, {args.epoch} ep  "
                 f"(1 = best)")
    for i in range(rank_table.shape[0]):
        for j in range(rank_table.shape[1]):
            v = rank_table[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="white" if v > len(methods_seen) / 2
                        else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="rank (1 = best)")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "rank_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"CSV     : {csv_path}")
    print(f"Heatmap : {osp.join(out_dir, 'rank_heatmap.png')}")


if __name__ == "__main__":
    main()
