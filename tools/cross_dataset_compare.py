"""
cross_dataset_compare.py
------------------------
Side-by-side comparison of methods that appear on BOTH datasets. Reads
from the bootstrap_ci.csv summaries already written by bootstrap_ci.py.

Answers questions like:
  - Does SimVP's advantage over recurrent baselines transfer between
    datasets, or is it Komatsuna-specific?
  - Is POI_MAE stable as a metric across datasets (do rankings agree)?
  - Which methods are sensitive to dataset dynamics (big rank changes),
    which are robust?

Outputs
-------
  work_dirs/cross_dataset_ep{epoch}.csv        long-form (dataset × method × metric)
  work_dirs/cross_dataset_ep{epoch}.md         side-by-side paper table
  work_dirs/cross_dataset_ranks_ep{epoch}.png  rank heatmap, methods × metrics ×
                                               datasets

Usage
-----
    python tools/cross_dataset_compare.py --epoch 150
"""

import argparse
import csv
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import METHOD_DISPLAY, METHODS


LOWER_BETTER = {"mae", "mse", "poi_mae", "lpips"}


def read_bootstrap_ci(path):
    out = {}
    if not osp.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["method"], row["metric"])] = (
                float(row["mean"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--datasets", nargs="+",
                    default=["komatsuna", "arabidopsis"])
    ap.add_argument("--metrics", nargs="+",
                    default=["mae", "mse", "ssim", "psnr", "poi_mae"])
    args = ap.parse_args()

    # dataset → {(method, metric): (mean, lo, hi)}
    per_ds = {}
    for ds in args.datasets:
        per_ds[ds] = read_bootstrap_ci(
            osp.join(args.work_dir, f"{ds}_ep{args.epoch}_analysis",
                     "bootstrap_ci", "bootstrap_ci.csv"))

    # Methods that appear on every dataset (intersection).
    methods_per_ds = [{m for (m, _) in per_ds[ds].keys()} for ds in args.datasets]
    common = set.intersection(*methods_per_ds) if methods_per_ds else set()
    common_sorted = [m for m in METHODS if m in common]
    for m in common:
        if m not in common_sorted:
            common_sorted.append(m)

    if not common_sorted:
        sys.exit("No method appears on all datasets — nothing to compare.")
    print(f"Methods common to all {len(args.datasets)} datasets: "
          f"{common_sorted}")

    # ---- Long-form CSV (one row per (dataset, method, metric)) --------- #
    csv_path = osp.join(args.work_dir, f"cross_dataset_ep{args.epoch}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "metric", "mean", "ci_low", "ci_high"])
        for ds in args.datasets:
            for m in common_sorted:
                for metric in args.metrics:
                    v = per_ds[ds].get((m, metric))
                    if v is None:
                        continue
                    w.writerow([ds, m, metric,
                                f"{v[0]:.4g}", f"{v[1]:.4g}", f"{v[2]:.4g}"])

    # ---- Side-by-side markdown table ---------------------------------- #
    md_path = osp.join(args.work_dir, f"cross_dataset_ep{args.epoch}.md")
    with open(md_path, "w") as f:
        f.write(f"# Cross-dataset comparison — common methods only, "
                f"{args.epoch} epochs\n\n")
        f.write(f"Methods present on all {len(args.datasets)} datasets: "
                f"**{', '.join(common_sorted)}**. Arabidopsis scoreboard will "
                f"expand once CAMP variants and TAU finish training.\n\n")

        # Rank agreement summary: Spearman ρ and Kendall τ of method rankings
        # per metric between the two datasets.
        from scipy.stats import spearmanr, kendalltau
        f.write("## Rank agreement across datasets\n\n")
        f.write("For each metric, how much do the two datasets agree on method "
                "ranking? ρ close to 1 → same ranking; ρ close to 0 → "
                "uncorrelated.\n\n")
        f.write("| Metric | Spearman ρ | Kendall τ |\n|---|---:|---:|\n")
        ds_a, ds_b = args.datasets[0], args.datasets[1] if len(args.datasets) > 1 else args.datasets[0]
        for metric in args.metrics:
            va = [per_ds[ds_a].get((m, metric), (np.nan,))[0] for m in common_sorted]
            vb = [per_ds[ds_b].get((m, metric), (np.nan,))[0] for m in common_sorted]
            va = np.array(va); vb = np.array(vb)
            valid = np.isfinite(va) & np.isfinite(vb)
            if valid.sum() < 3:
                f.write(f"| {metric.upper()} | — | — |\n")
                continue
            rho, _ = spearmanr(va[valid], vb[valid])
            kt, _  = kendalltau(va[valid], vb[valid])
            f.write(f"| {metric.upper()} | {rho:+.3f} | {kt:+.3f} |\n")
        f.write("\n")

        # Absolute value table, method × metric × dataset.
        f.write("## Side-by-side values\n\n")
        f.write("For each method, the value on each dataset is shown with the "
                "relative gap `(Arabidopsis − Komatsuna) / |Komatsuna|`. "
                "Use with care: datasets have different sample counts and "
                "different dynamics, so absolute value comparison is "
                "informative only up to dataset-level scale.\n\n")
        for metric in args.metrics:
            f.write(f"### {metric.upper()}\n\n")
            cols = "| Method |" + "|".join(f" {ds} " for ds in args.datasets) + "| Gap % |\n"
            f.write(cols)
            f.write("|---|" + "|".join(["---:"] * (len(args.datasets) + 1)) + "|\n")
            for m in common_sorted:
                row = [METHOD_DISPLAY.get(m, m)]
                vals = []
                for ds in args.datasets:
                    v = per_ds[ds].get((m, metric))
                    if v is None:
                        row.append("—")
                        vals.append(np.nan)
                        continue
                    row.append(f"{v[0]:.3f}" if metric in {"ssim"}
                               else f"{v[0]:.2f}")
                    vals.append(v[0])
                # Simple gap between first two datasets.
                if len(args.datasets) >= 2 and np.isfinite(vals[0]) and np.isfinite(vals[1]) and vals[0] != 0:
                    gap = (vals[1] - vals[0]) / abs(vals[0]) * 100
                    row.append(f"{gap:+.1f} %")
                else:
                    row.append("—")
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

    # ---- Combined rank heatmap ---------------------------------------- #
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})
    n_ds = len(args.datasets)
    n_met = len(args.metrics)
    n_meth = len(common_sorted)

    # Rank per (method, metric, dataset). 1 = best within that dataset.
    ranks = np.full((n_meth, n_met * n_ds), np.nan)
    col_labels = []
    for d_idx, ds in enumerate(args.datasets):
        for m_idx, metric in enumerate(args.metrics):
            col_labels.append(f"{metric.upper()}\n{ds[:4]}")
            lower_better = metric in LOWER_BETTER
            vals = [(per_ds[ds].get((mm, metric), (np.nan,))[0], mm)
                    for mm in common_sorted]
            valid_vals = [v for v in vals if np.isfinite(v[0])]
            valid_vals.sort(key=lambda x: x[0], reverse=not lower_better)
            rank_map = {mm: i + 1 for i, (_, mm) in enumerate(valid_vals)}
            col = d_idx * n_met + m_idx
            for row, mm in enumerate(common_sorted):
                if mm in rank_map:
                    ranks[row, col] = rank_map[mm]

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_met * n_ds),
                                    max(3, 0.5 * n_meth)))
    im = ax.imshow(ranks, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_meth))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in common_sorted])
    ax.set_title(f"Ranks (1 = best) for common methods — "
                 f"{', '.join(args.datasets)}")
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            if np.isfinite(ranks[i, j]):
                ax.text(j, i, f"{int(ranks[i, j])}", ha="center",
                        va="center", fontsize=9,
                        color="white" if ranks[i, j] > n_meth / 2
                        else "black")
    # Vertical separator between datasets.
    for sep in range(1, n_ds):
        ax.axvline(sep * n_met - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, shrink=0.7, label="rank")
    fig.tight_layout()
    png_path = osp.join(args.work_dir, f"cross_dataset_ranks_ep{args.epoch}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"CSV    : {csv_path}")
    print(f"MD     : {md_path}")
    print(f"Heatmap: {png_path}")


if __name__ == "__main__":
    main()
