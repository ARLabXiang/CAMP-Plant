"""
failure_case_viewer.py
----------------------
Find samples where one method wins big and the other loses big, then
render the RGB frames and ExGI maps side by side. Addresses Reviewer 1's
comment: "Stronger numerical evidence is required to support claims."

Given two methods (e.g., camp_full vs tau) we rank every test sample by
the signed metric difference  Δ = metric(method_A) − metric(method_B)
and render the K most negative (method_A wins) and K most positive
(method_A loses) samples.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/failure_cases/
    {A}_vs_{B}_{metric}_A_wins_{sample_idx}.png
    {A}_vs_{B}_{metric}_A_loses_{sample_idx}.png
    ranking_{A}_vs_{B}_{metric}.csv

Usage
-----
    python tools/failure_case_viewer.py --dataset komatsuna --epoch 150 \\
        --method_a camp_full --method_b tau --metric poi_mae --topk 3
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
    compute_exgi, per_sample_mae, per_sample_mse, per_sample_ssim,
    per_sample_psnr, per_sample_poi_mae, METHOD_DISPLAY,
)


def get_per_sample(saved, metric):
    preds, trues = load_preds_trues(saved)
    run_name = osp.basename(osp.dirname(saved))
    if metric == "mae":     return per_sample_mae(preds, trues), preds, trues
    if metric == "mse":     return per_sample_mse(preds, trues), preds, trues
    if metric == "ssim":    return per_sample_ssim(preds, trues), preds, trues
    if metric == "psnr":    return per_sample_psnr(preds, trues), preds, trues
    if metric == "poi_mae":
        v = per_sample_poi_mae(saved, run_name)
        if v is None:
            sys.exit(f"POI data missing for {run_name} — run eval_poi.py.")
        return v, preds, trues
    raise ValueError(metric)


def plot_sample(sample_idx, preds_a, preds_b, trues,
                name_a, name_b, metric, delta, out_path, tau=0.1):
    """Render a grid: rows = {GT, A, B}, cols = T frames (RGB + ExGI)."""
    T = preds_a.shape[0]
    fig, axes = plt.subplots(3, 2 * T, figsize=(2.2 * T, 6.6))
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})

    def show_row(row_idx, frames, label):
        for t in range(T):
            # RGB
            ax = axes[row_idx, 2 * t]
            img = np.clip(frames[t].transpose(1, 2, 0), 0, 1)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(label, fontsize=10)
            if row_idx == 0:
                ax.set_title(f"t = {t + 1}", fontsize=9)

            # ExGI
            ax = axes[row_idx, 2 * t + 1]
            exgi = 2 * frames[t, 1] - frames[t, 0] - frames[t, 2]
            ax.imshow(exgi > tau, cmap="Greens", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

    show_row(0, trues, "Ground Truth")
    show_row(1, preds_a, METHOD_DISPLAY.get(name_a, name_a))
    show_row(2, preds_b, METHOD_DISPLAY.get(name_b, name_b))

    fig.suptitle(f"Sample {sample_idx}  |  Δ{metric} "
                 f"({name_a} − {name_b}) = {delta:.3f}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--method_a", required=True,
                    help="model that 'wins' when Δ is negative")
    ap.add_argument("--method_b", required=True)
    ap.add_argument("--metric", default="poi_mae",
                    choices=["mae", "mse", "ssim", "psnr", "poi_mae"])
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    for m in (args.method_a, args.method_b):
        if m not in models:
            sys.exit(f"Method {m} not found in {args.work_dir}. "
                     f"Available: {list(models)}")

    vals_a, preds_a, trues_a = get_per_sample(models[args.method_a], args.metric)
    vals_b, preds_b, trues_b = get_per_sample(models[args.method_b], args.metric)

    if vals_a.shape != vals_b.shape:
        sys.exit(f"Sample counts differ: {vals_a.shape} vs {vals_b.shape}")
    if not np.allclose(trues_a, trues_b):
        print("WARNING: ground-truth arrays differ between methods — "
              "test sets may not be aligned.")

    # Sign convention: lower-better metrics use delta = a − b (negative ⇒ A wins).
    # Higher-better metrics flip sign.
    lower_better = args.metric in {"mae", "mse", "poi_mae"}
    delta = vals_a - vals_b if lower_better else vals_b - vals_a

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "failure_cases")

    # Save full ranking.
    order = np.argsort(delta)
    csv_path = osp.join(out_dir,
                        f"ranking_{args.method_a}_vs_{args.method_b}_{args.metric}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx",
                    f"{args.method_a}_{args.metric}",
                    f"{args.method_b}_{args.metric}",
                    f"delta_A_minus_B (lower_better={lower_better})"])
        for i in order:
            w.writerow([int(i), f"{vals_a[i]:.4f}", f"{vals_b[i]:.4f}",
                        f"{delta[i]:.4f}"])

    # Render extremes.
    a_wins = order[:args.topk]
    a_loses = order[-args.topk:][::-1]

    for rank, i in enumerate(a_wins):
        png = osp.join(out_dir,
                       f"{args.method_a}_vs_{args.method_b}_{args.metric}_"
                       f"A_wins_rank{rank + 1}_sample{i}.png")
        plot_sample(int(i), preds_a[i], preds_b[i], trues_a[i],
                    args.method_a, args.method_b, args.metric,
                    float(delta[i]), png)
    for rank, i in enumerate(a_loses):
        png = osp.join(out_dir,
                       f"{args.method_a}_vs_{args.method_b}_{args.metric}_"
                       f"A_loses_rank{rank + 1}_sample{i}.png")
        plot_sample(int(i), preds_a[i], preds_b[i], trues_a[i],
                    args.method_a, args.method_b, args.metric,
                    float(delta[i]), png)

    print(f"CSV → {csv_path}")
    print(f"{args.topk} A-wins + {args.topk} A-loses grids → {out_dir}")


if __name__ == "__main__":
    main()
