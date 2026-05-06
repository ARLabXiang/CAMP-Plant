"""
ablation_table.py
-----------------
Cite-ready relative-improvement table: how much does each CAMP ablation
variant change over a chosen baseline (default: `camp_base`) on every
metric? This is the cleanest rebuttal to the reviewer's "CAMP_full is
just gaming its own metric" concern — if L_ExGI improves every metric,
not just POI_MAE, the gain is real.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/ablation/
    ablation_vs_{baseline}.csv      rows = variant, cols = metric
    ablation_vs_{baseline}.md       paper-ready Markdown table

Usage
-----
    python tools/ablation_table.py --dataset komatsuna --epoch 150
    python tools/ablation_table.py --dataset komatsuna --epoch 150 \\
        --baseline predrnn --variants camp camp_no_cls camp_full
"""

import argparse
import csv
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import ensure_analysis_dir, METHOD_DISPLAY


LOWER_BETTER = {"mae", "mse", "poi_mae", "lpips"}


def read_bootstrap_ci(path):
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["method"], row["metric"])] = (
                float(row["mean"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
            )
    return out


def read_lpips_csv(path):
    out = {}
    if not osp.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["method"]] = (
                float(row["lpips_mean"]),
                float(row["lpips_mean"]) - float(row["lpips_std"]),
                float(row["lpips_mean"]) + float(row["lpips_std"]),
            )
    return out


def relative_change(baseline, variant, metric):
    """Signed % change where positive = improvement."""
    if baseline == 0:
        return float("nan")
    raw = (variant - baseline) / abs(baseline) * 100
    return -raw if metric in LOWER_BETTER else raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--baseline", default="camp_base")
    ap.add_argument("--variants", nargs="+",
                    default=["camp", "camp_no_cls", "camp_full"])
    ap.add_argument("--metrics", nargs="+",
                    default=["mae", "mse", "ssim", "psnr", "poi_mae", "lpips"])
    args = ap.parse_args()

    analysis_root = osp.join(args.work_dir,
                             f"{args.dataset}_ep{args.epoch}_analysis")
    ci = read_bootstrap_ci(osp.join(analysis_root, "bootstrap_ci",
                                    "bootstrap_ci.csv"))
    ci.update({(m, "lpips"): v for m, v in
               read_lpips_csv(osp.join(analysis_root, "lpips",
                                       "lpips_per_method.csv")).items()})

    if (args.baseline, args.metrics[0]) not in ci:
        sys.exit(f"Baseline {args.baseline} not found in "
                 f"{analysis_root}/bootstrap_ci/bootstrap_ci.csv")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "ablation")

    # CSV with raw + delta columns.
    csv_path = osp.join(out_dir, f"ablation_vs_{args.baseline}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["variant"]
        for m in args.metrics:
            header += [f"{m}_value", f"{m}_pct_change"]
        w.writerow(header)
        # Baseline row first.
        row = [args.baseline]
        for m in args.metrics:
            v = ci.get((args.baseline, m), (None, None, None))[0]
            row += [f"{v:.4g}" if v is not None else "", ""]
        w.writerow(row)
        # Variant rows.
        for var in args.variants:
            row = [var]
            for m in args.metrics:
                base_v = ci.get((args.baseline, m), (None,))[0]
                var_v = ci.get((var, m), (None,))[0]
                if base_v is None or var_v is None:
                    row += ["", ""]
                    continue
                pct = relative_change(base_v, var_v, m)
                row += [f"{var_v:.4g}", f"{pct:+.2f}"]
            w.writerow(row)

    # Markdown (paper-ready).
    md_path = osp.join(out_dir, f"ablation_vs_{args.baseline}.md")
    with open(md_path, "w") as f:
        f.write(f"# CAMP ablation vs {args.baseline} — "
                f"{args.dataset}, {args.epoch} ep\n\n")
        f.write("Signed % change relative to baseline. Positive = improvement "
                "(direction-aware: for lower-is-better metrics, a decrease in "
                "raw value is reported as a positive gain).\n\n")
        f.write("| Variant | "
                + " | ".join(f"Δ {m.upper()}" for m in args.metrics)
                + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(args.metrics)) + "|\n")
        # Baseline row (all zeros).
        f.write(f"| {METHOD_DISPLAY.get(args.baseline, args.baseline)} "
                f"(baseline) | "
                + " | ".join(["0.00 %"] * len(args.metrics)) + " |\n")
        for var in args.variants:
            cells = []
            for m in args.metrics:
                base_v = ci.get((args.baseline, m), (None,))[0]
                var_v = ci.get((var, m), (None,))[0]
                if base_v is None or var_v is None:
                    cells.append("—")
                    continue
                pct = relative_change(base_v, var_v, m)
                tag = "**" if pct > 0 else ""
                cells.append(f"{tag}{pct:+.2f} %{tag}")
            f.write(f"| {METHOD_DISPLAY.get(var, var)} | "
                    + " | ".join(cells) + " |\n")
        f.write("\n")
        f.write(f"Raw values in `{osp.basename(csv_path)}`.\n")

    print(f"CSV : {csv_path}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
