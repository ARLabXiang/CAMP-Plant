"""
combined_scoreboard.py
----------------------
Single 'paper Table 1' CSV combining every metric + bootstrap 95% CI for
every trained method on every dataset. Reads from:

  - work_dirs/{dataset}_ep{epoch}_analysis/bootstrap_ci/bootstrap_ci.csv
  - (if missing, re-runs bootstrap internally — slow fallback)

Outputs
-------
  work_dirs/combined_scoreboard_ep{epoch}.csv
  work_dirs/combined_scoreboard_ep{epoch}.md     Markdown table for the paper

Usage
-----
    python tools/combined_scoreboard.py --epoch 150
    python tools/combined_scoreboard.py --epoch 150 --datasets komatsuna arabidopsis
"""

import argparse
import csv
import os.path as osp
import sys
from collections import defaultdict

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import METHOD_DISPLAY, METHODS


def read_bootstrap_ci(path):
    """Return {(method, metric): (mean, ci_low, ci_high)} from bootstrap_ci.csv."""
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

    # dataset -> {(method, metric): (mean, lo, hi)}
    data = {}
    for ds in args.datasets:
        ci_path = osp.join(args.work_dir, f"{ds}_ep{args.epoch}_analysis",
                           "bootstrap_ci", "bootstrap_ci.csv")
        data[ds] = read_bootstrap_ci(ci_path)
        if not data[ds]:
            print(f"  [warn] no bootstrap_ci.csv for {ds}; "
                  f"run tools/bootstrap_ci.py --dataset {ds} first")

    # Discover the union of methods that appear in any dataset.
    methods_seen = set()
    for ds, d in data.items():
        for (m, _metric), _ in d.items():
            methods_seen.add(m)
    methods_sorted = [m for m in METHODS if m in methods_seen]
    for m in methods_seen:
        if m not in methods_sorted:
            methods_sorted.append(m)

    # ---- Long-form CSV -------------------------------------------------- #
    csv_path = osp.join(args.work_dir, f"combined_scoreboard_ep{args.epoch}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "metric", "mean",
                    "ci_low", "ci_high", "ci_width"])
        for ds in args.datasets:
            for method in methods_sorted:
                for metric in args.metrics:
                    key = (method, metric)
                    if key not in data[ds]:
                        continue
                    mean, lo, hi = data[ds][key]
                    w.writerow([ds, method, metric,
                                f"{mean:.4g}", f"{lo:.4g}", f"{hi:.4g}",
                                f"{hi - lo:.4g}"])

    # ---- Wide Markdown table per dataset (paper-style) ------------------ #
    md_path = osp.join(args.work_dir, f"combined_scoreboard_ep{args.epoch}.md")
    with open(md_path, "w") as f:
        f.write(f"# Combined scoreboard — {args.epoch} epochs\n\n")
        f.write("Bootstrap 95% CI in brackets. Best in each column is "
                "**bolded**. ↓/↑ shows the desired direction.\n\n")
        for ds in args.datasets:
            if not data[ds]:
                continue
            f.write(f"## {ds.capitalize()}\n\n")
            f.write("| Method | " +
                    " | ".join(f"{m.upper()} "
                              f"{'↓' if m in {'mae','mse','poi_mae'} else '↑'}"
                              for m in args.metrics) + " |\n")
            f.write("|---|" + "|".join(["---:"] * len(args.metrics)) + "|\n")

            # Find the best method per metric in this dataset.
            best = {}
            for metric in args.metrics:
                vals = [(data[ds][(m, metric)][0], m) for m in methods_sorted
                        if (m, metric) in data[ds]]
                if not vals:
                    continue
                lower_better = metric in {"mae", "mse", "poi_mae"}
                vals.sort(key=lambda x: x[0], reverse=not lower_better)
                best[metric] = vals[0][1]

            for method in methods_sorted:
                cells = []
                for metric in args.metrics:
                    key = (method, metric)
                    if key not in data[ds]:
                        cells.append("—")
                        continue
                    mean, lo, hi = data[ds][key]
                    text = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]" \
                        if metric in {"ssim"} else \
                        f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"
                    if best.get(metric) == method:
                        text = f"**{text}**"
                    cells.append(text)
                any_entry = any(c != "—" for c in cells)
                if not any_entry:
                    continue
                f.write(f"| {METHOD_DISPLAY.get(method, method)} | "
                        + " | ".join(cells) + " |\n")
            f.write("\n")

    print(f"Long CSV : {csv_path}")
    print(f"Markdown : {md_path}")


if __name__ == "__main__":
    main()
