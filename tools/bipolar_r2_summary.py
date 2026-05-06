"""
bipolar_r2_summary.py
---------------------
Extract the bipolar-regression R² for 'new POIs' (growth) and 'missing POIs'
(shrinkage) from the .npz files written by eval_poi.py. Summarises every
trained method on every dataset in a single table.

R² is recomputed via sklearn so values are consistent even if the eval_poi
implementation changes.

Outputs
-------
  work_dirs/bipolar_r2_ep{epoch}.csv
  work_dirs/bipolar_r2_ep{epoch}.md

Usage
-----
    python tools/bipolar_r2_summary.py --epoch 150
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import discover_models, METHOD_DISPLAY


def r2_calibration(y_true, y_pred):
    """
    Calibration-aware R²: 1 − SS_res / SS_tot, penalises systematic bias.
    Negative values mean predictions are worse than a constant mean.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def r2_correlation(y_true, y_pred):
    """
    Correlation-only R² = (Pearson r)². Equivalent to the R² that
    scipy.stats.linregress reports. Does NOT penalise systematic bias —
    a perfectly-shaped but scaled prediction still scores high.

    eval_poi.py reports this version in the bipolar scatter titles.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--datasets", nargs="+",
                    default=["komatsuna", "arabidopsis"])
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        models = discover_models(args.work_dir, ds, args.epoch)
        for method, saved in models.items():
            run_name = osp.basename(osp.dirname(saved))
            npz_path = osp.join(saved, "poi_plots",
                                f"{run_name}_bipolar_regression_data.npz")
            if not osp.exists(npz_path):
                continue
            z = np.load(npz_path, allow_pickle=True)
            rows.append({
                "dataset": ds, "method": method,
                "n_transitions": int(len(z["all_new_true"])),
                "r2_new_corr":   r2_correlation(z["all_new_true"],  z["all_new_pred"]),
                "r2_new_cal":    r2_calibration(z["all_new_true"],  z["all_new_pred"]),
                "r2_miss_corr":  r2_correlation(z["all_miss_true"], z["all_miss_pred"]),
                "r2_miss_cal":   r2_calibration(z["all_miss_true"], z["all_miss_pred"]),
                "mean_new_true":  float(np.mean(z["all_new_true"])),
                "mean_new_pred":  float(np.mean(z["all_new_pred"])),
                "mean_miss_true": float(np.mean(z["all_miss_true"])),
                "mean_miss_pred": float(np.mean(z["all_miss_pred"])),
            })

    csv_path = osp.join(args.work_dir, f"bipolar_r2_ep{args.epoch}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "r2_new_corr":   f"{r['r2_new_corr']:.4f}",
                        "r2_new_cal":    f"{r['r2_new_cal']:.4f}",
                        "r2_miss_corr":  f"{r['r2_miss_corr']:.4f}",
                        "r2_miss_cal":   f"{r['r2_miss_cal']:.4f}",
                        "mean_new_true":  f"{r['mean_new_true']:.2f}",
                        "mean_new_pred":  f"{r['mean_new_pred']:.2f}",
                        "mean_miss_true": f"{r['mean_miss_true']:.2f}",
                        "mean_miss_pred": f"{r['mean_miss_pred']:.2f}"})

    md_path = osp.join(args.work_dir, f"bipolar_r2_ep{args.epoch}.md")
    with open(md_path, "w") as f:
        f.write(f"# Bipolar R² — {args.epoch} epochs\n\n")
        f.write("Two complementary R² definitions on the frame-to-frame "
                "bipolar-regression points:\n\n")
        f.write("- **R²_corr** (correlation-only) = Pearson r² — measures "
                "whether predictions track the shape of ground truth. "
                "This is what `eval_poi.py` prints on the bipolar scatter "
                "plot titles.\n")
        f.write("- **R²_cal** (calibration-aware) = 1 − SS_res / SS_tot — "
                "penalises systematic bias. A high r² with compressed or "
                "inflated slope can produce a low (or negative) R²_cal.\n\n")
        f.write("A model that predicts 2× the true values has R²_corr = 1.0 "
                "but low R²_cal. Always read both.\n\n")
        for ds in args.datasets:
            ds_rows = [r for r in rows if r["dataset"] == ds]
            if not ds_rows:
                continue
            f.write(f"## {ds.capitalize()}\n\n")
            f.write("| Method | R²_corr(new) | R²_cal(new) | R²_corr(miss) "
                    "| R²_cal(miss) | N |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for r in ds_rows:
                f.write(f"| {METHOD_DISPLAY.get(r['method'], r['method'])} | "
                        f"{r['r2_new_corr']:.3f} | {r['r2_new_cal']:.3f} | "
                        f"{r['r2_miss_corr']:.3f} | {r['r2_miss_cal']:.3f} | "
                        f"{r['n_transitions']} |\n")
            f.write("\n")

    print(f"CSV : {csv_path}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
