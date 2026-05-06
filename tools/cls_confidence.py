"""
cls_confidence.py
-----------------
For CAMP variants with a classification head (`camp`, `camp_full`), plot
the distribution of sigmoid(cls_logits) on the test set.

On Komatsuna every label is −1 (drought information unavailable) so the
classification loss is masked to zero and the head never receives a
gradient. If that story is correct, the predicted probabilities should
cluster near the dropout-driven prior (≈ 0.5 or whatever untrained noise
produces) instead of separating the classes. This script provides the
quantitative evidence for that claim.

On Arabidopsis (where labels ∈ {0, 1} are real) the same plot becomes the
primary classification-accuracy figure: a well-separated bimodal
distribution indicates the head learned the environment signal.

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/cls_confidence/
    cls_confidence_{method}.png       histogram coloured by label
    cls_confidence_summary.csv        one row per method

Usage
-----
    python tools/cls_confidence.py --dataset komatsuna --epoch 150
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
from _analysis_utils import discover_models, ensure_analysis_dir, METHOD_DISPLAY


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    args = ap.parse_args()

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "cls_confidence")
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "DejaVu Serif"]})

    rows = []
    for method, saved in models.items():
        logits_path = osp.join(saved, "cls_logits.npy")
        labels_path = osp.join(saved, "labels.npy")
        if not (osp.exists(logits_path) and osp.exists(labels_path)):
            continue
        logits = np.load(logits_path).squeeze(-1) if \
            np.load(logits_path).ndim > 1 else np.load(logits_path)
        labels = np.load(labels_path).squeeze(-1) if \
            np.load(labels_path).ndim > 1 else np.load(labels_path)
        probs = sigmoid(logits)

        valid = labels >= 0
        n_valid = int(valid.sum())
        if n_valid > 0:
            preds = (logits[valid] > 0).astype(np.float32)
            acc = float((preds == labels[valid]).mean())
        else:
            acc = np.nan

        rows.append({
            "method": method,
            "N": int(len(probs)),
            "n_labeled": n_valid,
            "prob_mean": float(probs.mean()),
            "prob_std":  float(probs.std(ddof=1) if len(probs) > 1 else 0.0),
            "prob_min":  float(probs.min()),
            "prob_max":  float(probs.max()),
            "cls_accuracy": acc,
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        if n_valid > 0:
            ax.hist(probs[labels == 0], bins=20, range=(0, 1), alpha=0.55,
                    color="steelblue", label=f"label = 0 (n={int((labels == 0).sum())})")
            ax.hist(probs[labels == 1], bins=20, range=(0, 1), alpha=0.55,
                    color="firebrick", label=f"label = 1 (n={int((labels == 1).sum())})")
        else:
            ax.hist(probs, bins=20, range=(0, 1), color="gray",
                    alpha=0.7, label=f"label = -1 (masked, n={len(probs)})")
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.4)
        ax.set_xlabel("sigmoid(cls_logit)")
        ax.set_ylabel("count")
        title = f"{METHOD_DISPLAY.get(method, method)} — {args.dataset}"
        if not np.isnan(acc):
            title += f" (acc = {acc:.3f})"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, f"cls_confidence_{method}.png"), dpi=150)
        plt.close(fig)
        print(f"{method:15s}  prob ∈ [{probs.min():.3f}, {probs.max():.3f}]  "
              f"mean={probs.mean():.3f}  "
              f"acc={'n/a' if np.isnan(acc) else f'{acc:.3f}'}")

    if not rows:
        print("No CAMP-style runs with cls_logits.npy found.")
        return

    csv_path = osp.join(out_dir, "cls_confidence_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "N", "n_labeled",
                                          "prob_mean", "prob_std",
                                          "prob_min", "prob_max",
                                          "cls_accuracy"])
        w.writeheader()
        for r in rows:
            r = {**r,
                 "prob_mean": f"{r['prob_mean']:.4f}",
                 "prob_std":  f"{r['prob_std']:.4f}",
                 "prob_min":  f"{r['prob_min']:.4f}",
                 "prob_max":  f"{r['prob_max']:.4f}",
                 "cls_accuracy": "" if np.isnan(r["cls_accuracy"])
                                  else f"{r['cls_accuracy']:.4f}"}
            w.writerow(r)

    print(f"\nCSV → {csv_path}")
    print(f"Plots → {out_dir}")


if __name__ == "__main__":
    main()
