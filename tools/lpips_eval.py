"""
lpips_eval.py
-------------
Compute the Learned Perceptual Image Patch Similarity (LPIPS) distance
between predicted and ground-truth frames for every trained method.
Addresses Reviewer 1's comment: "No comparison with existing […]
perceptual metrics."

LPIPS is a learned perceptual distance (Zhang et al. 2018) that correlates
with human judgement better than MSE or SSIM on natural images. We use
the AlexNet backbone because it's fastest and the CAMP paper's figures
are already qualitative; perceptual ordering matters more than absolute
value.

Requires the `lpips` package and torch. If either is missing the script
exits with a clear message (no training code change needed).

Outputs
-------
  work_dirs/{dataset}_ep{epoch}_analysis/lpips/
    lpips_per_method.csv       method, mean LPIPS ↓, std, N
    per_sample_lpips.npz       raw per-sample arrays keyed by method

Usage
-----
    python tools/lpips_eval.py --dataset komatsuna --epoch 150
    python tools/lpips_eval.py --dataset komatsuna --epoch 150 --net vgg
"""

import argparse
import csv
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _analysis_utils import (
    discover_models, load_preds_trues, ensure_analysis_dir, METHOD_DISPLAY,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work_dirs")
    ap.add_argument("--dataset", default="komatsuna")
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cpu",
                    help="torch device for LPIPS; cpu avoids fighting the "
                         "training GPU")
    args = ap.parse_args()

    try:
        import torch
        import lpips  # type: ignore
    except ImportError as e:
        sys.exit(f"lpips/torch not available ({e}). "
                 f"Install with: pip install lpips")

    models = discover_models(args.work_dir, args.dataset, args.epoch)
    if not models:
        sys.exit(f"No ready models in {args.work_dir}")

    device = torch.device(args.device)
    net = lpips.LPIPS(net=args.net, verbose=False).to(device).eval()

    out_dir = ensure_analysis_dir(args.work_dir, args.dataset, args.epoch,
                                  "lpips")
    rows = []
    per_sample_all = {}

    for method, saved in models.items():
        preds, trues = load_preds_trues(saved)
        N, T = preds.shape[:2]
        per_sample = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            for i in range(N):
                # [-1, 1] range for LPIPS.
                p = torch.from_numpy(preds[i] * 2 - 1).float().to(device)
                t = torch.from_numpy(trues[i] * 2 - 1).float().to(device)
                # lpips returns per-frame distances; mean over frames.
                d = net(p, t).squeeze().cpu().numpy()
                per_sample[i] = float(np.mean(d))
        per_sample_all[method] = per_sample
        rows.append({"method": method,
                     "lpips_mean": float(per_sample.mean()),
                     "lpips_std":  float(per_sample.std(ddof=1)),
                     "N": int(N)})
        print(f"{method:15s}  LPIPS (↓) = {per_sample.mean():.4f} "
              f"± {per_sample.std(ddof=1):.4f}")

    csv_path = osp.join(out_dir, "lpips_per_method.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "lpips_mean",
                                          "lpips_std", "N"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["lpips_mean"]):
            r = {**r,
                 "lpips_mean": f"{r['lpips_mean']:.4f}",
                 "lpips_std":  f"{r['lpips_std']:.4f}"}
            w.writerow(r)

    np.savez(osp.join(out_dir, "per_sample_lpips.npz"), **per_sample_all)
    print(f"\nCSV → {csv_path}")
    print(f"Per-sample arrays → {osp.join(out_dir, 'per_sample_lpips.npz')}")


if __name__ == "__main__":
    main()
