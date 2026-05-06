"""
_analysis_utils.py
------------------
Shared helpers for the post-hoc analysis scripts (significance_test,
bootstrap_ci, metric_correlation, threshold_sensitivity, lpips_eval,
per_frame_error, error_distribution, failure_case_viewer, cls_confidence).

These scripts are read-only: they consume the .npy / .npz artifacts that
every model's test step saves to work_dirs/{dataset}_{method}_ep{N}/saved/.
They never touch training code and can run while training is in progress
on other GPUs.

Conventions
-----------
- Saved arrays have shape (N, T, C, H, W) where values are in [0, 1].
- OpenSTL's aggregated MAE/MSE in CSV = per-sample-sum, averaged over N.
- POI per-sample curves live in saved/poi_plots/*_growth_curves_data.npz.
- Per-sample bipolar pairs (N*(T-1) frame transitions) live in
  saved/poi_plots/*_bipolar_regression_data.npz.
"""

import glob
import os
import os.path as osp
import re
from typing import Dict, Optional

import numpy as np


# Canonical display order (matches the paper and work_dirs CSV).
METHODS = [
    "convlstm", "predrnn", "phydnet", "simvp", "mim", "tau",
    "camp", "camp_base", "camp_no_cls", "camp_full",
    "tau_full", "simvp_full", "mim_full", "tau_no_cls", "tau_full_clsw01", "simvp_no_cls",
    "tau_predcls", "tau_full_detached", "simvp_predcls",
]
METHOD_DISPLAY = {
    "convlstm": "ConvLSTM", "predrnn": "PredRNN", "phydnet": "PhyDNet",
    "simvp": "SimVP", "mim": "MIM", "tau": "TAU",
    "camp": "CAMP", "camp_base": "CAMP_base", "camp_no_cls": "CAMP_no_cls",
    "camp_full": "CAMP_full",
    "tau_full": "TAU_full",
    "simvp_full": "SimVP_full",
    "mim_full": "MIM_full",
    "tau_no_cls": "TAU_no_cls",
    "tau_full_clsw01": "TAU_full (cls_w=0.1)",
    "simvp_no_cls": "SimVP_no_cls",
    "tau_predcls": "TAU_PredCls",
    "tau_full_detached": "TAU_full (detached)",
    "simvp_predcls": "SimVP_PredCls",
}


def discover_models(work_dir: str, dataset: str, epoch: int) -> Dict[str, str]:
    """
    Scan work_dir for {dataset}_<method>_ep<epoch> directories that contain
    saved/preds.npy. Return {method: path_to_saved_dir} ordered by METHODS.

    A method is considered "ready" only if both preds.npy and trues.npy
    exist — skips runs that are still training.
    """
    pattern = osp.join(work_dir, f"{dataset}_*_ep{epoch}", "saved")
    found = {}
    for saved in glob.glob(pattern):
        if not (osp.exists(osp.join(saved, "preds.npy")) and
                osp.exists(osp.join(saved, "trues.npy"))):
            continue
        run_dir = osp.basename(osp.dirname(saved))
        m = re.match(rf"^{dataset}_(.+)_ep{epoch}$", run_dir)
        if m:
            found[m.group(1)] = saved

    # Return in canonical order, plus any extras at the end.
    ordered = {k: found[k] for k in METHODS if k in found}
    for k in found:
        if k not in ordered:
            ordered[k] = found[k]
    return ordered


def load_preds_trues(saved_dir: str):
    """Load (preds, trues) as float32 arrays of shape (N, T, C, H, W)."""
    preds = np.load(osp.join(saved_dir, "preds.npy")).astype(np.float32)
    trues = np.load(osp.join(saved_dir, "trues.npy")).astype(np.float32)
    # Clip preds to valid image range — occasional tiny overshoots in models.
    preds = np.clip(preds, 0.0, 1.0)
    return preds, trues


def per_sample_mae(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """Sum-of-absolute-errors per sample, shape (N,). Matches OpenSTL CSV."""
    return np.abs(preds - trues).reshape(preds.shape[0], -1).sum(axis=1)


def per_sample_mse(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """Sum-of-squared-errors per sample, shape (N,). Matches OpenSTL CSV."""
    return ((preds - trues) ** 2).reshape(preds.shape[0], -1).sum(axis=1)


def per_sample_ssim(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """Mean SSIM per sample over the T frames and channels, shape (N,)."""
    from skimage.metrics import structural_similarity as ssim
    N, T, C, H, W = preds.shape
    out = np.empty(N, dtype=np.float32)
    for i in range(N):
        vals = []
        for t in range(T):
            for c in range(C):
                vals.append(ssim(trues[i, t, c], preds[i, t, c],
                                 data_range=1.0))
        out[i] = float(np.mean(vals))
    return out


def per_sample_psnr(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """Mean PSNR per sample over the T frames, shape (N,). Uses dB, data_range=1."""
    mse = ((preds - trues) ** 2).reshape(preds.shape[0], preds.shape[1], -1).mean(axis=2)
    # Avoid div-by-zero; clamp mse to a tiny positive.
    mse = np.maximum(mse, 1e-12)
    psnr_per_frame = 10.0 * np.log10(1.0 / mse)
    return psnr_per_frame.mean(axis=1).astype(np.float32)


def compute_exgi(frames: np.ndarray) -> np.ndarray:
    """
    Excess Green Index on a batch of frames.

    Input  : frames [..., C, H, W] with C >= 3 (R, G, B) in [0, 1]
    Output : exgi  [..., H, W]  values roughly in [−1, 2]
    """
    r = frames[..., 0, :, :]
    g = frames[..., 1, :, :]
    b = frames[..., 2, :, :]
    return 2.0 * g - r - b


def per_sample_poi_mae(saved_dir: str, model_name: str) -> Optional[np.ndarray]:
    """
    Load per-sample POI_MAE from the growth_curves_data.npz written by
    eval_poi.py. Returns shape (N,) or None if the npz is absent.
    """
    npz_path = osp.join(saved_dir, "poi_plots",
                        f"{model_name}_growth_curves_data.npz")
    if not osp.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=True)
    pred_poi = z["pred_poi"]
    true_poi = z["true_poi"]
    N = len(pred_poi)
    out = np.empty(N, dtype=np.float32)
    for i in range(N):
        p = np.asarray(pred_poi[i], dtype=np.float32)
        t = np.asarray(true_poi[i], dtype=np.float32)
        out[i] = np.abs(p - t).mean()
    return out


def filter_to_majority_N(models: Dict[str, str]) -> Dict[str, str]:
    """
    Drop methods whose sample count differs from the majority count.
    Returns the filtered dict; prints a warning for each skipped method.

    This matters for paired statistical tests: MIM's dataloader drops the
    tail when batch_size != dataset_size % batch_size, so MIM often has
    fewer samples than the other methods.
    """
    from collections import Counter
    ns = {}
    for method, saved in models.items():
        preds = np.load(osp.join(saved, "preds.npy"), mmap_mode="r")
        ns[method] = preds.shape[0]
    counts = Counter(ns.values())
    majority_n = counts.most_common(1)[0][0]
    kept = {}
    for method, saved in models.items():
        if ns[method] == majority_n:
            kept[method] = saved
        else:
            print(f"  [warn] dropping {method}: N={ns[method]} != majority "
                  f"N={majority_n} (paired tests require equal N)")
    return kept


def ensure_analysis_dir(work_dir: str, dataset: str, epoch: int,
                       subdir: Optional[str] = None) -> str:
    """Create work_dirs/{dataset}_ep{epoch}_analysis[/subdir] and return it."""
    root = osp.join(work_dir, f"{dataset}_ep{epoch}_analysis")
    path = osp.join(root, subdir) if subdir else root
    os.makedirs(path, exist_ok=True)
    return path
