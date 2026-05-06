"""
visualize_results.py
--------------------
Reproduce all main figures from the CAMP paper
(Multi-Task Spatio-Temporal Modelling for Plant Growth Prediction).

This script reads the .npy prediction artefacts saved by on_test_epoch_end()
and produces publication-quality figures.  It is designed to be run *after*
all methods have been tested via tools/run_plant_experiments.sh.

Figures reproduced
------------------
  Fig 2  — Pareto-front scatter (SSIM vs MAE / SSIM vs MSE).
            One marker per model; CAMP is highlighted.
  Fig 3  — ExGI pipeline: RGB → ExGI map → binary POI mask.
            Shows the green-pixel extraction process for one example frame.
  Fig 4/5 — Qualitative prediction grids (one per dataset).
            Rows = models; columns = 5 predicted frames.
            Left half = RGB frames; right half = ExGI maps.
  Fig 6/7 — POI-based bipolar regression curves (one per dataset).
            Three panels: total POI count, new POIs, missing POIs.
            One line per model, averaged over the test set.

Usage
-----
    # All figures for both datasets
    python tools/visualize_results.py --work_dir work_dirs --dataset komatsuna

    # Single figure type
    python tools/visualize_results.py --work_dir work_dirs --dataset arabidopsis \\
        --figures pareto qualitative poi_curves

    # Save to a specific output directory
    python tools/visualize_results.py --work_dir work_dirs --dataset komatsuna \\
        --out_dir figures/ --sample_idx 0

Arguments
---------
    --work_dir   : Root directory containing method subdirectories
                   (e.g.  work_dirs/komatsuna_camp/saved/preds.npy).
                   Default: work_dirs
    --dataset    : 'komatsuna' or 'arabidopsis'.  Used to construct
                   subdirectory names.  Default: komatsuna
    --figures    : Which figures to generate.
                   Choices: pareto exgi_pipeline qualitative poi_curves.
                   Default: all four.
    --out_dir    : Where to save figures.  Default: figures/{dataset}/
    --sample_idx : Which test-set sample to show in the qualitative grid
                   and ExGI pipeline.  Default: 0
    --threshold  : ExGI threshold τ for POI extraction.  Default: 0.1
    --dpi        : Figure DPI.  Default: 150
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # server-safe non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
from matplotlib.lines import Line2D

# Global font sizes — paper-ready (large enough for publication)
matplotlib.rcParams.update({
    "font.size":        15,
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "figure.titlesize": 19,
})

# ---------------------------------------------------------------------------
# We reuse the ExGI helpers from eval_poi.py.  Add tools/ to sys.path so
# the import works regardless of where this script is invoked from.
# ---------------------------------------------------------------------------
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from eval_poi import (
    compute_exgi,
    compute_poi_mask,
    poi_growth_curve,
)


# ===========================================================================
# Colour scheme — consistent across all figures
# ===========================================================================

#  Each method has a (display_name, matplotlib_color, marker_style).
# Note: "predrnn" and "camp_base" are the same architecture. We display only
# "camp_base" (labeled "CAMP_base") throughout all figures. PredRNN is noted
# in a table footnote in the paper.
METHOD_STYLES = {
    "convlstm"    : ("ConvLSTM",    "#4e79a7", "o"),
    "phydnet"     : ("PhyDNet",     "#59a14f", "^"),
    "simvp"       : ("SimVP",       "#b07aa1", "D"),
    "mim"         : ("MIM",         "#76b7b2", "P"),
    "tau"         : ("TAU",         "#e15759", "X"),
    "camp"        : ("CAMP",        "#ffae6b", "p"),
    "camp_base"   : ("CAMP_base",   "#f28e2b", "s"),   # same color as old PredRNN
    "camp_no_cls" : ("CAMP_no_cls", "#d94801", "v"),
    "camp_full"   : ("CAMP_full",   "#7f0000", "*"),   # dark red star — highlighted
}

# Ordered for display (baselines first, CAMP family last).
# predrnn is omitted; camp_base uses the same model and is shown instead.
METHOD_ORDER = [
    "convlstm", "phydnet", "simvp", "mim", "tau",
    "camp", "camp_base", "camp_no_cls", "camp_full",
]


# ===========================================================================
# Helpers
# ===========================================================================

_EPOCH_SUFFIX = ""  # set by main() from --epoch; "" means no suffix


def _saved_dir(work_dir: str, dataset: str, method: str) -> str:
    """Return the 'saved' sub-directory for a given method + dataset.

    If --epoch was passed, directories are named {dataset}_{method}_ep{N};
    otherwise the legacy {dataset}_{method} naming is used.
    """
    run_name = f"{dataset}_{method}{_EPOCH_SUFFIX}"
    return os.path.join(work_dir, run_name, "saved")


def _load_npy(path: str, name: str):
    """Load a .npy file, return None if it does not exist."""
    full = os.path.join(path, name)
    if not os.path.exists(full):
        return None
    return np.load(full, allow_pickle=True)


def _available_methods(work_dir: str, dataset: str) -> list:
    """Return the subset of METHOD_ORDER whose saved/ dir contains preds.npy."""
    available = []
    for m in METHOD_ORDER:
        d = _saved_dir(work_dir, dataset, m)
        if os.path.exists(os.path.join(d, "preds.npy")):
            available.append(m)
    return available


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] frame (C,H,W) to uint8 (H,W,C) for imshow."""
    # Accept (C, H, W) or (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)   # → (H, W, C)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _exgi_colormap(exgi: np.ndarray) -> np.ndarray:
    """
    Map a (H, W) ExGI array to an RGB image for display.
    Clips negatives to 0 (background = dark) and scales positives with
    the Greens colormap — consistent with visualize_predictions.py.
    Returns uint8 (H, W, 3).
    """
    exgi_clipped = np.clip(exgi, 0.0, None)
    exgi_norm = np.clip(exgi_clipped / 0.5, 0.0, 1.0)
    cmap = plt.get_cmap("Greens")
    rgb = (cmap(exgi_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def _poi_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay the POI mask (True = leaf pixel) on top of the frame as a
    semi-transparent green tint.  Returns uint8 (H, W, 3).
    """
    rgb = _clip_uint8(frame).copy()
    green = np.zeros_like(rgb)
    green[..., 1] = 180   # pure green channel
    alpha = 0.45
    rgb[mask] = ((1 - alpha) * rgb[mask] + alpha * green[mask]).astype(np.uint8)
    return rgb


# ===========================================================================
# Figure 2 — Pareto front scatter plot
# ===========================================================================

def figure_pareto(work_dir: str, dataset: str, out_dir: str, dpi: int):
    """
    Scatter plot of SSIM (higher is better, x-axis) vs MAE (lower is better,
    y-axis) and SSIM vs MSE, one marker per model.

    Metrics are loaded from  saved/metrics.npy  which on_test_epoch_end saves
    as a flat array [mae, mse, ssim, psnr, ...].  See
    openstl/methods/base_method.py for the exact order.
    """
    methods  = _available_methods(work_dir, dataset)
    if not methods:
        print("[visualize] No metrics found; skipping Pareto figure.")
        return

    mae_vals, mse_vals, ssim_vals, names, colors, markers = [], [], [], [], [], []

    for m in methods:
        d    = _saved_dir(work_dir, dataset, m)
        mets = _load_npy(d, "metrics.npy")
        if mets is None:
            continue
        # metrics.npy is a 1-D array saved as object — flatten scalars
        mets = np.array(mets, dtype=float).ravel()
        # Index order defined in base_method.py evaluate():
        #   [0]=mae, [1]=mse, [2]=ssim, [3]=psnr
        if len(mets) < 3:
            # SSIM not saved — compute from preds/trues directly
            sys.path.insert(0, _TOOLS_DIR)
            from _analysis_utils import per_sample_ssim as _ps_ssim
            preds_arr = _load_npy(d, "preds.npy")
            trues_arr = _load_npy(d, "trues.npy")
            if preds_arr is None or trues_arr is None:
                continue
            preds_arr = np.clip(preds_arr.astype(np.float32), 0.0, 1.0)
            trues_arr = trues_arr.astype(np.float32)
            ssim_val  = float(np.mean(_ps_ssim(preds_arr, trues_arr)))
        else:
            ssim_val = float(mets[2])
        mae_vals.append(float(mets[0]))
        mse_vals.append(float(mets[1]))
        ssim_vals.append(ssim_val)
        label, color, marker = METHOD_STYLES[m]
        names.append(label)
        colors.append(color)
        markers.append(marker)

    if not mae_vals:
        print("[visualize] No metrics loaded; skipping Pareto figure.")
        return

    try:
        from adjustText import adjust_text
        _has_adjusttext = True
    except ImportError:
        _has_adjusttext = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(
        f"Pareto Front — {dataset.capitalize()} dataset",
        fontweight="bold",
    )

    def _repulsion_offsets(xs, ys, x_scale, y_scale, strength=0.12):
        """Pre-compute per-label offset by pushing each label away from its neighbours."""
        n = len(xs)
        offsets = []
        for i in range(n):
            dx, dy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                ddx = (xs[i] - xs[j]) / x_scale
                ddy = (ys[i] - ys[j]) / y_scale
                dist2 = ddx**2 + ddy**2 + 1e-6
                dx += ddx / dist2
                dy += ddy / dist2
            mag = (dx**2 + dy**2)**0.5 + 1e-9
            offsets.append((dx / mag * strength * x_scale,
                             dy / mag * strength * y_scale))
        return offsets

    for ax, y_vals, y_label, title in [
        (axes[0], mae_vals,  "MAE ↓",  "SSIM vs MAE"),
        (axes[1], mse_vals,  "MSE ↓",  "SSIM vs MSE"),
    ]:
        x_range = max(ssim_vals) - min(ssim_vals) or 1e-6
        y_range = max(y_vals)    - min(y_vals)    or 1e-6

        # Generous padding so pushed labels have room
        x_pad = x_range * 0.35
        y_pad = y_range * 0.30
        ax.set_xlim(min(ssim_vals) - x_pad, max(ssim_vals) + x_pad)
        ax.set_ylim(min(y_vals)    - y_pad, max(y_vals)    + y_pad)

        offsets = _repulsion_offsets(ssim_vals, y_vals, x_range, y_range)

        texts = []
        for (ssim, y, name, color, marker), (ox, oy) in zip(
            zip(ssim_vals, y_vals, names, colors, markers), offsets
        ):
            size = 300 if name == "CAMP_full" else (200 if name.startswith("CAMP") else 140)
            zorder = 5 if name == "CAMP_full" else 3
            ax.scatter(
                ssim, y,
                s=size, c=color, marker=marker,
                zorder=zorder, label=name,
                edgecolors="black", linewidths=0.6,
            )
            # Start text already displaced from the marker
            texts.append(ax.text(ssim + ox, y + oy, name, fontsize=12, color=color))

        if _has_adjusttext:
            adjust_text(
                texts, ax=ax,
                expand_points=(2.0, 2.0),
                expand_text=(1.4, 1.4),
                force_points=(0.6, 0.6),
                force_text=(0.5, 0.5),
                lim=300,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, shrinkA=6),
            )

        ax.set_xlabel("SSIM ↑")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.annotate(
            "← better",
            xy=(0.02, 0.05),
            xycoords="axes fraction",
            fontsize=12, color="gray",
        )

    # Single shared legend
    handles = [
        Line2D(
            [0], [0],
            marker=METHOD_STYLES[m][2],
            color="w",
            markerfacecolor=METHOD_STYLES[m][1],
            markeredgecolor="black",
            markersize=10,
            label=METHOD_STYLES[m][0],
        )
        for m in methods if m in METHOD_STYLES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(methods), 5),
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save_fig(fig, out_dir, "fig2_pareto_front.pdf", dpi)
    print(f"[visualize] Figure 2 (Pareto) saved.")


# ===========================================================================
# Figure 3 — ExGI pipeline
# ===========================================================================

def figure_exgi_pipeline(
    work_dir: str,
    dataset: str,
    out_dir: str,
    sample_idx: int,
    frame_idx: int,
    threshold: float,
    dpi: int,
):
    """
    Three-panel strip for one frame showing the ExGI extraction pipeline:
      Panel 1: Original RGB image
      Panel 2: ExGI map (RdYlGn colourmap; green = high ExGI = leaf)
      Panel 3: POI binary mask overlaid on RGB (green tint = leaf pixels)

    We use the ground-truth frames from the CAMP saved directory (any model
    that has trues.npy works).
    """
    # Find any method that has trues.npy
    frames = None
    for m in METHOD_ORDER:
        d = _saved_dir(work_dir, dataset, m)
        t = _load_npy(d, "trues.npy")
        if t is not None:
            frames = t
            break

    if frames is None:
        print("[visualize] No trues.npy found; skipping ExGI pipeline figure.")
        return

    # frames : [N, T, C, H, W]
    n = frames.shape[0]
    sample_idx = sample_idx % n
    frame = frames[sample_idx, frame_idx]   # (C, H, W) float32

    exgi = compute_exgi(frame)
    mask = compute_poi_mask(exgi, threshold)

    rgb_img   = _clip_uint8(frame)
    exgi_img  = _exgi_colormap(exgi)
    poi_img   = _poi_overlay(frame, mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"ExGI Extraction Pipeline  |  {dataset.capitalize()}, "
        f"sample {sample_idx}, frame {frame_idx}",
        fontsize=13,
    )

    panel_data = [
        (rgb_img,  "RGB Input",        None),
        (exgi_img, "ExGI Map",         "RdYlGn"),
        (poi_img,  f"POI Mask (τ={threshold})", None),
    ]

    for ax, (img, title, _) in zip(axes, panel_data):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Add a colourbar for the ExGI panel (Greens, 0–0.5, matches _exgi_colormap)
    sm = plt.cm.ScalarMappable(
        cmap="Greens",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("ExGI value", fontsize=9)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig3_exgi_pipeline.pdf", dpi)
    print(f"[visualize] Figure 3 (ExGI pipeline) saved.")


# ===========================================================================
# Figures 4 / 5 — Qualitative prediction grids
# ===========================================================================

def figure_qualitative(
    work_dir: str,
    dataset: str,
    out_dir: str,
    sample_idx: int,
    threshold: float,
    dpi: int,
    n_frames: int = 5,
):
    """
    Grid showing predicted frames and their ExGI maps side by side.

    Layout:
        Rows: Ground Truth | ConvLSTM | PredRNN | PhyDNet | SimVP | MIM | TAU | CAMP
        Column groups (each group = n_frames columns wide):
            Left group  : RGB predicted / ground-truth frames
            Right group : Corresponding ExGI maps

    We load preds.npy and trues.npy for each method.
    """
    methods = _available_methods(work_dir, dataset)
    if not methods:
        print("[visualize] No prediction files; skipping qualitative figure.")
        return

    # ------------------------------------------------------------------
    # Load data for each available method
    # ------------------------------------------------------------------
    all_preds = {}
    all_trues = {}

    for m in methods:
        d = _saved_dir(work_dir, dataset, m)
        p = _load_npy(d, "preds.npy")
        t = _load_npy(d, "trues.npy")
        if p is not None and t is not None:
            all_preds[m] = p
            all_trues[m] = t

    if not all_preds:
        print("[visualize] No preds/trues loaded; skipping qualitative figure.")
        return

    # Determine safe sample index
    first_preds = next(iter(all_preds.values()))
    n_samples   = first_preds.shape[0]
    sample_idx  = sample_idx % n_samples

    # ------------------------------------------------------------------
    # Build row list: ("row_label", frames_array)
    # 'frames_array' : (T, C, H, W) for this sample
    # First row = Ground Truth (from any method's trues.npy)
    # ------------------------------------------------------------------
    rows = []
    first_m = next(iter(all_trues))
    rows.append(("Ground\nTruth", all_trues[first_m][sample_idx]))
    for m in [m for m in METHOD_ORDER if m in all_preds]:
        rows.append((METHOD_STYLES[m][0], all_preds[m][sample_idx]))

    # Each row spans n_frames (RGB) + n_frames (ExGI) columns
    n_cols = n_frames * 2
    n_rows = len(rows)

    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 1.8))
    fig.suptitle(
        f"Qualitative Predictions — {dataset.capitalize()}  (sample {sample_idx})",
        fontweight="bold", y=1.00,
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.05, wspace=0.02,
        left=0.09, right=0.99, top=0.93, bottom=0.01,
    )

    # Column header annotations (done once at the top)
    header_ax = fig.add_axes([0.09, 0.94, 0.90, 0.025])
    header_ax.axis("off")
    half = 0.5 / n_cols
    for i in range(n_frames):
        header_ax.text(
            (i + 0.5) / n_cols, 0.5,
            f"Frame {i + 1}",
            ha="center", va="center", fontsize=11, color="#333333",
        )
        header_ax.text(
            (n_frames + i + 0.5) / n_cols, 0.5,
            f"ExGI {i + 1}",
            ha="center", va="center", fontsize=11, color="#007700",
        )
    # Vertical separator line between RGB and ExGI halves
    header_ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")

    for row_idx, (label, frames) in enumerate(rows):
        # Highlight CAMP row with a slightly different background
        is_camp = (label == "CAMP")

        for col_idx in range(n_frames):
            t_idx = min(col_idx, frames.shape[0] - 1)
            frame = frames[t_idx]          # (C, H, W)
            exgi  = compute_exgi(frame)
            mask  = compute_poi_mask(exgi, threshold)

            # ---- RGB panel -----------------------------------------------
            ax_rgb = fig.add_subplot(gs[row_idx, col_idx])
            ax_rgb.imshow(_clip_uint8(frame))
            ax_rgb.axis("off")
            if is_camp:
                for spine in ax_rgb.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(1.5)

            # ---- ExGI panel -----------------------------------------------
            ax_ex = fig.add_subplot(gs[row_idx, n_frames + col_idx])
            ax_ex.imshow(_exgi_colormap(exgi))
            ax_ex.axis("off")
            if is_camp:
                for spine in ax_ex.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(1.5)

        # Row label on the left
        # Attach to the first cell's position
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.set_ylabel(
            label,
            fontsize=12,
            labelpad=4,
            rotation=0,
            ha="right",
            va="center",
            fontweight="bold" if is_camp else "normal",
            color="red" if is_camp else "black",
        )
        ax0.axis("off")

    # Vertical separator between RGB and ExGI halves
    # Draw as a figure-level line
    left_frac  = 0.09
    right_frac = 0.99
    total_width = right_frac - left_frac
    sep_x = left_frac + total_width * 0.5
    line = matplotlib.lines.Line2D(
        [sep_x, sep_x], [0.01, 0.93],
        transform=fig.transFigure,
        color="gray", linewidth=1.0, linestyle="--",
    )
    fig.add_artist(line)

    # Section labels
    fig.text(left_frac + total_width * 0.25, 0.97,
             "RGB Predictions", ha="center", fontsize=13, color="#222222")
    fig.text(left_frac + total_width * 0.75, 0.97,
             "ExGI Maps", ha="center", fontsize=13, color="#007700")

    fname = f"fig4_qualitative_{dataset}.pdf"
    _save_fig(fig, out_dir, fname, dpi)
    print(f"[visualize] Qualitative grid saved.")


# ===========================================================================
# Figures 6 / 7 — POI-based bipolar regression curves
# ===========================================================================

def figure_poi_curves(
    work_dir: str,
    dataset: str,
    out_dir: str,
    threshold: float,
    dpi: int,
):
    """
    Three-panel figure showing the mean (±std) POI growth curves across the
    test set for every available model.

      Panel 1: Total POI count over time (g(t) — leaf area proxy)
      Panel 2: New POIs per frame (N_new — growth rate)
      Panel 3: Missing POIs per frame (N_miss — prediction error / shrinkage)

    All curves are averaged over the test set.  We show Ground Truth once
    (from any model's trues.npy) as a thick black reference line.
    """
    methods = _available_methods(work_dir, dataset)
    if not methods:
        print("[visualize] No prediction files; skipping POI-curves figure.")
        return

    # ---------------------------------------------------------------
    # Compute mean curves for every method and for Ground Truth
    # ---------------------------------------------------------------
    gt_loaded   = False
    gt_poi_mean = None
    gt_new_mean = None
    gt_miss_mean= None

    method_data = {}   # method → {"poi": ..., "new": ..., "miss": ...}

    for m in methods:
        d     = _saved_dir(work_dir, dataset, m)
        preds = _load_npy(d, "preds.npy")
        trues = _load_npy(d, "trues.npy")
        if preds is None or trues is None:
            continue

        N = preds.shape[0]

        # Ground truth curves (compute once)
        if not gt_loaded:
            gt_poi_all, gt_new_all, gt_miss_all = _batch_curves(trues, threshold)
            gt_poi_mean  = np.mean(gt_poi_all,  axis=0)
            gt_new_mean  = np.mean(gt_new_all,  axis=0)
            gt_miss_mean = np.mean(gt_miss_all, axis=0)
            gt_loaded = True

        # Predicted curves
        p_poi_all, p_new_all, p_miss_all = _batch_curves(preds, threshold)
        method_data[m] = {
            "poi_mean"  : np.mean(p_poi_all,  axis=0),
            "poi_std"   : np.std(p_poi_all,   axis=0),
            "new_mean"  : np.mean(p_new_all,  axis=0),
            "new_std"   : np.std(p_new_all,   axis=0),
            "miss_mean" : np.mean(p_miss_all, axis=0),
            "miss_std"  : np.std(p_miss_all,  axis=0),
        }

    if not method_data:
        print("[visualize] No data for POI curves; skipping.")
        return

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"POI-based Bipolar Regression Curves — {dataset.capitalize()}",
        fontweight="bold",
    )

    panel_keys = [
        ("poi_mean",  "poi_std",  "Total POI Count",       "N_POI",    gt_poi_mean),
        ("new_mean",  "new_std",  "New POIs (growth)",     "N_new",    gt_new_mean),
        ("miss_mean", "miss_std", "Missing POIs (error)",  "N_miss",   gt_miss_mean),
    ]

    for ax, (mean_key, std_key, title, ylabel, gt_curve) in zip(axes, panel_keys):
        # Ground truth reference
        x_gt = np.arange(len(gt_curve))
        ax.plot(
            x_gt, gt_curve,
            color="black", linewidth=2.0, linestyle="-",
            label="Ground Truth", zorder=4,
        )

        # Each model
        for m in [m for m in METHOD_ORDER if m in method_data]:
            d     = method_data[m]
            mean  = d[mean_key]
            std   = d[std_key]
            label_, color, marker = METHOD_STYLES[m]
            lw    = 2.2 if m == "camp" else 1.4
            x_m   = np.arange(len(mean))

            ax.plot(
                x_m, mean,
                color=color, linewidth=lw,
                linestyle="-", marker=marker,
                markersize=5 if m == "camp" else 3.5,
                label=label_, zorder=3,
            )
            # Shaded ±1 std band
            ax.fill_between(
                x_m,
                np.maximum(mean - std, 0),
                mean + std,
                color=color, alpha=0.12, zorder=2,
            )

        ax.set_title(title)
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = f"fig6_poi_curves_{dataset}.pdf"
    _save_fig(fig, out_dir, fname, dpi)
    print(f"[visualize] POI growth-curve figure saved.")


# ===========================================================================
# Internal helpers
# ===========================================================================

def _batch_curves(frames: np.ndarray, threshold: float):
    """
    Compute poi_counts, n_new, n_miss for every sample in *frames*.

    Args:
        frames : [N, T, C, H, W] float32 array.
        threshold : ExGI threshold τ.

    Returns:
        poi_array  : (N, T)   int32 — total POI count per frame.
        new_array  : (N, T-1) int32 — new POIs per transition.
        miss_array : (N, T-1) int32 — missing POIs per transition.
    """
    N, T = frames.shape[:2]
    poi_list, new_list, miss_list = [], [], []

    for i in range(N):
        counts, n_new, n_miss = poi_growth_curve(frames[i], threshold)
        poi_list.append(counts)
        new_list.append(n_new)
        miss_list.append(n_miss)

    # Pad to equal length (in case some samples have fewer frames)
    T_poi  = max(len(c) for c in poi_list)
    T_diff = max(len(c) for c in new_list)

    poi_array  = np.zeros((N, T_poi),  dtype=np.int32)
    new_array  = np.zeros((N, T_diff), dtype=np.int32)
    miss_array = np.zeros((N, T_diff), dtype=np.int32)

    for i in range(N):
        poi_array[i,  :len(poi_list[i])]  = poi_list[i]
        new_array[i,  :len(new_list[i])]  = new_list[i]
        miss_array[i, :len(miss_list[i])] = miss_list[i]

    return poi_array, new_array, miss_array


def _save_fig(fig, out_dir: str, filename: str, dpi: int):
    """Save figure as both PDF (vector) and PNG (raster, capped at 100 dpi)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    pdf_path = os.path.join(out_dir, filename)
    png_path = os.path.join(out_dir, base + ".png")

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=min(dpi, 100), bbox_inches="tight")
    plt.close(fig)
    print(f"  → {pdf_path}")
    print(f"  → {png_path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce CAMP paper figures from saved prediction artefacts."
    )
    parser.add_argument(
        "--work_dir", type=str, default="work_dirs",
        help="Root directory containing {dataset}_{method}/saved/ sub-dirs.",
    )
    parser.add_argument(
        "--dataset", type=str, default="komatsuna",
        choices=["komatsuna", "arabidopsis"],
        help="Dataset name (used to find work_dirs/{dataset}_{method}/).",
    )
    parser.add_argument(
        "--figures", nargs="+",
        default=["pareto", "exgi_pipeline", "qualitative", "poi_curves"],
        choices=["pareto", "exgi_pipeline", "qualitative", "poi_curves"],
        help="Which figure types to generate (default: all).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: figures/{dataset}/).",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0,
        help="Test-set sample index for qualitative / ExGI-pipeline figures.",
    )
    parser.add_argument(
        "--frame_idx", type=int, default=2,
        help="Frame index within the sample for the ExGI pipeline figure.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="ExGI threshold τ for POI extraction (default: 0.1).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI for raster outputs (default: 150).",
    )
    parser.add_argument(
        "--n_frames", type=int, default=5,
        help="Number of future frames to show in qualitative grid (default: 5).",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="If set, look in work_dirs/{dataset}_{method}_ep{epoch}/ instead of "
             "the legacy work_dirs/{dataset}_{method}/ naming.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure the directory-name suffix used by _saved_dir.
    global _EPOCH_SUFFIX
    _EPOCH_SUFFIX = f"_ep{args.epoch}" if args.epoch is not None else ""

    out_dir = args.out_dir or os.path.join(
        "figures", args.dataset + (f"_ep{args.epoch}" if args.epoch else ""))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  CAMP Paper — Figure Visualisation")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Work dir : {args.work_dir}")
    print(f"  Out dir  : {out_dir}")
    print("=" * 60)

    if "pareto" in args.figures:
        print("\n[Figure 2] Pareto front scatter ...")
        figure_pareto(args.work_dir, args.dataset, out_dir, args.dpi)

    if "exgi_pipeline" in args.figures:
        print("\n[Figure 3] ExGI extraction pipeline ...")
        figure_exgi_pipeline(
            args.work_dir, args.dataset, out_dir,
            sample_idx=args.sample_idx,
            frame_idx=args.frame_idx,
            threshold=args.threshold,
            dpi=args.dpi,
        )

    if "qualitative" in args.figures:
        print("\n[Figures 4/5] Qualitative prediction grid ...")
        figure_qualitative(
            args.work_dir, args.dataset, out_dir,
            sample_idx=args.sample_idx,
            threshold=args.threshold,
            dpi=args.dpi,
            n_frames=args.n_frames,
        )

    if "poi_curves" in args.figures:
        print("\n[Figures 6/7] POI-based bipolar regression curves ...")
        figure_poi_curves(
            args.work_dir, args.dataset, out_dir,
            threshold=args.threshold,
            dpi=args.dpi,
        )

    print("\n" + "=" * 60)
    print(f"  All figures saved to  {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
