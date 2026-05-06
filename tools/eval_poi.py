"""
eval_poi.py
-----------
Pixel-of-Interest (POI) based bipolar regression evaluation for plant growth
prediction, as described in Section 2.4.2 of the CAMP paper.

This script is run *after* training and testing.  It loads the .npy files
saved by the CAMP (or any other) model's test run and computes:

  1. ExGI (Excess Green Index) map for each predicted and ground-truth frame.
  2. POI mask: pixels with ExGI above threshold τ are considered leaf pixels.
  3. POI count curve g(t) over time.
  4. New POIs  (N_new):     leaf pixels that appeared since the previous frame.
  5. Missing POIs (N_miss): leaf pixels that disappeared since the previous frame.
  6. Bipolar regression curves and their MAE vs ground truth.
  7. Optional visualisation of growth curves.

Usage:
    python tools/eval_poi.py \\
        --pred_dir  work_dirs/camp_komatsuna/saved \\
        --threshold 0.1 \\
        --plot

Arguments:
    --pred_dir   : Directory containing preds.npy and trues.npy
                   (output of on_test_epoch_end).
    --threshold  : ExGI threshold τ for POI extraction (default: 0.1).
    --plot       : If set, saves growth-curve plots to pred_dir/poi_plots/.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

# Use Times New Roman throughout all plots, matching the paper's typography.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],  # fallback if TNR missing
})


# ===========================================================================
# ExGI and POI utilities
# ===========================================================================

def compute_exgi(frame: np.ndarray) -> np.ndarray:
    """
    Compute the Excess Green Index for a single RGB frame.

    ExGI(i,j) = 2·G(i,j) − R(i,j) − B(i,j)

    Args:
        frame : float32 array of shape (C, H, W) or (H, W, C), values in [0,1].

    Returns:
        exgi  : float32 array of shape (H, W), values in [−1, 2].
    """
    # Accept both channel-first (C,H,W) and channel-last (H,W,C)
    if frame.ndim == 3 and frame.shape[0] == 3:
        R, G, B = frame[0], frame[1], frame[2]
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    exgi = 2.0 * G - R - B
    return exgi.astype(np.float32)


def compute_poi_mask(exgi: np.ndarray, threshold: float) -> np.ndarray:
    """
    Threshold the ExGI map to produce a binary POI (leaf-pixel) mask.

    POI(i,j) = 1  if ExGI(i,j) > τ
              = 0  otherwise

    Args:
        exgi      : (H, W) ExGI map.
        threshold : scalar τ (default 0.1 in the paper).

    Returns:
        mask : boolean (H, W) array; True = leaf pixel.
    """
    return exgi > threshold


def count_poi(mask: np.ndarray) -> int:
    """Return the total number of POI pixels in *mask*."""
    return int(mask.sum())


def compute_new_missing_poi(mask_prev: np.ndarray,
                             mask_curr: np.ndarray):
    """
    Compare two consecutive POI masks to find new and missing pixels.

    New POIs     = pixels that are POI in curr but NOT in prev  (growth).
    Missing POIs = pixels that were POI in prev but NOT in curr (shrinkage/
                   prediction error).

    Args:
        mask_prev : boolean (H, W) mask for the previous frame.
        mask_curr : boolean (H, W) mask for the current frame.

    Returns:
        n_new     : int — number of newly appeared leaf pixels.
        n_missing : int — number of disappeared leaf pixels.
    """
    new_poi     = mask_curr & ~mask_prev   # appeared
    missing_poi = mask_prev & ~mask_curr   # disappeared
    return int(new_poi.sum()), int(missing_poi.sum())


# ===========================================================================
# Per-sample growth curves
# ===========================================================================

def poi_growth_curve(frames: np.ndarray, threshold: float):
    """
    Compute the POI count curve and bipolar regression curves for one sample.

    Args:
        frames    : float32 array [T, C, H, W] — sequence of (predicted or
                    ground-truth) frames.
        threshold : ExGI threshold τ.

    Returns:
        poi_counts : list[int] of length T — N_POI at each time step.
        n_new_list : list[int] of length T-1 — new POIs per step.
        n_miss_list: list[int] of length T-1 — missing POIs per step.
    """
    T = frames.shape[0]

    # Compute ExGI and POI masks for every frame
    masks = [compute_poi_mask(compute_exgi(frames[t]), threshold) for t in range(T)]
    poi_counts = [count_poi(m) for m in masks]

    # Compute new and missing POIs between consecutive frames
    n_new_list  = []
    n_miss_list = []
    for t in range(1, T):
        n_new, n_miss = compute_new_missing_poi(masks[t - 1], masks[t])
        n_new_list.append(n_new)
        n_miss_list.append(n_miss)

    return poi_counts, n_new_list, n_miss_list


# ===========================================================================
# Batch evaluation
# ===========================================================================

def evaluate_poi(preds: np.ndarray, trues: np.ndarray, threshold: float = 0.1):
    """
    Run full POI-based bipolar regression evaluation over a dataset.

    Args:
        preds     : float32 [N, T, C, H, W] — predicted frames (all samples).
        trues     : float32 [N, T, C, H, W] — ground-truth frames.
        threshold : ExGI threshold τ.

    Returns:
        dict with keys:
            'poi_mae'         : float — mean |g_pred(t) - g_true(t)| over all
                                        samples and time steps.
            'new_mae'         : float — MAE of N_new curves.
            'miss_mae'        : float — MAE of N_miss curves.
            'pred_poi_counts' : list[list[int]] — per-sample POI count curves
                                (predicted).
            'true_poi_counts' : list[list[int]] — per-sample POI count curves
                                (ground-truth).
            'pred_new'        : list[list[int]] — per-sample N_new (predicted).
            'true_new'        : list[list[int]] — per-sample N_new (GT).
            'pred_miss'       : list[list[int]] — per-sample N_miss (predicted).
            'true_miss'       : list[list[int]] — per-sample N_miss (GT).
    """
    N = preds.shape[0]

    pred_poi_counts, true_poi_counts = [], []
    pred_new, true_new               = [], []
    pred_miss, true_miss             = [], []

    for i in range(N):
        # Predicted growth curves
        p_counts, p_new, p_miss = poi_growth_curve(preds[i], threshold)
        # Ground-truth growth curves
        t_counts, t_new, t_miss = poi_growth_curve(trues[i], threshold)

        pred_poi_counts.append(p_counts)
        true_poi_counts.append(t_counts)
        pred_new.append(p_new)
        true_new.append(t_new)
        pred_miss.append(p_miss)
        true_miss.append(t_miss)

    # Compute MAEs as scalars
    poi_mae  = _curve_mae(pred_poi_counts, true_poi_counts)
    new_mae  = _curve_mae(pred_new,         true_new)
    miss_mae = _curve_mae(pred_miss,        true_miss)

    return {
        "poi_mae"         : poi_mae,
        "new_mae"         : new_mae,
        "miss_mae"        : miss_mae,
        "pred_poi_counts" : pred_poi_counts,
        "true_poi_counts" : true_poi_counts,
        "pred_new"        : pred_new,
        "true_new"        : true_new,
        "pred_miss"       : pred_miss,
        "true_miss"       : true_miss,
    }


def _curve_mae(pred_curves, true_curves):
    """
    Compute mean absolute error between two lists of curves.
    Each curve is a list of integers; curves may have different lengths so
    we average at the element level.
    """
    all_errors = []
    for p, t in zip(pred_curves, true_curves):
        # Align lengths (take the shorter one)
        n = min(len(p), len(t))
        for i in range(n):
            all_errors.append(abs(p[i] - t[i]))
    return float(np.mean(all_errors)) if all_errors else 0.0


# ===========================================================================
# Visualisation
# ===========================================================================

def _format_model_name(model_name: str) -> str:
    """Convert a folder name into a clean display name for figure titles.

    Examples
    --------
    "komatsuna_convlstm_ep200"  →  "ConvLSTM  |  KOMATSUNA  |  200 epochs"
    "arabidopsis_camp_ep50"     →  "CAMP  |  Arabidopsis  |  50 epochs"
    ""                          →  "Unknown model"
    """
    if not model_name:
        return "Unknown model"

    # Expected format: {dataset}_{method}_ep{epochs}
    # e.g.  komatsuna_convlstm_ep200
    parts = model_name.split("_")

    # Extract epoch count (last part, starts with "ep")
    epoch_str = ""
    if parts and parts[-1].startswith("ep") and parts[-1][2:].isdigit():
        epoch_str = f"{parts[-1][2:]} epochs"
        parts = parts[:-1]

    # First part is dataset name
    dataset = parts[0].capitalize() if parts else ""

    # Remaining parts form the method name — map to clean names
    method_raw = "_".join(parts[1:]).lower().replace("plant_", "")
    method_map = {
        "convlstm" : "ConvLSTM",
        "predrnn"  : "PredRNN",
        "phydnet"  : "PhyDNet",
        "simvp"    : "SimVP",
        "mim"      : "MIM",
        "tau"      : "TAU",
        "camp"     : "CAMP",
    }
    method = method_map.get(method_raw, method_raw.upper())

    parts_display = [p for p in [method, dataset, epoch_str] if p]
    return "  |  ".join(parts_display)


def plot_bipolar_regression(
    pred_new,  true_new,
    pred_miss, true_miss,
    out_dir: str,
    model_name: str = "",
):
    """
    Save the bipolar regression scatter plot as described in the CAMP paper.

    Two regression lines are fitted and plotted on a single axes:
      - Positive pole (blue):  N_new predicted  vs N_new ground-truth
      - Negative pole (red):   N_miss predicted vs N_miss ground-truth

    A perfect model would have both lines coinciding with the y = x diagonal.
    """
    import numpy as np
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)
    display_name = _format_model_name(model_name)

    # Flatten all (pred, true) pairs across samples and time steps
    all_new_true  = np.array([v for curve in true_new  for v in curve], dtype=float)
    all_new_pred  = np.array([v for curve in pred_new  for v in curve], dtype=float)
    all_miss_true = np.array([v for curve in true_miss for v in curve], dtype=float)
    all_miss_pred = np.array([v for curve in pred_miss for v in curve], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---- Positive pole: N_new -----------------------------------------------
    ax.scatter(all_new_true, all_new_pred,
               alpha=0.35, s=18, color="steelblue", label="New POIs (growth)")
    if len(all_new_true) > 1:
        slope, intercept, r, *_ = stats.linregress(all_new_true, all_new_pred)
        x_line = np.linspace(all_new_true.min(), all_new_true.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color="steelblue", linewidth=2,
                label=f"Regression (new)  R²={r**2:.3f}")

    # ---- Negative pole: N_miss -----------------------------------------------
    ax.scatter(all_miss_true, all_miss_pred,
               alpha=0.35, s=18, color="tomato", label="Missing POIs (shrinkage)")
    if len(all_miss_true) > 1:
        slope, intercept, r, *_ = stats.linregress(all_miss_true, all_miss_pred)
        x_line = np.linspace(all_miss_true.min(), all_miss_true.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color="tomato", linewidth=2,
                label=f"Regression (miss)  R²={r**2:.3f}")

    # ---- y = x identity line ------------------------------------------------
    all_vals = np.concatenate([all_new_true, all_new_pred,
                                all_miss_true, all_miss_pred])
    lo, hi = all_vals.min(), all_vals.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x (ideal)")

    ax.set_xlabel("Ground Truth", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"{display_name}\nBipolar Regression: Predicted vs Ground Truth",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prefix = f"{model_name}_" if model_name else ""
    save_path = os.path.join(out_dir, f"{prefix}bipolar_regression.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_poi] Saved bipolar regression plot to {save_path}")

    # Save raw data so the plot can be regenerated later without retraining.
    # Load with: d = np.load(path); d['all_new_true'], d['all_new_pred'], ...
    data_path = save_path.replace(".png", "_data.npz")
    np.savez_compressed(
        data_path,
        all_new_true=all_new_true,
        all_new_pred=all_new_pred,
        all_miss_true=all_miss_true,
        all_miss_pred=all_miss_pred,
        model_name=np.array(model_name),
    )
    print(f"[eval_poi] Saved bipolar regression data to {data_path}")


def plot_growth_curves(
    pred_poi, true_poi,
    pred_new,  true_new,
    pred_miss, true_miss,
    sample_indices,
    out_dir: str,
    model_name: str = "",
):
    """
    Save growth-curve plots for a selection of samples.

    Each figure has three subplots:
      (1) Total POI count over time.
      (2) New POIs (growth) over time.
      (3) Missing POIs (shrinkage / error) over time.

    Args:
        pred_poi / true_poi   : lists of per-sample count curves.
        pred_new  / true_new  : lists of per-sample new-POI curves.
        pred_miss / true_miss : lists of per-sample missing-POI curves.
        sample_indices        : indices of samples to plot.
        out_dir               : directory where PNG files are saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Format model name for display — e.g. "komatsuna_convlstm_ep200" → "ConvLSTM (200 epochs)"
    display_name = _format_model_name(model_name)

    for idx in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        fig.suptitle(f"{display_name}\nSample {idx} — POI-based Bipolar Regression",
                     fontsize=13, fontweight="bold")

        # Paper color convention: orange = predicted, blue = ground truth
        PRED_COLOR = '#ff7f0e'   # orange  (matches paper Figure 6 & 7)
        GT_COLOR   = '#4c72b0'   # steel-blue (matches paper Figure 6 & 7)

        # ---- (1) Total POI count curve -----------------------------------
        ax = axes[0]
        t_total = range(len(pred_poi[idx]))
        ax.plot(t_total, pred_poi[idx], '-o', color=PRED_COLOR, label="Predicted",    markersize=4)
        ax.plot(t_total, true_poi[idx], '-o', color=GT_COLOR,   label="Ground Truth", markersize=4)
        ax.set_title("POI Count (leaf area proxy)", fontsize=11)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Number of POIs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- (2) New POIs ------------------------------------------------
        ax = axes[1]
        t_diff = range(len(pred_new[idx]))
        ax.plot(t_diff, pred_new[idx], '-o', color=PRED_COLOR, label="Predicted",    markersize=4)
        ax.plot(t_diff, true_new[idx], '-o', color=GT_COLOR,   label="Ground Truth", markersize=4)
        ax.set_title("New POIs per Frame (growth)", fontsize=11)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("N_new")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- (3) Missing POIs -------------------------------------------
        ax = axes[2]
        ax.plot(t_diff, pred_miss[idx], '-o', color=PRED_COLOR, label="Predicted",    markersize=4)
        ax.plot(t_diff, true_miss[idx], '-o', color=GT_COLOR,   label="Ground Truth", markersize=4)
        ax.set_title("Missing POIs per Frame (shrinkage / error)", fontsize=11)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("N_miss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        prefix = f"{model_name}_" if model_name else ""
        save_path = os.path.join(out_dir, f"{prefix}poi_sample_{idx:04d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"[eval_poi] Saved {len(sample_indices)} growth-curve plots to {out_dir}")

    # Save all curve data once so every per-sample plot can be regenerated.
    # Arrays are object arrays (ragged curves); use allow_pickle=True on load.
    # Load with: d = np.load(path, allow_pickle=True)
    prefix = f"{model_name}_" if model_name else ""
    data_path = os.path.join(out_dir, f"{prefix}growth_curves_data.npz")
    np.savez_compressed(
        data_path,
        pred_poi      = np.array(pred_poi,  dtype=object),
        true_poi      = np.array(true_poi,  dtype=object),
        pred_new      = np.array(pred_new,  dtype=object),
        true_new      = np.array(true_new,  dtype=object),
        pred_miss     = np.array(pred_miss, dtype=object),
        true_miss     = np.array(true_miss, dtype=object),
        sample_indices= np.array(sample_indices),
        model_name    = np.array(model_name),
    )
    print(f"[eval_poi] Saved growth curve data to {data_path}")


# ===========================================================================
# Command-line entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="POI-based bipolar regression evaluation for CAMP"
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing preds.npy and trues.npy"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="ExGI threshold τ for POI extraction (default: 0.1)"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Save growth-curve plots for up to 10 test samples"
    )
    parser.add_argument(
        "--n_plot", type=int, default=10,
        help="Number of sample plots to generate when --plot is set"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Derive model name from pred_dir path:
    # e.g. "work_dirs/komatsuna_convlstm/saved" → "komatsuna_convlstm"
    model_name = os.path.basename(os.path.dirname(args.pred_dir))

    # ---- Load saved predictions and ground-truth -------------------------
    preds_path = os.path.join(args.pred_dir, "preds.npy")
    trues_path = os.path.join(args.pred_dir, "trues.npy")

    if not os.path.exists(preds_path) or not os.path.exists(trues_path):
        raise FileNotFoundError(
            f"Could not find preds.npy / trues.npy in {args.pred_dir}.\n"
            "Run model testing first to generate these files."
        )

    print(f"[eval_poi] Loading predictions from {preds_path}")
    preds = np.load(preds_path)   # [N, T, C, H, W]
    trues = np.load(trues_path)   # [N, T, C, H, W]

    print(f"[eval_poi] preds shape: {preds.shape}")
    print(f"[eval_poi] trues shape: {trues.shape}")
    print(f"[eval_poi] ExGI threshold τ = {args.threshold}")

    # ---- Run POI evaluation ----------------------------------------------
    results = evaluate_poi(preds, trues, threshold=args.threshold)

    # ---- Report results --------------------------------------------------
    print("\n" + "=" * 60)
    print("POI-based Bipolar Regression Results")
    print("=" * 60)
    print(f"  POI count MAE  (total leaf area): {results['poi_mae']:.2f}")
    print(f"  New POI   MAE  (growth curve)   : {results['new_mae']:.2f}")
    print(f"  Missing POI MAE (error/shrink)   : {results['miss_mae']:.2f}")
    print("=" * 60 + "\n")

    # ---- Optional plots --------------------------------------------------
    if args.plot:
        n = min(args.n_plot, len(results["pred_poi_counts"]))
        sample_indices = list(range(n))
        plot_dir = os.path.join(args.pred_dir, "poi_plots")

        # Bipolar regression scatter plot (one figure per model)
        plot_bipolar_regression(
            results["pred_new"],  results["true_new"],
            results["pred_miss"], results["true_miss"],
            out_dir=plot_dir,
            model_name=model_name,
        )

        plot_growth_curves(
            results["pred_poi_counts"], results["true_poi_counts"],
            results["pred_new"],        results["true_new"],
            results["pred_miss"],       results["true_miss"],
            sample_indices,
            out_dir=plot_dir,
            model_name=model_name,
        )

    # ---- Save numeric results -------------------------------------------
    results_path = os.path.join(args.pred_dir, "poi_results.npy")
    np.save(results_path, {
        "poi_mae" : results["poi_mae"],
        "new_mae" : results["new_mae"],
        "miss_mae": results["miss_mae"],
    })
    print(f"[eval_poi] Numeric results saved to {results_path}")


if __name__ == "__main__":
    main()
