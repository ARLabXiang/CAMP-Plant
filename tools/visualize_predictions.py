"""
visualize_predictions.py
------------------------
Reproduces Figures 4–7 from the CAMP paper.

  Figure 4/5 : Qualitative RGB + ExGI grid
               Rows = Ground Truth, then each model
               Columns = 5 predicted frames (left: RGB, right: ExGI)

  Figure 6   : POI count curves — all models overlaid on one subplot per
               sample, so the reader can compare all models vs GT at once.

  Figure 7   : POI grid — rows = models, columns = samples.
               Each cell shows that model's POI curve vs GT.

Usage
-----
    python3 tools/visualize_predictions.py \\
        --work_dir  work_dirs \\
        --dataset   komatsuna \\
        --epoch     200 \\
        --n_samples 5

Outputs are written to:
    figures/{dataset}_ep{epoch}/
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Paper-ready typography
matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        15,
    'axes.titlesize':   17,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'figure.titlesize': 19,
})

# Add project root to Python path so the script can be run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Constants
# ===========================================================================

# Display order matches the paper.
# predrnn omitted — camp_base is the same architecture and is shown instead.
METHODS = ['convlstm', 'phydnet', 'simvp', 'mim', 'tau',
           'camp', 'camp_base', 'camp_no_cls', 'camp_full']

METHOD_DISPLAY = {
    'convlstm':    'ConvLSTM',
    'phydnet':     'PhyDNet',
    'simvp':       'SimVP',
    'mim':         'MIM',
    'tau':         'TAU',
    'camp':        'CAMP',
    'camp_base':   'CAMP_base',
    'camp_no_cls': 'CAMP_no_cls',
    'camp_full':   'CAMP_full',
}

# One distinct color per model for overlaid plots
METHOD_COLORS = {
    # Blue (#4c72b0) is reserved for Ground Truth — no model uses blue here.
    # ConvLSTM moved from blue to teal so it doesn't clash with GT.
    'convlstm':    '#00b4d8',   # teal
    'predrnn':     '#ff7f0e',   # orange
    'phydnet':     '#2ca02c',   # green
    'simvp':       '#d62728',   # red
    'mim':         '#9467bd',   # purple
    'tau':         '#8c564b',   # brown
    'camp':        '#e377c2',   # pink  (full CAMP — highlighted)
    # Ablation variants — shades of magenta/maroon to group them visually
    'camp_base':   '#f28e2b',   # orange (matches old PredRNN color for continuity)
    'camp_no_cls': '#d4548a',   # deep pink
    'camp_full':   '#8b0000',   # dark red
}

GT_COLOR     = '#4c72b0'   # steel-blue dashed = ground truth (matches paper Figures 6 & 7)
GT_LINEWIDTH = 2.5


# ===========================================================================
# Shared utilities  (inline copies — no dependency on eval_poi.py)
# ===========================================================================

def _exgi(frame: np.ndarray) -> np.ndarray:
    """ExGI = 2G - R - B for a CHW float32 frame."""
    R, G, B = frame[0], frame[1], frame[2]
    return (2.0 * G - R - B).astype(np.float32)


def _poi_mask(exgi: np.ndarray, threshold: float) -> np.ndarray:
    return exgi > threshold


def _poi_count(mask: np.ndarray) -> int:
    return int(mask.sum())


def _poi_curve(frames: np.ndarray, threshold: float):
    """Return list of POI counts for a [T, C, H, W] sequence."""
    return [_poi_count(_poi_mask(_exgi(frames[t]), threshold))
            for t in range(frames.shape[0])]


# ===========================================================================
# Data loading
# ===========================================================================

def load_model_data(work_dir: str, dataset: str, epoch: int) -> dict:
    """
    Scan work_dir for completed runs named {dataset}_{method}_ep{epoch}.

    Returns
    -------
    dict  method_short -> {'preds': ndarray [N,T,C,H,W],
                           'trues': ndarray [N,T,C,H,W]}
    """
    data = {}
    for method in METHODS:
        folder = os.path.join(
            work_dir, f"{dataset}_{method}_ep{epoch}", "saved"
        )
        preds_path = os.path.join(folder, "preds.npy")
        trues_path = os.path.join(folder, "trues.npy")

        if os.path.exists(preds_path) and os.path.exists(trues_path):
            data[method] = {
                'preds': np.load(preds_path),
                'trues': np.load(trues_path),
            }
            print(f"  [ok] {METHOD_DISPLAY[method]:10s}  "
                  f"preds {data[method]['preds'].shape}")
        else:
            print(f"  [--] {METHOD_DISPLAY[method]:10s}  not found, skipping")
    return data


def get_trues(data: dict) -> np.ndarray:
    """All models share the same test set; return trues from any one of them."""
    for v in data.values():
        return v['trues']
    raise ValueError("No model data loaded.")


def _save_both(path_png: str, dpi: int = 100):
    """Save the current figure as both PNG (raster) and PDF (vector for Illustrator)."""
    os.makedirs(os.path.dirname(os.path.abspath(path_png)), exist_ok=True)
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight')
    pdf_path = os.path.splitext(path_png)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches='tight')


# ===========================================================================
# Figure 4 / 5  —  Qualitative RGB + ExGI grid
# ===========================================================================

def plot_qualitative_grid(
    trues: np.ndarray,
    preds_dict: dict,
    sample_idx: int,
    dataset: str,
    epoch: int,
    out_path: str,
):
    """
    One figure per sample.

    Layout
    ------
    Rows : Ground Truth, then each available model
    Cols : T frames  (left block = RGB, right block = ExGI)
    A narrow blank column separates the two blocks.
    """
    T = trues.shape[1]
    available = [m for m in METHODS if m in preds_dict]
    row_labels = ['Ground Truth'] + [METHOD_DISPLAY[m] for m in available]
    n_rows = len(row_labels)

    # Build sequences: GT first, then models
    sequences = [trues[sample_idx]] + [preds_dict[m][sample_idx] for m in available]

    # Column widths: label(1) | T×RGB(1 each) | spacer(0.3) | T×ExGI(1 each)
    width_ratios = [1.2] + [1] * T + [0.3] + [1] * T
    fig_w = sum(width_ratios) * 1.4
    fig_h = n_rows * 1.6 + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Qualitative Prediction — {dataset.capitalize()}  |  {epoch} epochs  |  "
        f"Sample {sample_idx}",
        fontsize=13, fontweight='bold', y=0.99,
    )

    gs = gridspec.GridSpec(
        n_rows, 1 + T + 1 + T,
        figure=fig,
        width_ratios=width_ratios,
        left=0.01, right=0.99, top=0.94, bottom=0.01,
        hspace=0.06, wspace=0.04,
    )

    for row_i, (label, frames) in enumerate(zip(row_labels, sequences)):
        # ---- Row label (first column) ------------------------------------
        ax_lbl = fig.add_subplot(gs[row_i, 0])
        ax_lbl.axis('off')
        is_gt = (label == 'Ground Truth')
        ax_lbl.text(
            0.95, 0.5, label,
            ha='right', va='center',
            fontsize=10,
            fontweight='bold' if is_gt else 'normal',
            color='#b22222' if is_gt else 'black',
            transform=ax_lbl.transAxes,
        )

        for t in range(T):
            frame = frames[t]                                # [C, H, W]
            rgb   = np.clip(frame.transpose(1, 2, 0), 0, 1) # [H, W, C]
            exgi  = _exgi(frame)                             # [H, W]

            # ---- RGB (columns 1 … T) -------------------------------------
            ax_rgb = fig.add_subplot(gs[row_i, 1 + t])
            ax_rgb.imshow(rgb)
            ax_rgb.axis('off')
            if row_i == 0:
                ax_rgb.set_title(f"t+{t + 1}", fontsize=10, pad=3)

            # ---- ExGI (columns T+2 … 2T+1) -------------------------------
            ax_eg = fig.add_subplot(gs[row_i, 1 + T + 1 + t])
            ax_eg.imshow(np.clip(exgi, 0, None), cmap='Greens', vmin=0, vmax=0.5)
            ax_eg.axis('off')
            if row_i == 0:
                ax_eg.set_title(f"ExGI t+{t + 1}", fontsize=10, pad=3)

        # Spacer column — leave blank
        ax_sp = fig.add_subplot(gs[row_i, 1 + T])
        ax_sp.axis('off')

    _save_both(out_path)
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # Save raw image frames so the figure can be regenerated without retraining.
    # Stored as: true_frames (T,C,H,W) + one array per model named pred_{method}.
    # Load with: d = np.load(path, allow_pickle=True)
    data_path = out_path.replace(".png", "_data.npz")
    save_dict = {
        'true_frames': trues[sample_idx],
        'sample_idx':  np.array(sample_idx),
        'dataset':     np.array(dataset),
        'epoch':       np.array(epoch),
        'methods':     np.array(available),
    }
    for m in available:
        save_dict[f'pred_{m}'] = preds_dict[m][sample_idx]
    np.savez_compressed(data_path, **save_dict)
    print(f"    Saved data: {data_path}")


# ===========================================================================
# Figure 6  —  All models overlaid on one subplot per sample
# ===========================================================================

def plot_fig6_overlaid(
    trues: np.ndarray,
    preds_dict: dict,
    sample_indices: list,
    dataset: str,
    epoch: int,
    threshold: float,
    out_path: str,
):
    """
    Reproduces Figure 6 style.

    One subplot per sample.  Each subplot shows:
      - Black dashed line : Ground Truth POI count curve
      - Coloured lines    : Each model's predicted POI count curve
    """
    available = [m for m in METHODS if m in preds_dict]
    n         = len(sample_indices)
    ncols     = min(n, 5)
    nrows     = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.4, nrows * 3.0),
        squeeze=False,
    )
    fig.suptitle(
        f"POI Growth Curve Comparison — {dataset.capitalize()}  |  {epoch} epochs",
        fontsize=15, fontweight='bold',
    )

    for plot_i, sample_idx in enumerate(sample_indices):
        ax = axes[plot_i // ncols][plot_i % ncols]

        # Ground truth curve
        gt_curve = _poi_curve(trues[sample_idx], threshold)
        ax.plot(gt_curve,
                color=GT_COLOR, linewidth=GT_LINEWIDTH,
                linestyle='--', label='Ground Truth', zorder=10)

        # Each model's predicted curve
        for method in available:
            pred_curve = _poi_curve(preds_dict[method][sample_idx], threshold)
            ax.plot(pred_curve,
                    color=METHOD_COLORS.get(method, '#888888'),
                    linewidth=1.6,
                    label=METHOD_DISPLAY[method])

        ax.set_title(f"Sample {sample_idx}", fontsize=11)
        ax.set_xlabel("Frame index", fontsize=10)
        ax.set_ylabel("POI count", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(sample_indices), nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    # Shared legend at the bottom
    legend_handles = [
        Line2D([0], [0], color=GT_COLOR, linewidth=GT_LINEWIDTH,
               linestyle='--', label='Ground Truth')
    ] + [
        Line2D([0], [0], color=METHOD_COLORS.get(m, '#888'),
               linewidth=1.6, label=METHOD_DISPLAY[m])
        for m in available
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(legend_handles), 5),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _save_both(out_path)
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # Save all POI curve arrays so fig6 and fig7 can be regenerated without
    # retraining.  One array per model named poi_curve_{method}; GT is
    # poi_curve_ground_truth.  threshold, dataset, epoch stored as metadata.
    # Load with: d = np.load(path, allow_pickle=True)
    data_path = out_path.replace(".png", "_data.npz")
    save_dict = {
        'sample_indices': np.array(sample_indices),
        'threshold':      np.array(threshold),
        'dataset':        np.array(dataset),
        'epoch':          np.array(epoch),
        'methods':        np.array(available),
        'poi_curve_ground_truth': np.array(
            [_poi_curve(trues[i], threshold) for i in sample_indices]
        ),
    }
    for m in available:
        save_dict[f'poi_curve_{m}'] = np.array(
            [_poi_curve(preds_dict[m][i], threshold) for i in sample_indices]
        )
    np.savez_compressed(data_path, **save_dict)
    print(f"    Saved data: {data_path}")


# ===========================================================================
# Figure 7  —  POI grid: rows = models, columns = samples
# ===========================================================================

def plot_fig7_grid(
    trues: np.ndarray,
    preds_dict: dict,
    sample_indices: list,
    dataset: str,
    epoch: int,
    threshold: float,
    out_path: str,
):
    """
    Reproduces Figure 7 style.

    Grid layout:
      Rows    = one per model (in paper order)
      Columns = one per sample
    Each cell shows that model's POI curve (coloured) vs GT (black dashed).
    """
    available = [m for m in METHODS if m in preds_dict]
    n_rows    = len(available)
    n_cols    = len(sample_indices)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.6, n_rows * 2.4),
        squeeze=False,
    )
    fig.suptitle(
        f"POI Growth Curves by Model — {dataset.capitalize()}  |  {epoch} epochs",
        fontsize=15, fontweight='bold',
    )

    for row_i, method in enumerate(available):
        for col_i, sample_idx in enumerate(sample_indices):
            ax = axes[row_i][col_i]

            gt_curve   = _poi_curve(trues[sample_idx],                  threshold)
            pred_curve = _poi_curve(preds_dict[method][sample_idx], threshold)

            ax.plot(gt_curve,
                    color=GT_COLOR, linewidth=2.0,
                    linestyle='--', label='Ground Truth')
            ax.plot(pred_curve,
                    color=METHOD_COLORS.get(method, '#888'),
                    linewidth=1.8,
                    label=METHOD_DISPLAY[method])

            # Model name label on leftmost column only
            if col_i == 0:
                ax.set_ylabel(METHOD_DISPLAY[method],
                               fontsize=11, fontweight='bold',
                               color=METHOD_COLORS.get(method, 'black'))

            # Sample index on top row only
            if row_i == 0:
                ax.set_title(f"Sample {sample_idx}", fontsize=11)

            ax.set_xlabel("Frame", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)

    # Shared legend: blue dashed = GT (paper convention), each model its own colour
    legend_handles = [
        Line2D([0], [0], color=GT_COLOR, linewidth=2.0,
               linestyle='--', label='Ground Truth (blue)'),
        Line2D([0], [0], color='gray', linewidth=1.8,
               label='Predicted (model colour)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=2, fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save_both(out_path)
    plt.close(fig)
    print(f"    Saved: {out_path}")
    # Note: fig7 shares underlying POI curve data with fig6.
    # The data file is saved by plot_fig6_overlaid() as fig6_..._data.npz.


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce CAMP paper Figures 4–7"
    )
    p.add_argument('--work_dir',  type=str, default='work_dirs',
                   help="Directory containing model output folders")
    p.add_argument('--dataset',   type=str, default='komatsuna',
                   choices=['komatsuna', 'arabidopsis'])
    p.add_argument('--epoch',     type=int, default=200,
                   help="Epoch count used in folder names (e.g. 200)")
    p.add_argument('--n_samples', type=int, default=5,
                   help="Number of test samples to visualise")
    p.add_argument('--threshold', type=float, default=0.1,
                   help="ExGI threshold τ for POI extraction")
    p.add_argument('--out_dir',   type=str, default=None,
                   help="Output directory (default: figures/{dataset}_ep{epoch}/)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join(
        'figures', f"{args.dataset}_ep{args.epoch}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Reproducing CAMP paper Figures 4–7")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epoch    : {args.epoch}")
    print(f"  Samples  : {args.n_samples}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading model outputs...\n")
    data = load_model_data(args.work_dir, args.dataset, args.epoch)

    if not data:
        print("\nERROR: No model outputs found. Run training first.")
        return

    trues      = get_trues(data)
    preds_dict = {m: d['preds'] for m, d in data.items()}

    N = trues.shape[0]

    # Skip samples that contain any near-black frame (acquisition anomaly).
    # A frame is anomalous if fewer than 1% of pixels have any signal.
    min_pixels = int(0.01 * trues.shape[3] * trues.shape[4])
    clean = [
        s for s in range(N)
        if all((trues[s, t].mean(0) > 0.02).sum() >= min_pixels
               for t in range(trues.shape[1]))
    ]
    sample_indices = clean[:args.n_samples]

    print(f"\n{len(data)} model(s) loaded, {N} test samples available "
          f"({len(clean)} clean).")
    print(f"Visualising samples: {sample_indices}\n")

    # ------------------------------------------------------------------
    # Figure 4 / 5  —  qualitative RGB + ExGI per sample
    # ------------------------------------------------------------------
    print("── Figure 4/5  (qualitative grids) ──")
    for idx in sample_indices:
        out_path = os.path.join(
            out_dir, f"{args.dataset}_fig45_qualitative_sample_{idx:04d}.png"
        )
        plot_qualitative_grid(
            trues, preds_dict, idx,
            dataset=args.dataset, epoch=args.epoch,
            out_path=out_path,
        )

    # ------------------------------------------------------------------
    # Figure 6  —  overlaid POI curves
    # ------------------------------------------------------------------
    print("\n── Figure 6  (overlaid POI curves) ──")
    plot_fig6_overlaid(
        trues, preds_dict, sample_indices,
        dataset=args.dataset, epoch=args.epoch,
        threshold=args.threshold,
        out_path=os.path.join(out_dir, f"{args.dataset}_fig6_poi_overlaid.png"),
    )

    # ------------------------------------------------------------------
    # Figure 7  —  POI grid (rows = models, cols = samples)
    # ------------------------------------------------------------------
    print("\n── Figure 7  (POI grid by model) ──")
    plot_fig7_grid(
        trues, preds_dict, sample_indices,
        dataset=args.dataset, epoch=args.epoch,
        threshold=args.threshold,
        out_path=os.path.join(out_dir, f"{args.dataset}_fig7_poi_grid.png"),
    )

    print(f"\n{'='*60}")
    print(f"  All figures saved to:  {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
