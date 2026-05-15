"""
Plot prediction-vs-classification trade-off using ONLY methods that actually
have classification logits saved on disk. This replaces an earlier version
which displayed hardcoded ~0.99 cls accuracies for baselines (TAU, SimVP,
*_no_cls variants) that DON'T have classification heads — those values were
inherited from a hallucinated source figure and have now been removed.

Honest data sources:
  work_dirs/arabidopsis_<method>_ep150/saved/cls_logits.npy
  work_dirs/arabidopsis_<method>_ep150/saved/labels.npy
For each method with both files present, classification accuracy is
computed as mean( (sigmoid(logit) > 0.5) == label ) over valid labels.

POI-MAE is read from work_dirs/combined_scoreboard_ep150.csv.

Outputs
-------
figures/may01_final/fig3_pred_vs_cls_tradeoff.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

REPO = Path(__file__).resolve().parents[1]
WORKDIRS = REPO / "work_dirs"
OUT_DIR = REPO / "figures" / "may01_final"
SCOREBOARD = WORKDIRS / "combined_scoreboard_ep150.csv"

# Methods we expect to have cls_logits + labels on Arabidopsis
CANDIDATES = [
    # method_key, display_name, backbone_family, marker, color
    ("camp",              "CAMP",              "Recurrent (PredRNN)", "o", "#1f77b4"),
    ("camp_base",         "CAMP_base",         "Recurrent (PredRNN)", "s", "#1f77b4"),
    ("camp_no_cls",       "CAMP_no_cls",       "Recurrent (PredRNN)", "D", "#1f77b4"),
    ("camp_full",         "CAMP_full",         "Recurrent (PredRNN)", "*", "#1f77b4"),
    ("simvp_predcls",     "SimVP_PredCls",     "Non-recurrent (SimVP)","P", "#2ca02c"),
    ("tau_full_detached", "TAU_full_detached", "Non-recurrent (TAU)", "X", "#d62728"),
    ("tau_predcls",       "TAU_PredCls",       "Non-recurrent (TAU)", "^", "#d62728"),
]

sb = pd.read_csv(SCOREBOARD)

def poi(method, dataset="arabidopsis"):
    sub = sb[(sb.dataset == dataset) & (sb.method == method) & (sb.metric == "poi_mae")]
    return float(sub.iloc[0]["mean"]) if len(sub) else None

def cls_acc(method, dataset="arabidopsis", epoch=150):
    cl = WORKDIRS / f"{dataset}_{method}_ep{epoch}" / "saved" / "cls_logits.npy"
    lb = WORKDIRS / f"{dataset}_{method}_ep{epoch}" / "saved" / "labels.npy"
    if not (cl.exists() and lb.exists()): return None
    L = np.load(cl).squeeze()
    Y = np.load(lb).squeeze()
    valid = Y >= 0
    if valid.sum() == 0: return None
    pred = (L > 0).astype(int)   # sigmoid > 0.5 ⇔ logit > 0
    return float((pred[valid] == Y[valid].astype(int)).mean())

# Collect rows
rows = []
for key, disp, family, marker, color in CANDIDATES:
    p = poi(key); a = cls_acc(key)
    if p is None or a is None:
        print(f"  skipping {key} — missing data (poi={p}, acc={a})")
        continue
    rows.append({"key": key, "label": disp, "family": family,
                 "poi": p, "acc": a, "marker": marker, "color": color})
df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ------------------------------------------------------------------ plotting
plt.rcParams.update({
    "font.size":       14,
    "axes.titlesize":  17,
    "axes.labelsize":  18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

fig, ax = plt.subplots(figsize=(11, 7.5))

for _, r in df.iterrows():
    ax.scatter(r["poi"], r["acc"], s=200, marker=r["marker"],
               c=r["color"], edgecolors="black", linewidths=1.4,
               zorder=3, alpha=0.95)

# Chance/majority reference (Arabidopsis class distribution ≈ 57.7% drought)
ax.axhline(0.577, color="#7f7f7f", linestyle=":", linewidth=1.8,
           label="chance/majority (57.7%)", zorder=1)
ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.2,
           label="random (50%)", zorder=1)

ax.set_xlabel("POI-MAE  (lower = better prediction)")
ax.set_ylabel("Classification accuracy  (higher = better)")
ax.set_title("Prediction–classification trade-off (Arabidopsis, ep 150)\n"
             "Only methods with saved classification logits are shown",
             fontsize=15)
ax.set_xlim(260, 820)
ax.set_ylim(0.35, 0.95)
ax.grid(alpha=0.25, zorder=0)

legend_handles = [
    Line2D([], [], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=11, markeredgecolor="black", label="Recurrent (PredRNN) backbone"),
    Line2D([], [], marker="P", color="w", markerfacecolor="#2ca02c",
           markersize=11, markeredgecolor="black", label="Non-recurrent (SimVP) backbone"),
    Line2D([], [], marker="X", color="w", markerfacecolor="#d62728",
           markersize=11, markeredgecolor="black", label="Non-recurrent (TAU) backbone"),
    Line2D([], [], color="#7f7f7f", linestyle=":", linewidth=1.8,
           label="chance/majority (57.7%)"),
]
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95)

texts = []
for _, r in df.iterrows():
    texts.append(ax.text(r["poi"], r["acc"], r["label"],
                         fontsize=13, fontweight="bold",
                         color=r["color"], zorder=5,
                         bbox=dict(boxstyle="round,pad=0.18",
                                   facecolor="white", edgecolor=r["color"],
                                   alpha=0.92, linewidth=0.9)))
adjust_text(texts,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8),
            expand_points=(1.6, 1.6),
            expand_text=(1.5, 1.5),
            ax=ax)

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    plt.savefig(OUT_DIR / f"fig3_pred_vs_cls_tradeoff.{ext}",
                dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Saved {OUT_DIR}/fig3_pred_vs_cls_tradeoff.{{png,pdf}}")
