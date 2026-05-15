"""
Plot cross-dataset zero-shot transfer of POI-MAE.

Replaces figures/may04_transfer/fig_cross_dataset_transfer.png with a
print-legible version: bigger fonts, two clean grouped-bar panels (K→A
and A→K), in-domain (green) vs zero-shot transfer (red), and an honest
NaN annotation for TAU-family methods whose BatchNorm running stats
collapse on transfer.

Inputs
------
  work_dirs/transfer_<src>_to_<dst>_<method>_ep150/saved/poi_results.npy
  work_dirs/transfer_<src>_to_<dst>_<method>_ep150/saved/nan_frac.npy
  work_dirs/combined_scoreboard_ep150.csv  (for in-domain POI-MAE)

Output
------
  figures/may04_transfer/fig_cross_dataset_transfer.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORKDIRS = REPO / "work_dirs"
OUT_DIR  = REPO / "figures" / "may04_transfer"
SCOREBOARD = WORKDIRS / "combined_scoreboard_ep150.csv"

# Methods to include, in display order. (Mirrors notebook §9 ordering.)
METHODS = [
    "tau", "tau_no_cls", "simvp", "mim_full", "mim",
    "camp", "camp_base", "predrnn", "camp_no_cls",
    "phydnet", "convlstm", "camp_full", "tau_full", "simvp_full",
]
DISPLAY = {
    "tau": "TAU", "tau_no_cls": "TAU_no_cls", "simvp": "SimVP",
    "mim_full": "MIM_full", "mim": "MIM", "camp": "CAMP",
    "camp_base": "CAMP_base", "predrnn": "PredRNN",
    "camp_no_cls": "CAMP_no_cls", "phydnet": "PhyDNet",
    "convlstm": "ConvLSTM", "camp_full": "CAMP_full",
    "tau_full": "TAU_full", "simvp_full": "SimVP_full",
}

sb = pd.read_csv(SCOREBOARD)

def in_domain_poi(method, dataset):
    sub = sb[(sb.dataset == dataset) & (sb.method == method) & (sb.metric == "poi_mae")]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["mean"])

def transfer_poi(src, dst, method):
    """Return (poi_mae, nan_frac_pct) or (None, None) if missing."""
    p = WORKDIRS / f"transfer_{src}_to_{dst}_{method}_ep150" / "saved" / "poi_results.npy"
    n = WORKDIRS / f"transfer_{src}_to_{dst}_{method}_ep150" / "saved" / "nan_frac.npy"
    if not p.exists():
        return None, None
    poi = np.load(p, allow_pickle=True).item()["poi_mae"]
    nf = 0.0
    if n.exists():
        nf_arr = np.load(n, allow_pickle=True)
        nf = float(nf_arr.item() if nf_arr.shape == () else nf_arr.mean()) * 100.0
    return float(poi), nf

# Collect into dataframes for both directions
def build_panel(src, dst):
    rows = []
    for m in METHODS:
        ind = in_domain_poi(m, dst)
        tr, nf = transfer_poi(src, dst, m)
        if tr is None:
            continue
        rows.append({"method": m, "display": DISPLAY[m],
                     "in_domain": ind, "transfer": tr, "nan_pct": nf})
    df = pd.DataFrame(rows)
    # Sort by in_domain to make patterns visible
    df = df.sort_values("in_domain").reset_index(drop=True)
    return df

df_ka = build_panel("komatsuna", "arabidopsis")
df_ak = build_panel("arabidopsis", "komatsuna")

# ------------------------------------------------------------------ plotting
plt.rcParams.update({
    "font.size":       14,
    "axes.titlesize":  20,
    "axes.labelsize":  18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))

def draw_panel(ax, df, title):
    x = np.arange(len(df))
    bar_w = 0.4
    bars_id = ax.bar(x - bar_w/2, df["in_domain"], width=bar_w,
                     color="#2ca02c", edgecolor="black", linewidth=0.8,
                     label="In-domain (trained on this dataset)")
    bars_tr = ax.bar(x + bar_w/2, df["transfer"], width=bar_w,
                     color="#d62728", edgecolor="black", linewidth=0.8,
                     label="Zero-shot transfer")
    # NaN annotations above the red bars whose NaN fraction is high
    for i, row in df.iterrows():
        if row["nan_pct"] is not None and row["nan_pct"] >= 5:
            y = row["transfer"]
            ax.text(i + bar_w/2, y + (df["transfer"].max() * 0.015),
                    f"NaN={row['nan_pct']:.0f}%",
                    ha="center", va="bottom",
                    fontsize=11, color="#d62728", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display"], rotation=45, ha="right")
    ax.set_ylabel("POI-MAE  (lower = better)")
    ax.set_title(title, pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.95)

draw_panel(axes[0], df_ka,
           f"Komatsuna → Arabidopsis transfer\n"
           f"(train on Komatsuna, test on Arabidopsis test split, "
           f"{int(df_ka['transfer'].count())} methods)")
draw_panel(axes[1], df_ak,
           f"Arabidopsis → Komatsuna transfer\n"
           f"(train on Arabidopsis, test on Komatsuna test split, "
           f"{int(df_ak['transfer'].count())} methods)")

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    plt.savefig(OUT_DIR / f"fig_cross_dataset_transfer.{ext}",
                dpi=180, bbox_inches="tight")
plt.close()
print(f"✓ Saved {OUT_DIR}/fig_cross_dataset_transfer.{{png,pdf}}")
print(f"  K→A: {len(df_ka)} methods | A→K: {len(df_ak)} methods")
