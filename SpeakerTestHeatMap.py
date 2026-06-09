#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_heatmaps_rocket.py
-----------------------
Generates a 2×2 grid of rocket-themed KDE heatmaps — one per condition —
averaged across all animals in that group.

White background, rocket fire colormap (white → yellow → orange → red → black).
Single vertical colorbar on the far right. No stats box. No cluttered legend.
Only the speaker box and ROI zone are labeled.

Usage
-----
  python plot_heatmaps_rocket.py

Outputs
-------
  <BASE_DIR>/heatmaps/heatmap_all_conditions_rocket.png
  <BASE_DIR>/heatmaps/heatmap_<condition>_rocket.png   (one per condition)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/Male.Video.CSV")

FRAME_WIDTH  = 1600
FRAME_HEIGHT = 1000

SPEAKER_X = 362.14
SPEAKER_Y = 387.86
SPEAKER_W = 144.29
SPEAKER_H = 170.00

SPEAKER_ROI_X_MAX = 600

GRID_COLS  = 320
GRID_ROWS  = 200
BLUR_SIGMA = 6.0
GAMMA      = 0.55
DPI        = 180

condition_files = {
    "Distress_Nonsocial": [
        BASE_DIR / "tracked_B2C_D1_Diss_Nonsocial_M.csv",
        BASE_DIR / "tracked.AD6.D2.Dis.NS.M.csv",
     
    ],
    "Distress_Social": [
        BASE_DIR / "tracked.AD4.D2.Diss.S.M.csv",
        BASE_DIR / "tracked.B1D.D1.Diss.Social.M.csv",
     
    ],
    "WN_Nonsocial": [
        BASE_DIR / "tracked.B1D.D2.WN.NS.M.csv",
        BASE_DIR / "tracked.AD4.D1.WN.NS.M.csv",
   ],     
    "WN_Social": [
        BASE_DIR / "tracked_B2C_D2_WN_S_M.csv",
        BASE_DIR / "tracked.AD6.D1.WN.S.M.csv",
        
    ],
}
# condition_files = {
#     "Distress_Nonsocial": [
#         BASE_DIR / "tracked.BIB.Diss.D1.NS.csv",
#         BASE_DIR / "tracked.AF8.Diss.D1.NS.csv",
#         BASE_DIR / "tracked.AFE.D2.Diss.NS.csv",
#         BASE_DIR / "tracked.B31.D1.Diss.NS.F.csv",
#         BASE_DIR / "tracked.B0C.D2.Diss.NS.csv",
#         BASE_DIR / "tracked.B06.NS.Diss.D2.F.csv",
#     ],
#     "Distress_Social": [
#         BASE_DIR / "tracked.B07.Diss.D1.S.csv",
#         BASE_DIR / "tracked.B0F.Diss.D1.S.csv",
#         BASE_DIR / "tracked.AFD.D2.Diss.S.csv",
#         BASE_DIR / "tracked.ADB.D1.Diss.Sl.F.csv",
#         BASE_DIR / "tracked.B26.D2.Diss.S.F.csv",
#         BASE_DIR / "tracked.AE7.Diss.D2.S.csv",
#     ],
#     "WN_Nonsocial": [
#         BASE_DIR / "tracked.AFD.WN.D1.NS.csv",
#         BASE_DIR / "tracked.AE7.WN.D1.NS.csv",
#         BASE_DIR / "tracked.B0F.D2.WN.NS.csv",
#         BASE_DIR / "tracked.B07.D2.WN.NS.csv",
#         BASE_DIR / "tracked.B26.D1.WN.NS.F.csv",
#         BASE_DIR / "tracked.ADB.D2.WN.NS.F.csv",
#     ],
#     "WN_Social": [
#         BASE_DIR / "tracked.AFE.WN.D1.S.csv",
#         BASE_DIR / "tracked.B0C.WN.D1.S.csv",
#         BASE_DIR / "tracked.AF8.D2.WN.S.csv",
#         BASE_DIR / "tracked.BIB.D2.WN.S.csv",
#         BASE_DIR / "tracked.B06.D1.WS.S.F.csv",
#         BASE_DIR / "tracked.B31.D2.WN.S.csv",
#     ],
# }

CONDITION_LABELS = {
    "Distress_Nonsocial": "Distress · Nonsocial",
    "Distress_Social":    "Distress · Social",
    "WN_Nonsocial":       "WN · Nonsocial",
    "WN_Social":          "WN · Social",
}

# ── COLORMAP ──────────────────────────────────────────────────────────────────

rocket_cmap = LinearSegmentedColormap.from_list(
    "rocket_fire_white",
    [
        (0.00, "#ffffff"),
        (0.08, "#fff5cc"),
        (0.20, "#ffe566"),
        (0.38, "#ffaa00"),
        (0.55, "#ee4400"),
        (0.72, "#bb0000"),
        (0.88, "#550000"),
        (1.00, "#100000"),
    ],
    N=512,
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_condition_points(file_list):
    all_x, all_y = [], []
    n_loaded, n_missing = 0, 0
    for fp in file_list:
        if not fp.exists():
            print(f"    [MISSING] {fp.name}")
            n_missing += 1
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"    [ERROR] {fp.name}: {e}")
            n_missing += 1
            continue
        det = df[df["detected"] == True].dropna(subset=["x_px", "y_px"])
        if det.empty:
            print(f"    [WARNING] No detected frames in {fp.name}")
            continue
        all_x.extend(det["x_px"].values)
        all_y.extend(det["y_px"].values)
        n_loaded += 1
        print(f"    Loaded {fp.name}  ({len(det):,} frames)")
    return np.array(all_x), np.array(all_y), n_loaded, n_missing

# ── KDE ───────────────────────────────────────────────────────────────────────

def build_heatmap(x, y):
    heatmap, _, _ = np.histogram2d(
        x, y,
        bins=[GRID_COLS, GRID_ROWS],
        range=[[0, FRAME_WIDTH], [0, FRAME_HEIGHT]],
    )
    heatmap = heatmap.T
    heatmap = gaussian_filter(heatmap, sigma=BLUR_SIGMA)
    vmax = heatmap.max()
    if vmax > 0:
        heatmap /= vmax
    return heatmap

# ── SINGLE PANEL ──────────────────────────────────────────────────────────────

def plot_single_condition(ax, cond_name, file_list):
    label = CONDITION_LABELS[cond_name]
    print(f"\n  [{cond_name}]")
    x, y, n_loaded, _ = load_condition_points(file_list)

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="-", zorder=0)
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="-", zorder=0)

    if len(x) > 0:
        heatmap = build_heatmap(x, y)
        display = np.power(heatmap, GAMMA)
        ax.imshow(
            display,
            cmap=rocket_cmap,
            origin="upper",
            extent=[0, FRAME_WIDTH, FRAME_HEIGHT, 0],
            aspect="auto",
            interpolation="bilinear",
            vmin=0, vmax=1,
            zorder=1,
        )

    # ROI zone
    ax.add_patch(patches.Rectangle(
        (0, 0), SPEAKER_ROI_X_MAX, FRAME_HEIGHT,
        linewidth=0, facecolor="#ed9761", alpha=0.08, zorder=2,
    ))
    ax.axvline(x=SPEAKER_ROI_X_MAX, color="#ed9761", linewidth=1.2,
               linestyle="--", alpha=0.75, zorder=3,
               label="ROI zone (x ≤ 600)")

    # Speaker box
    ax.add_patch(patches.Rectangle(
        (SPEAKER_X, SPEAKER_Y), SPEAKER_W, SPEAKER_H,
        linewidth=2.0, edgecolor="#e63946", facecolor="#e63946",
        alpha=0.18, zorder=4, label="Speaker",
    ))

    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)
    ax.set_xlabel("X position (pixels)", fontsize=9, color="#333333", labelpad=4)
    ax.set_ylabel("Y position (pixels)", fontsize=9, color="#333333", labelpad=4)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=7, color="#222222")
    return x, y

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = BASE_DIR / "heatmaps"
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    cond_list = list(condition_files.items())

    # ── 2×2 combined figure ──────────────────────────────────────────────────
    # Leave room on the right for a single shared vertical colorbar
    fig = plt.figure(figsize=(20, 16), facecolor="white")

    # Grid: 2 rows × 2 cols of axes, then one thin colorbar column on the right
    gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1, 1, 0.045],
    wspace=0.22,
    hspace=0.55,        # ← increase this (was 0.38)
    left=0.06, right=0.96,
    top=0.88, bottom=0.10,
)

    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axes = [fig.add_subplot(gs[r, c]) for r, c in ax_positions]

    for ax, (cond_name, file_list) in zip(axes, cond_list):
        plot_single_condition(ax, cond_name, file_list)

    # Shared vertical colorbar spanning both rows on the right
    cax = fig.add_subplot(gs[:, 2])
    sm  = plt.cm.ScalarMappable(cmap=rocket_cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"], fontsize=8, color="#444444")
    cbar.outline.set_edgecolor("#aaaaaa")
    cbar.ax.tick_params(colors="#444444", size=3)
    cbar.set_label("Occupancy density", fontsize=9, color="#333333", labelpad=8)

    # Super-title
    fig.suptitle(
        "",
        fontsize=15, fontweight="bold", color="#111111", y=0.95,
    
    )

    # Minimal legend: only Speaker and ROI — bottom centre, clean
  

    out_path = out_dir / "heatmap_all_conditions_rocket.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved combined figure → {out_path}")
    plt.show()
    plt.close(fig)

    # ── Individual condition figures ─────────────────────────────────────────
    for cond_name, file_list in cond_list:
        fig2 = plt.figure(figsize=(11, 7), facecolor="white")
        gs2  = fig2.add_gridspec(
            1, 2,
            width_ratios=[1, 0.04],
            wspace=0.08,
            left=0.09, right=0.95,
            top=0.88, bottom=0.12,
        )
        ax2  = fig2.add_subplot(gs2[0, 0])
        cax2 = fig2.add_subplot(gs2[0, 1])

        plot_single_condition(ax2, cond_name, file_list)

        sm2 = plt.cm.ScalarMappable(cmap=rocket_cmap, norm=mcolors.Normalize(0, 1))
        sm2.set_array([])
        cbar2 = fig2.colorbar(sm2, cax=cax2, orientation="vertical")
        cbar2.set_ticks([0.0, 1.0])
        cbar2.set_ticklabels(["Low", "High"], fontsize=8, color="#444444")
        cbar2.outline.set_edgecolor("#aaaaaa")
        cbar2.ax.tick_params(colors="#444444", size=3)
        cbar2.set_label("Occupancy density", fontsize=9, color="#333333", labelpad=8)

        out2 = out_dir / f"heatmap_{cond_name}_rocket.png"
        fig2.savefig(out2, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out2.name}")
        plt.close(fig2)

    print("\nDone!")


if __name__ == "__main__":
    main()