#!/usr/bin/env python3
"""
plot_approach_direction.py
--------------------------
For each bat, computes the average SIGNED x-displacement per second
during stimulus windows:
  - Negative values = moving TOWARD the speaker (decreasing x)
  - Positive values  = moving AWAY from speaker (increasing x)

Only consecutive frames (no tracking gaps) are used so reappearance
jumps don't pollute the displacement signal.

The result is one value per bat → plotted as a boxplot with individual
dots, one box per condition.

Output: <BASE_DIR>/plots/approach_direction_boxplot.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Qt5Agg')

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/CSV_Tracking/Batch1.Man1")

FRAME_WIDTH  = 1600
FRAME_HEIGHT = 1000

# Speaker is on the LEFT side of the arena (low x values)
SPEAKER_X = 362.14
SPEAKER_W = 144.29

# Stimulus windows (seconds)
STIMULUS_WINDOWS = [
    (120, 150),   # 2:00–2:30
    (270, 300),   # 4:30–5:00
    (420, 450),   # 7:00–7:30
    (570, 600),   # 9:30–10:00
]

DPI = 180

# ── CONDITIONS ────────────────────────────────────────────────────────────────

condition_files = {
    "Distress · Nonsocial": [
        BASE_DIR / "tracked.BIB.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AF8.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AFE.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B31.D1.Diss.NS.F.csv",
        BASE_DIR / "tracked.B0C.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B06.NS.Diss.D2.F.csv",
    ],
    "Distress · Social": [
        BASE_DIR / "tracked.B07.Diss.D1.S.csv",
        BASE_DIR / "tracked.B0F.Diss.D1.S.csv",
        BASE_DIR / "tracked.AFD.D2.Diss.S.csv",
        BASE_DIR / "tracked.ADB.D1.Diss.Sl.F.csv",
        BASE_DIR / "tracked.B26.D2.Diss.S.F.csv",
        BASE_DIR / "tracked.AE7.Diss.D2.S.csv",
    ],
    "WN · Nonsocial": [
        BASE_DIR / "tracked.AFD.WN.D1.NS.csv",
        BASE_DIR / "tracked.AE7.WN.D1.NS.csv",
        BASE_DIR / "tracked.B0F.D2.WN.NS.csv",
        BASE_DIR / "tracked.B07.D2.WN.NS.csv",
        BASE_DIR / "tracked.B26.D1.WN.NS.F.csv",
        BASE_DIR / "tracked.ADB.D2.WN.NS.F.csv",
    ],
    "WN · Social": [
        BASE_DIR / "tracked.AFE.WN.D1.S.csv",
        BASE_DIR / "tracked.B0C.WN.D1.S.csv",
        BASE_DIR / "tracked.AF8.D2.WN.S.csv",
        BASE_DIR / "tracked.BIB.D2.WN.S.csv",
        BASE_DIR / "tracked.B06.D1.WS.S.F.csv",
        BASE_DIR / "tracked.B31.D2.WN.S.csv",
    ],
}

# Bottom → top on horizontal plot
CONDITION_ORDER = [
    "WN · Social",
    "WN · Nonsocial",
    "Distress · Social",
    "Distress · Nonsocial",
]

CONDITION_COLORS = {
    "Distress · Nonsocial": "#221449",
    "Distress · Social":    "#862781",
    "WN · Nonsocial":       "#e34e64",
    "WN · Social":          "#feb77f",
}

# ── CORE METRIC ───────────────────────────────────────────────────────────────

def closest_avg_position(csv_path: Path) -> float | None:
    """
    Returns the minimum x_px the bat reached during any stimulus window.
    Lower = closer to speaker. Captures brief approach behavior even if
    the bat didn't stay near the speaker.
    """
    if not csv_path.exists():
        print(f"  [MISSING]  {csv_path.name}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR]    {csv_path.name}: {e}")
        return None

    det = (df[df["detected"] == True]
             .dropna(subset=["x_px", "time_sec"])
             .sort_values("time_sec")
             .reset_index(drop=True))

    if det.empty:
        print(f"  [WARNING]  No detected frames: {csv_path.name}")
        return None

    # Filter to stimulus windows
    stim_mask = pd.Series(False, index=det.index)
    for t0, t1 in STIMULUS_WINDOWS:
        stim_mask |= (det["time_sec"] >= t0) & (det["time_sec"] <= t1)
    det = det[stim_mask]

    if det.empty:
        print(f"  [WARNING]  No stimulus-window frames: {csv_path.name}")
        return None

    min_x = float(det["x_px"].min())
    print(f"  {csv_path.name:<45}  min x = {min_x:,.1f} px")
    return min_x


# ── COLLECT ───────────────────────────────────────────────────────────────────

def collect_data() -> dict:
    data = {}
    for cond, files in condition_files.items():
        print(f"\n[{cond}]")
        vals = []
        for fp in files:
            v = closest_avg_position(fp)
            if v is not None:
                vals.append(v)
        data[cond] = vals
        print(f"  → n={len(vals)}  values: {[f'{v:,.1f}' for v in vals]}")
    return data


def zscore_data(data: dict) -> dict:
    """
    Z-score all raw pixel values across ALL bats (pooled),
    then return the same dict structure with z-scores instead.
    """
    all_vals = [v for vals in data.values() for v in vals]
    grand_mean = np.mean(all_vals)
    grand_sd   = np.std(all_vals, ddof=1)
    print(f"\nZ-score: grand mean = {grand_mean:,.0f} px, SD = {grand_sd:,.0f} px")
    return {
        cond: [(v - grand_mean) / grand_sd for v in vals]
        for cond, vals in data.items()
    }


# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e4e4e4", linewidth=0.6, zorder=0)
    ax.yaxis.grid(False)

    n_conds   = len(CONDITION_ORDER)
    positions = list(range(n_conds))

    # Speaker position reference line
    ax.axvline(x=500, color="#e63946", linewidth=1.2, linestyle="--",
               alpha=0.6, zorder=3, label="Zone threshold (x=500)")

    ax.axvspan(0, 500, color="#e63946", alpha=0.05, zorder=1)
    ax.axvspan(500, 1600, color="#457b9d", alpha=0.04, zorder=1)

    for i, cond in enumerate(CONDITION_ORDER):
        vals  = data.get(cond, [])
        color = CONDITION_COLORS[cond]

        if len(vals) >= 2:
            ax.boxplot(
                vals,
                positions=[i],
                vert=False,
                widths=0.42,
                patch_artist=True,
                manage_ticks=False,
                zorder=5,
                boxprops=dict(facecolor=color, alpha=0.25, linewidth=1.4,
                              edgecolor=color),
                medianprops=dict(color=color, linewidth=2.4),
                whiskerprops=dict(color=color, linewidth=1.4),
                capprops=dict(color=color, linewidth=1.6),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markeredgecolor=color, markersize=5, alpha=0.7),
            )
        elif len(vals) == 1:
            ax.scatter(vals, [i], color=color, s=80, zorder=7,
                       edgecolors="white", linewidths=0.8)

        # Individual dots with jitter
        rng    = np.random.default_rng(seed=42 + i)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(vals, np.array([i]*len(vals)) + jitter,
                   color=color, s=55, zorder=7, alpha=0.9,
                   edgecolors="white", linewidths=0.7)

    # Normal x axis 0-1600, lower x = closer to speaker
    ax.set_xlim(0, 1600)

    # Axes
    ax.set_ylim(-0.6, n_conds - 0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(CONDITION_ORDER, fontsize=10.5, color="#222222")
    ax.set_xlabel(
        "Closest x position reached during stimulus windows (pixels)\n",
        fontsize=9, color="#444444", labelpad=8
    )
    ax.set_title(
        "Closest Position to Speaker During Stimulus Windows",
        fontsize=13, fontweight="bold", color="#111111", pad=10
    )

    # Annotations
    ax.text(0.01, 0.02, "← Closer to speaker", transform=ax.transAxes,
            fontsize=8, color="#e63946", alpha=0.8, va="bottom")
    ax.text(0.99, 0.02, "Further away →", transform=ax.transAxes,
            fontsize=8, color="#457b9d", alpha=0.8, va="bottom", ha="right")

    # Legend
    legend_handles = [
        patches.Patch(facecolor="#e63946", alpha=0.2, edgecolor="#e63946",
                      label="Near speaker zone (x ≤ 500)"),
        patches.Patch(facecolor="#457b9d", alpha=0.2, edgecolor="#457b9d",
                      label="Away from speaker (x > 500)"),
        plt.Line2D([0], [0], color="#e63946", linewidth=1.2, linestyle="--",
                   label="Zone threshold (x = 500)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left",
              framealpha=0.9, edgecolor="#dddddd")

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#555555", labelsize=8)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = BASE_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    # Step 1: collect raw pixel distances
    raw_data = collect_data()

    print("\n── Raw pixel summary ─────────────────────────────────")
    for cond in CONDITION_ORDER:
        vals = raw_data.get(cond, [])
        if vals:
            print(f"  {cond}: mean={np.mean(vals):,.1f} px  "
                  f"sem={stats.sem(vals):,.1f}  n={len(vals)}")
        else:
            print(f"  {cond}: no data")
    print("──────────────────────────────────────────────────────\n")

    fig = plot(raw_data)
    out_path = out_dir / "avg_position_boxplot.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    from scipy import stats
    main()