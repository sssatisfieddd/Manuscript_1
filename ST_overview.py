#!/usr/bin/env python3
"""
plot_approach_twopart.py
------------------------
Two-part analysis of bat approach behavior during stimulus windows:

  PART 1 — Proportion of bats that approached at all (bar chart with
            exact % labels). This is the primary comparison across conditions.

  PART 2 — Among approachers only: 3 metrics plotted as dot plots with
            mean ± SEM bars (boxplots need more data points to be meaningful):
              • Number of approaches
              • Total time near speaker (s)
              • Latency to first approach (s)


"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Qt5Agg')

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/Male.Video.CSV")

APPROACH_X_MAX = 500       # x ≤ this = near speaker
MIN_GAP_SEC    = 0.5       # min seconds outside zone before new approach counts
BASELINE_SEC   = 5.0       # seconds before each window to check for pre-presence

STIMULUS_WINDOWS = [
    (120, 150),
    (270, 300),
    (420, 450),
    (570, 600),
]

DPI = 180

# ── CONDITIONS ────────────────────────────────────────────────────────────────
condition_files = {
    "Distress\nNonsocial": [
        BASE_DIR / "tracked_B2C_D1_Diss_Nonsocial_M.csv",
        BASE_DIR / "tracked.AD6.D2.Dis.NS.M.csv",
     
    ],
    "Distress\nSocial": [
        BASE_DIR / "tracked.AD4.D2.Diss.S.M.csv",
        BASE_DIR / "tracked.B1D.D1.Diss.Social.M.csv",
     
    ],
    "WN\nNonsocial": [
        BASE_DIR / "tracked.B1D.D2.WN.NS.M.csv",
        BASE_DIR / "tracked.AD4.D1.WN.NS.M.csv",
   ],     
    "WN\nSocial": [
        BASE_DIR / "tracked_B2C_D2_WN_S_M.csv",
        BASE_DIR / "tracked.AD6.D1.WN.S.M.csv",
        
    ],
}
# condition_files = {
#     "Distress\nNonsocial": [
#         BASE_DIR / "tracked.BIB.Diss.D1.NS.csv",
#         BASE_DIR / "tracked.AF8.Diss.D1.NS.csv",
#         BASE_DIR / "tracked.AFE.D2.Diss.NS.csv",
#         BASE_DIR / "tracked.B31.D1.Diss.NS.F.csv",
#         BASE_DIR / "tracked.B0C.D2.Diss.NS.csv",
#         BASE_DIR / "tracked.B06.NS.Diss.D2.F.csv",
#     ],
#     "Distress\nSocial": [
#         BASE_DIR / "tracked.B07.Diss.D1.S.csv",
#         BASE_DIR / "tracked.B0F.Diss.D1.S.csv",
#         BASE_DIR / "tracked.AFD.D2.Diss.S.csv",
#         BASE_DIR / "tracked.ADB.D1.Diss.Sl.F.csv",
#         BASE_DIR / "tracked.B26.D2.Diss.S.F.csv",
#         BASE_DIR / "tracked.AE7.Diss.D2.S.csv",
#     ],
#     "WN\nNonsocial": [
#         BASE_DIR / "tracked.AFD.WN.D1.NS.csv",
#         BASE_DIR / "tracked.AE7.WN.D1.NS.csv",
#         BASE_DIR / "tracked.B0F.D2.WN.NS.csv",
#         BASE_DIR / "tracked.B07.D2.WN.NS.csv",
#         BASE_DIR / "tracked.B26.D1.WN.NS.F.csv",
#         BASE_DIR / "tracked.ADB.D2.WN.NS.F.csv",
#     ],
#     "WN\nSocial": [
#         BASE_DIR / "tracked.AFE.WN.D1.S.csv",
#         BASE_DIR / "tracked.B0C.WN.D1.S.csv",
#         BASE_DIR / "tracked.AF8.D2.WN.S.csv",
#         BASE_DIR / "tracked.BIB.D2.WN.S.csv",
#         BASE_DIR / "tracked.B06.D1.WS.S.F.csv",
#         BASE_DIR / "tracked.B31.D2.WN.S.csv",
#     ],
# }

CONDITION_ORDER = [
    "Distress\nNonsocial",
    "Distress\nSocial",
    "WN\nNonsocial",
    "WN\nSocial",
]

CONDITION_COLORS = {
    "Distress\nNonsocial": "#221449",   # dark navy/purple
    "Distress\nSocial":    "#862781",   # deep magenta purple
    "WN\nNonsocial":       "#e34e64",   # coral pink/red
    "WN\nSocial":          "#feb77f",   # warm peach orange
}

# ── CORE METRICS PER BAT ─────────────────────────────────────────────────────

def compute_metrics(csv_path: Path):
    """
    Returns dict with:
      approached     : bool
      n_approaches   : int
      total_time_sec : float
      latency_sec    : float or None  (seconds from onset of the window
                                       in which the first approach occurred)
    Returns None on file error.
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
        return None

    t = det["time_sec"].values
    x = det["x_px"].values

    total_n   = 0
    total_t   = 0.0
    first_app              = None   # absolute time of first approach entry
    first_app_window_onset = None   # onset of the window it occurred in
    skipped_windows = 0

    for (win_onset, win_offset) in STIMULUS_WINDOWS:

        # ── Pre-stimulus check ────────────────────────────────────────────
        # If the bat was in the zone at any point in the 5s before onset,
        # skip this entire window for this bat
        pre_mask = (t >= win_onset - BASELINE_SEC) & (t < win_onset)
        pre_x    = x[pre_mask]
        if len(pre_x) > 0 and np.any(pre_x <= APPROACH_X_MAX):
            print(f"    [SKIP window {win_onset}-{win_offset}s] "
                  f"bat already in zone before stimulus onset")
            skipped_windows += 1
            continue

        wm = (t >= win_onset) & (t <= win_offset)
        wt = t[wm];  wx = x[wm]
        if len(wt) < 2:
            continue

        in_zone = wx <= APPROACH_X_MAX

        # Total time: both endpoints in zone
        dt = np.diff(wt)
        total_t += dt[in_zone[:-1] & in_zone[1:]].sum()

        # Count approaches with debounce
        in_approach = False
        gap_start   = None
        for idx in range(len(wt)):
            if in_zone[idx]:
                if not in_approach:
                    if gap_start is None or (wt[idx] - gap_start) >= MIN_GAP_SEC:
                        total_n += 1
                        in_approach = True
                        # Record first approach: store both the time and the
                        # window onset so latency is relative to THIS window
                        if first_app is None:
                            first_app              = wt[idx]
                            first_app_window_onset = win_onset
                    else:
                        in_approach = True   # short gap — continuation
            else:
                if in_approach:
                    gap_start   = wt[idx]
                    in_approach = False

    # Latency = time from the onset of the window the bat first entered,
    # NOT from the very first stimulus window.  Will always be 0–30 s.
    latency = None
    if first_app is not None:
        latency = max(0.0, float(first_app - first_app_window_onset))

    approached = total_n > 0

    label = (f"approaches={total_n}  time={total_t:.1f}s  "
             f"latency={'--' if latency is None else f'{latency:.1f}s'}  "
             f"skipped_windows={skipped_windows}")
    status = "APPROACHED" if approached else "no approach"
    print(f"  [{status}]  {csv_path.name:<42} {label}")

    return {
        "approached":       approached,
        "n_approaches":     total_n,
        "total_time_sec":   total_t,
        "latency_sec":      latency,
        "skipped_windows":  skipped_windows,
    }


# ── COLLECT ───────────────────────────────────────────────────────────────────

def collect_all():
    all_data = {}   # cond → list of metric dicts

    for cond, files in condition_files.items():
        print(f"\n[{cond.replace(chr(10), ' ')}]")
        rows = []
        for fp in files:
            m = compute_metrics(fp)
            if m is not None:
                rows.append(m)
        all_data[cond] = rows

    return all_data


# ── PLOT ──────────────────────────────────────────────────────────────────────

def mean_sem(vals):
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return vals[0], 0.0
    from scipy import stats
    return float(np.mean(vals)), float(stats.sem(vals))


def plot_twopart(all_data: dict):
    fig = plt.figure(figsize=(13, 7), facecolor="white")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                           width_ratios=[1.4, 1, 1])

    ax_prop  = fig.add_subplot(gs[0])
    ax_n     = fig.add_subplot(gs[1])
    ax_lat   = fig.add_subplot(gs[2])

    conds    = CONDITION_ORDER
    colors   = [CONDITION_COLORS[c] for c in conds]
    x_pos    = np.arange(len(conds))
    xlabels  = [c.replace("\n", "\n") for c in conds]

    # ── PART 1: Proportion bar chart ─────────────────────────────────────────
    proportions  = []
    n_approached = []
    n_total      = []
    for cond in conds:
        rows = all_data[cond]
        n    = len(rows)
        k    = sum(r["approached"] for r in rows)
        proportions.append(k / n * 100 if n > 0 else 0)
        n_approached.append(k)
        n_total.append(n)

    bars = ax_prop.bar(x_pos, proportions, color=colors, width=0.55,
                       edgecolor="white", linewidth=1.2, zorder=3, alpha=0.85)

    for bar, pct, k, n in zip(bars, proportions, n_approached, n_total):
        ax_prop.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{k}/{n}\n{pct:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#333333"
        )

    ax_prop.set_xticks(x_pos)
    ax_prop.set_xticklabels(xlabels, fontsize=9.5)
    ax_prop.set_ylabel("Bats that approached (%)", fontsize=9.5, labelpad=6)
    ax_prop.set_ylim(0, 115)
    ax_prop.set_title("Part 1\nProportion Approaching", fontsize=10.5,
                      fontweight="bold", pad=8)
    ax_prop.yaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax_prop.set_axisbelow(True)
    ax_prop.tick_params(axis="x", length=0)
    for spine in ax_prop.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.8)

    # ── PART 2: Dot plots with mean ± SEM — approachers only ─────────────────
    metric_axes = [
        (ax_n,   "n_approaches", "Number of Approaches",
         "Approach count"),
        (ax_lat, "latency_sec",  "Latency to First Approach",
         "Seconds from stimulus onset"),
    ]

    for ax, key, title, ylabel in metric_axes:
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        for i, cond in enumerate(conds):
            color = CONDITION_COLORS[cond]
            rows  = all_data[cond]

            if key == "latency_sec":
                vals = [r[key] for r in rows
                        if r["approached"] and r[key] is not None]
            else:
                vals = [r[key] for r in rows if r["approached"]]

            m, se = mean_sem(vals)
            if not np.isnan(m):
                ax.bar(i, m, width=0.5, color=color, alpha=0.25,
                       edgecolor=color, linewidth=1.3, zorder=2)
                ax.errorbar(i, m, yerr=se, fmt="none",
                            ecolor=color, elinewidth=2, capsize=5,
                            capthick=2, zorder=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5, color="#444444", labelpad=5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=8, colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.8)

    fig.suptitle(
        f"Bat Approach Behavior During Stimulus Windows  "
        f"(zone: x ≤ {APPROACH_X_MAX} px,  gap threshold: ≥ {MIN_GAP_SEC}s)",
        fontsize=11, fontweight="bold", y=1.01
    )

    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = BASE_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    all_data = collect_all()

    # Console summary
    print("\n── Proportion summary ────────────────────────────────────────────")
    for cond in CONDITION_ORDER:
        rows = all_data[cond]
        k = sum(r["approached"] for r in rows)
        n = len(rows)
        print(f"  {cond.replace(chr(10),' '):<22} {k}/{n} approached  "
              f"({k/n*100:.0f}%)" if n else f"  {cond}: no data")

    print("\n── Approacher metrics ────────────────────────────────────────────")
    for cond in CONDITION_ORDER:
        rows = all_data[cond]
        app  = [r for r in rows if r["approached"]]
        if not app:
            print(f"  {cond.replace(chr(10),' '):<22} no approachers")
            continue
        from scipy import stats
        for key, label in [("n_approaches","approaches"),
                           ("total_time_sec","time(s)"),
                           ("latency_sec","latency(s)")]:
            vals = [r[key] for r in app if r[key] is not None]
            if vals:
                se = stats.sem(vals) if len(vals) > 1 else 0
                print(f"  {cond.replace(chr(10),' '):<22} {label:<14} "
                      f"mean={np.mean(vals):.2f}  sem={se:.2f}  n={len(vals)}")
    print("──────────────────────────────────────────────────────────────────\n")

    fig = plot_twopart(all_data)
    out_path = out_dir / "approach_twopart.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.show()
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()