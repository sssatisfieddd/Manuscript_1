"""
-----------------
Reads tracked CSV files for 4 experimental conditions and produces:
  1. Average Number of Approaches During Stimulus Presentation
  2. Average Time Spent Near Speaker During Stimulus Presentations (seconds)

Statistics:
  - Shapiro-Wilk normality test per condition per metric
  - If ALL groups normal → one-way ANOVA + Tukey HSD post-hoc
  - If ANY group non-normal → Kruskal-Wallis + Dunn post-hoc (Bonferroni)
  - Significance stars drawn on plots for all pairs
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scikit_posthocs import posthoc_dunn, posthoc_tukey
import matplotlib
matplotlib.use('Qt5Agg')

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/Male.Video.CSV")

SPEAKER_ROI_X_MAX = 600
MIN_GAP_SEC       = 0.5

STIMULUS_WINDOWS = [
    (2*60,      2*60+30),
    (4*60+30,   5*60),
]

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

CONDITION_ORDER = list(condition_files.keys())

# ── 2. COLOURS ────────────────────────────────────────────────────────────────

rocket_cmap = plt.cm.get_cmap('magma')
_n = 4
COLORS_LIST = [rocket_cmap(0.15 + 0.7 * i / (_n - 1)) for i in range(_n)]

# ── 3. HELPERS ────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARNING] Could not read {path.name}: {e}")
        return pd.DataFrame()


def count_approaches(df: pd.DataFrame) -> int:
    det = (df[df["detected"] == True]
             .dropna(subset=["x_px", "time_sec"])
             .sort_values("time_sec")
             .reset_index(drop=True))
    if det.empty:
        return 0

    t = det["time_sec"].values
    x = det["x_px"].values
    total_n = 0

    for (win_start, win_end) in STIMULUS_WINDOWS:
        wm = (t >= win_start) & (t <= win_end)
        wt = t[wm];  wx = x[wm]
        if len(wt) < 2:
            continue
        in_zone     = wx <= SPEAKER_ROI_X_MAX
        in_approach = False
        gap_start   = None
        for idx in range(len(wt)):
            if in_zone[idx]:
                if not in_approach:
                    if gap_start is None or (wt[idx] - gap_start) >= MIN_GAP_SEC:
                        total_n    += 1
                        in_approach = True
                    else:
                        in_approach = True
            else:
                if in_approach:
                    gap_start   = wt[idx]
                    in_approach = False

    return total_n


def time_near_speaker(df: pd.DataFrame) -> float:
    det = (df[df["detected"] == True]
             .dropna(subset=["x_px", "time_sec"])
             .sort_values("time_sec")
             .reset_index(drop=True))
    if det.empty:
        return 0.0

    t = det["time_sec"].values
    x = det["x_px"].values
    total = 0.0

    for (win_start, win_end) in STIMULUS_WINDOWS:
        wm = (t >= win_start) & (t <= win_end)
        wt = t[wm];  wx = x[wm]
        if len(wt) < 2:
            continue
        in_zone = wx <= SPEAKER_ROI_X_MAX
        dt      = np.diff(wt)
        total  += dt[in_zone[:-1] & in_zone[1:]].sum()

    return float(total)


# ── 4. COLLECT RAW VALUES PER CONDITION ──────────────────────────────────────

def collect_raw(files):
    approaches, roi_times = [], []
    for fp in files:
        df = load_csv(fp)
        if df.empty:
            continue
        approaches.append(count_approaches(df))
        roi_times.append(time_near_speaker(df))
    return approaches, roi_times


raw = {}   # cond → {"approaches": [...], "roi_time": [...]}
for cond, files in condition_files.items():
    app, roi = collect_raw(files)
    raw[cond] = {"approaches": app, "roi_time": roi}

# ── 5. STATISTICS ─────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_stats(raw_data: dict, metric_key: str, metric_label: str):
    """
    1. Shapiro-Wilk per group
    2. ANOVA (if all normal) or Kruskal-Wallis
    3. Tukey HSD or Dunn post-hoc for all pairs
    Returns dict of {(cond_i, cond_j): p_value} for all pairs.
    """
    conds  = CONDITION_ORDER
    groups = [raw_data[c][metric_key] for c in conds]

    print(f"\n{'─'*60}")
    print(f"  METRIC: {metric_label}")
    print(f"{'─'*60}")

    # Shapiro-Wilk
    all_normal = True
    print("  Shapiro-Wilk normality test:")
    for cond, grp in zip(conds, groups):
        if len(grp) < 3:
            print(f"    {cond.replace(chr(10),' '):<22} n<3, skipping normality test")
            all_normal = False
            continue
        stat, p = stats.shapiro(grp)
        normal = p > 0.05
        if not normal:
            all_normal = False
        flag = "  ✓ normal" if normal else "  ✗ NON-NORMAL"
        print(f"    {cond.replace(chr(10),' '):<22} W={stat:.3f}  p={p:.4f}{flag}")

    # Omnibus test
    if all_normal:
        f_stat, p_omni = stats.f_oneway(*groups)
        print(f"\n  One-way ANOVA: F={f_stat:.3f}  p={p_omni:.4f}  {stars(p_omni)}")
        test_used = "Tukey HSD"
    else:
        h_stat, p_omni = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis: H={h_stat:.3f}  p={p_omni:.4f}  {stars(p_omni)}")
        test_used = "Dunn (Bonferroni)"

    # Post-hoc pairwise
    print(f"\n  Post-hoc: {test_used}")
    all_vals   = []
    group_labels = []
    for cond, grp in zip(conds, groups):
        all_vals.extend(grp)
        group_labels.extend([cond] * len(grp))

    df_ph = pd.DataFrame({"value": all_vals, "group": group_labels})

    if all_normal:
        ph = posthoc_tukey(df_ph, val_col="value", group_col="group")
    else:
        ph = posthoc_dunn(df_ph, val_col="value", group_col="group",
                          p_adjust="bonferroni")

    pair_pvals = {}
    for c1, c2 in combinations(conds, 2):
        p = ph.loc[c1, c2]
        pair_pvals[(c1, c2)] = p
        label1 = c1.replace('\n', ' ')
        label2 = c2.replace('\n', ' ')
        print(f"    {label1:<22} vs {label2:<22}  p={p:.4f}  {stars(p)}")

    return pair_pvals, all_normal


pair_pvals_app, normal_app = run_stats(raw, "approaches", "Number of Approaches")
pair_pvals_roi, normal_roi = run_stats(raw, "roi_time",   "Time Near Speaker (s)")

# ── 6. SUMMARY STATS FOR PLOTTING ────────────────────────────────────────────

app_means, app_sems, roi_means, roi_sems = [], [], [], []
for cond in CONDITION_ORDER:
    a = raw[cond]["approaches"]
    r = raw[cond]["roi_time"]
    app_means.append(np.mean(a));  app_sems.append(stats.sem(a))
    roi_means.append(np.mean(r));  roi_sems.append(stats.sem(r))

# ── 7. PLOT ───────────────────────────────────────────────────────────────────

def add_significance_brackets(ax, pair_pvals, cond_order, y_max, y_step_frac=0.10):
    """
    Draw significance brackets above bars for significant pairs only.
    ns pairs are skipped to keep the plot clean.
    """
    cond_to_x = {c: i for i, c in enumerate(cond_order)}
    current_y  = y_max * (1 + y_step_frac)
    y_step     = y_max * y_step_frac

    # Sort pairs by distance so closer pairs are drawn lower
    sorted_pairs = sorted(pair_pvals.items(),
                          key=lambda kv: abs(cond_to_x[kv[0][0]] - cond_to_x[kv[0][1]]))

    # Track the highest bracket drawn over each x position
    max_y_at = [y_max] * len(cond_order)

    for (c1, c2), p in sorted_pairs:
        s = stars(p)
        if s == "ns":
            continue
        x1 = cond_to_x[c1]
        x2 = cond_to_x[c2]
        bracket_y = max(max_y_at[min(x1,x2):max(x1,x2)+1]) + y_step

        # Draw bracket
        ax.plot([x1, x1, x2, x2],
                [bracket_y - y_step*0.3, bracket_y,
                 bracket_y, bracket_y - y_step*0.3],
                lw=1.2, color="black")
        ax.text((x1 + x2) / 2, bracket_y + y_step * 0.05,
                s, ha="center", va="bottom", fontsize=10, color="black")

        for xi in range(min(x1,x2), max(x1,x2)+1):
            max_y_at[xi] = bracket_y

    # Expand y-axis to fit brackets
    new_top = max(max_y_at) + y_step * 1.5
    ax.set_ylim(0, new_top)


fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor("white")

bar_kwargs = dict(width=0.6, edgecolor="none", capsize=6,
                  error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4))

x      = np.arange(len(CONDITION_ORDER))
colors = COLORS_LIST

# — Plot 1: Number of Approaches —
ax1 = axes[0]
ax1.bar(x, app_means, color=colors, yerr=app_sems, **bar_kwargs)
ax1.set_xticks(x)
ax1.set_xticklabels(CONDITION_ORDER, fontsize=10)
ax1.set_xlabel("Experimental Condition", fontsize=11, labelpad=8)
ax1.set_ylabel("Number of Approaches", fontsize=11)
ax1.set_title("Average Number of Approaches\nDuring Stimulus Presentation",
              fontsize=12, fontweight="bold", pad=12)
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

test_label1 = "One-way ANOVA + Tukey" if normal_app else "Kruskal-Wallis + Dunn"
ax1.text(0.98, 0.98, test_label1, transform=ax1.transAxes,
         fontsize=7.5, color="#666666", ha="right", va="top", style="italic")

add_significance_brackets(ax1, pair_pvals_app, CONDITION_ORDER,
                           y_max=max(app_means) + max(app_sems))

# — Plot 2: Time Near Speaker —
ax2 = axes[1]
ax2.bar(x, roi_means, color=colors, yerr=roi_sems, **bar_kwargs)
ax2.set_xticks(x)
ax2.set_xticklabels(CONDITION_ORDER, fontsize=10)
ax2.set_xlabel("Experimental Condition", fontsize=11, labelpad=8)
ax2.set_ylabel("Time in ROI (seconds)", fontsize=11)
ax2.set_title("Average Time Spent Near Speaker\nDuring Stimulus Presentations",
              fontsize=12, fontweight="bold", pad=12)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

test_label2 = "One-way ANOVA + Tukey" if normal_roi else "Kruskal-Wallis + Dunn"
ax2.text(0.98, 0.98, test_label2, transform=ax2.transAxes,
         fontsize=7.5, color="#666666", ha="right", va="top", style="italic")

add_significance_brackets(ax2, pair_pvals_roi, CONDITION_ORDER,
                           y_max=max(roi_means) + max(roi_sems))

plt.tight_layout(pad=3)

out_path = Path("condition_plots.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out_path.resolve()}")
plt.show()