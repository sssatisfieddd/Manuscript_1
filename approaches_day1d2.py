#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:45:32 2026

@author: zariageorge
"""

#Over time apprach by trial 
"""
Average Number of Approaches to Speaker During Stimulus Presentation
4 conditions × 2 days (D1 solid, D2 hatched) — same colour per condition.
Mann-Whitney U test between D1 and D2 within each condition.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib
matplotlib.use('Qt5Agg')

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Users/zariageorge/Desktop/CSV_Tracking/Batch1.Man1")

SPEAKER_ROI_X_MAX = 600
MIN_GAP_SEC       = 0.5

STIMULUS_WINDOWS = [
    (2*60,      2*60+30),
    (4*60+30,   5*60),
]

condition_files = {
    "Distress\nNonsocial": [
        BASE_DIR / "tracked.BIB.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AF8.Diss.D1.NS.csv",
        BASE_DIR / "tracked.AFE.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B31.D1.Diss.NS.F.csv",
        BASE_DIR / "tracked.B0C.D2.Diss.NS.csv",
        BASE_DIR / "tracked.B06.NS.Diss.D2.F.csv",
    ],
    "Distress\nSocial": [
        BASE_DIR / "tracked.B07.Diss.D1.S.csv",
        BASE_DIR / "tracked.BOF.Diss.D1.S.csv",
        BASE_DIR / "tracked.AFD.D2.Diss.S.csv",
        BASE_DIR / "tracked.ADB.D1.Diss.Sl.F.csv",
        BASE_DIR / "tracked.B26.D2.Diss.S.F.csv",
        BASE_DIR / "tracked.AE7.Diss.D2.S.csv",
    ],
    "WN\nNonsocial": [
        BASE_DIR / "tracked.AFD.WN.D1.NS.csv",
        BASE_DIR / "tracked.AE7.WN.D1.NS.csv",
        BASE_DIR / "tracked.B0F.D2.WN.NS.csv",
        BASE_DIR / "tracked.B07.D2.WN.NS.csv",
        BASE_DIR / "tracked.B26.D1.WN.NS.F.csv",
        BASE_DIR / "tracked.ADB.D2.WN.NS.F.csv",
    ],
    "WN\nSocial": [
        BASE_DIR / "tracked.AFE.WN.D1.S.csv",
        BASE_DIR / "tracked.B0C.WN.D1.S.csv",
        BASE_DIR / "tracked.AF8.D2.WN.S.csv",
        BASE_DIR / "tracked.BIB.D2.WN.S.csv",
        BASE_DIR / "tracked.B06.D1.WS.S.F.csv",
        BASE_DIR / "tracked.B31.D2.WN.S.csv",
    ],
}

CONDITION_ORDER = list(condition_files.keys())

# ── 2. COLOURS — one per condition ───────────────────────────────────────────

magma = plt.cm.get_cmap('magma')
_n = len(CONDITION_ORDER)
COLORS = [magma(0.15 + 0.65 * i / (_n - 1)) for i in range(_n)]

# ── 3. HELPERS ────────────────────────────────────────────────────────────────

def get_day(filepath: Path) -> str:
    for part in filepath.stem.upper().split('.'):
        if part in ('D1', 'D2'):
            return part
    name = filepath.stem.upper()
    if 'D1' in name: return 'D1'
    if 'D2' in name: return 'D2'
    return 'Unknown'


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARNING] Could not read {path.name}: {e}")
        return pd.DataFrame()


def count_approaches(df: pd.DataFrame) -> int:
    det = (df[df["detected"] == True]
             .dropna(subset=["x_px", "time_sec"])
             .sort_values("time_sec"))
    if det.empty:
        return 0
    t = det["time_sec"].values
    x = det["x_px"].values
    total_n = 0
    for (ws, we) in STIMULUS_WINDOWS:
        wm = (t >= ws) & (t <= we)
        wt, wx = t[wm], x[wm]
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


# ── 4. COLLECT DATA SPLIT BY DAY ─────────────────────────────────────────────


raw = {}
for cond, files in condition_files.items():
    raw[cond] = {'D1': [], 'D2': []}
    for fp in files:
        day = get_day(fp)
        if day not in ('D1', 'D2'):
            print(f"  [WARNING] Could not detect day for {fp.name} — skipping")
            continue
        df = load_csv(fp)
        if df.empty:
            continue
        n_approaches = count_approaches(df)
        raw[cond][day].append(n_approaches)

        # ── Print file approach info ──────────────────────────────────────────
        if n_approaches > 0:
            print(f"  {fp.name:<45}  {day}  →  {n_approaches} approach(es)")

# ── Summary: files with approaches ───────────────────────────────────────────
print(f"\n{'─'*60}")
print("  Summary: Files With At Least One Approach")
print(f"{'─'*60}")
for cond, files in condition_files.items():
    label = cond.replace('\n', ' ')
    for fp in files:
        day = get_day(fp)
        if day not in ('D1', 'D2'):
            continue
        # Re-use already collected data
        pass  # data already printed above

total_with_approaches = sum(
    1
    for cond in raw
    for day in raw[cond]
    for n in raw[cond][day]
    if n > 0
)
total_files = sum(
    len(raw[cond][day])
    for cond in raw
    for day in raw[cond]
)
print(f"  {total_with_approaches} of {total_files} files had at least one approach\n")
# ── 5. MANN-WHITNEY U ─────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print(f"\n{'─'*60}")
print("  Mann-Whitney U  —  Number of Approaches  (D1 vs D2)")
print(f"{'─'*60}")

mw_results = {}
for cond in CONDITION_ORDER:
    d1 = np.array(raw[cond]['D1'], dtype=float)
    d2 = np.array(raw[cond]['D2'], dtype=float)
    label = cond.replace('\n', ' ')
    if len(d1) >= 2 and len(d2) >= 2:
        u, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        print(f"  {label:<24}  D1 n={len(d1)}  D2 n={len(d2)}  "
              f"U={u:.1f}  p={p:.4f}  {stars(p)}")
        mw_results[cond] = (u, p)
    else:
        print(f"  {label:<24}  insufficient data — skipped")
        mw_results[cond] = (np.nan, np.nan)

# ── 6. PLOT ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')

bar_width  = 0.35
group_gap  = 0.5                          # extra space between condition groups
n_conds    = len(CONDITION_ORDER)
centres    = np.arange(n_conds) * (2 * bar_width + group_gap + 0.1)
x_d1       = centres - bar_width / 2
x_d2       = centres + bar_width / 2

bar_kwargs = dict(
    width=bar_width, edgecolor='black', linewidth=0.8,
    capsize=6, error_kw=dict(elinewidth=1.5, ecolor='black', capthick=1.5)
)

rng    = np.random.default_rng(42)
y_tops = []

for i, cond in enumerate(CONDITION_ORDER):
    col    = COLORS[i]
    d1_arr = np.array(raw[cond]['D1'], dtype=float)
    d2_arr = np.array(raw[cond]['D2'], dtype=float)

    m1 = d1_arr.mean() if len(d1_arr) else 0
    m2 = d2_arr.mean() if len(d2_arr) else 0
    s1 = stats.sem(d1_arr) if len(d1_arr) > 1 else 0
    s2 = stats.sem(d2_arr) if len(d2_arr) > 1 else 0

    # Day 1 — solid fill
    ax.bar(x_d1[i], m1, yerr=s1, color=col, alpha=0.90, **bar_kwargs)

    # Day 2 — same colour but hatched
    ax.bar(x_d2[i], m2, yerr=s2, color=col, alpha=0.90,
           hatch='////', **bar_kwargs)

    # Individual scatter points
    for vals, xc in [(d1_arr, x_d1[i]), (d2_arr, x_d2[i])]:
        if len(vals):
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(xc + jitter, vals, color='black', s=50, zorder=5,
                       edgecolors='white', linewidths=0.7, alpha=0.95)

    # Significance bracket between D1 and D2
    _, p = mw_results.get(cond, (np.nan, np.nan))
    s_label = stars(p) if not np.isnan(p) else ''
    bracket_y = max(m1 + s1, m2 + s2) * 1.18
    h = max(bracket_y * 0.05, 0.05)
    ax.plot([x_d1[i], x_d1[i], x_d2[i], x_d2[i]],
            [bracket_y, bracket_y + h, bracket_y + h, bracket_y],
            lw=1.2, color='black')
    ax.text((x_d1[i] + x_d2[i]) / 2, bracket_y + h * 1.4,
            s_label, ha='center', va='bottom', fontsize=11)

    y_tops.append(bracket_y + h * 4)

    # n labels
    ax.text(x_d1[i], -0.055, f'n={len(d1_arr)}', ha='center',
            transform=ax.get_xaxis_transform(), fontsize=8, style='italic')
    ax.text(x_d2[i], -0.055, f'n={len(d2_arr)}', ha='center',
            transform=ax.get_xaxis_transform(), fontsize=8, style='italic')

# ── Axis styling ──────────────────────────────────────────────────────────────

ax.set_xticks(centres)
ax.set_xticklabels(CONDITION_ORDER, fontsize=11)
ax.set_xlabel("Experimental Condition", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Approaches to Speaker", fontsize=12)
ax.set_title("Average Number of Approaches During Stimulus Presentation",
             fontsize=13, fontweight='bold', pad=15)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')
ax.set_xlim(x_d1[0] - 0.5, x_d2[-1] + 0.5)
ax.set_ylim(0, max(y_tops) * 1.05 if y_tops else 1)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.text(0.99, 0.99, 'Mann-Whitney U', transform=ax.transAxes,
        fontsize=8, color='#777777', ha='right', va='top', style='italic')

# ── Legend ────────────────────────────────────────────────────────────────────

legend_handles = [
    mpatches.Patch(facecolor='grey', edgecolor='black', linewidth=0.8,
                   alpha=0.90, label='Day 1'),
    mpatches.Patch(facecolor='grey', edgecolor='black', linewidth=0.8,
                   alpha=0.90, hatch='////', label='Day 2'),
]

ax.legend(handles=legend_handles, fontsize=10, frameon=True,
          framealpha=0.9, loc='upper right')

plt.tight_layout(pad=2.5)

out_path = Path("approaches_d1_vs_d2.png")
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nSaved → {out_path.resolve()}")
plt.show()