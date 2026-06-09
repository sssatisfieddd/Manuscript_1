"""
Bat Heart Rate Timeseries Analysis
- Average within bat first across all presentations of each stimulus
- Then average those bat-level timeseries across all bats of that sex
- Each bat weighted equally regardless of number of valid presentations
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# ── Styling ───────────────────────────────────────────────────────────────────
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--'
})

# ── Configuration ─────────────────────────────────────────────────────────────
female_folder_path = '/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Females'
male_folder_path   = '/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Males'

female_csv_files = glob.glob(os.path.join(female_folder_path, "*.csv"))
male_csv_files   = glob.glob(os.path.join(male_folder_path,   "*.csv"))

MALE_BATS   = {"A88", "AEO", "B05", "B1D", "B07", "AD6", "AF4"}
FEMALE_BATS = {"693", "B04", "CD3", "B30", "B32", "AF6"}
ALT_BATS    = {"AF4", "AF6", "B32", "AD6"}

sex_colors = {
    "Female": "#800080",
    "Male":   "#FFA500"
}

# ── Time windows ──────────────────────────────────────────────────────────────
time_windows = {
    "Aggression":   [(61.302, 90.813),  (691.400, 720.914)], # (871.504, 901.015)],
                    # (961.503, 991.014),  (1231.604, 1261.114),(1411.604, 1441.113)],
    "Distress":     [(151.302, 180.851),  (241.303, 270.072)],#  (601.403, 630.953)],
                     #(1051.501, 1081.052),(1951.803, 1981.352),(2041.804, 2071.353)],
    "Echolocation": [(421.402, 450.44),   (511.746, 540.944)],#  (1321.602, 1351.145)],
                     #(1501.602, 1531.146),(1771.803, 1801.345),(1861.803, 1891.345)],
    "Distortion":   [(331.302, 361.104),  (781.401, 811.205)],#(1141.503, 1171.305)],#(1591.603, 1621.406),
                     #(1681.703, 1711.504),(2131.428, 2161.605)],
}

alternative_time_windows = {
    "Aggression":   [(60.189, 90.189),  (150.189, 180.189)],#  (510.389, 540.389)],
                    # (1590.790, 1620.790),(1770.790, 1800.790),(2130.791, 2160.791)],
    "Distress":     [(420.389, 450.389),  (600.389, 630.389)],#  (780.489, 810.489)],
                   #  (870.489, 900.489),  (1050.490, 1080.490),(1950.790, 1980.790)],
    "Echolocation": [(960.490, 990.490), (1320.690, 1350.690)],#(1500.790, 1530.790)],
                   #  (1680.790, 1710.790),(1860.790, 1890.790),(2040.790, 2070.790)],
    "Distortion":   [(691.400, 720.914),  (1141.503, 1171.305)],#(1231.604, 1261.114)],
                     #(1411.604, 1441.113),(241.303, 270.072),  (331.302, 361.104)],
}

def get_time_windows(bat_id):
    return alternative_time_windows if bat_id in ALT_BATS else time_windows

# ── Baseline ──────────────────────────────────────────────────────────────────
def calculate_robust_baseline(df):
    valid_hr = df[(df["Heart_Rate"] > 0) & (df["Heart_Rate"] <= 310)]["Heart_Rate"]
    if len(valid_hr) < 5:
        return None

    window_size = 30
    min_avg_hr  = float('inf')

    for start_time in np.arange(df["Time"].min(), df["Time"].max() - window_size, 5):
        window_hr = df[
            (df["Time"] >= start_time) &
            (df["Time"] <= start_time + window_size) &
            (df["Heart_Rate"] > 0) &
            (df["Heart_Rate"] <= 310)
        ]["Heart_Rate"]
        if len(window_hr) >= 5:
            avg = np.mean(window_hr)
            if avg < min_avg_hr:
                min_avg_hr = avg

    percentile_baseline = np.percentile(valid_hr, 5)

    if min_avg_hr != float('inf'):
        return min(min_avg_hr, percentile_baseline)
    else:
        return percentile_baseline

# ── Timeseries extractor ──────────────────────────────────────────────────────
def get_hr_timeseries_around_stimulus(df, stimulus_start, baseline_hr,
                                       time_before=10, time_after=30):
    segment = df[
        (df["Time"] >= stimulus_start - time_before) &
        (df["Time"] <= stimulus_start + time_after)
    ].copy()

    if len(segment) == 0:
        return None

    segment["Relative_Time"] = segment["Time"] - stimulus_start

    # Clean noisy pre-stimulus HR
    pre_mask = (segment["Relative_Time"] >= -time_before) & (segment["Relative_Time"] < 0)
    segment.loc[pre_mask & (segment["Heart_Rate"] > 300), "Heart_Rate"] = np.nan

    segment["HR_Change"] = segment["Heart_Rate"] - baseline_hr
    segment.loc[segment["HR_Change"].abs() > 400, "HR_Change"] = np.nan

    # 1-second bins
    time_bins    = np.arange(-time_before, time_after + 1, 1)
    hr_by_second = {}
    for i in range(len(time_bins) - 1):
        bin_data = segment[
            (segment["Relative_Time"] >= time_bins[i]) &
            (segment["Relative_Time"] <  time_bins[i + 1])
        ]["HR_Change"].dropna()
        if len(bin_data) > 0:
            hr_by_second[time_bins[i] + 0.5] = bin_data.mean()

    return hr_by_second if hr_by_second else None


# ── Average presentations within a bat into one timeseries ───────────────────
def average_presentations(timeseries_list):
    """Average multiple presentation timeseries into one bat-level timeseries."""
    if not timeseries_list:
        return None
    all_timepoints = sorted(set().union(*[set(ts.keys()) for ts in timeseries_list]))
    bat_avg = {}
    for t in all_timepoints:
        vals = [ts[t] for ts in timeseries_list if t in ts]
        if vals:
            bat_avg[t] = np.mean(vals)
    return bat_avg if bat_avg else None


# ── Average bat-level timeseries across bats (with SE) ───────────────────────
def average_across_bats(bat_timeseries_list):
    """
    Input: list of per-bat averaged timeseries dicts
    Output: group mean and SE dicts — each bat weighted equally
    """
    if not bat_timeseries_list:
        return {}, {}
    all_timepoints = sorted(set().union(*[set(ts.keys()) for ts in bat_timeseries_list]))
    avg, se = {}, {}
    for t in all_timepoints:
        vals = [ts[t] for ts in bat_timeseries_list if t in ts]
        if vals:
            avg[t] = np.mean(vals)
            se[t]  = np.std(vals) / np.sqrt(len(vals))
    return avg, se

# ── Data processing ───────────────────────────────────────────────────────────
categories = ["Aggression", "Distress", "Echolocation", "Distortion"]

# sex -> category -> list of per-BAT averaged timeseries
# (one dict per bat, already averaged across that bat's presentations)
bat_level_data = {sex: {cat: [] for cat in categories} for sex in ["Female", "Male"]}
bat_counts     = {sex: {cat: set() for cat in categories} for sex in ["Female", "Male"]}

for csv_files, folder_sex in [(female_csv_files, "Female"), (male_csv_files, "Male")]:
    for file in csv_files:
        try:
            df        = pd.read_csv(file)
            file_name = os.path.basename(file)
            bat_id    = file_name[:3]

            # Sex from ID lookup
            if bat_id in MALE_BATS:
                sex = "Male"
            elif bat_id in FEMALE_BATS:
                sex = "Female"
            else:
                print(f"  WARNING: {bat_id} not in sex lookup — skipping")
                continue

            if any(c not in df.columns for c in ["Time", "Heart_Rate", "File_Marker"]):
                print(f"  Missing columns in {file_name} — skipping")
                continue

            if 1 not in df["File_Marker"].values:
                print(f"  No File_Marker == 1 in {file_name} — skipping")
                continue
            df["Time"] = df["Time"] - df[df["File_Marker"] == 1]["Time"].min()

            baseline_hr = calculate_robust_baseline(df)
            if baseline_hr is None:
                print(f"  Could not calculate baseline for {file_name} — skipping")
                continue

            print(f"  {bat_id} ({sex}) — baseline: {baseline_hr:.1f} BPM")

            bat_time_windows = get_time_windows(bat_id)

            for category, intervals in bat_time_windows.items():

                # Step 1: collect all presentation timeseries for this bat
                presentation_timeseries = []
                for start, end in intervals:
                    ts = get_hr_timeseries_around_stimulus(df, start, baseline_hr)
                    if ts is not None:
                        presentation_timeseries.append(ts)

                # Step 2: average across presentations → one timeseries per bat
                bat_avg_ts = average_presentations(presentation_timeseries)
                if bat_avg_ts is not None:
                    bat_level_data[sex][category].append(bat_avg_ts)
                    bat_counts[sex][category].add(bat_id)
                    print(f"    {category}: averaged {len(presentation_timeseries)} presentations")
                else:
                    print(f"    {category}: no valid presentations for {bat_id}")

        except Exception as e:
            print(f"  Error processing {file}: {e}")
            continue

# ── 2x2 Plot ──────────────────────────────────────────────────────────────────
category_order = ["Aggression", "Distortion", "Distress", "Echolocation"]
ax_positions   = [(0, 0), (0, 1), (1, 0), (1, 1)]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

for cat, (row, col) in zip(category_order, ax_positions):
    ax = axes[row][col]

    for sex in ["Female", "Male"]:

        # Step 3: average bat-level timeseries across bats of this sex
        avg, se = average_across_bats(bat_level_data[sex][cat])
        if not avg:
            print(f"  No data for {sex} {cat}")
            continue

        times      = np.array(sorted(avg.keys()))
        hr_changes = np.array([avg[t] for t in times])
        errors     = np.array([se[t]  for t in times])
        n_bats     = len(bat_counts[sex][cat])

        ax.plot(times, hr_changes,
                color=sex_colors[sex],
                linewidth=2.5, marker='o', markersize=4,
                label=f"{sex} (n={n_bats})",
                alpha=0.9)
        ax.fill_between(times,
                        hr_changes - errors,
                        hr_changes + errors,
                        color=sex_colors[sex], alpha=0.2)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.8,
               alpha=0.7, label='Stimulus Onset')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)

    ax.set_title(cat, fontweight='bold', pad=10)
    ax.set_xlabel('Time Relative to Stimulus Onset (s)')
    ax.set_ylabel('Δ Heart Rate (BPM)')
    ax.set_xlim(-10, 30)
    ax.set_ylim(-50, 250)
    ax.legend(loc='upper left', framealpha=0.9)

plt.suptitle('Heart Rate Response by Sex and Stimulus Category',
             fontsize=16, fontweight='bold', y=1.02)

output_path = '/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/timeseries_by_sex.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.show()