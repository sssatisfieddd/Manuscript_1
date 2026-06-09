"""
Bat Heart Rate Analysis with Robust Baseline Detection and Statistical Analysis
FIXES:
  1. bat_id is now raw (e.g. "AF4") for time-window and sex lookup;
     "Bat AF4" label is kept separately for display only
  2. HR Change filter > 0 and <= 300 added (matches R pipeline)
  3. Sex assigned from hardcoded bat ID lookup (matches R pipeline)
  4. Category spoke labels removed from radar chart body;
     placed outside the outermost ring instead. Ring numbers kept.
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, false_discovery_control
import itertools

# ── Font setup ────────────────────────────────────────────────────────────────
calibri_light_path = None
for font in fm.findSystemFonts(fontext='ttf') + fm.findSystemFonts(fontext='otf'):
    name = os.path.basename(font).lower()
    if 'calibri' in name and ('light' in name or 'l.' in name):
        calibri_light_path = font
        break

if not calibri_light_path:
    search_dirs = [
        os.path.expanduser('~/Library/Fonts'),
        '/Library/Fonts',
        '/System/Library/Fonts',
        '/System/Library/Fonts/Supplemental',
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            fl = fname.lower()
            if 'calibri' in fl and ('light' in fl or fl == 'calibril.ttf'):
                calibri_light_path = os.path.join(d, fname)
                break
        if calibri_light_path:
            break

if calibri_light_path:
    fm.fontManager.addfont(calibri_light_path)
    calibri_light_props = fm.FontProperties(fname=calibri_light_path)
    FONT_NAME = calibri_light_props.get_name()
    print(f"Calibri Light found: {calibri_light_path} -> '{FONT_NAME}'")
else:
    print("Calibri Light not found — falling back to DejaVu Sans")
    FONT_NAME = 'DejaVu Sans'

plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['font.size']   = 8

# ── Folder paths ──────────────────────────────────────────────────────────────
female_folder_path = '/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Females'
male_folder_path   = '/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Males'

female_csv_files = glob.glob(os.path.join(female_folder_path, "*.csv"))
male_csv_files   = glob.glob(os.path.join(male_folder_path,   "*.csv"))

# ── Time windows ──────────────────────────────────────────────────────────────
time_windows = {
    "Aggression":   [(61.302,90.813),   (691.400,720.914),  (871.504,901.015),
                     (961.503,991.014), (1231.604,1261.114),(1411.604,1441.113)],
    "Distress":     [(151.302,180.851), (241.303,270.072),  (601.403,630.953),
                     (1051.501,1081.052),(1951.803,1981.352),(2041.804,2071.353)],
    "Echolocation": [(421.402,450.44),  (511.746,540.944),  (1321.602,1351.145),
                     (1501.602,1531.146),(1771.803,1801.345),(1861.803,1891.345)],
    "Distortion":   [(2131.428,2161.605),(331.302,361.104), (781.401,811.205),
                     (1141.503,1171.305),(1591.603,1621.406),(1681.703,1711.504)],
}

alternative_time_windows = {
    "Aggression":   [(60.189,90.189),   (150.189,180.189),  (510.389,540.389),
                     (1590.790,1620.790),(1770.790,1800.790),(2130.791,2160.791)],
    "Distress":     [(420.389,450.389), (600.389,630.389),  (780.489,810.489),
                     (870.489,900.489), (1050.490,1080.490),(1950.790,1980.790)],
    "Echolocation": [(960.490,990.490), (1320.690,1350.690),(1500.790,1530.790),
                     (1680.790,1710.790),(1860.790,1890.790),(2040.790,2070.790)],
    "Distortion":   [(691.400,720.914), (1141.503,1171.305),(1231.604,1261.114),
                     (1411.604,1441.113),(241.303,270.072),  (331.302,361.104)],
}

# ── Bat ID lookups ────────────────────────────────────────────────────────────
ALT_BATS    = {"AF4", "AF6", "B32", "AD6"}
MALE_BATS   = {"A88", "AEO", "B05", "B1D", "B07", "AD6", "AF4"}
FEMALE_BATS = {"693", "B04", "CD3", "B30", "B32", "AF6"}

def get_time_windows(bat_id):
    return alternative_time_windows if bat_id in ALT_BATS else time_windows

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
            avg_hr = np.mean(window_hr)
            if avg_hr < min_avg_hr:
                min_avg_hr = avg_hr

    percentile_baseline = np.percentile(valid_hr, 5)
    return min(min_avg_hr, percentile_baseline) if min_avg_hr < 50 else percentile_baseline

# ── Data processing ───────────────────────────────────────────────────────────
plot_data           = []
baseline_comparison = []

for csv_files, gender_label in [(female_csv_files, "Female"), (male_csv_files, "Male")]:
    for file in csv_files:
        try:
            df        = pd.read_csv(file)
            file_name = os.path.basename(file)

            bat_id    = file_name[:3]
            bat_label = "Bat " + bat_id

            if bat_id in MALE_BATS:
                sex = "Male"
            elif bat_id in FEMALE_BATS:
                sex = "Female"
            else:
                sex = gender_label

            original_baseline = df[
                (df["File_Marker"] == 0) &
                (df["Heart_Rate"] > 0) &
                (df["Heart_Rate"] <= 310)
            ]["Heart_Rate"].mean()

            robust_baseline = calculate_robust_baseline(df)
            baseline_hr     = robust_baseline if robust_baseline is not None else original_baseline

            baseline_comparison.append({
                "Bat":               bat_label,
                "File":              file_name,
                "Original Baseline": original_baseline,
                "Robust Baseline":   baseline_hr,
                "Method Used":       "robust" if robust_baseline is not None else "original",
                "Difference":        original_baseline - baseline_hr if robust_baseline is not None else 0
            })

            if len(df[df["File_Marker"] == 1]) > 0:
                first_time = df[df["File_Marker"] == 1]["Time"].min()
                df["Time"] = df["Time"] - first_time

            bat_time_windows = get_time_windows(bat_id)

            for category, intervals in bat_time_windows.items():
                hr_values = []
                for start, end in intervals:
                    hr_segment = df[(df["Time"] >= start) & (df["Time"] <= end)]["Heart_Rate"]
                    hr_segment = hr_segment[(hr_segment > 0) & (hr_segment <= 500)]
                    hr_values.extend(hr_segment.tolist())

                if hr_values:
                    avg_hr               = np.mean(hr_values)
                    change_from_baseline = avg_hr - baseline_hr
                    plot_data.append({
                        "Bat":       bat_label,
                        "Category":  category,
                        "HR Change": change_from_baseline,
                        "Gender":    sex,
                        "Raw HR":    avg_hr,
                        "Baseline":  baseline_hr
                    })
                else:
                    print(f"No valid HR data for {bat_label} in {category}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

# ── DataFrames ────────────────────────────────────────────────────────────────
plot_df     = pd.DataFrame(plot_data)
baseline_df = pd.DataFrame(baseline_comparison)
baseline_df.to_csv("baseline_comparison.csv", index=False)

print(f"Avg baseline difference: {baseline_df['Difference'].mean():.2f} BPM")
print(f"Max baseline difference: {baseline_df['Difference'].max():.2f} BPM")

if plot_df.empty:
    print("No valid data.")
    raise SystemExit

plot_df = plot_df[(plot_df["HR Change"] > 0) & (plot_df["HR Change"] <= 300)].copy()

print(f"\nRows after filter: {len(plot_df)}")
print(plot_df.groupby(["Gender", "Category"])["HR Change"].median().unstack())

# ── Statistical analysis ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

categories = list(time_windows.keys())

for gender in ['Female', 'Male']:
    gender_data     = plot_df[plot_df['Gender'] == gender]
    category_groups = [gender_data[gender_data['Category'] == cat]['HR Change'].values
                       for cat in categories]

    print(f"\nCounts for {gender}:")
    for cat, grp in zip(categories, category_groups):
        print(f"  {cat}: {len(grp)}")

    if any(len(g) < 2 for g in category_groups):
        print(f"  Insufficient data — skipping.")
        continue

    h_stat, p_value = kruskal(*category_groups)
    print(f"\n{gender} — Kruskal-Wallis: H={h_stat:.4f}, p={p_value:.4f}")

    if p_value < 0.05:
        print("  Post-hoc pairwise (Mann-Whitney, FDR corrected):")
        p_values    = []
        comparisons = []
        for cat1, cat2 in itertools.combinations(categories, 2):
            d1 = gender_data[gender_data['Category'] == cat1]['HR Change'].values
            d2 = gender_data[gender_data['Category'] == cat2]['HR Change'].values
            if len(d1) > 0 and len(d2) > 0:
                u, p = mannwhitneyu(d1, d2, alternative='two-sided')
                p_values.append(p)
                comparisons.append((cat1, cat2))
        corrected_p = false_discovery_control(p_values)
        for (cat1, cat2), cp in zip(comparisons, corrected_p):
            sig = '***' if cp < 0.001 else '**' if cp < 0.01 else '*' if cp < 0.05 else 'ns'
            print(f"    {cat1} vs {cat2}: p={cp:.4f} {sig}")

print("\n--- Between-sex comparisons (Mann-Whitney U) ---")
for category in categories:
    f_data = plot_df[(plot_df['Gender'] == 'Female') & (plot_df['Category'] == category)]['HR Change'].values
    m_data = plot_df[(plot_df['Gender'] == 'Male')   & (plot_df['Category'] == category)]['HR Change'].values
    if len(f_data) > 0 and len(m_data) > 0:
        u, p = mannwhitneyu(f_data, m_data, alternative='two-sided')
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {category}: U={u:.4f}, p={p:.4f} {sig}")

# ── Radar chart ───────────────────────────────────────────────────────────────
radar_data = plot_df.pivot_table(
    values='HR Change', index='Gender', columns='Category', aggfunc='median'
)
print("\nRadar medians (after filter):")
print(radar_data)

N      = len(radar_data.columns)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

values_female = radar_data.loc['Female'].values.tolist() + [radar_data.loc['Female'].values[0]]
values_male   = radar_data.loc['Male'].values.tolist()   + [radar_data.loc['Male'].values[0]]

print("Female medians:", values_female)
print("Male medians:  ", values_male)

legend_font = fm.FontProperties(family=FONT_NAME, size=8)

fig = plt.figure(figsize=(12, 12), facecolor='white')
ax  = plt.subplot(111, polar=True)

ax.plot(angles, values_female, 'o-', linewidth=2.5, color='#800080',
        label='Female', alpha=0.9, markersize=8)
ax.fill(angles, values_female, color='#800080', alpha=0.15)

ax.plot(angles, values_male, 'o-', linewidth=2.5, color='#FFA500',
        label='Male', alpha=0.9, markersize=8)
ax.fill(angles, values_male, color='#FFA500', alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ── Remove category names from the spokes; keep ring (y-axis) numbers ────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])          # blank spoke labels — no category text on chart

# Keep y-axis ring numbers but apply font
ax.tick_params(axis='y', labelsize=8)
for label in ax.get_yticklabels():
    label.set_fontfamily(FONT_NAME)
    label.set_fontsize(8)

max_value = max(max(values_female), max(values_male)) + 10
ax.set_ylim(0, max_value)

# ── Place category labels outside the outermost ring ─────────────────────────
# Increase LABEL_OFFSET multiplier if labels feel too close to the chart
LABEL_OFFSET = max_value * 1.22

category_labels = list(radar_data.columns)
for angle, label in zip(angles[:-1], category_labels):
    ax.text(
        angle,
        LABEL_OFFSET,
        label,
        ha='center',
        va='center',
        fontsize=8,
        fontfamily=FONT_NAME,
        fontweight='bold',
        color='black'
    )

plt.legend(
    loc='upper right',
    bbox_to_anchor=(1.3, 1.1),
    fontsize=8,
    frameon=True,
    fancybox=True,
    shadow=True,
    prop=legend_font
)

plt.title(
    'Median Change in Heart Rate by Sex and Stimulus',
    fontsize=8,
    fontweight='bold',
    fontfamily=FONT_NAME,
    pad=20
)

plt.tight_layout()
plt.savefig('radar_chart_with_statistics.png', dpi=300, bbox_inches='tight')
plt.show()