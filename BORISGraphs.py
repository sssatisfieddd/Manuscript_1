import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# FILE PARSING
# ─────────────────────────────────────────────────────────────

def calculate_behavior_durations(csv_path):
    """
    Calculate durations for each behavior in a BORIS CSV file.
    Returns a dictionary with behavior names as keys and list of durations as values.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return {}

    behaviors = df['Behavior'].dropna().unique()
    behavior_durations = {}

    for behavior in behaviors:
        behavior_df = df[df['Behavior'] == behavior].copy()
        behavior_df = behavior_df.sort_values('Time')

        durations = []
        i = 0
        while i < len(behavior_df):
            row = behavior_df.iloc[i]
            if row['Behavior type'] == 'START':
                j = i + 1
                while j < len(behavior_df):
                    next_row = behavior_df.iloc[j]
                    if next_row['Behavior type'] == 'STOP':
                        duration = next_row['Time'] - row['Time']
                        durations.append(duration)
                        i = j
                        break
                    j += 1
            i += 1

        if durations:
            behavior_durations[behavior] = durations

    return behavior_durations


def process_csv_files(csv_files, condition_name):
    """
    Process a list of CSV files for a given condition.
    Returns a DataFrame with columns: Condition, Behavior, Duration, File
    """
    all_data = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue

        print(f"  Processing {csv_path.name}...")
        behavior_durations = calculate_behavior_durations(csv_path)

        for behavior, durations in behavior_durations.items():
            for duration in durations:
                all_data.append({
                    'Condition': condition_name,
                    'Behavior': behavior,
                    'Duration': duration,
                    'File': csv_path.name
                })

    return pd.DataFrame(all_data)


def process_csv_files_per_file(csv_files, condition_name):
    """
    Process CSV files and calculate MEAN duration per file (n=1 per file).
    This treats each file as an independent observation.
    """
    all_data = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue

        print(f"  Processing {csv_path.name}...")
        behavior_durations = calculate_behavior_durations(csv_path)

        for behavior, durations in behavior_durations.items():
            mean_duration = np.mean(durations) if len(durations) > 0 else 0
            count = len(durations)

            all_data.append({
                'Condition': condition_name,
                'Behavior': behavior,
                'MeanDuration': mean_duration,
                'Count': count,
                'File': csv_path.name,
                'Subject': csv_path.stem
            })

    return pd.DataFrame(all_data)


# ─────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────

CONDITIONS = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
BEHAVIORS_TO_KEEP = ['Grooming', 'Pointing', 'Touching Box']

CONDITION_COLORS = {
    'Distress_Nonsocial': '#1a0b2e',
    'Distress_Social':    '#9b4f96',
    'WN_Nonsocial':       '#ff6b9d',
    'WN_Social':          '#ffd23f',
}


# ─────────────────────────────────────────────────────────────
# BASIC PLOTS (no statistics)
# ─────────────────────────────────────────────────────────────

def create_duration_plot(data, output_path='behavior_duration_barplot.png'):
    """
    Create subplots for each behavior showing mean duration across conditions.
    """
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})

    print(f"\n*** ALL BEHAVIORS FOUND IN DATA: {sorted(data['Behavior'].unique())}")
    data = data[data['Behavior'].isin(BEHAVIORS_TO_KEEP)]
    print(f"*** BEHAVIORS AFTER FILTERING: {sorted(data['Behavior'].unique())}")

    behaviors = sorted(data['Behavior'].unique())
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 6))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        means, sems = [], []

        for condition in CONDITIONS:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            means.append(subset['Duration'].mean() if len(subset) > 0 else 0)
            sems.append(subset['Duration'].sem()  if len(subset) > 0 else 0)

        x_positions = np.arange(len(CONDITIONS))

        for i, condition in enumerate(CONDITIONS):
            ax.bar(x_positions[i], means[i],
                   color=CONDITION_COLORS[condition],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=sems[i], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        ax.set_xticks(x_positions)
        ax.set_ylim(0, 40)
        ax.set_xticklabels(CONDITIONS, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Duration (s)', fontsize=11, fontweight='bold')
        ax.set_title(behavior, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    fig.suptitle('Behavior Durations Across Conditions', fontsize=16, fontweight='bold', y=0.98)

    legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                     facecolor=CONDITION_COLORS[c],
                                     edgecolor='black', alpha=0.85, label=c)
                       for c in CONDITIONS]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               title='Conditions', title_fontsize=11, bbox_to_anchor=(0.98, 0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nDuration plot saved to {output_path}")
    plt.show()
    return fig


def create_frequency_plot(data, output_path='behavior_frequency_barplot.png'):
    """
    Create subplots for each behavior showing frequency across conditions.
    """
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    data = data[data['Behavior'].isin(BEHAVIORS_TO_KEEP)]

    count_data = data.groupby(['Condition', 'File', 'Behavior']).size().reset_index(name='Count')

    behaviors = sorted(count_data['Behavior'].unique())
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 6))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        means, sems = [], []

        for condition in CONDITIONS:
            subset = count_data[(count_data['Condition'] == condition) &
                                (count_data['Behavior'] == behavior)]
            means.append(subset['Count'].mean() if len(subset) > 0 else 0)
            sems.append(subset['Count'].sem()  if len(subset) > 0 else 0)

        x_positions = np.arange(len(CONDITIONS))

        for i, condition in enumerate(CONDITIONS):
            ax.bar(x_positions[i], means[i],
                   color=CONDITION_COLORS[condition],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=sems[i], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONDITIONS, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Frequency (count)', fontsize=11, fontweight='bold')
        ax.set_title(behavior, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(0, 6)

    fig.suptitle('Behavior Frequency Across Conditions', fontsize=16, fontweight='bold', y=0.98)

    legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                     facecolor=CONDITION_COLORS[c],
                                     edgecolor='black', alpha=0.8, label=c)
                       for c in CONDITIONS]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               title='Conditions', title_fontsize=11, bbox_to_anchor=(0.98, 0.86))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFrequency plot saved to {output_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# STATISTICS (scipy only — no statsmodels, no scikit-posthocs)
# ─────────────────────────────────────────────────────────────

def perform_kruskal_wallis(data, behavior):
    """
    Kruskal-Wallis H-test across all 4 conditions.
    Returns (result_table, behavior_data) or None.
    """
    behavior_data = data[data['Behavior'] == behavior].copy()

    groups = [
        behavior_data[behavior_data['Condition'] == c]['MeanDuration'].values
        for c in CONDITIONS
        if len(behavior_data[behavior_data['Condition'] == c]) > 0
    ]

    if len(groups) < 2:
        return None

    try:
        h_stat, p_val = stats.kruskal(*groups)
        result_table = pd.DataFrame({
            'H-statistic': [round(h_stat, 4)],
            'p-value':     [round(p_val, 4)],
            'significant': [p_val < 0.05]
        })
        print("\nKruskal-Wallis Test:")
        print(result_table.to_string(index=False))
        print(f"\nSample sizes per condition:")
        print(behavior_data.groupby('Condition').size().to_string())
        return result_table, behavior_data
    except Exception as e:
        print(f"    Kruskal-Wallis failed for {behavior}: {e}")
        return None


def perform_posthoc_mannwhitney(data, behavior):
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.
    Returns a symmetric p-value DataFrame (same shape as scikit-posthocs output).
    """
    behavior_data = data[data['Behavior'] == behavior].copy()
    available = [c for c in CONDITIONS if c in behavior_data['Condition'].values]

    if len(available) < 2:
        return None

    try:
        n = len(available)
        p_matrix = pd.DataFrame(np.ones((n, n)), index=available, columns=available)
        pairs = list(combinations(available, 2))
        n_comparisons = len(pairs)

        for c1, c2 in pairs:
            g1 = behavior_data[behavior_data['Condition'] == c1]['MeanDuration']
            g2 = behavior_data[behavior_data['Condition'] == c2]['MeanDuration']

            if len(g1) > 1 and len(g2) > 1:
                _, p_raw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                p_bonf = min(p_raw * n_comparisons, 1.0)
            else:
                p_bonf = 1.0

            p_matrix.loc[c1, c2] = p_bonf
            p_matrix.loc[c2, c1] = p_bonf

        return p_matrix
    except Exception as e:
        print(f"    Post-hoc failed for {behavior}: {e}")
        return None


def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'ns'


def add_significance_bracket(ax, x1, x2, y, p_value, height_multiplier=0.05):
    sig_text = get_significance_stars(p_value)
    if sig_text == 'ns':
        return
    bracket_h = y * height_multiplier
    ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y],
            lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + bracket_h * 1.2, sig_text,
            ha='center', va='bottom', fontsize=11, fontweight='bold')


def create_duration_plot_with_stats(data, output_path='behavior_duration_with_stats.png'):
    """
    Duration bar plots with Kruskal-Wallis + Bonferroni-corrected
    Mann-Whitney post-hoc significance brackets.
    """
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    data = data[data['Behavior'].isin(BEHAVIORS_TO_KEEP)]

    behaviors = sorted(data['Behavior'].unique())
    n_behaviors = len(behaviors)

    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None

    fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 7))
    plt.subplots_adjust(wspace=0.4)

    if n_behaviors == 1:
        axes = [axes]

    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS — DURATION (File-Level Analysis)")
    print("=" * 70)

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]

        # Compute per-condition summaries
        summary = []
        for condition in CONDITIONS:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            if len(subset) > 0:
                summary.append({
                    'condition': condition,
                    'mean': subset['MeanDuration'].mean(),
                    'sem':  subset['MeanDuration'].sem(),
                    'n':    len(subset)
                })
            else:
                summary.append({'condition': condition, 'mean': 0, 'sem': 0, 'n': 0})

        summary_df = pd.DataFrame(summary)
        x_positions = np.arange(len(CONDITIONS))

        for i, row in summary_df.iterrows():
            ax.bar(x_positions[i], row['mean'],
                   color=CONDITION_COLORS[row['condition']],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=row['sem'], capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})

        print(f"\n{'─'*50}")
        print(f"BEHAVIOR: {behavior}")
        print(f"{'─'*50}")

        # Kruskal-Wallis
        kw_result = perform_kruskal_wallis(data, behavior)

        # Post-hoc pairwise tests
        posthoc = perform_posthoc_mannwhitney(data, behavior)
        if posthoc is not None:
            print("\nPost-hoc Mann-Whitney U (Bonferroni-corrected p-values):")
            print(posthoc.round(4).to_string())

            y_max = summary_df['mean'].max() + summary_df['sem'].max()
            comparisons = [
                ('Distress_Nonsocial', 'Distress_Social', 0),
                ('WN_Nonsocial',       'WN_Social',       1),
                ('Distress_Nonsocial', 'WN_Nonsocial',    2),
                ('Distress_Social',    'WN_Social',       3),
            ]
            bracket_y_start = y_max * 1.1

            for c1, c2, offset in comparisons:
                if c1 in posthoc.index and c2 in posthoc.columns:
                    p_val = posthoc.loc[c1, c2]
                    if p_val < 0.05:
                        i1 = CONDITIONS.index(c1)
                        i2 = CONDITIONS.index(c2)
                        bracket_y = bracket_y_start + offset * y_max * 0.15
                        add_significance_bracket(ax, i1, i2, bracket_y, p_val)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONDITIONS, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Mean Duration (s)', fontsize=11, fontweight='bold')
        ax.set_title(behavior, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        max_y = summary_df['mean'].max() + summary_df['sem'].max()
        ax.set_ylim(bottom=0, top=max(max_y * 1.8, 1))

    fig.suptitle('Behavior Durations Across Conditions',
                 fontsize=16, fontweight='bold', y=0.98)

    legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                     facecolor=CONDITION_COLORS[c],
                                     edgecolor='black', alpha=0.85, label=c)
                       for c in CONDITIONS]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               title='Conditions', title_fontsize=11, bbox_to_anchor=(0.98, 0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Duration plot saved to {output_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    file_lists = {
        'Distress_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AEF.D2.Diss.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BOC.D2.Diss.NS.csv',
            '/Volumes/T7/UPDATEBORIS/BIB.D1.Diss.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AF8.Diss.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D2.NS.Diss.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.B31.F.D1.NS.Diss.csv',
        ],
        'Distress_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AFD.D2.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B0F.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B07.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D1.S.Diss.Batch2.csv',
            '/Volumes/T7/UPDATEBORIS/B26.D2.Soc.Diss.csv',
            '/Volumes/T7/UPDATEBORIS/B31.WN.D2.Social.csv'
            ],
        'WN_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B07.D2.Wn.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B0F.WN.D2.NS.csv',
            '/Volumes/T7/UPDATEBORIS/AE7.WN.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AFD.D1.WN.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D2.NS.Wn.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.F.D1.WN.NS.B26.csv',
        ],
        'WN_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AF8.WN.D2.S.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BIB.WN.D2.S.csv',
            '/Volumes/T7/UPDATEBORIS/AFE.WN.D1.Social.csv',
            '/Volumes/T7/UPDATEBORIS/BOC.WN.D1.S.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D1.S.WN.batch2.csv',
            '/Volumes/T7/UPDATEBORIS/AE7.Diss.D2.Social.csv'
        ],
    }

    # ── Basic plots ───────────────────────────────────────────
    print("\n" + "="*50)
    print("PROCESSING FILES FOR BASIC PLOTS")
    print("="*50)

    all_data = []
    for condition_name, csv_files in file_lists.items():
        print(f"\nProcessing condition: {condition_name}")
        try:
            condition_data = process_csv_files(csv_files, condition_name)
            all_data.append(condition_data)
            print(f"  Found {len(condition_data)} behavior instances")
        except Exception as e:
            print(f"  Error: {e}")

    all_data = [df for df in all_data if len(df) > 0]
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        print(f"\nTotal observations : {len(combined_data)}")
        print(f"Behaviors found    : {sorted(combined_data['Behavior'].unique())}")
        print("\nObservations per condition:")
        print(combined_data.groupby('Condition').size().to_string())
        print("\nMean duration by behavior and condition:")
        print(combined_data.groupby(['Condition', 'Behavior'])['Duration']
              .agg(['mean', 'std', 'count']).round(2).to_string())

        create_duration_plot(combined_data)
        create_frequency_plot(combined_data)

        combined_data.to_csv('processed_behavior_data.csv', index=False)
        print("\nProcessed data saved to 'processed_behavior_data.csv'")
    else:
        print("\nNo data was processed. Please check your file paths.")

    # ── Statistical analysis ──────────────────────────────────
    print("\n" + "="*50)
    print("RUNNING STATISTICAL ANALYSIS (file-level means)")
    print("="*50)

    all_data_stats = []
    for condition_name, csv_files in file_lists.items():
        print(f"\nProcessing condition: {condition_name}")
        try:
            condition_data = process_csv_files_per_file(csv_files, condition_name)
            all_data_stats.append(condition_data)
            print(f"  Processed {len(condition_data)} file-behavior rows")
        except Exception as e:
            print(f"  Error: {e}")

    all_data_stats = [df for df in all_data_stats if len(df) > 0]
    if all_data_stats:
        combined_stats = pd.concat(all_data_stats, ignore_index=True)

        print(f"\nTotal files analyzed : {combined_stats['File'].nunique()}")
        print(f"Behaviors found      : {sorted(combined_stats['Behavior'].unique())}")
        print("\nFiles per condition:")
        print(combined_stats.groupby('Condition')['File'].nunique().to_string())
        print("\nMean duration by behavior and condition:")
        print(combined_stats.groupby(['Condition', 'Behavior'])['MeanDuration']
              .agg(['mean', 'std', 'count']).round(2).to_string())

        create_duration_plot_with_stats(combined_stats)

        combined_stats.to_csv('processed_behavior_data_file_level.csv', index=False)
        print("\nProcessed data saved to 'processed_behavior_data_file_level.csv'")
    else:
        print("\nNo statistical data was processed. Please check your file paths.")