import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    
    # Get unique behaviors in this file
    behaviors = df['Behavior'].dropna().unique()
    
    behavior_durations = {}
    
    for behavior in behaviors:
        # Filter for this specific behavior
        behavior_df = df[df['Behavior'] == behavior].copy()
        behavior_df = behavior_df.sort_values('Time')
        
        durations = []
        
        # Find START-STOP pairs
        i = 0
        while i < len(behavior_df):
            row = behavior_df.iloc[i]
            
            if row['Behavior type'] == 'START':
                # Look for the corresponding STOP
                j = i + 1
                while j < len(behavior_df):
                    next_row = behavior_df.iloc[j]
                    if next_row['Behavior type'] == 'STOP':
                        duration = next_row['Time'] - row['Time']
                        durations.append(duration)
                        i = j  # Move to the STOP event
                        break
                    j += 1
            i += 1
        
        if durations:  # Only add if we found at least one duration
            behavior_durations[behavior] = durations
    
    return behavior_durations

def process_csv_files(csv_files, condition_name):
    """
    Process a list of CSV files for a given condition.
    Returns a DataFrame with columns: Condition, Behavior, Duration
    """
    all_data = []
    
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue
            
        print(f"  Processing {csv_path.name}...")
        behavior_durations = calculate_behavior_durations(csv_path)
        
        # Convert to list of records
        for behavior, durations in behavior_durations.items():
            for duration in durations:
                all_data.append({
                    'Condition': condition_name,
                    'Behavior': behavior,
                    'Duration': duration,
                    'File': csv_path.name
                })
    
    return pd.DataFrame(all_data)

def create_duration_plot(data, output_path='behavior_duration_barplot.png'):
    """
    Create subplots for each behavior showing duration across conditions.
    Each behavior gets its own subplot arranged horizontally (side by side).
    """
    
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    
    # Print all unique behaviors found before filtering
    print(f"\n*** ALL BEHAVIORS FOUND IN DATA: {sorted(data['Behavior'].unique())}")
    
    # Print all unique behaviors found before filtering
    print(f"\n*** ALL BEHAVIORS FOUND IN DATA: {sorted(data['Behavior'].unique())}")
    
    # FILTER FOR ONLY SPECIFIC BEHAVIORS
    behaviors_to_keep = ['Grooming', 'Pointing', 'Touching Box']
    data = data[data['Behavior'].isin(behaviors_to_keep)]
    
    print(f"*** BEHAVIORS AFTER FILTERING: {sorted(data['Behavior'].unique())}")
    
    print(f"*** BEHAVIORS AFTER FILTERING: {sorted(data['Behavior'].unique())}")
    
    # Define condition order - ALWAYS show all 4 conditions
    conditions = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
    n_conditions = len(conditions)
    
    # NEW ROCKET PALETTE - Different shades but same vibe
    condition_colors = {
        'Distress_Nonsocial': '#1a0b2e',  # Deep midnight purple
        'Distress_Social': '#9b4f96',      # Magenta purple
        'WN_Nonsocial': '#ff6b9d',         # Hot pink
        'WN_Social': '#ffd23f'             # Golden yellow
    }
    
    # Get unique behaviors
    behaviors = sorted(data['Behavior'].unique())
    n_behaviors = len(behaviors)
    
    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None
    
    # Set up the subplots - arranged HORIZONTALLY (1 row, multiple columns)
    fig, axes = plt.subplots(1, n_behaviors, figsize=(5*n_behaviors, 6))
    
    # Add more spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Make axes always iterable
    if n_behaviors == 1:
        axes = [axes]
    
    # Plot each behavior in its own subplot
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        
        means = []
        sems = []
        
        for condition in conditions:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            if len(subset) > 0:
                means.append(subset['Duration'].mean())
                sems.append(subset['Duration'].sem())
            else:
                means.append(0)
                sems.append(0)
        
        # Create bar positions
        x_positions = np.arange(n_conditions)
        
        # Plot bars for this behavior
        bars = []
        for i, condition in enumerate(conditions):
            bar = ax.bar(x_positions[i], means[i], 
                   color=condition_colors[condition],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.5,
                   yerr=sems[i],
                   capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})
            bars.append(bar)
        
        # Customize subplot
        ax.set_xticks(x_positions)
        ax.set_ylim(0, 40)
        ax.set_xticklabels(conditions, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Duration (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{behavior}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    
    # Add overall title and legend
    fig.suptitle('Behavior Durations Across Conditions', fontsize=16, fontweight='bold', y=0.98)
    
    # Create custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=condition_colors[c], 
                                     edgecolor='black', alpha=0.85, label=c) 
                      for c in conditions]
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
    Each behavior gets its own subplot arranged horizontally (side by side).
    """
    
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    
    # FILTER FOR ONLY SPECIFIC BEHAVIORS
    behaviors_to_keep = ['Grooming', 'Pointing', 'Touching Box']
    data = data[data['Behavior'].isin(behaviors_to_keep)]
    
    # Count occurrences per file for each behavior-condition combination
    count_data = data.groupby(['Condition', 'File', 'Behavior']).size().reset_index(name='Count')
    
    # Define condition order - ALWAYS show all 4 conditions
    conditions = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
    n_conditions = len(conditions)
    
    # NEW ROCKET PALETTE - Same as duration plot
    condition_colors = {
        'Distress_Nonsocial': '#1a0b2e',  # Deep midnight purple
        'Distress_Social': '#9b4f96',      # Magenta purple
        'WN_Nonsocial': '#ff6b9d',         # Hot pink
        'WN_Social': '#ffd23f'             # Golden yellow
    }
    
    # Get unique behaviors
    behaviors = sorted(count_data['Behavior'].unique())
    n_behaviors = len(behaviors)
    
    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None
    
    # Set up the subplots - arranged HORIZONTALLY (1 row, multiple columns)
    fig, axes = plt.subplots(1, n_behaviors, figsize=(5*n_behaviors, 6))
    
    # Add more spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    
    # Make axes always iterable
    if n_behaviors == 1:
        axes = [axes]
    
    # Plot each behavior in its own subplot
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        
        means = []
        sems = []
        
        for condition in conditions:
            subset = count_data[(count_data['Condition'] == condition) & (count_data['Behavior'] == behavior)]
            if len(subset) > 0:
                means.append(subset['Count'].mean())
                sems.append(subset['Count'].sem())
            else:
                means.append(0)
                sems.append(0)
        
        # Create bar positions
        x_positions = np.arange(n_conditions)
        
        # Plot bars for this behavior
        bars = []
        for i, condition in enumerate(conditions):
            bar = ax.bar(x_positions[i], means[i], 
                   color=condition_colors[condition],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.5,
                   yerr=sems[i],
                   capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})
            bars.append(bar)
        
        # Customize subplot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Frequency (count)', fontsize=11, fontweight='bold')
        ax.set_title(f'{behavior}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(0, 6)
    
    # Add overall title and legend
    fig.suptitle('Behavior Frequency Across Conditions', fontsize=16, fontweight='bold', y=0.98)
    
    # Create custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=condition_colors[c], 
                                     edgecolor='black', alpha=0.8, label=c) 
                      for c in conditions]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              title='Conditions', title_fontsize=11, bbox_to_anchor=(0.98, 0.86))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFrequency plot saved to {output_path}")
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Define your CSV files for each condition
    file_lists = {
        'Distress_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AEF.D2.Diss.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BOC.D2.Diss.NS.csv',
            '/Volumes/T7/UPDATEBORIS/BIB.D1.Diss.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AF8.Diss.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D2.NS.Diss.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.B31.F.D1.NS.Diss.csv'
        ],
        
        'Distress_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AFD.D2.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B0F.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/B07.D1.Diss.S.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D1.S.Diss.Batch2.csv',
            '/Volumes/T7/UPDATEBORIS/B26.D2.Soc.Diss.csv'
        ],
    
        'WN_Nonsocial': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B07.D2.Wn.NS.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/B0F.WN.D2.NS.csv',
            '/Volumes/T7/UPDATEBORIS/AE7.WN.D1.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/AFD.D1.WN.Nonsocial.csv',
            '/Volumes/T7/UPDATEBORIS/ADB.D2.NS.Wn.csv',
            '/Volumes/T7/UPDATEBORIS/Batch2.F.D1.WN.NS.B26.csv'
        ],
        
        'WN_Social': [
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/AF8.WN.D2.S.csv',
            '/Volumes/T7/Day2.Redo.2weekstent/Batch1.Day2/BIB.WN.D2.S.csv',
            '/Volumes/T7/UPDATEBORIS/AFE.WN.D1.Social.csv',
            '/Volumes/T7/UPDATEBORIS/BOC.WN.D1.S.csv',
            '/Volumes/T7/UPDATEBORIS/B06.D1.S.WN.batch2.csv'
        ],
    }
    
    # Process all conditions
    all_data = []
    
    for condition_name, csv_files in file_lists.items():
        print(f"\n{'='*50}")
        print(f"Processing condition: {condition_name}")
        print(f"{'='*50}")
        
        try:
            condition_data = process_csv_files(csv_files, condition_name)
            all_data.append(condition_data)
            print(f"  Found {len(condition_data)} behavior instances in {condition_name}")
        except Exception as e:
            print(f"  Error processing {condition_name}: {e}")
    
    # Combine all data
    if all_data:
        # Filter out empty dataframes
        all_data = [df for df in all_data if len(df) > 0]
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Print summary statistics
            print(f"\n{'='*50}")
            print("SUMMARY STATISTICS")
            print(f"{'='*50}")
            print(f"\nTotal observations: {len(combined_data)}")
            print(f"\nBehaviors found: {sorted(combined_data['Behavior'].unique())}")
            print(f"\nObservations per condition:")
            print(combined_data.groupby('Condition').size())
            print(f"\nMean duration by behavior and condition:")
            print(combined_data.groupby(['Condition', 'Behavior'])['Duration'].agg(['mean', 'std', 'count']).round(2))
            
            # Create the duration plot
            create_duration_plot(combined_data)
            
            # Create the frequency plot
            create_frequency_plot(combined_data)
            
            # Optionally save the processed data
            combined_data.to_csv('processed_behavior_data.csv', index=False)
            print("\nProcessed data saved to 'processed_behavior_data.csv'")
        else:
            print("\nNo behavior data was extracted from the files.")
    else:
        print("\nNo data was processed. Please check your file paths.")


# STATISTICAL ANALYSIS SECTION
import pandas as pd
from pathlib import Path
import warnings
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

warnings.filterwarnings('ignore')

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
        
        # Calculate mean duration per behavior per file
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

def perform_two_way_anova(data, behavior):
    """Perform two-way ANOVA for a specific behavior using file-level means."""
    behavior_data = data[data['Behavior'] == behavior].copy()
    
    if len(behavior_data) < 4:
        return None
    
    # Add factors for ANOVA
    behavior_data['CallType'] = behavior_data['Condition'].apply(
        lambda x: 'Distress' if 'Distress' in x else 'WN'
    )
    behavior_data['SocialContext'] = behavior_data['Condition'].apply(
        lambda x: 'Social' if 'Social' in x else 'Nonsocial'
    )
    
    try:
        model = ols('MeanDuration ~ C(CallType) * C(SocialContext)', data=behavior_data).fit()
        anova_table = anova_lm(model, typ=2)
        return anova_table, behavior_data
    except Exception as e:
        print(f"    ANOVA failed for {behavior}: {e}")
        return None

def perform_posthoc_tukey(data, behavior):
    """Perform Tukey HSD post-hoc test for pairwise comparisons."""
    behavior_data = data[data['Behavior'] == behavior].copy()
    
    conditions = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
    available_conditions = [c for c in conditions if c in behavior_data['Condition'].values]
    
    if len(available_conditions) < 2:
        return None
    
    try:
        posthoc = sp.posthoc_tukey(behavior_data, val_col='MeanDuration', group_col='Condition')
        return posthoc
    except Exception as e:
        print(f"    Post-hoc failed for {behavior}: {e}")
        return None

def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def add_significance_bracket(ax, x1, x2, y, p_value, height_multiplier=0.05):
    """Add significance bracket to plot."""
    sig_text = get_significance_stars(p_value)
    
    if sig_text == 'ns':
        return
    
    bracket_h = y * height_multiplier
    ax.plot([x1, x1, x2, x2], [y, y+bracket_h, y+bracket_h, y], 
            lw=1.5, c='black')
    
    ax.text((x1+x2)/2, y+bracket_h*1.2, sig_text, 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

def create_duration_plot_with_stats(data, output_path='behavior_duration_with_stats.png'):
    """Create duration plots with statistical comparisons."""
    
    data = data.copy()
    data['Behavior'] = data['Behavior'].replace({'Flapping': 'Flying'})
    
    # FILTER FOR ONLY SPECIFIC BEHAVIORS
    behaviors_to_keep = ['Grooming', 'Pointing', 'Touching Box']
    data = data[data['Behavior'].isin(behaviors_to_keep)]
    
    conditions = ['Distress_Nonsocial', 'Distress_Social', 'WN_Nonsocial', 'WN_Social']
    n_conditions = len(conditions)
    
    # NEW ROCKET PALETTE
    condition_colors = {
        'Distress_Nonsocial': '#1a0b2e',  # Deep midnight purple
        'Distress_Social': '#9b4f96',      # Magenta purple
        'WN_Nonsocial': '#ff6b9d',         # Hot pink
        'WN_Social': '#ffd23f'             # Golden yellow
    }
    
    behaviors = sorted(data['Behavior'].unique())
    n_behaviors = len(behaviors)
    
    if n_behaviors == 0:
        print("No data found for the specified behaviors!")
        return None
    
    fig, axes = plt.subplots(1, n_behaviors, figsize=(5*n_behaviors, 7))
    plt.subplots_adjust(wspace=0.4)
    
    if n_behaviors == 1:
        axes = [axes]
    
    print("\n" + "="*70)
    print("STATISTICAL RESULTS - DURATION (File-Level Analysis)")
    print("="*70)
    
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        
        summary_stats = []
        for condition in conditions:
            subset = data[(data['Condition'] == condition) & (data['Behavior'] == behavior)]
            if len(subset) > 0:
                mean_val = subset['MeanDuration'].mean()
                sem_val = subset['MeanDuration'].sem()
                n = len(subset)
                summary_stats.append({
                    'condition': condition,
                    'mean': mean_val,
                    'sem': sem_val,
                    'n': n
                })
            else:
                summary_stats.append({
                    'condition': condition,
                    'mean': 0,
                    'sem': 0,
                    'n': 0
                })
        
        summary_df = pd.DataFrame(summary_stats)
        x_positions = np.arange(n_conditions)
        
        for i, row in summary_df.iterrows():
            ax.bar(x_positions[i], row['mean'], 
                   color=condition_colors[row['condition']],
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.5,
                   yerr=row['sem'],
                   capsize=5,
                   error_kw={'linewidth': 2, 'elinewidth': 1.5})
        
        print(f"\n{behavior}:")
        print("-" * 50)
        
        anova_results = perform_two_way_anova(data, behavior)
        if anova_results is not None:
            anova_table, behavior_data = anova_results
            print("\nTwo-Way ANOVA:")
            print(anova_table)
            print(f"\nSample sizes per condition:")
            print(behavior_data.groupby('Condition').size())
        
        posthoc_results = perform_posthoc_tukey(data, behavior)
        if posthoc_results is not None:
            print("\nPost-hoc Tukey HSD (p-values):")
            print(posthoc_results)
            
            y_max = summary_df['mean'].max() + summary_df['sem'].max()
            
            comparisons = [
                ('Distress_Nonsocial', 'Distress_Social', 0),
                ('WN_Nonsocial', 'WN_Social', 1),
                ('Distress_Nonsocial', 'WN_Nonsocial', 2),
                ('Distress_Social', 'WN_Social', 3),
            ]
            
            bracket_y_start = y_max * 1.1
            
            for comp1, comp2, offset in comparisons:
                if comp1 in posthoc_results.index and comp2 in posthoc_results.columns:
                    p_val = posthoc_results.loc[comp1, comp2]
                    
                    if p_val < 0.05:
                        idx1 = conditions.index(comp1)
                        idx2 = conditions.index(comp2)
                        
                        bracket_y = bracket_y_start + (offset * y_max * 0.15)
                        add_significance_bracket(ax, idx1, idx2, bracket_y, p_val)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Mean Duration (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{behavior}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        max_y = summary_df['mean'].max() + summary_df['sem'].max()
        ax.set_ylim(bottom=0, top=max_y * 1.8)
    
    fig.suptitle('Behavior Durations Across Conditions', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=condition_colors[c], 
                                     edgecolor='black', alpha=0.85, label=c) 
                      for c in conditions]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              title='Conditions', title_fontsize=11, bbox_to_anchor=(0.98, 0.85))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Duration plot saved to {output_path}")
    plt.show()
    
    return fig

# Run the statistical analysis
print("\n" + "="*70)
print("RUNNING STATISTICAL ANALYSIS")
print("="*70)

all_data_stats = []

for condition_name, csv_files in file_lists.items():
    print(f"\n{'='*50}")
    print(f"Processing condition: {condition_name}")
    print(f"{'='*50}")
    
    try:
        condition_data = process_csv_files_per_file(csv_files, condition_name)
        all_data_stats.append(condition_data)
        print(f"  Processed {len(condition_data)} files in {condition_name}")
    except Exception as e:
        print(f"  Error processing {condition_name}: {e}")

if all_data_stats:
    all_data_stats = [df for df in all_data_stats if len(df) > 0]
    
    if all_data_stats:
        combined_data_stats = pd.concat(all_data_stats, ignore_index=True)
        
        print(f"\n{'='*50}")
        print("SUMMARY STATISTICS (File-Level)")
        print(f"{'='*50}")
        print(f"\nTotal files analyzed: {combined_data_stats['File'].nunique()}")
        print(f"\nBehaviors found: {sorted(combined_data_stats['Behavior'].unique())}")
        print(f"\nFiles per condition:")
        print(combined_data_stats.groupby('Condition')['File'].nunique())
        print(f"\nMean duration by behavior and condition:")
        print(combined_data_stats.groupby(['Condition', 'Behavior'])['MeanDuration'].agg(['mean', 'std', 'count']).round(2))
        
        create_duration_plot_with_stats(combined_data_stats)
        
        combined_data_stats.to_csv('processed_behavior_data_file_level.csv', index=False)
        print("\nProcessed data saved to 'processed_behavior_data_file_level.csv'")