#!/usr/bin/env python3
"""
Visualize Innovation Test Results
================================

Analyze and visualize the standardized innovation testing results 
from ARMA-GARCH model fitting across all subjects.

This shows model adequacy assessment on the standardized innovations 
(z_t = ε_t/σ_t) using Ljung-Box whiteness tests and goodness-of-fit 
tests against the selected distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_innovation_results():
    """Load innovation test results from fitted models."""
    
    # Try to load comprehensive results first
    results_file = Path("residual_analysis_clean/fitted_models/fit_summary_clean.csv")
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        subjects_in_main = set(df['subject'].unique())
        print(f"✅ Loaded {len(df)} models from main file")
        print(f"   Subjects in main: {sorted(subjects_in_main)}")
        
        # Check for additional subject-specific files
        fitted_models_dir = Path("residual_analysis_clean/fitted_models")
        additional_dfs = []
        
        for subject_dir in fitted_models_dir.glob("subject_*"):
            if subject_dir.is_dir():
                subject_name = subject_dir.name
                if subject_name not in subjects_in_main:
                    subject_csv = subject_dir / f"{subject_name}_results.csv"
                    if subject_csv.exists():
                        subject_df = pd.read_csv(subject_csv)
                        additional_dfs.append(subject_df)
                        print(f"   + Added {len(subject_df)} models from {subject_name}")
        
        # Combine all data
        if additional_dfs:
            all_dfs = [df] + additional_dfs
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"✅ Combined total: {len(combined_df)} fitted models")
        else:
            combined_df = df
            
    else:
        # Load from individual subject files
        print("📂 Loading from individual subject files...")
        fitted_models_dir = Path("residual_analysis_clean/fitted_models") 
        all_dfs = []
        
        for subject_dir in fitted_models_dir.glob("subject_*"):
            if subject_dir.is_dir():
                subject_csv = subject_dir / f"{subject_dir.name}_results.csv"
                if subject_csv.exists():
                    subject_df = pd.read_csv(subject_csv)
                    all_dfs.append(subject_df)
                    print(f"   ✅ {subject_dir.name}: {len(subject_df)} models")
        
        if not all_dfs:
            print(f"❌ No individual results files found")
            return None
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Combined total: {len(combined_df)} fitted models")
    
    print(f"   Final subjects: {sorted(combined_df['subject'].unique())}")
    return combined_df

def analyze_innovation_tests(df):
    """Analyze innovation test results across subjects."""
    
    print("\n" + "="*60)
    print("STANDARDIZED INNOVATION TEST ANALYSIS")
    print("="*60)
    
    # Overall statistics
    total_models = len(df)
    white_noise_count = df['is_white_noise'].sum()
    good_fit_count = df['good_fit'].sum()
    
    print(f"\nOverall Innovation Test Results:")
    print(f"  Total models fitted: {total_models}")
    print(f"  White noise (Ljung-Box p > 0.05): {white_noise_count} ({white_noise_count/total_models*100:.1f}%)")
    print(f"  Good distribution fit: {good_fit_count} ({good_fit_count/total_models*100:.1f}%)")
    
    # By subject analysis
    print(f"\nBy Subject:")
    for subject in sorted(df['subject'].unique()):
        subj_df = df[df['subject'] == subject]
        n_trials = len(subj_df)
        white_count = subj_df['is_white_noise'].sum()
        fit_count = subj_df['good_fit'].sum()
        
        print(f"  {subject}:")
        print(f"    Trials: {n_trials}")
        print(f"    White noise: {white_count}/{n_trials} ({white_count/n_trials*100:.1f}%)")
        print(f"    Good fit: {fit_count}/{n_trials} ({fit_count/n_trials*100:.1f}%)")
    
    # Distribution breakdown
    print(f"\nBy Distribution:")
    for dist in sorted(df['best_distribution'].unique()):
        dist_df = df[df['best_distribution'] == dist]
        n_models = len(dist_df)
        white_count = dist_df['is_white_noise'].sum()
        fit_count = dist_df['good_fit'].sum()
        
        print(f"  {dist}:")
        print(f"    Models: {n_models}")
        print(f"    White noise: {white_count}/{n_models} ({white_count/n_models*100:.1f}%)")
        print(f"    Good fit: {fit_count}/{n_models} ({fit_count/n_models*100:.1f}%)")
    
    return df

def create_innovation_heatmaps(df):
    """Create heatmaps showing innovation test results."""
    
    # Prepare data for heatmaps
    subjects = sorted(df['subject'].unique())
    
    # Create summary matrices for heatmaps
    whiteness_data = []
    fit_data = []
    combined_data = []
    
    for subject in subjects:
        subj_df = df[df['subject'] == subject]
        
        # Whiteness test results (percentage passing)
        white_rate = subj_df['is_white_noise'].mean() * 100
        whiteness_data.append(white_rate)
        
        # Distribution fit results (percentage passing) 
        fit_rate = subj_df['good_fit'].mean() * 100
        fit_data.append(fit_rate)
        
        # Combined adequacy (both tests pass)
        both_pass = (subj_df['is_white_noise'] & subj_df['good_fit']).mean() * 100
        combined_data.append(both_pass)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Clean subject labels
    subject_labels = [s.replace('subject_', '') for s in subjects]
    
    # 1. Whiteness test results
    ax1 = axes[0]
    sns.heatmap([[w] for w in whiteness_data], 
                yticklabels=subject_labels,
                xticklabels=['Ljung-Box\nWhiteness'],
                annot=True, fmt='.1f', 
                cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Pass Rate (%)'}, ax=ax1)
    ax1.set_title('Innovation Whiteness Test\n(Ljung-Box p > 0.05)')
    ax1.set_ylabel('Subject')
    
    # 2. Distribution fit results  
    ax2 = axes[1]
    sns.heatmap([[f] for f in fit_data],
                yticklabels=subject_labels, 
                xticklabels=['Distribution\nFit'],
                annot=True, fmt='.1f',
                cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Pass Rate (%)'}, ax=ax2)
    ax2.set_title('Innovation Distribution Fit\n(KS/Shapiro-Wilk)')
    ax2.set_ylabel('')
    
    # 3. Combined model adequacy
    ax3 = axes[2] 
    sns.heatmap([[c] for c in combined_data],
                yticklabels=subject_labels,
                xticklabels=['Combined\nAdequacy'], 
                annot=True, fmt='.1f',
                cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Pass Rate (%)'}, ax=ax3)
    ax3.set_title('Combined Model Adequacy\n(Both Tests Pass)')
    ax3.set_ylabel('')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("residual_analysis_clean/innovation_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_file = output_dir / "innovation_test_heatmaps.png"
    svg_file = output_dir / "innovation_test_heatmaps.svg"
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(svg_file, bbox_inches='tight')
    
    print(f"\n✅ Innovation test heatmaps saved:")
    print(f"   📊 PNG: {png_file}")
    print(f"   📊 SVG: {svg_file}")
    
    plt.show()
    
    return whiteness_data, fit_data, combined_data

def create_detailed_heatmap(df):
    """Create detailed trial-by-trial heatmap of innovation tests."""
    
    # Pivot data for detailed heatmap
    heatmap_data = df.pivot_table(
        values=['is_white_noise', 'good_fit'],
        index='subject',
        columns='trial',
        fill_value=0
    )
    
    # Create combined adequacy matrix (both tests must pass)
    subjects = sorted(df['subject'].unique())
    trials_per_subject = {}
    
    for subject in subjects:
        subj_df = df[df['subject'] == subject]
        trials_per_subject[subject] = sorted(subj_df['trial'].unique())
    
    # Find maximum number of trials for padding
    max_trials = max(len(trials) for trials in trials_per_subject.values())
    
    # Create combined adequacy matrix
    combined_matrix = []
    subject_labels = []
    
    for subject in subjects:
        subj_df = df[df['subject'] == subject]
        subject_labels.append(subject.replace('subject_', ''))
        
        # Get combined adequacy for each trial
        trial_adequacy = []
        for trial in trials_per_subject[subject]:
            trial_data = subj_df[subj_df['trial'] == trial]
            if len(trial_data) > 0:
                both_pass = (trial_data['is_white_noise'].iloc[0] & 
                           trial_data['good_fit'].iloc[0])
                trial_adequacy.append(int(both_pass))
            else:
                trial_adequacy.append(0)
        
        # Pad with NaN for missing trials
        while len(trial_adequacy) < max_trials:
            trial_adequacy.append(np.nan)
            
        combined_matrix.append(trial_adequacy)
    
    # Create detailed heatmap
    fig, ax = plt.subplots(1, 1, figsize=(max_trials * 0.3 + 3, len(subjects) * 0.5 + 2))
    
    combined_matrix = np.array(combined_matrix)
    
    # Custom colormap: white for NaN, red for fail (0), green for pass (1)
    from matplotlib.colors import ListedColormap
    colors = ['#ff4444', '#44ff44', '#ffffff']  # red, green, white
    cmap = ListedColormap(colors)
    
    im = ax.imshow(combined_matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=1.5)
    
    # Set ticks and labels
    ax.set_yticks(range(len(subject_labels)))
    ax.set_yticklabels(subject_labels)
    ax.set_xticks(range(max_trials))
    ax.set_xticklabels([f'{i+1:03d}' for i in range(max_trials)], rotation=90, fontsize=8)
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('Subject')
    ax.set_title('Innovation Test Adequacy by Trial\n(Green=Pass Both Tests, Red=Fail, White=No Data)')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, max_trials, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed heatmap
    output_dir = Path("residual_analysis_clean/innovation_tests")
    png_file = output_dir / "innovation_test_detailed.png"
    svg_file = output_dir / "innovation_test_detailed.svg"
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(svg_file, bbox_inches='tight')
    
    print(f"✅ Detailed innovation test heatmap saved:")
    print(f"   📊 PNG: {png_file}")  
    print(f"   📊 SVG: {svg_file}")
    
    plt.show()

def main():
    """Main analysis workflow."""
    
    print("🔬 STANDARDIZED INNOVATION TEST ANALYSIS")
    print("=" * 60)
    print("Analyzing model adequacy on standardized innovations z_t = ε_t/σ_t")
    print("Tests: Ljung-Box whiteness + Distribution goodness-of-fit")
    
    # Load results
    df = load_innovation_results()
    if df is None:
        return
    
    # Analyze test results
    analyze_innovation_tests(df)
    
    # Create visualizations
    print(f"\n📊 Creating innovation test visualizations...")
    whiteness_data, fit_data, combined_data = create_innovation_heatmaps(df)
    
    # Create detailed trial-level heatmap
    print(f"\n📊 Creating detailed trial-level heatmap...")
    create_detailed_heatmap(df)
    
    # Summary report
    print(f"\n📋 SUMMARY:")
    print(f"   • Raw residual diagnostic tests → justify ARMA-GARCH modeling")  
    print(f"   • Innovation adequacy tests → validate final model quality")
    print(f"   • Low pass rates indicate need for model refinement")
    print(f"   • Results available in residual_analysis_clean/innovation_tests/")

if __name__ == "__main__":
    main()