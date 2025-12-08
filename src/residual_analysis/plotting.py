#!/usr/bin/env python3
"""
Multi-Subject ARMA-GARCH Results Visualization (Clean Implementation)
====================================================================

Interactive analysis and visualization for all subjects using clean implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from scipy import stats
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_multi_subject_data(subjects: Optional[List[str]] = None) -> Dict:
    """
    Load fitted models and results for multiple subjects.
    
    Parameters:
    -----------
    subjects : list, optional
        List of subject IDs. Default: all available subjects
    
    Returns:
    --------
    dict : Combined data for all subjects
    """
    
    if subjects is None:
        # Auto-detect available subjects
        fitted_models_dir = Path("residual_analysis_clean/fitted_models")
        if fitted_models_dir.exists():
            subjects = [d.name for d in fitted_models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('subject_')]
        else:
            subjects = [f'subject_{i:03d}' for i in range(1, 6)]
    
    print(f"Loading data for subjects: {subjects}")
    
    all_data = {
        'subjects': {},
        'combined_df': None,
        'summary_stats': {}
    }
    
    combined_results = []
    
    for subject_id in subjects:
        subject_data = {}
        
        # Load fitted models
        models_path = Path(f"residual_analysis_clean/fitted_models/{subject_id}/{subject_id}_fitted_models.pkl")
        if models_path.exists():
            with open(models_path, 'rb') as f:
                subject_data['fitted_models'] = pickle.load(f)
            print(f"  ✓ Loaded fitted models for {subject_id}")
        else:
            print(f"  ⚠️ No fitted models found for {subject_id}")
            continue
        
        # Load results CSV
        results_path = Path(f"residual_analysis_clean/fitted_models/{subject_id}/{subject_id}_results.csv")
        if results_path.exists():
            df = pd.read_csv(results_path)
            df['subject'] = subject_id
            subject_data['results_df'] = df
            combined_results.append(df)
            print(f"  ✓ Loaded {len(df)} trial results for {subject_id}")
        else:
            print(f"  ⚠️ No results CSV found for {subject_id}")
        
        # Load summary
        summary_path = Path(f"residual_analysis_clean/fitted_models/{subject_id}/{subject_id}_summary.json")
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                subject_data['summary'] = json.load(f)
            print(f"  ✓ Loaded summary for {subject_id}")
        
        all_data['subjects'][subject_id] = subject_data
    
    # Combine all results
    if combined_results:
        all_data['combined_df'] = pd.concat(combined_results, ignore_index=True)
        print(f"\n✅ Combined data: {len(all_data['combined_df'])} trials across {len(subjects)} subjects")
    else:
        print("\n❌ No data loaded for any subject")
        return None
    
    # Generate summary statistics
    for subject_id, subject_data in all_data['subjects'].items():
        if 'results_df' in subject_data:
            df = subject_data['results_df']
            all_data['summary_stats'][subject_id] = {
                'n_trials': len(df),
                'mean_bic': float(df['bic'].mean()),
                'mean_nu': float(df['nu'].mean()),
                'mean_loglik': float(df['log_likelihood'].mean()),
                'mean_persistence': float(df.get('volatility_persistence', pd.Series([np.nan])).mean())
            }
    
    return all_data


def plot_cross_subject_comparison(data: Dict, output_dir: str = None):
    """
    Create comprehensive cross-subject comparison plots.
    
    Parameters:
    -----------
    data : dict
        Combined multi-subject data
    output_dir : str, optional
        Output directory for plots
    """
    
    if output_dir is None:
        output_dir = "residual_analysis_clean/plots_multi_subject"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_df = data['combined_df']
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Subject ARMA-GARCH Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. AIC comparison
    sns.boxplot(data=combined_df, x='subject', y='aic', ax=axes[0, 0])
    axes[0, 0].set_title('AIC by Subject')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. BIC comparison
    sns.boxplot(data=combined_df, x='subject', y='bic', ax=axes[0, 1])
    axes[0, 1].set_title('BIC by Subject')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Nu (Student-t parameter) comparison
    sns.boxplot(data=combined_df, x='subject', y='nu', ax=axes[0, 2])
    axes[0, 2].set_title('Student-t Parameter (ν) by Subject')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Log-likelihood comparison
    sns.boxplot(data=combined_df, x='subject', y='log_likelihood', ax=axes[1, 0])
    axes[1, 0].set_title('Log-Likelihood by Subject')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Trial count by subject
    trial_counts = combined_df['subject'].value_counts().sort_index()
    trial_counts.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Number of Trials by Subject')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. AIC vs BIC scatter
    subjects = combined_df['subject'].unique()
    colors = sns.color_palette("husl", len(subjects))
    
    for i, subject in enumerate(subjects):
        subject_data = combined_df[combined_df['subject'] == subject]
        axes[1, 2].scatter(subject_data['aic'], subject_data['bic'], 
                          label=subject, color=colors[i], alpha=0.7)
    
    axes[1, 2].set_xlabel('AIC')
    axes[1, 2].set_ylabel('BIC')
    axes[1, 2].set_title('AIC vs BIC by Subject')
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    comparison_plot_path = output_path / "multi_subject_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {comparison_plot_path}")
    
    plt.show()


def plot_parameter_distributions(data: Dict, output_dir: str = None):
    """
    Plot parameter distributions across all subjects.
    
    Parameters:
    -----------
    data : dict
        Combined multi-subject data
    output_dir : str, optional
        Output directory for plots
    """
    
    if output_dir is None:
        output_dir = "residual_analysis_clean/plots_multi_subject"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_df = data['combined_df']
    
    # Create parameter distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ARMA-GARCH Parameter Distributions Across All Subjects', fontsize=16, fontweight='bold')
    
    # 1. AIC distribution
    combined_df['aic'].hist(bins=30, alpha=0.7, ax=axes[0, 0])
    axes[0, 0].axvline(combined_df['aic'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {combined_df["aic"].mean():.2f}')
    axes[0, 0].set_title('AIC Distribution')
    axes[0, 0].set_xlabel('AIC')
    axes[0, 0].legend()
    
    # 2. BIC distribution
    combined_df['bic'].hist(bins=30, alpha=0.7, ax=axes[0, 1])
    axes[0, 1].axvline(combined_df['bic'].mean(), color='red', linestyle='--',
                       label=f'Mean: {combined_df["bic"].mean():.2f}')
    axes[0, 1].set_title('BIC Distribution')
    axes[0, 1].set_xlabel('BIC')
    axes[0, 1].legend()
    
    # 3. Nu (Student-t) distribution
    combined_df['nu'].hist(bins=30, alpha=0.7, ax=axes[1, 0])
    axes[1, 0].axvline(combined_df['nu'].mean(), color='red', linestyle='--',
                       label=f'Mean: {combined_df["nu"].mean():.2f}')
    axes[1, 0].set_title('Student-t Parameter (ν) Distribution')
    axes[1, 0].set_xlabel('ν')
    axes[1, 0].legend()
    
    # 4. Log-likelihood distribution
    combined_df['log_likelihood'].hist(bins=30, alpha=0.7, ax=axes[1, 1])
    axes[1, 1].axvline(combined_df['log_likelihood'].mean(), color='red', linestyle='--',
                       label=f'Mean: {combined_df["log_likelihood"].mean():.2f}')
    axes[1, 1].set_title('Log-Likelihood Distribution')
    axes[1, 1].set_xlabel('Log-Likelihood')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    distributions_plot_path = output_path / "parameter_distributions.png"
    plt.savefig(distributions_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved parameter distributions plot to: {distributions_plot_path}")
    
    plt.show()


def print_detailed_summary(data: Dict):
    """
    Print detailed summary statistics for all subjects.
    
    Parameters:
    -----------
    data : dict
        Combined multi-subject data
    """
    
    print("\n" + "="*80)
    print("DETAILED MULTI-SUBJECT SUMMARY")
    print("="*80)
    
    combined_df = data['combined_df']
    summary_stats = data['summary_stats']
    
    print(f"Total trials across all subjects: {len(combined_df)}")
    print(f"Subjects analyzed: {len(summary_stats)}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"AIC: {combined_df['aic'].mean():.2f} ± {combined_df['aic'].std():.2f} "
          f"[{combined_df['aic'].min():.2f}, {combined_df['aic'].max():.2f}]")
    print(f"BIC: {combined_df['bic'].mean():.2f} ± {combined_df['bic'].std():.2f} "
          f"[{combined_df['bic'].min():.2f}, {combined_df['bic'].max():.2f}]")
    print(f"ν (Student-t): {combined_df['nu'].mean():.2f} ± {combined_df['nu'].std():.2f} "
          f"[{combined_df['nu'].min():.2f}, {combined_df['nu'].max():.2f}]")
    print(f"Log-likelihood: {combined_df['log_likelihood'].mean():.2f} ± {combined_df['log_likelihood'].std():.2f}")
    
    # Per-subject breakdown
    print(f"\n📋 PER-SUBJECT BREAKDOWN:")
    print(f"{'Subject':<15} {'Trials':<10} {'Mean AIC':<12} {'Mean BIC':<12} {'Mean ν':<10} {'Mean LogLik':<12}")
    print("-" * 75)
    
    for subject_id, stats in summary_stats.items():
        print(f"{subject_id:<15} {stats['n_trials']:<10} {stats['mean_aic']:<12.2f} "
              f"{stats['mean_bic']:<12.2f} {stats['mean_nu']:<10.2f} {stats['mean_loglik']:<12.2f}")
    
    # Statistical tests across subjects
    print(f"\n🔬 CROSS-SUBJECT STATISTICAL TESTS:")
    subjects = list(summary_stats.keys())
    
    if len(subjects) > 1:
        # ANOVA tests for differences between subjects
        aic_groups = [combined_df[combined_df['subject'] == s]['aic'].values for s in subjects]
        bic_groups = [combined_df[combined_df['subject'] == s]['bic'].values for s in subjects]
        nu_groups = [combined_df[combined_df['subject'] == s]['nu'].values for s in subjects]
        
        try:
            aic_f, aic_p = stats.f_oneway(*aic_groups)
            print(f"AIC across subjects: F={aic_f:.3f}, p={aic_p:.4f}")
            
            bic_f, bic_p = stats.f_oneway(*bic_groups)
            print(f"BIC across subjects: F={bic_f:.3f}, p={bic_p:.4f}")
            
            nu_f, nu_p = stats.f_oneway(*nu_groups)
            print(f"ν across subjects: F={nu_f:.3f}, p={nu_p:.4f}")
            
        except Exception as e:
            print(f"Statistical tests failed: {e}")
    
    # Heavy-tail analysis
    print(f"\n🎯 HEAVY-TAIL ANALYSIS:")
    nu_values = combined_df['nu'].values
    light_tail_count = np.sum(nu_values > 30)  # Approximately normal for high ν
    moderate_tail_count = np.sum((nu_values > 5) & (nu_values <= 30))
    heavy_tail_count = np.sum(nu_values <= 5)
    
    total_trials = len(nu_values)
    print(f"Heavy tails (ν ≤ 5): {heavy_tail_count}/{total_trials} ({heavy_tail_count/total_trials*100:.1f}%)")
    print(f"Moderate tails (5 < ν ≤ 30): {moderate_tail_count}/{total_trials} ({moderate_tail_count/total_trials*100:.1f}%)")
    print(f"Light tails (ν > 30): {light_tail_count}/{total_trials} ({light_tail_count/total_trials*100:.1f}%)")
    
    if heavy_tail_count > total_trials * 0.5:
        print("➜ Strong evidence for heavy-tailed EMG residual innovations")
    elif heavy_tail_count > total_trials * 0.25:
        print("➜ Moderate evidence for heavy-tailed EMG residual innovations")
    else:
        print("➜ Limited evidence for heavy-tailed innovations")


def interactive_multi_subject_menu():
    """
    Interactive menu for multi-subject analysis.
    """
    
    print("="*80)
    print("MULTI-SUBJECT ARMA-GARCH ANALYSIS (CLEAN IMPLEMENTATION)")
    print("="*80)
    
    # Load data
    print("Loading multi-subject data...")
    data = load_multi_subject_data()
    
    if data is None:
        print("❌ No data available. Run run_all_subjects_workflow.py first.")
        return
    
    while True:
        print(f"\n" + "="*60)
        print("Multi-Subject Analysis Options:")
        print("1. Cross-subject comparison plots")
        print("2. Parameter distribution plots")  
        print("3. Detailed summary statistics")
        print("4. Single subject analysis (from individual fitted models)")
        print("5. Export combined data")
        print("6. Exit")
        print("="*60)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            plot_cross_subject_comparison(data)
            
        elif choice == '2':
            plot_parameter_distributions(data)
            
        elif choice == '3':
            print_detailed_summary(data)
            
        elif choice == '4':
            available_subjects = list(data['subjects'].keys())
            print(f"\nAvailable subjects: {available_subjects}")
            subject_id = input("Enter subject ID: ").strip()
            
            if subject_id in data['subjects']:
                # Use single-subject analysis from the original comprehensive script
                try:
                    from plot_results_comprehensive import plot_comprehensive_comparison
                    
                    # Load fitted models for this subject
                    subject_fitted_models = {}
                    fitted_models_df = data['subjects'][subject_id]['fitted_models']
                    
                    # Convert DataFrame to dictionary format expected by the original script
                    for _, row in fitted_models_df.iterrows():
                        trial_name = row['trial_name']
                        subject_fitted_models[trial_name] = row.to_dict()
                    
                    # Get trial name from user
                    trials = list(subject_fitted_models.keys())
                    print(f"Available trials: {trials[:5]}... (showing first 5 of {len(trials)})")
                    trial_name = input("Enter trial name: ").strip()
                    
                    if trial_name in subject_fitted_models:
                        plot_comprehensive_comparison(trial_name, subject_fitted_models)
                    else:
                        print(f"❌ Trial {trial_name} not found")
                        
                except ImportError:
                    print("❌ Cannot import single-subject analysis. Check plot_results_comprehensive.py")
            else:
                print(f"❌ Subject {subject_id} not found")
                
        elif choice == '5':
            output_path = Path("residual_analysis_clean/plots_multi_subject/combined_results.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data['combined_df'].to_csv(output_path, index=False)
            print(f"✅ Exported combined data to: {output_path}")
            
        elif choice == '6':
            print("Exiting multi-subject analysis...")
            break
            
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    interactive_multi_subject_menu()