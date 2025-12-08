#!/usr/bin/env python3
"""
Generate Specific Parameter Plots
=================================

This script generates specific parameter plots from the ARMA-GARCH-t analysis:
1. GARCH parameters by subject (box plots)
2. Combined ARMA analysis (order distribution + coefficients)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette('muted')

def load_fitted_parameters():
    """Load fitted ARMA-GARCH parameters from residual_analysis_clean directory."""
    
    import pickle
    
    # Use the correct path for clean fitted models
    models_dir = Path("residual_analysis_clean/fitted_models")
    
    if not models_dir.exists():
        print("❌ No fitted parameters found. Please run residual analysis first.")
        return None
    
    # Load parameters from each subject directory
    all_params = {}
    subjects_loaded = []
    
    # First try to load the main fitted_models_clean.pkl file
    main_file = models_dir / "fitted_models_clean.pkl"
    if main_file.exists():
        try:
            with open(main_file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                all_params.update(data)
                print(f"✅ Loaded main fitted models file: {len(data)} trials")
        except Exception as e:
            print(f"⚠️  Warning: Could not load main file {main_file}: {e}")
    
    # Also load individual subject files
    subject_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('subject_')]
    
    for subject_dir in sorted(subject_dirs):
        subject_file = subject_dir / f"{subject_dir.name}_fitted_models.pkl"
        if subject_file.exists():
            try:
                with open(subject_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    all_params.update(data)
                    subjects_loaded.append(subject_dir.name)
                    print(f"✅ Loaded {subject_dir.name}: {len(data)} trials")
            except Exception as e:
                print(f"⚠️  Warning: Could not load {subject_file}: {e}")
                continue
    
    if subjects_loaded:
        print(f"✅ Total loaded subjects: {subjects_loaded}")
    
    return all_params

def extract_subject_id(trial_name):
    """Extract subject ID from trial name."""
    if 'subject_' in trial_name:
        return trial_name.split('/')[0] if '/' in trial_name else trial_name.split('_')[0] + '_' + trial_name.split('_')[1]
    return 'unknown'

def extract_parameter_data(all_params):
    """Extract parameter data from fitted models for analysis."""
    
    parameter_data = []
    
    for trial_name, trial_data in all_params.items():
        subject = extract_subject_id(trial_name)
        
        # Extract GARCH parameters
        garch_params = trial_data.get('garch_params', {})
        
        # Handle both numpy and regular float types
        omega = float(garch_params.get('omega', np.nan))
        alpha = float(garch_params.get('alpha_1', garch_params.get('alpha', np.nan)))
        beta = float(garch_params.get('beta_1', garch_params.get('beta', np.nan)))
        
        # Extract other key parameters
        nu = trial_data.get('nu', np.nan)  # t-distribution DoF
        bic = trial_data.get('bic', np.nan)
        aic = trial_data.get('aic', np.nan)
        
        # Extract ARMA orders
        optimal_orders = trial_data.get('optimal_orders', {})
        p = optimal_orders.get('p', 0)
        q = optimal_orders.get('q', 0)
        r = optimal_orders.get('r', 1)
        s = optimal_orders.get('s', 1)
        
        # Calculate derived parameters
        volatility_persistence = trial_data.get('volatility_persistence', alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan)
        volatility_halflife = trial_data.get('volatility_halflife', np.nan)
        
        # Extract ARMA parameters
        arma_params = trial_data.get('arma_params', {})
        
        # Extract individual ARMA coefficients
        phi_coeffs = []
        theta_coeffs = []
        
        # Extract AR coefficients (phi)
        for i in range(1, p+1):
            phi_key = f'phi_{i}'
            phi_val = arma_params.get(phi_key, np.nan)
            phi_coeffs.append(phi_val)
        
        # Extract MA coefficients (theta)  
        for i in range(1, q+1):
            theta_key = f'theta_{i}'
            theta_val = arma_params.get(theta_key, np.nan)
            theta_coeffs.append(theta_val)
        
        # Calculate ARMA parameter summary statistics
        phi_mean = np.mean(phi_coeffs) if phi_coeffs else np.nan
        phi_std = np.std(phi_coeffs) if len(phi_coeffs) > 1 else np.nan
        theta_mean = np.mean(theta_coeffs) if theta_coeffs else np.nan
        theta_std = np.std(theta_coeffs) if len(theta_coeffs) > 1 else np.nan
        
        parameter_data.append({
            'trial_name': trial_name,
            'subject': subject,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'nu': nu,
            'bic': bic,
            'aic': aic,
            'arma_p': p,
            'arma_q': q,
            'garch_r': r,
            'garch_s': s,
            'volatility_persistence': volatility_persistence,
            'volatility_halflife': volatility_halflife,
            'arma_params': arma_params,
            'phi_coeffs': phi_coeffs,
            'theta_coeffs': theta_coeffs,
            'phi_mean': phi_mean,
            'phi_std': phi_std,
            'theta_mean': theta_mean,
            'theta_std': theta_std
        })
    
    return pd.DataFrame(parameter_data)

def create_garch_parameters_by_subject_plots(df, output_dir):
    """Create separate GARCH parameter distribution plots by subject."""
    
    # Filter out invalid data
    valid_df = df.dropna(subset=['omega', 'alpha', 'beta', 'nu'])
    
    if len(valid_df) == 0:
        print("❌ No valid GARCH parameter data found")
        return
    
    print(f"📊 Creating GARCH parameters by subject plots for {len(valid_df)} valid trials")
    
    # Get subject order and colors - manually set correct order
    subjects = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
    # Filter to only subjects that exist in the data
    subjects = [s for s in subjects if s in valid_df['subject'].unique()]
    colors = sns.color_palette('muted', len(subjects))
    subject_colors = dict(zip(subjects, colors))
    
    # Helper function to create styled boxplots
    def create_styled_boxplot(data, x_col, y_col, ylabel, filename):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # First create stripplot for data points with same colors as boxplots
        sns.stripplot(data=data, x=x_col, y=y_col, ax=ax, 
                     alpha=0.6, size=2, palette=subject_colors, order=subjects)
        
        # Create custom boxplots for each subject
        for i, subject in enumerate(subjects):
            subject_data = data[data[x_col] == subject][y_col]
            if len(subject_data) > 0:
                color = subject_colors[subject]
                parts = ax.boxplot(subject_data, positions=[i], widths=0.5,
                                  patch_artist=True, showfliers=False, showcaps=False)
                parts['boxes'][0].set_facecolor('white')
                parts['boxes'][0].set_edgecolor(color)
                parts['boxes'][0].set_alpha(0.3)
                parts['medians'][0].set_color(color)
                parts['medians'][0].set_linewidth(2)
                parts['whiskers'][0].set_color(color)
                parts['whiskers'][1].set_color(color)
        
        ax.set_xticklabels(['001', '002', '003', '004', '005'], rotation=45, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        ax.set_xlabel('Subject', fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', labelsize=18)

        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}.svg', dpi=300, bbox_inches='tight')
        # plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Omega (intercept)
    create_styled_boxplot(valid_df, 'subject', 'omega', r'$\hat\omega$', 'garch_omega_by_subject')
    
    # Alpha (ARCH coefficient)
    create_styled_boxplot(valid_df, 'subject', 'alpha', r'$\hat\alpha$', 'garch_alpha_by_subject')
    
    # Beta (GARCH coefficient)
    create_styled_boxplot(valid_df, 'subject', 'beta', r'$\hat\beta$', 'garch_beta_by_subject')
    
    # Nu (t-distribution DoF)
    create_styled_boxplot(valid_df, 'subject', 'nu', r'$\hat\nu$', 'garch_nu_by_subject')
    
    print(f"✅ GARCH parameters by subject plots saved")

def create_volatility_persistence_plot(df, output_dir):
    """Create volatility persistence (α + β) analysis plot."""
    
    # Filter out invalid data
    valid_df = df.dropna(subset=['alpha', 'beta'])
    
    if len(valid_df) == 0:
        print("❌ No valid volatility persistence data found")
        return
    
    # Calculate volatility persistence
    valid_df = valid_df.copy()
    valid_df['volatility_persistence'] = valid_df['alpha'] + valid_df['beta']
    
    print(f"📊 Creating volatility persistence plot for {len(valid_df)} valid trials")
    
    # Create single plot for volatility persistence
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get subject order and colors - manually set correct order
    subjects = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
    # Filter to only subjects that exist in the data
    subjects = [s for s in subjects if s in valid_df['subject'].unique()]
    colors = sns.color_palette('muted', len(subjects))
    subject_colors = dict(zip(subjects, colors))
    
    # Create stripplot for data points
    sns.stripplot(data=valid_df, x='subject', y='volatility_persistence', ax=ax, 
                 alpha=0.6, size=3, palette=subject_colors, order=subjects)
    
    # Create custom boxplots for each subject
    for i, subject in enumerate(subjects):
        subject_data = valid_df[valid_df['subject'] == subject]['volatility_persistence']
        if len(subject_data) > 0:
            color = subject_colors[subject]
            parts = ax.boxplot(subject_data, positions=[i], widths=0.5,
                              patch_artist=True, showfliers=False, showcaps=False)
            parts['boxes'][0].set_facecolor('white')
            parts['boxes'][0].set_edgecolor(color)
            parts['boxes'][0].set_alpha(0.3)
            parts['medians'][0].set_color(color)
            parts['medians'][0].set_linewidth(2)
            parts['whiskers'][0].set_color(color)
            parts['whiskers'][1].set_color(color)
    
    # Formatting
    ax.set_xticklabels(['001', '002', '003', '004', '005'], rotation=45, fontsize=18)
    ax.set_ylabel(r'$\hat\alpha + \hat\beta$', fontsize=18, fontweight='bold')
    ax.set_ylim([0.55, 1])
    ax.set_xlabel('Subject', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)

    
    plt.tight_layout()
    plt.savefig(output_dir / 'volatility_persistence_by_subject.svg', dpi=300, bbox_inches='tight')
    # plt.savefig(output_dir / 'volatility_persistence_by_subject.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Volatility persistence plot saved")

def create_arma_order_distribution_plots(df, output_dir):
    """Create separate ARMA order distribution plots by subject."""
    
    # Filter for trials with ARMA parameters
    arma_df = df[(df['arma_p'] > 0) | (df['arma_q'] > 0)].copy()
    
    if len(arma_df) == 0:
        print("❌ No ARMA parameters found")
        return
    
    print(f"📊 Creating ARMA order distribution plots for {len(arma_df)} trials with ARMA components")
    
    # AR Order Distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Manually set subject order to ensure correct sequence
    subject_order = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
    sns.countplot(data=arma_df, x='subject', hue='arma_p', ax=ax, palette='flare', order=subject_order)
    ax.set_ylabel('Count', fontweight='bold', fontsize=18)
    ax.set_xlabel('Subject', fontweight='bold', fontsize=18)
    ax.set_xticklabels(['001', '002', '003', '004', '005'], rotation=45, fontsize=18)
    ax.legend(title='AR Order (p)', title_fontsize=12,fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'arma_ar_order_distribution.svg', dpi=300, bbox_inches='tight')
    # plt.savefig(output_dir / 'arma_ar_order_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # MA Order Distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Manually set subject order to ensure correct sequence
    subject_order = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
    sns.countplot(data=arma_df, x='subject', hue='arma_q', ax=ax, palette='flare', order=subject_order)
    ax.set_ylabel('Count', fontweight='bold', fontsize=18)
    ax.set_xlabel('Subject', fontweight='bold', fontsize=18)
    ax.set_xticklabels(['001', '002', '003', '004', '005'], rotation=45, fontsize=18)
    ax.legend(title='MA Order (q)', title_fontsize=12,fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'arma_ma_order_distribution.svg', dpi=300, bbox_inches='tight')
    # plt.savefig(output_dir / 'arma_ma_order_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ARMA order distribution plots saved")

def create_arma_coefficient_plots(df, output_dir):
    """Create separate ARMA coefficient plots by subject."""
    
    # Filter for trials with ARMA parameters
    arma_df = df[(df['arma_p'] > 0) | (df['arma_q'] > 0)].copy()
    
    if len(arma_df) == 0:
        print("❌ No ARMA parameters found")
        return
    
    print(f"📊 Creating ARMA coefficient plots for {len(arma_df)} trials with ARMA components")
    
    # Collect all AR and MA coefficients
    ar_by_subject = defaultdict(list)
    ma_by_subject = defaultdict(list)
    
    for _, row in arma_df.iterrows():
        subject = row['subject']
        phi_coeffs = row['phi_coeffs']
        theta_coeffs = row['theta_coeffs']
        
        # Collect AR coefficients
        for phi in phi_coeffs:
            if not np.isnan(phi):
                ar_by_subject[subject].append(phi)
        
        # Collect MA coefficients
        for theta in theta_coeffs:
            if not np.isnan(theta):
                ma_by_subject[subject].append(theta)
    
    # Helper function to create styled coefficient boxplots
    def create_styled_coeff_boxplot(coeff_data, ylabel, filename):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if not coeff_data:
            ax.text(0.5, 0.5, f'No {ylabel} coefficients found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=18)
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=18)
            ax.set_xlabel('Subject', fontweight='bold', fontsize=18)
            ax.set_xticklabels(ax.get_xticks(), fontsize=18)
            ax.set_yticklabels(ax.get_yticks(), fontsize=18)
            plt.tight_layout()
            plt.savefig(output_dir / f'{filename}.svg', dpi=300, bbox_inches='tight')
            # plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
            
        # Get colors for each subject first - manually set correct order
        subjects = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
        # Filter to only subjects that exist in the data
        subjects = [s for s in subjects if s in arma_df['subject'].unique()]
        colors = sns.color_palette('muted', len(subjects))
        subject_colors = dict(zip(subjects, colors))
        
        # Create data for plotting
        plot_data = []
        for subject, coeffs in coeff_data.items():
            for coeff in coeffs:
                plot_data.append({'subject': subject, 'coeff': coeff})
        plot_df = pd.DataFrame(plot_data)
        
        # Use consistent ordering for both stripplot and boxplot - manual order
        subj_list = ['subject_001', 'subject_002', 'subject_003', 'subject_004', 'subject_005']
        # Filter to only subjects that exist in the coefficient data
        subj_list = [s for s in subj_list if s in coeff_data.keys()]
        
        # Create stripplot first with matching colors and explicit order
        sns.stripplot(data=plot_df, x='subject', y='coeff', ax=ax, 
                     alpha=0.6, size=2, palette=subject_colors, order=subj_list)
        
        # Create custom boxplots for each subject
        for i, subject in enumerate(subj_list):
            coeffs = coeff_data[subject]
            if len(coeffs) > 0:
                color = subject_colors.get(subject, 'steelblue')
                parts = ax.boxplot(coeffs, positions=[i], widths=0.5,
                                  patch_artist=True, showfliers=False, showcaps=False)
                parts['boxes'][0].set_facecolor('white')
                parts['boxes'][0].set_edgecolor(color)
                parts['boxes'][0].set_alpha(0.3)
                parts['medians'][0].set_color(color)
                parts['medians'][0].set_linewidth(2)
                parts['whiskers'][0].set_color(color)
                parts['whiskers'][1].set_color(color)
        
        ax.set_xticklabels(['001', '002', '003', '004', '005'][:len(subj_list)], rotation=45, fontsize=18)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=18)
        ax.set_xlabel('Subject', fontweight='bold', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)

        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}.svg', dpi=300, bbox_inches='tight')
        # plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # AR coefficients by subject
    create_styled_coeff_boxplot(ar_by_subject, r'$\hat\phi$', 'arma_ar_coefficients_by_subject')
    
    # MA coefficients by subject
    create_styled_coeff_boxplot(ma_by_subject, r'$\hat\theta$', 'arma_ma_coefficients_by_subject')
    
    print(f"✅ ARMA coefficient plots saved")

def main():
    """Main function to generate specific parameter plots."""
    
    print("="*70)
    print("SPECIFIC PARAMETER PLOTS GENERATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path("results_plots/residual_modeling/specific_parameter_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fitted parameters
    print("\n1. Loading fitted model parameters...")
    all_params = load_fitted_parameters()
    
    if all_params is None or len(all_params) == 0:
        print("❌ No fitted parameters found.")
        return
    
    print(f"✅ Loaded parameters for {len(all_params)} trials")
    
    # Extract parameter data
    print("\n2. Extracting parameter data...")
    parameter_df = extract_parameter_data(all_params)
    
    print(f"✅ Extracted parameters for {len(parameter_df)} trials")
    print(f"   Valid GARCH parameters: {parameter_df.dropna(subset=['omega', 'alpha', 'beta']).shape[0]} trials")
    print(f"   ARMA trials: {parameter_df[(parameter_df['arma_p'] > 0) | (parameter_df['arma_q'] > 0)].shape[0]} trials")
    
    # Create plots
    print("\n3. Creating GARCH parameters by subject plots...")
    create_garch_parameters_by_subject_plots(parameter_df, output_dir)
    
    print("\n4. Creating volatility persistence plot...")
    create_volatility_persistence_plot(parameter_df, output_dir)
    
    print("\n5. Creating ARMA order distribution plots...")
    create_arma_order_distribution_plots(parameter_df, output_dir)
    
    print("\n6. Creating ARMA coefficient plots...")
    create_arma_coefficient_plots(parameter_df, output_dir)
    
    # Print summary
    print("\n✅ Specific parameter plots completed!")
    print(f"📁 Output files:")
    print(f"   - GARCH ω by subject: {output_dir}/garch_omega_by_subject.svg")
    print(f"   - GARCH α by subject: {output_dir}/garch_alpha_by_subject.svg")
    print(f"   - GARCH β by subject: {output_dir}/garch_beta_by_subject.svg")
    print(f"   - GARCH ν by subject: {output_dir}/garch_nu_by_subject.svg")
    print(f"   - Volatility persistence: {output_dir}/volatility_persistence_by_subject.svg")
    print(f"   - ARMA AR order distribution: {output_dir}/arma_ar_order_distribution.svg")
    print(f"   - ARMA MA order distribution: {output_dir}/arma_ma_order_distribution.svg")
    print(f"   - ARMA AR coefficients: {output_dir}/arma_ar_coefficients_by_subject.svg")
    print(f"   - ARMA MA coefficients: {output_dir}/arma_ma_coefficients_by_subject.svg")

if __name__ == "__main__":
    main()