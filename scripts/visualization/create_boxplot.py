#!/usr/bin/env python3
"""
Create boxplot comparing ARMA-GARCH vs White Gaussian model adequacy validation results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from datetime import datetime

sns.set_palette('muted')

def load_model_adequacy_results():
    """Load model adequacy validation results - find latest file"""
    import glob
    
    # Try to find latest model adequacy validation file
    validation_files = glob.glob('model_adequacy_validation_*.json')
    if validation_files:
        # Use the most recent one
        latest_file = max(validation_files)
        print(f"📂 Loading latest validation results: {latest_file}")
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"❌ Error loading {latest_file}: {e}")
            return None
    
    # Fallback to old filename
    try:
        with open('model_adequacy_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print("❌ No model adequacy results files found. Run validation first.")
        return None

def create_adequacy_boxplot():
    """Create boxplot comparing ARMA-GARCH vs White Gaussian model adequacy"""
    
    # Load results
    results = load_model_adequacy_results()
    if not results:
        return
    
    # Extract test results
    arma_garch_tests = results.get('arma_garch_tests', [])
    white_gaussian_tests = results.get('white_gaussian_tests', [])
    trial_names = results.get('trial_names', [])
    
    # Also load fitted models to get sigma ratios
    from create_comprehensive_trial_analysis_v2 import load_clean_models
    fitted_models = load_clean_models()
    
    print(f"📊 Creating boxplot from {len(arma_garch_tests)} ARMA-GARCH and {len(white_gaussian_tests)} White Gaussian results")
    
    # Prepare data for plotting - using p-values from different tests and sigma ratios
    data_rows = []
    sigma_ratio_data = []  # Separate storage for actual sigma ratios
    
    # Process all trials
    for i, trial_name in enumerate(trial_names):
        if i < len(arma_garch_tests) and i < len(white_gaussian_tests):
            arma_test = arma_garch_tests[i]
            white_test = white_gaussian_tests[i]
            
            if arma_test and white_test:
                # Add ARMA-GARCH results for statistical tests
                for test_name, test_label in [
                    ('ljung_box_pvalue', 'Ljung-Box'),
                    ('arch_lm_pvalue', 'ARCH-LM'), 
                    ('lilliefors_pvalue', 'Lilliefors')
                ]:
                    if test_name in arma_test:
                        data_rows.append({
                            'Trial': trial_name,
                            'Model': 'ARMA-GARCH',
                            'Test': test_label,
                            'P_Value': float(arma_test[test_name])
                        })
                    
                    if test_name in white_test:
                        data_rows.append({
                            'Trial': trial_name,
                            'Model': 'White Gaussian',
                            'Test': test_label,
                            'P_Value': float(white_test[test_name])
                        })
                
                # Add sigma ratio for ARMA-GARCH (convert to p-value-like metric)
                if trial_name in fitted_models:
                    trial_data = fitted_models[trial_name]
                    original_residuals = trial_data.get('original_residuals', [])
                    
                    if len(original_residuals) > 0:
                        try:
                            # Calculate sigma ratio like in Option 6
                            from create_comprehensive_trial_analysis_v2 import simulate_enhanced_arma_garch
                            import numpy as np
                            
                            simulation_result = simulate_enhanced_arma_garch(
                                trial_data, 
                                n_periods=len(original_residuals),
                                random_seed=42
                            )
                            
                            if isinstance(simulation_result, dict):
                                simulated_residuals = simulation_result['residuals']
                            else:
                                simulated_residuals = simulation_result
                            
                            orig_std = np.std(original_residuals)
                            sim_std = np.std(simulated_residuals)
                            sigma_ratio = sim_std / max(orig_std, 1e-8)
                            
                            # Store actual sigma ratios for separate plotting
                            sigma_ratio_data.append({
                                'Trial': trial_name,
                                'Model': 'ARMA-GARCH',
                                'Test': 'Sigma Ratio',
                                'Sigma_Ratio': float(sigma_ratio)
                            })
                            
                            # White Gaussian gets sigma ratio of original vs original (always 1.0)
                            sigma_ratio_data.append({
                                'Trial': trial_name,
                                'Model': 'White Gaussian', 
                                'Test': 'Sigma Ratio',
                                'Sigma_Ratio': 1.0  # Perfect sigma ratio for baseline
                            })
                            
                        except Exception:
                            # Skip sigma ratio for this trial if calculation fails
                            pass
    
    if not data_rows and not sigma_ratio_data:
        print("❌ No valid data for plotting")
        return
    
    # Create DataFrames
    df = pd.DataFrame(data_rows) if data_rows else pd.DataFrame()
    df_sigma = pd.DataFrame(sigma_ratio_data) if sigma_ratio_data else pd.DataFrame()
    
    print(f"✅ Created datasets:")
    if len(df) > 0:
        print(f"   P-value points: {len(df)} ({len(df[df['Model'] == 'ARMA-GARCH'])} ARMA-GARCH, {len(df[df['Model'] == 'White Gaussian'])} White Gaussian)")
    if len(df_sigma) > 0:
        print(f"   Sigma ratio points: {len(df_sigma)} ({len(df_sigma[df_sigma['Model'] == 'ARMA-GARCH'])} ARMA-GARCH, {len(df_sigma[df_sigma['Model'] == 'White Gaussian'])} White Gaussian)")
    
    # Create single plot with secondary y-axis
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # Set color palette for the two models  
    colors = ['steelblue', 'orange']
    sns.set_palette('muted')
    
    # Create stripplot for p-values (excluding Sigma Ratio)
    df_pvalues = df[df['Test'] != 'Sigma Ratio'] if len(df) > 0 else pd.DataFrame()
    if len(df_pvalues) > 0:
        sns.stripplot(data=df_pvalues, x='Test', y='P_Value', hue='Model', ax=ax, 
                      alpha=0.6, size=2, dodge=True, legend=False)
    
    # Create boxplots for p-values only (statistical tests)
    df_arma_pvalues = df_pvalues[df_pvalues['Model'] == 'ARMA-GARCH'] if len(df_pvalues) > 0 else pd.DataFrame()
    df_gauss_pvalues = df_pvalues[df_pvalues['Model'] == 'White Gaussian'] if len(df_pvalues) > 0 else pd.DataFrame()
    
    # Create box plots for statistical tests only
    for i, test in enumerate(['Ljung-Box', 'ARCH-LM', 'Lilliefors']):
        # ARMA-GARCH boxes (left side)
        arma_data = df_arma_pvalues[df_arma_pvalues['Test'] == test]['P_Value']
        if len(arma_data) > 0:
            parts = ax.boxplot(arma_data, positions=[i-0.2], widths=0.2, 
                              patch_artist=True, showfliers=False, showcaps=False)
            parts['boxes'][0].set_facecolor('white')
            parts['boxes'][0].set_edgecolor('steelblue')
            parts['boxes'][0].set_alpha(0.3)
            parts['medians'][0].set_color('steelblue')
            parts['medians'][0].set_linewidth(2)
            parts['whiskers'][0].set_color('steelblue')
            parts['whiskers'][1].set_color('steelblue')
        
        # White Gaussian boxes (right side)  
        gauss_data = df_gauss_pvalues[df_gauss_pvalues['Test'] == test]['P_Value']
        if len(gauss_data) > 0:
            parts = ax.boxplot(gauss_data, positions=[i+0.2], widths=0.2,
                              patch_artist=True, showfliers=False, showcaps=False)
            parts['boxes'][0].set_facecolor('white')
            parts['boxes'][0].set_edgecolor('orange')
            parts['boxes'][0].set_alpha(0.3)
            parts['medians'][0].set_color('orange')
            parts['medians'][0].set_linewidth(2)
            parts['whiskers'][0].set_color('orange')
            parts['whiskers'][1].set_color('orange')
    
    # Primary y-axis formatting (p-values)
    # Add significance line only in the p-value region (x=0-2.5)
    ax.plot([-0.5, 2.5], [0.05, 0.05], color="r", linestyle="--", alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('p-value', fontsize=16, fontweight='bold')
    ax.set_xlabel('Test', fontsize=16, fontweight='bold')
    
    # Set x-axis ticks for statistical tests only
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Ljung-Box', 'ARCH-LM', 'Lilliefors'], fontsize = 14)

    
    # Create secondary y-axis for sigma ratios
    ax2 = ax.twinx()
    
    # Plot sigma ratios on the secondary axis with rocket colormap  
    if len(df_sigma) > 0:
        # Use rocket colormap colors
        rocket_colors = sns.color_palette("flare", n_colors=2)
        
        # Create stripplot for sigma ratios using seaborn (consistent with p-values)
        sns.stripplot(data=df_sigma, x='Test', y='Sigma_Ratio', hue='Model', ax=ax2,
                      alpha=0.6, size=2, dodge=True, palette=rocket_colors, legend=False)
        
        # Create boxplots for sigma ratios (consistent with p-value boxplots)
        df_sigma_arma = df_sigma[df_sigma['Model'] == 'ARMA-GARCH']
        df_sigma_gauss = df_sigma[df_sigma['Model'] == 'White Gaussian']
        
        # ARMA-GARCH sigma boxes (at x=3 position)
        if len(df_sigma_arma) > 0:
            arma_sigma_data = df_sigma_arma['Sigma_Ratio']
            parts = ax2.boxplot(arma_sigma_data, positions=[3-0.2], widths=0.2,
                               patch_artist=True, showfliers=False, showcaps=False)
            parts['boxes'][0].set_facecolor('white')
            parts['boxes'][0].set_edgecolor(rocket_colors[0])
            parts['boxes'][0].set_alpha(0.3)
            parts['medians'][0].set_color(rocket_colors[0])
            parts['medians'][0].set_linewidth(2)
            parts['whiskers'][0].set_color(rocket_colors[0])
            parts['whiskers'][1].set_color(rocket_colors[0])
        
        # White Gaussian sigma boxes (at x=3 position)
        if len(df_sigma_gauss) > 0:
            gauss_sigma_data = df_sigma_gauss['Sigma_Ratio']
            parts = ax2.boxplot(gauss_sigma_data, positions=[3+0.2], widths=0.2,
                               patch_artist=True, showfliers=False, showcaps=False)
            parts['boxes'][0].set_facecolor('white')
            parts['boxes'][0].set_edgecolor(rocket_colors[1])
            parts['boxes'][0].set_alpha(0.3)
            parts['medians'][0].set_color(rocket_colors[1])
            parts['medians'][0].set_linewidth(2)
            parts['whiskers'][0].set_color(rocket_colors[1])
            parts['whiskers'][1].set_color(rocket_colors[1])
        
        # Set y-axis limits and add reference line for perfect sigma ratio (only in sigma ratio region)
        sigma_min = df_sigma['Sigma_Ratio'].min() * 0.95
        sigma_max = df_sigma['Sigma_Ratio'].max() * 1.05
        ax2.set_ylim(sigma_min, sigma_max)
        
        # Align gridlines between both y-axes
        # Set matching tick positions for both axes
        p_value_ticks = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        
        # For sigma ratios: middle=1.0, bottom=0.15, calculate top for equal spacing
        # With 11 ticks, middle is at index 5, bottom at index 0
        middle_sigma = 1.0
        bottom_sigma = 0.15
        step = (middle_sigma - bottom_sigma) / 5  # 5 steps from bottom to middle
        top_sigma = middle_sigma + 5 * step  # 5 steps from middle to top
        
        sigma_ticks = np.linspace(bottom_sigma, top_sigma, 11)
        
        ax.set_yticks(p_value_ticks)
        ax2.set_yticks(sigma_ticks)
        ax2.set_ylim(bottom_sigma, top_sigma)
        
        # Add perfect ratio line only in the sigma ratio region (x=3)
        ax2.plot([2.5, 3.5], [1.0, 1.0], color="green", linestyle="--", linewidth=1, alpha=0.8)
        ax2.set_ylabel(r'$\sigma$ ratio', fontsize=16, fontweight='bold')
        ax2.set_xticklabels(ax.get_xticks(), fontsize=16)
    
    # Set default grid if no sigma data
    if len(df_sigma) == 0:
        ax.set_yticks(np.linspace(0, 1, 11), fontsize = 14)
    
    # Add grid after setting ticks
    ax.grid(True, alpha=0.3)
    
    # Update x-axis to include sigma ratio
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim([-0.5,3.5])
    ax.set_xticklabels(['Ljung-Box', 'ARCH-LM', 'Lilliefors', 'Sigma Ratio'], fontsize = 14)
    
    # Create comprehensive legend (at the end to appear on top)
    import matplotlib.lines as mlines
    
    # P-value legend items
    arma_dot = mlines.Line2D([], [], color='steelblue', marker='o', linestyle='None',
                            markersize=6, alpha=0.6, label='ARMA-GARCH (p-values)')
    gauss_dot = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                             markersize=6, alpha=0.6, label='White Gaussian (p-values)')
    sig_line = mlines.Line2D([], [], color='red', linestyle='--', 
                            label=r'$\alpha$ = 0.05')
    
    # Sigma ratio legend items  
    legend_handles = [arma_dot, gauss_dot, sig_line]
    
    if len(df_sigma) > 0:
        flare_colors = sns.color_palette("flare", n_colors=2)
        sigma_arma_dot = mlines.Line2D([], [], color=flare_colors[0], marker='o', linestyle='None',
                                      markersize=6, alpha=0.6, label='ARMA-GARCH (σ ratio)')
        sigma_gauss_dot = mlines.Line2D([], [], color=flare_colors[1], marker='o', linestyle='None',
                                       markersize=6, alpha=0.6, label='White Gaussian (σ ratio)')
        perfect_line = mlines.Line2D([], [], color='green', linestyle='--', 
                                    label='Perfect σ ratio = 1.0')
        legend_handles.extend([sigma_arma_dot, sigma_gauss_dot, perfect_line])
    
    # Add legend last so it appears on top of gridlines
    ax.legend(handles=legend_handles, loc='upper right', fontsize = 10)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'model_adequacy_grouped_boxplot_{timestamp}.svg'
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    print(f"💾 Saved grouped boxplot: {filename}")
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating Model Adequacy Validation Boxplots")
    print("=" * 60)
    create_adequacy_boxplot()
    print("✅ Boxplot creation complete!")