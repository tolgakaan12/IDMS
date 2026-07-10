#!/usr/bin/env python3
"""
Synthetic Residual Generation and Validation
===========================================

This script:
1. Loads the best fitted ARMA-GARCH model
2. Generates synthetic residuals using the fitted parameters
3. Compares real vs synthetic residuals with statistical tests
4. Creates validation plots showing:
   - Time series comparison
   - Distribution comparison
   - Statistical test results comparison
   - Sigma ratio analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, lilliefors
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def load_fitted_models():
    """Load fitted ARMA-GARCH models from saved results."""
    
    # Try different possible locations
    possible_dirs = [
        Path("residual_analysis_clean/fitted_models"),
        Path("residual_analysis/arma_garch_fits"),
        Path("residual_analysis_final/fitted_models")
    ]
    
    for model_dir in possible_dirs:
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pkl"))
            if model_files:
                print(f"✅ Found fitted models in: {model_dir}")
                
                import pickle
                fitted_models = {}
                
                for file_path in model_files:
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        if isinstance(data, dict):
                            fitted_models.update(data)
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load {file_path}: {e}")
                
                return fitted_models
    
    print("❌ No fitted models found. Creating synthetic example...")
    return create_synthetic_example()

def create_synthetic_example():
    """Create a synthetic example for demonstration."""
    
    # Generate synthetic residuals
    n_samples = 1000
    np.random.seed(42)
    
    # ARMA(1,1) process
    arma_residuals = np.random.normal(0, 1, n_samples)
    for i in range(1, n_samples):
        arma_residuals[i] += 0.3 * arma_residuals[i-1] - 0.2 * np.random.normal(0, 1)
    
    # GARCH(1,1) volatility
    omega, alpha, beta = 0.01, 0.1, 0.8
    volatility = np.zeros(n_samples)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for i in range(1, n_samples):
        volatility[i] = np.sqrt(omega + alpha * arma_residuals[i-1]**2 + beta * volatility[i-1]**2)
    
    # Final residuals
    final_residuals = arma_residuals * volatility
    
    # Create synthetic fitted model data
    fitted_models = {
        'subject_005/trial_030': {
            'optimal_orders': {'p': 1, 'q': 1, 'r': 1, 's': 1},
            'arma_params': {'ar': [0.3], 'ma': [-0.2]},
            'garch_params': {'omega': omega, 'alpha': alpha, 'beta': beta},
            'distribution_params': {'nu': 5.0},
            'best_distribution': 't',
            'original_residuals': final_residuals,
            'volatility_persistence': alpha + beta,
            'bic_score': 2800.5,
            'test_results': {
                'ljung_box_pass': True,
                'lilliefors_pass': True,
                'std_ratio': 0.95
            }
        }
    }
    
    return fitted_models

def simulate_arma_garch(arma_params, garch_params, distribution_params, n_samples, distribution='t'):
    """Simulate ARMA-GARCH process."""
    
    # Extract parameters
    ar_params = arma_params.get('ar', [])
    ma_params = arma_params.get('ma', [])
    omega = garch_params.get('omega', 0.01)
    alpha = garch_params.get('alpha', 0.1)
    beta = garch_params.get('beta', 0.8)
    
    # Initialize
    max_lag = max(len(ar_params), len(ma_params), 1)
    residuals = np.zeros(n_samples + max_lag)
    volatility = np.zeros(n_samples + max_lag)
    innovations = np.zeros(n_samples + max_lag)
    
    # Initial volatility
    initial_vol = np.sqrt(omega / max(1 - alpha - beta, 0.01))
    volatility[:max_lag] = initial_vol
    
    # Generate innovations
    if distribution == 't':
        nu = distribution_params.get('nu', 5.0)
        innovations[max_lag:] = stats.t.rvs(nu, size=n_samples)
    else:
        innovations[max_lag:] = np.random.standard_normal(n_samples)
    
    # Generate process
    for t in range(max_lag, n_samples + max_lag):
        # ARMA component
        arma_component = 0
        
        # AR component
        for i, ar_coef in enumerate(ar_params):
            if t - i - 1 >= 0:
                arma_component += ar_coef * residuals[t - i - 1]
        
        # MA component  
        for i, ma_coef in enumerate(ma_params):
            if t - i - 1 >= 0:
                arma_component += ma_coef * innovations[t - i - 1]
        
        # GARCH volatility
        volatility[t] = np.sqrt(omega + alpha * residuals[t-1]**2 + beta * volatility[t-1]**2)
        
        # Combined residual
        residuals[t] = arma_component + volatility[t] * innovations[t]
    
    return residuals[max_lag:], volatility[max_lag:], innovations[max_lag:]

def run_statistical_tests(residuals, test_name=""):
    """Run comprehensive statistical tests on residuals."""
    
    results = {}
    
    # Clean residuals
    clean_residuals = residuals[np.isfinite(residuals)]
    clean_residuals = clean_residuals - np.mean(clean_residuals)
    n = len(clean_residuals)
    
    if n < 10:
        return {test: False for test in ['shapiro', 'jarque_bera', 'lilliefors', 'adf', 'kpss', 'ljung_box', 'arch_lm', 'runs']}
    
    # 1. Shapiro-Wilk test (normality)
    try:
        if n <= 5000:
            stat, p_val = stats.shapiro(clean_residuals)
            results['shapiro'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
        else:
            results['shapiro'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    except:
        results['shapiro'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 2. Jarque-Bera test (normality)
    try:
        stat, p_val = stats.jarque_bera(clean_residuals)
        results['jarque_bera'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
    except:
        results['jarque_bera'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 3. Lilliefors test (normality)
    try:
        stat, p_val = lilliefors(clean_residuals, dist='norm')
        results['lilliefors'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
    except:
        results['lilliefors'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 4. ADF test (stationarity)
    try:
        stat, p_val, _, _, _, _ = adfuller(clean_residuals, autolag='AIC')
        results['adf'] = {'stat': stat, 'p_value': p_val, 'pass': p_val < 0.05}  # Reject unit root
    except:
        results['adf'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 5. KPSS test (stationarity)
    try:
        stat, p_val, _, _ = kpss(clean_residuals, regression='c')
        if isinstance(p_val, str):
            p_val = float(p_val.replace('>', '').replace('<', ''))
        results['kpss'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}  # Fail to reject stationarity
    except:
        results['kpss'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 6. Ljung-Box test (autocorrelation)
    try:
        if n > 20:
            result = acorr_ljungbox(clean_residuals, lags=20)
            p_val = result['lb_pvalue'].iloc[-1]
            stat = result['lb_stat'].iloc[-1]
            results['ljung_box'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
        else:
            results['ljung_box'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    except:
        results['ljung_box'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 7. ARCH-LM test (heteroscedasticity)
    try:
        if n > 10:
            stat, p_val, _, _ = het_arch(clean_residuals, maxlag=10)
            results['arch_lm'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
        else:
            results['arch_lm'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    except:
        results['arch_lm'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    # 8. Runs test (randomness)
    try:
        median_val = np.median(clean_residuals)
        binary_seq = (clean_residuals > median_val).astype(int)
        stat, p_val = runstest_1samp(binary_seq)
        results['runs'] = {'stat': stat, 'p_value': p_val, 'pass': p_val > 0.05}
    except:
        results['runs'] = {'stat': np.nan, 'p_value': np.nan, 'pass': False}
    
    return results

def create_validation_plots(real_residuals, synthetic_residuals, real_tests, synthetic_tests, output_path, trial_info):
    """Create comprehensive validation plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Real vs Synthetic Residual Validation\n{trial_info["trial_name"]}', fontsize=16, fontweight='bold')
    
    # 1. Time series comparison
    time_idx = np.arange(min(len(real_residuals), len(synthetic_residuals)))
    min_len = len(time_idx)
    
    axes[0, 0].plot(time_idx, real_residuals[:min_len], 'b-', alpha=0.7, linewidth=0.8, label='Real Residuals')
    axes[0, 0].plot(time_idx, synthetic_residuals[:min_len], 'r-', alpha=0.7, linewidth=0.8, label='Synthetic Residuals')
    axes[0, 0].set_title('Time Series Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution comparison
    axes[0, 1].hist(real_residuals, bins=50, density=True, alpha=0.7, color='blue', label='Real', edgecolor='black')
    axes[0, 1].hist(synthetic_residuals, bins=50, density=True, alpha=0.7, color='red', label='Synthetic', edgecolor='black')
    axes[0, 1].set_title('Distribution Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    real_sorted = np.sort(real_residuals)
    synthetic_sorted = np.sort(synthetic_residuals)
    min_len = min(len(real_sorted), len(synthetic_sorted))
    
    axes[0, 2].scatter(real_sorted[:min_len], synthetic_sorted[:min_len], alpha=0.6, s=8)
    min_val = min(real_sorted[:min_len].min(), synthetic_sorted[:min_len].min())
    max_val = max(real_sorted[:min_len].max(), synthetic_sorted[:min_len].max())
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    axes[0, 2].set_title('Q-Q Plot: Real vs Synthetic', fontweight='bold')
    axes[0, 2].set_xlabel('Real Quantiles')
    axes[0, 2].set_ylabel('Synthetic Quantiles')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Statistical test comparison
    test_names = ['Shapiro', 'Jarque-Bera', 'Lilliefors', 'ADF', 'KPSS', 'Ljung-Box', 'ARCH-LM', 'Runs']
    test_keys = ['shapiro', 'jarque_bera', 'lilliefors', 'adf', 'kpss', 'ljung_box', 'arch_lm', 'runs']
    
    real_passes = [real_tests[key]['pass'] for key in test_keys]
    synthetic_passes = [synthetic_tests[key]['pass'] for key in test_keys]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, real_passes, width, label='Real', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, synthetic_passes, width, label='Synthetic', color='red', alpha=0.7)
    axes[1, 0].set_title('Statistical Test Results', fontweight='bold')
    axes[1, 0].set_xlabel('Statistical Tests')
    axes[1, 0].set_ylabel('Test Pass (1=Pass, 0=Fail)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. P-value comparison
    real_pvals = [real_tests[key]['p_value'] for key in test_keys]
    synthetic_pvals = [synthetic_tests[key]['p_value'] for key in test_keys]
    
    # Filter out NaN values for plotting
    valid_indices = [i for i in range(len(real_pvals)) if not np.isnan(real_pvals[i]) and not np.isnan(synthetic_pvals[i])]
    
    if valid_indices:
        valid_real = [real_pvals[i] for i in valid_indices]
        valid_synthetic = [synthetic_pvals[i] for i in valid_indices]
        valid_names = [test_names[i] for i in valid_indices]
        
        axes[1, 1].scatter(valid_real, valid_synthetic, s=50, alpha=0.7)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=2)
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
        axes[1, 1].axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
        
        for i, name in enumerate(valid_names):
            axes[1, 1].annotate(name, (valid_real[i], valid_synthetic[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_title('P-value Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Real P-values')
    axes[1, 1].set_ylabel('Synthetic P-values')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    # Calculate sigma ratio
    sigma_real = np.std(real_residuals)
    sigma_synthetic = np.std(synthetic_residuals)
    sigma_ratio = sigma_real / sigma_synthetic
    
    # Calculate test agreement
    agreement = sum(1 for i in range(len(real_passes)) if real_passes[i] == synthetic_passes[i])
    agreement_pct = agreement / len(real_passes) * 100
    
    summary_text = f"""Validation Summary

Model: ARMA({trial_info["arma_order"][0]},{trial_info["arma_order"][1]})
       GARCH({trial_info["garch_order"][0]},{trial_info["garch_order"][1]})
       {trial_info["distribution"]}-distribution

Sample Sizes:
  Real: {len(real_residuals):,}
  Synthetic: {len(synthetic_residuals):,}

Statistics:
  σ_real: {sigma_real:.4f}
  σ_synthetic: {sigma_synthetic:.4f}
  σ_ratio: {sigma_ratio:.4f}

Distribution:
  Real mean: {np.mean(real_residuals):.4f}
  Synthetic mean: {np.mean(synthetic_residuals):.4f}
  
  Real skew: {stats.skew(real_residuals):.3f}
  Synthetic skew: {stats.skew(synthetic_residuals):.3f}
  
  Real kurtosis: {stats.kurtosis(real_residuals):.3f}
  Synthetic kurtosis: {stats.kurtosis(synthetic_residuals):.3f}

Test Agreement: {agreement_pct:.0f}%
  ({agreement}/{len(real_passes)} tests agree)

Quality Assessment:
  σ_ratio ∈ [0.8, 1.2]: {'✓' if 0.8 <= sigma_ratio <= 1.2 else '✗'}
  Agreement > 75%: {'✓' if agreement_pct > 75 else '✗'}
"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'sigma_ratio': sigma_ratio,
        'agreement_percentage': agreement_pct,
        'real_stats': {'mean': np.mean(real_residuals), 'std': sigma_real, 'skew': stats.skew(real_residuals), 'kurtosis': stats.kurtosis(real_residuals)},
        'synthetic_stats': {'mean': np.mean(synthetic_residuals), 'std': sigma_synthetic, 'skew': stats.skew(synthetic_residuals), 'kurtosis': stats.kurtosis(synthetic_residuals)}
    }

def main():
    """Main function for synthetic residual validation."""
    
    print("="*70)
    print("SYNTHETIC RESIDUAL GENERATION AND VALIDATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path("results_plots/residual_modeling/synthetic_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fitted models
    print("\n1. Loading fitted ARMA-GARCH models...")
    fitted_models = load_fitted_models()
    
    if not fitted_models:
        print("❌ No fitted models found.")
        return
    
    print(f"✅ Loaded {len(fitted_models)} fitted models")
    
    # Find best trial for demonstration
    print("\n2. Selecting best trial for validation...")
    
    # Simple selection - first trial with complete data
    best_trial_name = None
    best_trial_data = None
    
    for trial_name, trial_data in fitted_models.items():
        if (trial_data and 'original_residuals' in trial_data and 
            'arma_params' in trial_data and 'garch_params' in trial_data):
            best_trial_name = trial_name
            best_trial_data = trial_data
            break
    
    if not best_trial_name:
        print("❌ No suitable trial found for validation.")
        return
    
    print(f"✅ Selected trial: {best_trial_name}")
    
    # Extract real residuals
    real_residuals = best_trial_data['original_residuals']
    print(f"   Real residuals length: {len(real_residuals)}")
    
    # Generate synthetic residuals
    print("\n3. Generating synthetic residuals...")
    
    arma_params = best_trial_data['arma_params']
    garch_params = best_trial_data['garch_params']
    distribution_params = best_trial_data.get('distribution_params', {'nu': 5.0})
    distribution = best_trial_data.get('best_distribution', 't')
    
    n_samples = len(real_residuals)
    synthetic_residuals, volatility, innovations = simulate_arma_garch(
        arma_params, garch_params, distribution_params, n_samples, distribution)
    
    print(f"   Generated {len(synthetic_residuals)} synthetic residuals")
    print(f"   Real σ: {np.std(real_residuals):.4f}")
    print(f"   Synthetic σ: {np.std(synthetic_residuals):.4f}")
    print(f"   σ ratio: {np.std(real_residuals) / np.std(synthetic_residuals):.4f}")
    
    # Run statistical tests
    print("\n4. Running statistical tests...")
    
    print("   Testing real residuals...")
    real_tests = run_statistical_tests(real_residuals, "Real")
    
    print("   Testing synthetic residuals...")
    synthetic_tests = run_statistical_tests(synthetic_residuals, "Synthetic")
    
    # Create validation plots
    print("\n5. Creating validation plots...")
    
    trial_info = {
        'trial_name': best_trial_name,
        'arma_order': [best_trial_data['optimal_orders']['p'], best_trial_data['optimal_orders']['q']],
        'garch_order': [best_trial_data['optimal_orders']['r'], best_trial_data['optimal_orders']['s']],
        'distribution': distribution
    }
    
    output_file = output_dir / f"validation_comparison_{best_trial_name.replace('/', '_')}.png"
    
    validation_results = create_validation_plots(
        real_residuals, synthetic_residuals, real_tests, synthetic_tests, output_file, trial_info)
    
    # Save detailed results
    print("\n6. Saving validation results...")
    
    results_summary = {
        'trial_info': trial_info,
        'validation_metrics': validation_results,
        'statistical_tests': {
            'real': {test: {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else v
                           for k, v in results.items()} 
                    for test, results in real_tests.items()},
            'synthetic': {test: {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else v
                                for k, v in results.items()} 
                         for test, results in synthetic_tests.items()}
        }
    }
    
    import json
    summary_file = output_dir / f"validation_summary_{best_trial_name.replace('/', '_')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)  # default=str handles any remaining numpy types
    
    print(f"✅ Synthetic residual validation completed!")
    print(f"📁 Output files:")
    print(f"   - Validation plots: {output_file}")
    print(f"   - Summary: {summary_file}")
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   σ ratio (real/synthetic): {validation_results['sigma_ratio']:.4f} {'✓' if 0.8 <= validation_results['sigma_ratio'] <= 1.2 else '✗'}")
    print(f"   Test agreement: {validation_results['agreement_percentage']:.0f}% {'✓' if validation_results['agreement_percentage'] > 75 else '✗'}")
    
    # Test-by-test comparison
    test_names = ['Shapiro', 'Jarque-Bera', 'Lilliefors', 'ADF', 'KPSS', 'Ljung-Box', 'ARCH-LM', 'Runs']
    test_keys = ['shapiro', 'jarque_bera', 'lilliefors', 'adf', 'kpss', 'ljung_box', 'arch_lm', 'runs']
    
    print(f"\n   Test-by-test comparison:")
    print(f"   {'Test':<12} {'Real':<6} {'Synthetic':<10} {'Match'}")
    print("-" * 40)
    
    for i, (name, key) in enumerate(zip(test_names, test_keys)):
        real_pass = '✓' if real_tests[key]['pass'] else '✗'
        synth_pass = '✓' if synthetic_tests[key]['pass'] else '✗'
        match = '✓' if real_tests[key]['pass'] == synthetic_tests[key]['pass'] else '✗'
        print(f"   {name:<12} {real_pass:<6} {synth_pass:<10} {match}")

if __name__ == "__main__":
    main()