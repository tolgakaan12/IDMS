#!/usr/bin/env python3
"""
Statistical Tests on PyTorch Elbow Model Residuals
==================================================

Conduct comprehensive statistical tests on velocity residuals from subject 005
before fitting ARMA-GARCH and Kalman filter models.

Tests performed at α = 0.05 significance level:
- Normality: Shapiro-Wilk, Jarque-Bera, Anderson-Darling
- Stationarity: Augmented Dickey-Fuller (ADF), KPSS
- Autocorrelation: Ljung-Box test
- Heteroscedasticity: Breusch-Pagan, ARCH-LM, White test
- Independence: Runs test
- Outliers: Grubbs test (approximate)

Date: 2025-01-15
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, het_white
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
from statsmodels.sandbox.stats.runs import runstest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_pytorch_residuals(residual_dir: str = "residual_analysis_clean/pytorch_elbow_residuals") -> Dict[str, np.ndarray]:
    """Load PyTorch residuals for all splits."""
    residual_path = Path(residual_dir)
    residuals = {}
    
    for split in ['test', 'val', 'train']:
        file_path = residual_path / f"{split}_residuals.npy"
        if file_path.exists():
            data = np.load(file_path)
            residuals[split] = data
            print(f"Loaded {split} residuals: {len(data):,} samples")
            print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
            print(f"  Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
        else:
            print(f"Warning: {file_path} not found")
            
    return residuals


def test_normality(data: np.ndarray, name: str) -> Dict:
    """Test for normality using multiple methods."""
    results = {'component': name}
    
    # Shapiro-Wilk (best for small samples, but we'll subsample for large data)
    if len(data) > 5000:
        sample_idx = np.random.choice(len(data), 5000, replace=False)
        sample_data = data[sample_idx]
    else:
        sample_data = data
        
    sw_stat, sw_p = stats.shapiro(sample_data)
    results['shapiro_stat'] = sw_stat
    results['shapiro_p'] = sw_p
    results['shapiro_normal'] = sw_p > 0.05
    
    # Jarque-Bera test
    jb_result = jarque_bera(data)
    results['jarque_bera_stat'] = jb_result[0]
    results['jarque_bera_p'] = jb_result[1]
    results['jarque_bera_normal'] = jb_result[1] > 0.05
    
    # Anderson-Darling test
    ad_stat, ad_crit, ad_sig = stats.anderson(data, dist='norm')
    results['anderson_stat'] = ad_stat
    results['anderson_normal'] = ad_stat < ad_crit[2]  # 5% significance level
    
    # Kolmogorov-Smirnov test against standard normal
    # First standardize the data
    standardized = (data - np.mean(data)) / np.std(data)
    ks_stat, ks_p = stats.kstest(standardized, 'norm')
    results['ks_stat'] = ks_stat
    results['ks_p'] = ks_p
    results['ks_normal'] = ks_p > 0.05
    
    return results


def test_stationarity(data: np.ndarray, name: str) -> Dict:
    """Test for stationarity using ADF and KPSS tests."""
    results = {'component': name}
    
    # Augmented Dickey-Fuller test (H0: has unit root / non-stationary)
    adf_stat, adf_p, adf_lags, adf_obs, adf_crit, adf_icbest = adfuller(data, autolag='AIC', maxlag=20)
    results['adf_stat'] = adf_stat
    results['adf_p'] = adf_p
    results['adf_stationary'] = adf_p < 0.05  # Reject H0 = stationary
    results['adf_critical_1%'] = adf_crit['1%']
    results['adf_critical_5%'] = adf_crit['5%']
    
    # KPSS test (H0: is stationary)
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(data, regression='c', nlags='auto')
    results['kpss_stat'] = kpss_stat
    results['kpss_p'] = kpss_p
    results['kpss_stationary'] = kpss_p > 0.05  # Fail to reject H0 = stationary
    results['kpss_critical_5%'] = kpss_crit['5%']
    
    # Combined interpretation
    if results['adf_stationary'] and results['kpss_stationary']:
        results['stationarity_conclusion'] = 'Stationary'
    elif not results['adf_stationary'] and not results['kpss_stationary']:
        results['stationarity_conclusion'] = 'Non-stationary'
    else:
        results['stationarity_conclusion'] = 'Inconclusive'
    
    return results


def test_autocorrelation(data: np.ndarray, name: str, lags: int = 20) -> Dict:
    """Test for autocorrelation using Ljung-Box test."""
    results = {'component': name}
    
    # Ljung-Box test (H0: no autocorrelation up to lag h)
    lb_result = acorr_ljungbox(data, lags=lags, return_df=True, auto_lag=False)
    
    # Get results for multiple lag values
    for lag in [1, 5, 10, 20]:
        if lag <= len(lb_result):
            results[f'ljung_box_lag{lag}_stat'] = lb_result.loc[lag, 'lb_stat']
            results[f'ljung_box_lag{lag}_p'] = lb_result.loc[lag, 'lb_pvalue']
            results[f'ljung_box_lag{lag}_independent'] = lb_result.loc[lag, 'lb_pvalue'] > 0.05
    
    # Overall assessment (use lag 20)
    if 20 <= len(lb_result):
        results['autocorrelated'] = lb_result.loc[20, 'lb_pvalue'] < 0.05
    else:
        results['autocorrelated'] = lb_result.iloc[-1]['lb_pvalue'] < 0.05
    
    return results


def test_heteroscedasticity(data: np.ndarray, name: str) -> Dict:
    """Test for heteroscedasticity using multiple methods."""
    results = {'component': name}
    
    # Create a simple time trend for regression
    n = len(data)
    time_trend = np.arange(n).reshape(-1, 1)
    
    # Breusch-Pagan test (H0: homoscedastic)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(time_trend)
        bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(data, X)
        results['breusch_pagan_lm'] = bp_lm
        results['breusch_pagan_p'] = bp_p
        results['breusch_pagan_homoscedastic'] = bp_p > 0.05
    except:
        results['breusch_pagan_p'] = np.nan
        results['breusch_pagan_homoscedastic'] = np.nan
    
    # ARCH-LM test for conditional heteroscedasticity (H0: no ARCH effects)
    try:
        arch_lm, arch_p, arch_f, arch_f_p = het_arch(data, maxlag=10)
        results['arch_lm'] = arch_lm
        results['arch_p'] = arch_p
        results['arch_homoscedastic'] = arch_p > 0.05
    except:
        results['arch_p'] = np.nan
        results['arch_homoscedastic'] = np.nan
    
    # White test (H0: homoscedastic)
    try:
        white_lm, white_p, white_f, white_f_p = het_white(data, X)
        results['white_lm'] = white_lm
        results['white_p'] = white_p
        results['white_homoscedastic'] = white_p > 0.05
    except:
        results['white_p'] = np.nan
        results['white_homoscedastic'] = np.nan
    
    return results


def test_independence(data: np.ndarray, name: str) -> Dict:
    """Test for independence using runs test."""
    results = {'component': name}
    
    try:
        # Runs test (H0: data is random/independent)
        # Convert to binary sequence (above/below median)
        median_val = np.median(data)
        binary_sequence = (data > median_val).astype(int)
        
        runs_stat, runs_p = runstest_1samp(binary_sequence)
        results['runs_stat'] = runs_stat
        results['runs_p'] = runs_p
        results['runs_independent'] = runs_p > 0.05
        
    except Exception as e:
        results['runs_p'] = np.nan
        results['runs_independent'] = np.nan
        
    return results


def test_outliers(data: np.ndarray, name: str) -> Dict:
    """Detect outliers using statistical methods."""
    results = {'component': name}
    
    # IQR method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = ((data < lower_bound) | (data > upper_bound))
    results['outliers_iqr_count'] = np.sum(outliers_iqr)
    results['outliers_iqr_percent'] = 100 * np.sum(outliers_iqr) / len(data)
    
    # Z-score method (|z| > 3)
    z_scores = np.abs(stats.zscore(data))
    outliers_zscore = z_scores > 3
    results['outliers_zscore_count'] = np.sum(outliers_zscore)
    results['outliers_zscore_percent'] = 100 * np.sum(outliers_zscore) / len(data)
    
    # Modified Z-score method (using median)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
    outliers_modified_z = np.abs(modified_z_scores) > 3.5
    results['outliers_modified_z_count'] = np.sum(outliers_modified_z)
    results['outliers_modified_z_percent'] = 100 * np.sum(outliers_modified_z) / len(data)
    
    return results


def comprehensive_statistical_analysis(residuals: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Run comprehensive statistical analysis on all residual splits."""
    all_results = []
    
    for split_name, data in residuals.items():
        print(f"\n{'='*60}")
        print(f"STATISTICAL ANALYSIS: {split_name.upper()} SPLIT")
        print(f"{'='*60}")
        print(f"Sample size: {len(data):,}")
        print(f"Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
        print(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
        print(f"Skewness: {stats.skew(data):.4f}, Kurtosis: {stats.kurtosis(data):.4f}")
        
        # Run all tests
        normality_results = test_normality(data, split_name)
        stationarity_results = test_stationarity(data, split_name)
        autocorr_results = test_autocorrelation(data, split_name)
        heterosced_results = test_heteroscedasticity(data, split_name)
        independence_results = test_independence(data, split_name)
        outlier_results = test_outliers(data, split_name)
        
        # Combine all results
        combined_results = {**normality_results, **stationarity_results, 
                          **autocorr_results, **heterosced_results,
                          **independence_results, **outlier_results}
        
        all_results.append(combined_results)
        
        # Print summary for this split
        print(f"\nTest Summary for {split_name.upper()}:")
        print(f"  Normality (Shapiro): {'✓' if normality_results.get('shapiro_normal') else '✗'} (p={normality_results.get('shapiro_p', 0):.6f})")
        print(f"  Normality (Jarque-Bera): {'✓' if normality_results.get('jarque_bera_normal') else '✗'} (p={normality_results.get('jarque_bera_p', 0):.6f})")
        print(f"  Stationarity: {stationarity_results.get('stationarity_conclusion', 'Unknown')}")
        print(f"  Autocorrelation: {'✓ Present' if autocorr_results.get('autocorrelated') else '✗ None'}")
        print(f"  Heteroscedasticity (ARCH): {'✓ Present' if not heterosced_results.get('arch_homoscedastic', True) else '✗ None'}")
        print(f"  Independence (Runs): {'✓' if independence_results.get('runs_independent') else '✗'}")
        print(f"  Outliers (IQR): {outlier_results.get('outliers_iqr_count', 0)} ({outlier_results.get('outliers_iqr_percent', 0):.2f}%)")
    
    return pd.DataFrame(all_results)


def create_diagnostic_plots(residuals: Dict[str, np.ndarray], output_dir: str = "residual_analysis/pytorch_elbow_residuals"):
    """Create diagnostic plots for residual analysis."""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('PyTorch Elbow Model - Residual Statistical Analysis', fontsize=16)
    
    colors = {'test': 'red', 'val': 'blue', 'train': 'green'}
    
    for i, (split_name, data) in enumerate(residuals.items()):
        color = colors.get(split_name, 'black')
        
        # Row 1: Distribution plots
        axes[0, 0].hist(data, bins=50, alpha=0.7, color=color, density=True, 
                       label=f'{split_name} (n={len(data):,})')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel('Residual Value')
        axes[0, 0].set_ylabel('Density')
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal)')
        
        # Box plot
        axes[0, 2].boxplot([data], positions=[i], widths=0.6, 
                          patch_artist=True, 
                          boxprops=dict(facecolor=color, alpha=0.7))
        
        # Row 2: Time series plots
        if len(data) > 1000:
            sample_idx = np.linspace(0, len(data)-1, 1000, dtype=int)
            sample_data = data[sample_idx]
            sample_time = np.arange(len(sample_data))
        else:
            sample_data = data
            sample_time = np.arange(len(data))
            
        axes[1, 0].plot(sample_time, sample_data, alpha=0.8, color=color, 
                       linewidth=0.5, label=split_name)
        axes[1, 0].set_title('Time Series (Sample)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residual')
        
        # Autocorrelation function
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(data, ax=axes[1, 1], lags=40, alpha=0.05, color=color)
        axes[1, 1].set_title('Autocorrelation Function')
        
        # Lag plot
        if len(data) > 1:
            axes[1, 2].scatter(data[:-1], data[1:], alpha=0.5, s=1, color=color)
            axes[1, 2].set_title('Lag-1 Plot')
            axes[1, 2].set_xlabel('Residual(t)')
            axes[1, 2].set_ylabel('Residual(t+1)')
        
        # Row 3: Advanced diagnostics
        # Rolling statistics
        window_size = min(100, len(data) // 10)
        if window_size > 1:
            rolling_mean = pd.Series(data).rolling(window_size).mean()
            rolling_std = pd.Series(data).rolling(window_size).std()
            
            axes[2, 0].plot(rolling_mean, color=color, label=f'{split_name} mean')
            axes[2, 0].fill_between(range(len(rolling_mean)), 
                                  rolling_mean - rolling_std,
                                  rolling_mean + rolling_std, 
                                  alpha=0.3, color=color)
        axes[2, 0].set_title('Rolling Statistics')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Value')
        
        # Squared residuals (for heteroscedasticity)
        squared_residuals = data**2
        if len(squared_residuals) > 1000:
            sample_squared = squared_residuals[sample_idx]
        else:
            sample_squared = squared_residuals
            
        axes[2, 1].plot(sample_time, sample_squared, alpha=0.8, color=color, 
                       linewidth=0.5, label=split_name)
        axes[2, 1].set_title('Squared Residuals')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Residual²')
        
    # Format plots
    axes[0, 0].legend()
    axes[0, 2].set_xticks(range(len(residuals)))
    axes[0, 2].set_xticklabels(list(residuals.keys()))
    axes[0, 2].set_title('Box Plots by Split')
    
    axes[1, 0].legend()
    axes[2, 0].legend()
    axes[2, 1].legend()
    
    # Remove empty subplot if we have less than 3 splits
    if len(residuals) < 3:
        axes[2, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_path / "statistical_analysis_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic plots saved to: {output_path / 'statistical_analysis_diagnostics.png'}")


def main():
    """Main function for statistical analysis of PyTorch residuals."""
    print("PyTorch Elbow Model - Statistical Analysis of Residuals")
    print("=" * 60)
    
    # Load residuals
    residuals = load_pytorch_residuals()
    
    if not residuals:
        print("ERROR: No residuals found! Please run extract_pytorch_residuals.py first.")
        return
    
    # Run comprehensive analysis
    results_df = comprehensive_statistical_analysis(residuals)
    
    # Save results
    output_dir = "residual_analysis/pytorch_elbow_residuals"
    results_df.to_csv(f"{output_dir}/statistical_test_results.csv", index=False)
    
    # Create diagnostic plots
    create_diagnostic_plots(residuals, output_dir)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/statistical_test_results.csv")
    print(f"Diagnostic plots: {output_dir}/statistical_analysis_diagnostics.png")
    
    # Print modeling recommendations
    print(f"\n{'='*60}")
    print("MODELING RECOMMENDATIONS")
    print(f"{'='*60}")
    
    for _, row in results_df.iterrows():
        split = row['component']
        print(f"\n{split.upper()} Split Recommendations:")
        
        # Normality
        if row.get('shapiro_normal') or row.get('jarque_bera_normal'):
            print("  ✓ Normal distribution - Gaussian models appropriate")
        else:
            print("  ⚠ Non-normal distribution - Consider Student-t or skewed models")
            
        # Stationarity
        if row.get('stationarity_conclusion') == 'Stationary':
            print("  ✓ Stationary - ARMA models appropriate")
        elif row.get('stationarity_conclusion') == 'Non-stationary':
            print("  ⚠ Non-stationary - Consider differencing or ARIMA models")
        else:
            print("  ? Stationarity unclear - Further investigation needed")
            
        # Autocorrelation
        if row.get('autocorrelated'):
            print("  ✓ Autocorrelated - ARMA modeling justified")
        else:
            print("  ⚠ No autocorrelation - Simple models may suffice")
            
        # Heteroscedasticity
        if not row.get('arch_homoscedastic', True):
            print("  ✓ ARCH effects present - GARCH modeling justified")
        else:
            print("  ⚠ Homoscedastic - GARCH may be unnecessary")
            
        # Outliers
        outlier_pct = row.get('outliers_iqr_percent', 0)
        if outlier_pct > 5:
            print(f"  ⚠ High outlier rate ({outlier_pct:.1f}%) - Robust methods recommended")
        else:
            print(f"  ✓ Low outlier rate ({outlier_pct:.1f}%) - Standard methods OK")


if __name__ == "__main__":
    main()