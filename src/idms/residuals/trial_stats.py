#!/usr/bin/env python3
"""
Trial-Level Statistical Tests on PyTorch Elbow Model Residuals
==============================================================

Conduct statistical tests on individual trials to identify specific violations
and failed tests per trial.

Format similar to residual_statistical_tests.py output with trial-specific
test failures and violations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, het_white
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def center_residuals(residuals: np.ndarray) -> np.ndarray:
    """Center residuals to zero mean."""
    return residuals - np.mean(residuals)


def extract_all_subject005_residuals() -> pd.DataFrame:
    """Extract residuals for ALL trials of subject 005, not just train/val/test splits."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_models'))
    
    from extract_pytorch_residuals import PyTorchResidualExtractor
    
    print("Extracting residuals for ALL subject 005 trials...")
    
    # Initialize extractor (pass directory path, not file path)
    model_dir = "pytorch_models/experiments/multistage_3108/stage2_finetune_subject_005_tcn_frozen"
    extractor = PyTorchResidualExtractor(model_dir)
    
    # Get all trials for subject 005 from the dataset
    import h5py
    dataset_path = "data/idms_ready_dataset.h5"
    
    with h5py.File(dataset_path, 'r') as f:
        # Get all subject 005 trials (correct path structure)
        subject_005_trials = [f'subject_005/{trial}' for trial in f['subjects/subject_005'].keys()]
        print(f"Found {len(subject_005_trials)} trials for subject 005")
    
    all_residuals = []
    
    for trial_name in tqdm(subject_005_trials, desc="Processing trials"):
        try:
            # Extract residuals for this trial
            residuals = extractor.extract_residuals_for_trial(trial_name)
            
            if residuals is not None and len(residuals) > 0:
                # Get split information from the extractor
                split = extractor.get_trial_split(trial_name) if hasattr(extractor, 'get_trial_split') else 'unknown'
                
                all_residuals.append({
                    'subject': 'subject_005',
                    'trial': trial_name.split('/')[-1],
                    'trial_name': trial_name, 
                    'split': split,
                    'n_windows': len(residuals),
                    'residuals': residuals
                })
            else:
                print(f"Warning: No residuals extracted for {trial_name}")
                
        except Exception as e:
            print(f"Error processing {trial_name}: {e}")
            continue
    
    return pd.DataFrame(all_residuals)


def test_normality(residuals: np.ndarray) -> Dict:
    """Test for normality using multiple tests."""
    results = {}
    alpha = 0.05
    
    # Shapiro-Wilk test (best for small to medium samples)
    if len(residuals) <= 5000:  # Shapiro-Wilk limit
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results['shapiro_normal'] = shapiro_p > alpha
        results['shapiro_p'] = shapiro_p
    else:
        results['shapiro_normal'] = None
        results['shapiro_p'] = None
    
    # Jarque-Bera test
    jb_result = jarque_bera(residuals)
    jb_stat, jb_p = jb_result[0], jb_result[1]
    results['jarque_bera_normal'] = jb_p > alpha
    results['jarque_bera_p'] = jb_p
    
    # Anderson-Darling test
    ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
    results['anderson_normal'] = ad_stat < ad_critical[2]  # 5% significance level
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
    results['ks_normal'] = ks_p > alpha
    results['ks_p'] = ks_p
    
    return results


def test_stationarity(residuals: np.ndarray) -> Dict:
    """Test for stationarity using ADF and KPSS tests."""
    results = {}
    alpha = 0.05
    
    # Augmented Dickey-Fuller test (H0: non-stationary)
    adf_stat, adf_p, _, _, adf_critical, _ = adfuller(residuals, autolag='AIC')
    results['adf_stationary'] = adf_p < alpha
    results['adf_p'] = adf_p
    
    # KPSS test (H0: stationary)  
    kpss_stat, kpss_p, _, kpss_critical = kpss(residuals, regression='c')
    results['kpss_stationary'] = kpss_p > alpha
    results['kpss_p'] = kpss_p
    
    # Combined conclusion
    if results['adf_stationary'] and results['kpss_stationary']:
        results['stationarity_conclusion'] = 'Stationary'
    elif not results['adf_stationary'] and not results['kpss_stationary']:
        results['stationarity_conclusion'] = 'Non-stationary'
    else:
        results['stationarity_conclusion'] = 'Inconclusive'
    
    return results


def test_autocorrelation(residuals: np.ndarray) -> Dict:
    """Test for autocorrelation using Ljung-Box test."""
    results = {}
    alpha = 0.05
    
    # Ljung-Box test for different lags
    for lag in [1, 5, 10, 20]:
        if len(residuals) > lag:
            lb_result = acorr_ljungbox(residuals, lags=lag, return_df=True)
            lb_p = lb_result['lb_pvalue'].iloc[-1]  # Get p-value for highest lag
            
            results[f'ljung_box_lag{lag}_p'] = lb_p
            results[f'ljung_box_lag{lag}_independent'] = lb_p > alpha
        else:
            results[f'ljung_box_lag{lag}_p'] = None
            results[f'ljung_box_lag{lag}_independent'] = None
    
    # Overall autocorrelation assessment (based on lag 20)
    if results['ljung_box_lag20_p'] is not None:
        results['autocorrelated'] = results['ljung_box_lag20_p'] <= alpha
    else:
        results['autocorrelated'] = None
    
    return results


def test_heteroscedasticity(residuals: np.ndarray) -> Dict:
    """Test for heteroscedasticity using multiple tests."""
    results = {}
    alpha = 0.05
    
    # Center residuals first
    centered_residuals = center_residuals(residuals)
    
    # Create time trend as explanatory variable (same as residual_statistical_tests.py)
    n = len(centered_residuals)
    time_trend = np.arange(1, n + 1)
    X = sm.add_constant(time_trend)  # Add intercept
    
    # ARCH-LM test
    try:
        arch_stat, arch_p, _, _ = het_arch(centered_residuals, maxlag=1)
        results['arch_homoscedastic'] = arch_p > alpha
        results['arch_p'] = arch_p
    except:
        results['arch_homoscedastic'] = None
        results['arch_p'] = None
    
    # Breusch-Pagan test
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(centered_residuals, X)
        results['breusch_pagan_homoscedastic'] = bp_p > alpha
        results['breusch_pagan_p'] = bp_p
    except:
        results['breusch_pagan_homoscedastic'] = None
        results['breusch_pagan_p'] = None
    
    # White test
    try:
        white_stat, white_p, _, _ = het_white(centered_residuals, X)
        results['white_homoscedastic'] = white_p > alpha
        results['white_p'] = white_p
    except:
        results['white_homoscedastic'] = None
        results['white_p'] = None
    
    return results


def runs_test(residuals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Wald-Wolfowitz runs test for randomness.
    H0: residuals are randomly distributed
    """
    centered_residuals = center_residuals(residuals)
    
    # Convert to binary sequence (above/below median)
    median_val = np.median(centered_residuals)
    binary_sequence = (centered_residuals > median_val).astype(int)
    
    # Count runs
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    # Calculate expected runs and variance
    n1 = np.sum(binary_sequence == 1)  # Above median
    n2 = np.sum(binary_sequence == 0)  # Below median
    n = n1 + n2
    
    if n1 == 0 or n2 == 0:
        # Degenerate case
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'reject_h0': False,
            'runs': runs,
            'note': f'Runs: {runs}, Above median: {n1}, Below median: {n2}'
        }
    
    expected_runs = (2 * n1 * n2) / n + 1
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    
    # Standardized test statistic
    if variance_runs > 0:
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0
        p_value = 1.0
    
    return {
        'statistic': z_stat,
        'p_value': p_value,
        'reject_h0': p_value < alpha,
        'runs': runs,
        'expected_runs': expected_runs,
        'note': f'Runs: {runs}, Expected: {expected_runs:.2f}, Above median: {n1}, Below median: {n2}'
    }


def test_independence(residuals: np.ndarray) -> Dict:
    """Test for independence using runs test."""
    results = {}
    alpha = 0.05
    
    # Runs test
    try:
        runs_result = runs_test(residuals, alpha)
        results['runs_independent'] = not runs_result['reject_h0']
        results['runs_p'] = runs_result['p_value']
    except:
        results['runs_independent'] = None
        results['runs_p'] = None
    
    return results


def detect_outliers(residuals: np.ndarray) -> Dict:
    """Detect outliers using multiple methods."""
    results = {}
    
    # IQR method
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = (residuals < lower_bound) | (residuals > upper_bound)
    
    results['outliers_iqr_count'] = np.sum(iqr_outliers)
    results['outliers_iqr_percent'] = (np.sum(iqr_outliers) / len(residuals)) * 100
    
    # Z-score method (|z| > 3)
    z_scores = np.abs(stats.zscore(residuals))
    zscore_outliers = z_scores > 3
    
    results['outliers_zscore_count'] = np.sum(zscore_outliers)
    results['outliers_zscore_percent'] = (np.sum(zscore_outliers) / len(residuals)) * 100
    
    # Modified Z-score method (|modified z| > 3.5)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    modified_z_scores = 0.6745 * (residuals - median) / mad
    mod_z_outliers = np.abs(modified_z_scores) > 3.5
    
    results['outliers_modified_z_count'] = np.sum(mod_z_outliers)
    results['outliers_modified_z_percent'] = (np.sum(mod_z_outliers) / len(residuals)) * 100
    
    return results


def engle_ng_sign_bias_test(residuals: np.ndarray, alpha: float = 0.05) -> Dict:
    """Engle-Ng sign bias test for asymmetric volatility effects."""
    centered_residuals = center_residuals(residuals)
    
    if len(centered_residuals) < 10:
        return {
            'asymmetric': None,
            'p': None,
            'note': 'Insufficient data'
        }
    
    # Prepare data
    squared_resid = centered_residuals[1:]**2
    lagged_resid = centered_residuals[:-1]
    
    # Sign indicator (1 if negative shock, 0 if positive)
    S = (lagged_resid < 0).astype(int)
    
    try:
        # Joint Test (most important)
        X_joint = np.column_stack([
            np.ones(len(S)), 
            S, 
            S * lagged_resid, 
            (1-S) * lagged_resid
        ])
        model_joint = sm.OLS(squared_resid, X_joint).fit()
        
        # F-test for joint significance of asymmetric terms
        joint_pvalue = model_joint.f_pvalue
        joint_fstat = model_joint.fvalue
        
        return {
            'asymmetric': joint_pvalue < alpha,
            'p': joint_pvalue,
            'statistic': joint_fstat,
            'r_squared': model_joint.rsquared
        }
        
    except Exception as e:
        return {
            'asymmetric': None,
            'p': None,
            'error': str(e)
        }


def mcleod_li_test(residuals: np.ndarray, lags: int = 10, alpha: float = 0.05) -> Dict:
    """McLeod-Li test for asymmetric volatility via autocorrelation of |residuals|."""
    centered_residuals = center_residuals(residuals)
    
    if len(centered_residuals) < 2 * lags:
        return {
            'asymmetric': None,
            'p': None,
            'note': 'Insufficient data'
        }
    
    try:
        # Calculate autocorrelations
        abs_resid = np.abs(centered_residuals)
        squared_resid = centered_residuals**2
        
        # Ljung-Box test on |residuals|
        abs_ljung_result = acorr_ljungbox(abs_resid, lags=lags, return_df=True)
        abs_pvalue = abs_ljung_result['lb_pvalue'].iloc[-1]
        abs_statistic = abs_ljung_result['lb_stat'].iloc[-1]
        
        # Ljung-Box test on squared residuals  
        sq_ljung_result = acorr_ljungbox(squared_resid, lags=lags, return_df=True)
        sq_pvalue = sq_ljung_result['lb_pvalue'].iloc[-1]
        
        # McLeod-Li focuses on |residuals| autocorrelation
        is_asymmetric = abs_pvalue < alpha
        
        return {
            'asymmetric': is_asymmetric,
            'p': abs_pvalue,
            'statistic': abs_statistic,
            'abs_resid_pvalue': abs_pvalue,
            'squared_resid_pvalue': sq_pvalue
        }
        
    except Exception as e:
        return {
            'asymmetric': None,
            'p': None,
            'error': str(e)
        }


def fit_student_t(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit Student-t distribution to residuals."""
    centered_residuals = center_residuals(residuals)
    df, loc, scale = stats.t.fit(centered_residuals)
    return df, loc, scale


def fit_skew_normal(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit Skew-normal distribution to residuals."""
    centered_residuals = center_residuals(residuals)
    a, loc, scale = stats.skewnorm.fit(centered_residuals)
    return a, loc, scale


def distribution_comparison(residuals: np.ndarray) -> Dict:
    """Compare different distribution fits and calculate goodness of fit."""
    centered_residuals = center_residuals(residuals)
    n = len(centered_residuals)
    
    # Fit distributions
    normal_params = (0, np.std(centered_residuals))
    t_params = fit_student_t(residuals)
    skewnorm_params = fit_skew_normal(residuals)
    
    distributions = {
        'Normal': (stats.norm, normal_params),
        'Student-t': (stats.t, t_params),
        'Skew-Normal': (stats.skewnorm, skewnorm_params)
    }
    
    results = {}
    
    for dist_name, (dist, params) in distributions.items():
        try:
            # Calculate log-likelihood
            log_likelihood = np.sum(dist.logpdf(centered_residuals, *params))
            
            # Calculate AIC and BIC
            k = len(params)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(centered_residuals, 
                                             lambda x: dist.cdf(x, *params))
            
            results[dist_name] = {
                'params': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'n_params': k
            }
        except Exception as e:
            results[dist_name] = {
                'error': str(e),
                'aic': np.inf,
                'bic': np.inf
            }
    
    # Find best distribution by AIC
    valid_results = {k: v for k, v in results.items() if 'aic' in v and np.isfinite(v['aic'])}
    if valid_results:
        best_aic = min(valid_results.keys(), key=lambda x: valid_results[x]['aic'])
        best_bic = min(valid_results.keys(), key=lambda x: valid_results[x]['bic'])
    else:
        best_aic = 'None'
        best_bic = 'None'
    
    return {
        'distribution_results': results,
        'best_aic': best_aic,
        'best_bic': best_bic
    }


def identify_test_failures(test_results: Dict) -> List[str]:
    """Identify which tests failed and what violations occurred."""
    failures = []
    
    # Normality failures
    normality_tests = ['shapiro_normal', 'jarque_bera_normal', 'anderson_normal', 'ks_normal']
    failed_normality = [test for test in normality_tests if test in test_results and not test_results[test]]
    if len(failed_normality) >= 2:  # If 2+ normality tests fail
        failures.append(f"NON_NORMAL({len(failed_normality)}/4_tests)")
    
    # Stationarity failures  
    if test_results.get('stationarity_conclusion') == 'Non-stationary':
        failures.append("NON_STATIONARY")
    elif test_results.get('stationarity_conclusion') == 'Inconclusive':
        failures.append("STATIONARITY_INCONCLUSIVE")
    
    # Autocorrelation 
    if test_results.get('autocorrelated'):
        failures.append("AUTOCORRELATED")
    
    # Heteroscedasticity
    hetero_tests = ['arch_homoscedastic', 'breusch_pagan_homoscedastic', 'white_homoscedastic']
    failed_hetero = [test for test in hetero_tests if test in test_results and not test_results[test]]
    if len(failed_hetero) >= 2:  # If 2+ heteroscedasticity tests fail
        failures.append(f"HETEROSCEDASTIC({len(failed_hetero)}/3_tests)")
    
    # Independence
    if test_results.get('runs_independent') == False:
        failures.append("TEMPORAL_DEPENDENCE")
    
    # High outlier rate
    iqr_outlier_pct = test_results.get('outliers_iqr_percent', 0)
    if iqr_outlier_pct > 20:
        failures.append(f"HIGH_OUTLIERS({iqr_outlier_pct:.1f}%)")
    elif iqr_outlier_pct > 10:
        failures.append(f"MODERATE_OUTLIERS({iqr_outlier_pct:.1f}%)")
    
    return failures


def load_existing_trial_residuals() -> pd.DataFrame:
    """Load residuals for the 10 trials we already have from splits."""
    residual_dir = Path("residual_analysis_clean/pytorch_elbow_residuals")
    
    # Load trial details to get trial mapping
    trial_details = pd.read_csv(residual_dir / "trial_details.csv")
    
    # Load residuals for each split
    all_residuals = []
    
    for split in ['test', 'val', 'train']:
        residual_file = residual_dir / f"{split}_residuals.npy"
        if residual_file.exists():
            residuals = np.load(residual_file)
            
            # Get trials for this split
            split_trials = trial_details[trial_details['split'] == split].copy()
            
            # Calculate cumulative windows to assign residuals to trials
            split_trials = split_trials.sort_values(['subject', 'trial'])
            split_trials['cumsum_windows'] = split_trials['n_windows'].cumsum()
            split_trials['start_idx'] = split_trials['cumsum_windows'].shift(1).fillna(0).astype(int)
            split_trials['end_idx'] = split_trials['cumsum_windows'].astype(int)
            
            # Extract residuals for each trial
            for _, row in split_trials.iterrows():
                start_idx = row['start_idx'] 
                end_idx = row['end_idx']
                trial_residuals = residuals[start_idx:end_idx]
                
                all_residuals.append({
                    'split': split,
                    'subject': row['subject'],
                    'trial': row['trial'], 
                    'trial_name': row['trial_name'],
                    'n_windows': len(trial_residuals),
                    'residuals': trial_residuals
                })
    
    return pd.DataFrame(all_residuals)


def comprehensive_trial_statistical_analysis():
    """Run comprehensive statistical analysis on individual trials."""
    
    print("Loading trial-level residuals for all 61 trials...")
    trial_data = extract_all_subject005_residuals()
    
    print(f"Analyzing {len(trial_data)} trials...")
    
    results = []
    
    for idx, row in trial_data.iterrows():
        residuals = row['residuals']
        
        if len(residuals) < 10:  # Skip trials with too few samples
            print(f"Skipping {row['trial_name']} (only {len(residuals)} samples)")
            continue
        
        print(f"Analyzing {row['trial_name']} ({len(residuals)} samples)...")
        
        # Run all statistical tests
        result = {
            'split': row['split'],
            'subject': row['subject'], 
            'trial': row['trial'],
            'trial_name': row['trial_name'],
            'n_samples': len(residuals)
        }
        
        # Normality tests
        norm_results = test_normality(residuals)
        result.update(norm_results)
        
        # Stationarity tests
        stat_results = test_stationarity(residuals)
        result.update(stat_results)
        
        # Autocorrelation tests
        autocorr_results = test_autocorrelation(residuals)
        result.update(autocorr_results)
        
        # Heteroscedasticity tests  
        hetero_results = test_heteroscedasticity(residuals)
        result.update(hetero_results)
        
        # Independence tests
        indep_results = test_independence(residuals)
        result.update(indep_results)
        
        # Asymmetric volatility tests
        sign_bias_results = engle_ng_sign_bias_test(residuals)
        result.update({f'sign_bias_{k}': v for k, v in sign_bias_results.items()})
        
        mcleod_li_results = mcleod_li_test(residuals)
        result.update({f'mcleod_li_{k}': v for k, v in mcleod_li_results.items()})
        
        # Distribution analysis
        dist_results = distribution_comparison(residuals)
        result['best_distribution_aic'] = dist_results['best_aic']
        result['best_distribution_bic'] = dist_results['best_bic']
        
        # Extract key distribution parameters
        if 'Skew-Normal' in dist_results['distribution_results'] and 'params' in dist_results['distribution_results']['Skew-Normal']:
            skew_params = dist_results['distribution_results']['Skew-Normal']['params']
            result['skew_normal_a'] = skew_params[0]  # skewness parameter
            result['skew_normal_aic'] = dist_results['distribution_results']['Skew-Normal']['aic']
        
        if 'Student-t' in dist_results['distribution_results'] and 'params' in dist_results['distribution_results']['Student-t']:
            t_params = dist_results['distribution_results']['Student-t']['params']
            result['student_t_df'] = t_params[0]  # degrees of freedom
            result['student_t_aic'] = dist_results['distribution_results']['Student-t']['aic']
        
        # Outlier detection
        outlier_results = detect_outliers(residuals)
        result.update(outlier_results)
        
        # Identify test failures and violations
        test_failures = identify_test_failures(result)
        result['test_failures'] = '; '.join(test_failures) if test_failures else 'NONE'
        result['num_violations'] = len(test_failures)
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("residual_analysis_clean/pytorch_elbow_residuals")
    output_file = output_dir / "trial_level_statistical_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved trial-level statistical results to: {output_file}")
    
    # Print summary in residual_statistical_tests.py format
    print("\n" + "="*80)
    print("TRIAL-LEVEL STATISTICAL TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total trials analyzed: {len(results_df)}")
    print(f"   Trials with violations: {len(results_df[results_df['num_violations'] > 0])}")
    print(f"   Average violations per trial: {results_df['num_violations'].mean():.1f}")
    
    # Group by split
    for split in ['test', 'val', 'train']:
        split_data = results_df[results_df['split'] == split]
        if len(split_data) == 0:
            continue
            
        print(f"\n{'='*20} {split.upper()} SPLIT {'='*20}")
        print(f"Trials: {len(split_data)}")
        print(f"Violations per trial: {split_data['num_violations'].mean():.1f} ± {split_data['num_violations'].std():.1f}")
        
        # Show trials with most violations
        worst_trials = split_data.nlargest(3, 'num_violations')[['trial_name', 'num_violations', 'test_failures']]
        
        print(f"\nWorst trials:")
        for _, trial in worst_trials.iterrows():
            print(f"   {trial['trial_name']}: {trial['num_violations']} violations")
            print(f"     → {trial['test_failures']}")
    
    # Show most common violations across all trials
    print(f"\n{'='*20} MOST COMMON VIOLATIONS {'='*20}")
    all_failures = []
    for failures in results_df['test_failures']:
        if failures != 'NONE':
            all_failures.extend(failures.split('; '))
    
    from collections import Counter
    failure_counts = Counter(all_failures)
    
    print("Violation frequency across all trials:")
    for failure, count in failure_counts.most_common():
        pct = (count / len(results_df)) * 100
        print(f"   {failure}: {count}/{len(results_df)} trials ({pct:.1f}%)")
    
    return results_df


if __name__ == "__main__":
    results_df = comprehensive_trial_statistical_analysis()