#!/usr/bin/env python3
"""
Robust NaN Handling for Residual Analysis
========================================

Utilities to properly handle NaNs in residual extraction and ARMA-GARCH analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

def clean_residuals(residuals: np.ndarray, 
                   min_length: int = 50,
                   verbose: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Clean residuals by handling NaNs and outliers.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Raw residual time series
    min_length : int
        Minimum length after cleaning to consider valid
    verbose : bool
        Print cleaning statistics
        
    Returns:
    --------
    cleaned_residuals, cleaning_stats
    """
    
    original_length = len(residuals)
    stats = {
        'original_length': original_length,
        'n_nan': np.sum(np.isnan(residuals)),
        'n_inf': np.sum(np.isinf(residuals)),
        'n_zero': np.sum(residuals == 0.0),
        'cleaned_length': 0,
        'fraction_kept': 0.0,
        'valid_for_analysis': False
    }
    
    if verbose:
        print(f"  Cleaning residuals: n={original_length}")
        print(f"    NaN values: {stats['n_nan']}")
        print(f"    Inf values: {stats['n_inf']}")
        print(f"    Zero values: {stats['n_zero']}")
    
    # Remove NaNs and Infs
    valid_mask = np.isfinite(residuals)
    cleaned = residuals[valid_mask]
    
    stats['cleaned_length'] = len(cleaned)
    stats['fraction_kept'] = len(cleaned) / original_length if original_length > 0 else 0.0
    stats['valid_for_analysis'] = len(cleaned) >= min_length
    
    if verbose:
        print(f"    After cleaning: n={len(cleaned)} ({stats['fraction_kept']:.1%} kept)")
        print(f"    Valid for analysis: {stats['valid_for_analysis']}")
    
    return cleaned, stats


def detect_suspicious_residuals(residuals: np.ndarray, 
                               z_threshold: float = 5.0) -> dict:
    """
    Detect suspicious residual patterns that might indicate extraction errors.
    """
    
    diagnostics = {
        'mean': np.mean(residuals),
        'std': np.std(residuals), 
        'min': np.min(residuals),
        'max': np.max(residuals),
        'n_extreme': np.sum(np.abs(residuals) > z_threshold * np.std(residuals)),
        'fraction_zeros': np.mean(residuals == 0.0),
        'autocorr_lag1': np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else np.nan
    }
    
    # Check for suspicious patterns
    warnings = []
    
    if diagnostics['fraction_zeros'] > 0.1:
        warnings.append(f"High fraction of zeros: {diagnostics['fraction_zeros']:.1%}")
    
    if abs(diagnostics['mean']) > 0.5 * diagnostics['std']:
        warnings.append(f"Large mean bias: {diagnostics['mean']:.3f}")
    
    if diagnostics['n_extreme'] > len(residuals) * 0.01:  # >1% extreme values
        warnings.append(f"Many extreme outliers: {diagnostics['n_extreme']}")
    
    if abs(diagnostics['autocorr_lag1']) > 0.8:
        warnings.append(f"Very high autocorrelation: {diagnostics['autocorr_lag1']:.3f}")
    
    diagnostics['warnings'] = warnings
    diagnostics['suspicious'] = len(warnings) > 0
    
    return diagnostics


def robust_trial_filter(trial_data: pd.DataFrame, 
                       min_length: int = 50,
                       min_fraction_valid: float = 0.8,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Filter trials to keep only those suitable for ARMA-GARCH analysis.
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        DataFrame with 'trial_name' and 'residuals' columns
    min_length : int
        Minimum residual length after cleaning
    min_fraction_valid : float
        Minimum fraction of residuals that must be valid
    """
    
    if verbose:
        print(f"Filtering {len(trial_data)} trials for ARMA-GARCH analysis...")
        print(f"Requirements: min_length={min_length}, min_valid_fraction={min_fraction_valid}")
    
    filtered_trials = []
    cleaning_report = []
    
    for idx, row in trial_data.iterrows():
        trial_name = row['trial_name']
        residuals = row['residuals']
        
        # Clean residuals
        cleaned_residuals, clean_stats = clean_residuals(residuals, min_length, verbose=False)
        
        # Check quality
        quality_ok = (clean_stats['valid_for_analysis'] and 
                     clean_stats['fraction_kept'] >= min_fraction_valid)
        
        if quality_ok:
            # Detect suspicious patterns
            diagnostics = detect_suspicious_residuals(cleaned_residuals)
            
            # Update row with cleaned residuals
            filtered_row = row.copy()
            filtered_row['residuals'] = cleaned_residuals
            filtered_row['cleaning_stats'] = clean_stats
            filtered_row['diagnostics'] = diagnostics
            
            filtered_trials.append(filtered_row)
            
            if verbose and diagnostics['suspicious']:
                print(f"  ⚠️  {trial_name}: {', '.join(diagnostics['warnings'])}")
        else:
            if verbose:
                reason = "too short" if not clean_stats['valid_for_analysis'] else "too many invalid values"
                print(f"  ❌ {trial_name}: {reason} ({clean_stats['cleaned_length']}/{clean_stats['original_length']})")
        
        # Track all trials for report
        cleaning_report.append({
            'trial_name': trial_name,
            'original_length': clean_stats['original_length'],
            'cleaned_length': clean_stats['cleaned_length'],
            'fraction_kept': clean_stats['fraction_kept'],
            'included': quality_ok,
            'n_nan': clean_stats['n_nan'],
            'n_inf': clean_stats['n_inf']
        })
    
    filtered_df = pd.DataFrame(filtered_trials)
    report_df = pd.DataFrame(cleaning_report)
    
    if verbose:
        print(f"\n✅ Kept {len(filtered_df)}/{len(trial_data)} trials for analysis")
        
        # Summary statistics
        excluded = report_df[~report_df['included']]
        if len(excluded) > 0:
            print(f"\nExcluded trials summary:")
            print(f"  Due to NaNs: {(excluded['n_nan'] > 0).sum()}")
            print(f"  Due to length: {(excluded['cleaned_length'] < min_length).sum()}")
            print(f"  Due to quality: {(excluded['fraction_kept'] < min_fraction_valid).sum()}")
    
    return filtered_df


if __name__ == "__main__":
    # Test with synthetic data containing NaNs
    print("Testing robust residual handling...")
    
    # Create test data with various issues
    np.random.seed(42)
    good_residuals = np.random.normal(0, 0.1, 1000)
    
    # Add problematic values
    problematic_residuals = good_residuals.copy()
    problematic_residuals[100:110] = np.nan  # NaN block
    problematic_residuals[200:205] = 0.0     # Zero block  
    problematic_residuals[500] = np.inf      # Inf value
    problematic_residuals[600] = 100.0       # Extreme outlier
    
    # Test cleaning
    print("\n1. Testing residual cleaning:")
    cleaned, stats = clean_residuals(problematic_residuals, verbose=True)
    
    print("\n2. Testing suspicious pattern detection:")
    diagnostics = detect_suspicious_residuals(cleaned)
    print(f"  Diagnostics: {diagnostics}")
    
    print("\n3. Testing trial filtering:")
    # Create mock trial data
    test_trials = pd.DataFrame([
        {'trial_name': 'good_trial', 'residuals': good_residuals},
        {'trial_name': 'problematic_trial', 'residuals': problematic_residuals},
        {'trial_name': 'short_trial', 'residuals': np.random.normal(0, 0.1, 20)},
        {'trial_name': 'mostly_nan_trial', 'residuals': np.full(100, np.nan)}
    ])
    
    filtered = robust_trial_filter(test_trials, verbose=True)
    print(f"\n✅ Filtering complete: {len(filtered)} trials passed quality checks")