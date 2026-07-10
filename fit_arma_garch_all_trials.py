#!/usr/bin/env python3
"""
Fit ARMA-GARCH-t to All 61 Trials and Save Parameters
====================================================

This script:
1. Loads all 61 trial residuals
2. Fits ARMA-GARCH-t model to each trial individually
3. Saves model parameters for each trial
4. Creates summary statistics of fitted parameters

The saved parameters can then be used for visualization and simulation.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from save_individual_trial_residuals import load_trial_residuals, convert_to_dataframe_format
from residual_analysis.arma_garch_residual_model import EnhancedARMAGARCH

def fit_arma_garch_all_trials(output_dir: str = "residual_analysis/arma_garch_fits"):
    """
    Fit ARMA-GARCH-t models to all 61 trials and save parameters.
    
    Returns:
    --------
    fitted_models_dict: Dictionary with fitted model results for each trial
    """
    
    print("="*70)
    print("FITTING ARMA-GARCH-t MODELS TO ALL TRIALS")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load trial residuals
    print("\n1. Loading trial residuals...")
    try:
        residuals_dict, metadata_df = load_trial_residuals()
        trial_data = convert_to_dataframe_format(residuals_dict, metadata_df)
        print(f"✅ Loaded {len(trial_data)} trials")
    except Exception as e:
        print(f"❌ Error loading residuals: {e}")
        return None
    
    # Initialize results storage
    fitted_models = {}
    fit_summaries = []
    
    print(f"\n2. Fitting ARMA-GARCH-t models...")
    print(f"   Testing distributions: Normal, Student-t")
    print(f"   Using automatic order selection (BIC)")
    
    # Fit model to each trial
    for idx, trial_row in tqdm(trial_data.iterrows(), total=len(trial_data), 
                              desc="Fitting models"):
        
        trial_name = trial_row['trial_name']
        residuals = trial_row['residuals']
        
        # Skip trials with insufficient data
        if len(residuals) < 100:
            print(f"\n⚠️  Skipping {trial_name}: insufficient data ({len(residuals)} samples)")
            continue
        
        try:
            # Initialize enhanced ARMA-GARCH model with auto order selection
            model = EnhancedARMAGARCH(auto_select_orders=True)
            
            # Fit model with Normal and Student-t distributions
            fit_results = model.fit(
                residuals=residuals,
                distributions=['normal', 't'],
                verbose=False  # Suppress individual fitting output
            )
            
            # Extract key information for this trial
            best_dist = fit_results['best_distribution']
            optimal_orders = fit_results['optimal_orders']
            
            # Get distribution parameters
            if best_dist == 't':
                garch_model = model.garch_models[best_dist]
                nu = garch_model.params.get('nu', np.nan)
                distribution_params = {'nu': nu}
            else:
                distribution_params = {}
            
            # Get ARMA and GARCH parameters
            arma_params = fit_results['arma_params']
            garch_params = fit_results['garch_params']
            
            # Test innovation quality
            whiteness_results = model.test_innovation_whiteness(verbose=False)
            gof_results = model.test_distribution_goodness_of_fit(verbose=False)
            
            # Store comprehensive results for this trial
            trial_result = {
                'trial_name': trial_name,
                'subject': trial_row.get('subject', 'subject_005'),
                'trial': trial_row.get('trial', trial_name.split('/')[-1]),
                'split': trial_row.get('split', 'unknown'),
                'n_samples': len(residuals),
                
                # Model specification
                'best_distribution': best_dist,
                'optimal_orders': optimal_orders,
                
                # Model parameters
                'arma_params': arma_params,
                'garch_params': garch_params,
                'distribution_params': distribution_params,
                
                # Model quality
                'volatility_persistence': fit_results.get('volatility_persistence', np.nan),
                'volatility_halflife': fit_results.get('volatility_halflife', np.nan),
                
                # Innovation quality
                'whiteness_score': whiteness_results.get('whiteness_score', 0),
                'is_white_noise': whiteness_results.get('is_white_noise', False),
                'fit_score': gof_results.get('fit_score', 0),
                'good_fit': gof_results.get('good_fit', False),
                
                # Model comparison (BIC scores)
                'distribution_comparison': fit_results.get('distribution_comparison', {}),
                
                # Store the actual fitted model object for simulation
                'fitted_model': model,
                'standardized_residuals': model.standardized_residuals.get(best_dist, np.array([])),
                
                # Original residuals for comparison
                'original_residuals': residuals
            }
            
            fitted_models[trial_name] = trial_result
            
            # Create summary row for CSV export
            summary_row = {
                'trial_name': trial_name,
                'subject': trial_result['subject'],
                'trial': trial_result['trial'],
                'split': trial_result['split'],
                'n_samples': trial_result['n_samples'],
                'best_distribution': best_dist,
                'arma_p': optimal_orders['p'],
                'arma_q': optimal_orders['q'],
                'garch_r': optimal_orders['r'],
                'garch_s': optimal_orders['s'],
                'volatility_persistence': trial_result['volatility_persistence'],
                'volatility_halflife': trial_result['volatility_halflife'],
                'whiteness_score': trial_result['whiteness_score'],
                'is_white_noise': trial_result['is_white_noise'],
                'fit_score': trial_result['fit_score'],
                'good_fit': trial_result['good_fit']
            }
            
            # Add distribution-specific parameters
            if best_dist == 't':
                summary_row['student_t_nu'] = distribution_params.get('nu', np.nan)
            else:
                summary_row['student_t_nu'] = np.nan
                
            # Add key ARMA parameters
            summary_row['arma_phi1'] = arma_params.get('phi_1', np.nan)
            summary_row['arma_theta1'] = arma_params.get('theta_1', np.nan)
            
            # Add key GARCH parameters
            summary_row['garch_omega'] = garch_params.get('omega', np.nan)
            summary_row['garch_alpha'] = garch_params.get('alpha', np.nan)
            summary_row['garch_beta'] = garch_params.get('beta', np.nan)
            
            # Add BIC scores
            dist_comparison = fit_results.get('distribution_comparison', {})
            summary_row['normal_bic'] = dist_comparison.get('normal', {}).get('bic', np.nan)
            summary_row['student_t_bic'] = dist_comparison.get('t', {}).get('bic', np.nan)
            
            fit_summaries.append(summary_row)
            
        except Exception as e:
            print(f"\n❌ Error fitting {trial_name}: {str(e)}")
            continue
    
    print(f"\n3. Saving results...")
    
    # Save fitted models (without the actual model objects to reduce file size)
    models_for_saving = {}
    for trial_name, result in fitted_models.items():
        # Create a copy without the fitted_model object (too large for pickle)
        save_result = {k: v for k, v in result.items() if k != 'fitted_model'}
        models_for_saving[trial_name] = save_result
    
    # Save fitted model parameters
    models_file = output_path / "fitted_arma_garch_models.pkl"
    with open(models_file, 'wb') as f:
        pickle.dump(models_for_saving, f)
    print(f"✅ Saved model parameters: {models_file}")
    
    # Save summary as CSV for easy viewing
    if fit_summaries:
        summary_df = pd.DataFrame(fit_summaries)
        summary_file = output_path / "arma_garch_fit_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Saved fit summary: {summary_file}")
        
        # Print overall summary
        print_fit_summary(summary_df)
    
    # Save model parameters in JSON format for easy reading
    def convert_to_json_serializable(obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        else:
            return obj
    
    params_json = {}
    for trial_name, result in models_for_saving.items():
        params_json[trial_name] = {
            'best_distribution': result['best_distribution'],
            'optimal_orders': convert_to_json_serializable(result['optimal_orders']),
            'arma_params': convert_to_json_serializable(result['arma_params']),
            'garch_params': convert_to_json_serializable(result['garch_params']),
            'distribution_params': convert_to_json_serializable(result['distribution_params']),
            'model_quality': {
                'volatility_persistence': convert_to_json_serializable(result['volatility_persistence']),
                'whiteness_score': convert_to_json_serializable(result['whiteness_score']),
                'is_white_noise': convert_to_json_serializable(result['is_white_noise']),
                'good_fit': convert_to_json_serializable(result['good_fit'])
            }
        }
    
    params_file = output_path / "arma_garch_parameters.json"
    with open(params_file, 'w') as f:
        json.dump(params_json, f, indent=2)
    print(f"✅ Saved parameters JSON: {params_file}")
    
    print(f"\n✅ FITTING COMPLETE!")
    print(f"   Successfully fitted: {len(fitted_models)} trials")
    print(f"   Results saved to: {output_path}")
    
    return fitted_models


def print_fit_summary(summary_df: pd.DataFrame):
    """Print a summary of the fitting results."""
    
    print(f"\n" + "="*50)
    print("FIT SUMMARY STATISTICS")
    print("="*50)
    
    total_trials = len(summary_df)
    
    # Distribution preferences
    print(f"\nDistribution Preferences ({total_trials} trials):")
    dist_counts = summary_df['best_distribution'].value_counts()
    for dist, count in dist_counts.items():
        pct = (count / total_trials) * 100
        print(f"  {dist:<12}: {count:>3} trials ({pct:>5.1f}%)")
    
    # Student-t analysis
    t_trials = summary_df[summary_df['best_distribution'] == 't']
    if len(t_trials) > 0:
        nu_values = t_trials['student_t_nu'].dropna()
        if len(nu_values) > 0:
            print(f"\nStudent-t Analysis ({len(t_trials)} trials):")
            print(f"  Mean ν: {nu_values.mean():.2f} ± {nu_values.std():.2f}")
            print(f"  Range ν: [{nu_values.min():.1f}, {nu_values.max():.1f}]")
            print(f"  Heavy tails (ν < 5): {(nu_values < 5).sum()}/{len(nu_values)} trials")
    
    # Model orders
    print(f"\nModel Orders:")
    print(f"  ARMA(p,q): p={summary_df['arma_p'].mean():.1f}±{summary_df['arma_p'].std():.1f}, q={summary_df['arma_q'].mean():.1f}±{summary_df['arma_q'].std():.1f}")
    print(f"  GARCH(r,s): r={summary_df['garch_r'].mean():.1f}±{summary_df['garch_r'].std():.1f}, s={summary_df['garch_s'].mean():.1f}±{summary_df['garch_s'].std():.1f}")
    
    # Model quality
    print(f"\nModel Quality:")
    print(f"  White noise achieved: {summary_df['is_white_noise'].sum()}/{total_trials} trials ({(summary_df['is_white_noise'].sum()/total_trials)*100:.1f}%)")
    print(f"  Good distribution fit: {summary_df['good_fit'].sum()}/{total_trials} trials ({(summary_df['good_fit'].sum()/total_trials)*100:.1f}%)")
    print(f"  Mean whiteness score: {summary_df['whiteness_score'].mean():.2f}/3.0")
    
    # Volatility persistence
    persistence_vals = summary_df['volatility_persistence'].dropna()
    if len(persistence_vals) > 0:
        print(f"  Mean volatility persistence: {persistence_vals.mean():.3f} ± {persistence_vals.std():.3f}")
        unit_root_trials = (persistence_vals >= 0.99).sum()
        print(f"  Near unit root (≥0.99): {unit_root_trials}/{len(persistence_vals)} trials")


def load_fitted_models(models_dir: str = "residual_analysis/arma_garch_fits"):
    """
    Load previously fitted ARMA-GARCH models.
    
    Returns:
    --------
    fitted_models: Dictionary with fitted model results
    summary_df: DataFrame with summary statistics
    """
    
    models_path = Path(models_dir)
    
    # Load model parameters
    models_file = models_path / "fitted_arma_garch_models.pkl"
    with open(models_file, 'rb') as f:
        fitted_models = pickle.load(f)
    
    # Load summary
    summary_file = models_path / "arma_garch_fit_summary.csv"
    summary_df = pd.read_csv(summary_file)
    
    print(f"✅ Loaded {len(fitted_models)} fitted models from: {models_path}")
    
    return fitted_models, summary_df


if __name__ == "__main__":
    # Fit models to all trials
    fitted_models = fit_arma_garch_all_trials()
    
    if fitted_models:
        print(f"\n🎯 RESULTS:")
        print(f"Successfully fitted ARMA-GARCH-t models to {len(fitted_models)} trials")
        print(f"\nFiles created:")
        print(f"  - fitted_arma_garch_models.pkl    # Model parameters")
        print(f"  - arma_garch_fit_summary.csv      # Summary table") 
        print(f"  - arma_garch_parameters.json      # Human-readable parameters")
        print(f"\nNext step: Use plot_arma_garch_fits.py to visualize results!")
    else:
        print("❌ Fitting failed - check error messages above")