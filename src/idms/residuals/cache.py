#!/usr/bin/env python3
"""
Save Individual Trial Residuals
===============================

Extract and save residuals for each trial individually, so we don't need
to re-extract them every time we run ARMA-GARCH analysis.

This creates a more efficient data structure for trial-level analysis.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from idms.residuals.trial_stats import extract_all_subject005_residuals

def save_trial_residuals(output_dir: str = "residual_analysis_clean/trial_residuals"):
    """
    Extract and save individual trial residuals.
    
    Creates:
    - trial_residuals.pkl: Dictionary with trial residuals
    - trial_residuals_metadata.csv: Trial information
    """
    
    print("Extracting individual trial residuals...")
    
    # Extract all trial residuals
    trial_data = extract_all_subject005_residuals()
    print(f"Extracted {len(trial_data)} trials")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data structures
    residuals_dict = {}
    metadata_list = []
    
    for idx, row in trial_data.iterrows():
        trial_name = row['trial_name']
        residuals = row['residuals']
        
        # Store residuals
        residuals_dict[trial_name] = residuals
        
        # Store metadata
        metadata = {
            'trial_name': trial_name,
            'subject': row.get('subject', 'subject_005'),
            'trial': row.get('trial', trial_name.split('/')[-1]),
            'split': row.get('split', 'unknown'),
            'n_windows': len(residuals),
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals),
            'residual_skew': float(pd.Series(residuals).skew()),
            'residual_kurt': float(pd.Series(residuals).kurtosis())
        }
        metadata_list.append(metadata)
    
    # Save residuals as pickle (efficient for numpy arrays)
    residuals_file = output_path / "trial_residuals.pkl"
    with open(residuals_file, 'wb') as f:
        pickle.dump(residuals_dict, f)
    
    print(f"✅ Saved residuals to: {residuals_file}")
    print(f"   Total trials: {len(residuals_dict)}")
    print(f"   File size: {residuals_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata as CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_file = output_path / "trial_residuals_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"✅ Saved metadata to: {metadata_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"Trials by split:")
    split_counts = metadata_df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} trials")
    
    print(f"\nResidual statistics:")
    print(f"  Mean length: {metadata_df['n_windows'].mean():.0f} ± {metadata_df['n_windows'].std():.0f}")
    print(f"  Length range: [{metadata_df['n_windows'].min()}, {metadata_df['n_windows'].max()}]")
    print(f"  Mean residual std: {metadata_df['residual_std'].mean():.4f}")
    
    return residuals_dict, metadata_df


def load_trial_residuals(residuals_dir: str = "residual_analysis_clean/trial_residuals") -> tuple:
    """
    Load saved trial residuals.
    
    Returns:
    --------
    residuals_dict, metadata_df
    """
    
    residuals_path = Path(residuals_dir)
    
    # Load residuals
    residuals_file = residuals_path / "trial_residuals.pkl"
    if not residuals_file.exists():
        raise FileNotFoundError(f"Residuals file not found: {residuals_file}")
    
    with open(residuals_file, 'rb') as f:
        residuals_dict = pickle.load(f)
    
    # Load metadata
    metadata_file = residuals_path / "trial_residuals_metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata_df = pd.read_csv(metadata_file)
    
    print(f"✅ Loaded {len(residuals_dict)} trial residuals from: {residuals_path}")
    
    return residuals_dict, metadata_df


def convert_to_dataframe_format(residuals_dict: dict, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert loaded residuals back to the format expected by trial_innovation_analyzer.
    """
    
    trial_data = []
    
    for _, row in metadata_df.iterrows():
        trial_name = row['trial_name']
        if trial_name in residuals_dict:
            trial_entry = {
                'trial_name': trial_name,
                'subject': row['subject'],
                'trial': row['trial'], 
                'split': row['split'],
                'residuals': residuals_dict[trial_name],
                'n_windows': len(residuals_dict[trial_name])
            }
            trial_data.append(trial_entry)
    
    return pd.DataFrame(trial_data)


def test_saved_residuals():
    """Test that saved residuals work correctly."""
    
    print("Testing saved residuals...")
    
    # Load residuals
    residuals_dict, metadata_df = load_trial_residuals()
    
    # Convert to DataFrame format
    trial_data = convert_to_dataframe_format(residuals_dict, metadata_df)
    
    print(f"✅ Successfully loaded {len(trial_data)} trials")
    
    # Test a few trials
    for i in range(min(3, len(trial_data))):
        trial = trial_data.iloc[i]
        residuals = trial['residuals']
        print(f"  {trial['trial_name']}: {len(residuals)} residuals, std={np.std(residuals):.4f}")
    
    return trial_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Save individual trial residuals')
    parser.add_argument('--save', action='store_true', help='Save residuals (default: test loading)')
    parser.add_argument('--output-dir', default='residual_analysis_clean/trial_residuals', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.save:
        print("SAVING INDIVIDUAL TRIAL RESIDUALS")
        print("="*40)
        residuals_dict, metadata_df = save_trial_residuals(args.output_dir)
        
        print("\nTesting saved residuals...")
        test_saved_residuals()
        
    else:
        print("TESTING SAVED RESIDUALS")
        print("="*30)
        try:
            trial_data = test_saved_residuals()
            print("✅ Saved residuals are working correctly!")
        except FileNotFoundError:
            print("❌ No saved residuals found. Run with --save first:")
            print("   python save_individual_trial_residuals.py --save")