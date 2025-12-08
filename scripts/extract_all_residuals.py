#!/usr/bin/env python3
"""
Extract Residuals for All Subjects (1-5)
========================================

Generalized version to extract residuals for any subject using their respective models.
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
import h5py
from typing import Dict, List, Tuple

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from extract_pytorch_residuals import PyTorchResidualExtractor

# Model paths for each subject
SUBJECT_MODEL_PATHS = {
    'subject_001': 'pytorch_models/experiments/multistage_3108/stage2_finetune_subject_001_nothing_frozen',
    'subject_002': 'pytorch_models/experiments/multistage_3108/stage2_finetune_subject_002_tcn_frozen',
    'subject_003': 'pytorch_models/experiments/multistage_3108/stage2_finetune_subject_003_tcn_frozen',
    'subject_004': 'pytorch_models/experiments/multistage_3108/stage2_finetune_subject_004_nothing_frozen',
    'subject_005': 'pytorch_models/experiments/multistage_3108/stage2_finetune_subject_005_tcn_frozen'
}


def extract_subject_residuals(subject_id: str, model_dir: str = None) -> pd.DataFrame:
    """
    Extract residuals for all trials of a specific subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'subject_001')
    model_dir : str, optional
        Path to model directory. If None, uses default from SUBJECT_MODEL_PATHS
    
    Returns:
    --------
    pd.DataFrame with columns: trial_name, subject, trial, split, residuals, n_windows
    """
    
    if model_dir is None:
        if subject_id not in SUBJECT_MODEL_PATHS:
            raise ValueError(f"No default model path for {subject_id}")
        model_dir = SUBJECT_MODEL_PATHS[subject_id]
    
    print(f"\nExtracting residuals for {subject_id}...")
    print(f"Using model: {model_dir}")
    
    # Initialize extractor
    extractor = PyTorchResidualExtractor(model_dir)
    
    # Get all trials for this subject from the dataset
    dataset_path = "data/idms_ready_dataset.h5"
    
    with h5py.File(dataset_path, 'r') as f:
        # Get all trials for this subject
        subject_trials = [f'{subject_id}/{trial}' for trial in f[f'subjects/{subject_id}'].keys()]
        print(f"Found {len(subject_trials)} trials for {subject_id}")
    
    # Extract residuals for each trial
    trial_data = []
    
    for trial_name in subject_trials:
        try:
            # Extract residuals for this trial
            residuals = extractor.extract_residuals_for_trial(trial_name)
            
            # Determine split (train/val/test) from model's data_splits.json
            split = extractor.get_trial_split(trial_name)
            
            trial_entry = {
                'trial_name': trial_name,
                'subject': subject_id,
                'trial': trial_name.split('/')[-1],
                'split': split,
                'residuals': residuals,
                'n_windows': len(residuals)
            }
            trial_data.append(trial_entry)
            
        except Exception as e:
            print(f"  Warning: Could not extract residuals for {trial_name}: {e}")
            continue
    
    df = pd.DataFrame(trial_data)
    print(f"Successfully extracted residuals for {len(df)} trials")
    
    return df


def save_subject_residuals(subject_id: str, output_dir: str = None):
    """
    Extract and save residuals for a specific subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'subject_001')
    output_dir : str, optional
        Output directory. Default: residual_analysis/subject_residuals/{subject_id}
    """
    
    if output_dir is None:
        output_dir = f"residual_analysis/subject_residuals/{subject_id}"
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING RESIDUALS FOR {subject_id.upper()}")
    print(f"{'='*60}")
    
    # Extract residuals
    trial_data = extract_subject_residuals(subject_id)
    
    if len(trial_data) == 0:
        print(f"❌ No residuals extracted for {subject_id}")
        return None, None
    
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
            'subject': row['subject'],
            'trial': row['trial'],
            'split': row['split'],
            'n_windows': len(residuals),
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals),
            'residual_skew': float(pd.Series(residuals).skew()),
            'residual_kurt': float(pd.Series(residuals).kurtosis())
        }
        metadata_list.append(metadata)
    
    # Save residuals as pickle
    residuals_file = output_path / "trial_residuals.pkl"
    with open(residuals_file, 'wb') as f:
        pickle.dump(residuals_dict, f)
    
    print(f"\n✅ Saved residuals to: {residuals_file}")
    print(f"   Total trials: {len(residuals_dict)}")
    print(f"   File size: {residuals_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata as CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_file = output_path / "trial_residuals_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"✅ Saved metadata to: {metadata_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY FOR {subject_id}:")
    print(f"Trials by split:")
    split_counts = metadata_df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} trials")
    
    print(f"\nResidual statistics:")
    print(f"  Mean length: {metadata_df['n_windows'].mean():.0f} ± {metadata_df['n_windows'].std():.0f}")
    print(f"  Length range: [{metadata_df['n_windows'].min()}, {metadata_df['n_windows'].max()}]")
    print(f"  Mean residual std: {metadata_df['residual_std'].mean():.4f}")
    
    return residuals_dict, metadata_df


def extract_all_subjects(subjects: List[str] = None):
    """
    Extract residuals for multiple subjects.
    
    Parameters:
    -----------
    subjects : List[str], optional
        List of subject IDs. Default: all subjects (1-5)
    """
    
    if subjects is None:
        subjects = [f'subject_{i:03d}' for i in range(1, 6)]
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING RESIDUALS FOR {len(subjects)} SUBJECTS")
    print(f"{'='*60}")
    
    results = {}
    
    for subject_id in subjects:
        try:
            residuals_dict, metadata_df = save_subject_residuals(subject_id)
            if residuals_dict is not None:
                results[subject_id] = {
                    'n_trials': len(residuals_dict),
                    'metadata': metadata_df
                }
        except Exception as e:
            print(f"\n❌ Failed to extract residuals for {subject_id}: {e}")
            continue
    
    # Save combined summary
    if results:
        summary_path = Path("residual_analysis/subject_residuals/extraction_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            subject: {
                'n_trials': data['n_trials'],
                'mean_std': float(data['metadata']['residual_std'].mean()),
                'splits': data['metadata']['split'].value_counts().to_dict()
            }
            for subject, data in results.items()
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved extraction summary to: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully extracted residuals for {len(results)}/{len(subjects)} subjects")
        for subject, data in results.items():
            print(f"  {subject}: {data['n_trials']} trials")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract residuals for subjects')
    parser.add_argument('--subjects', nargs='+', 
                       help='Subject IDs to process (e.g., subject_001 subject_002)')
    parser.add_argument('--all', action='store_true',
                       help='Process all subjects (1-5)')
    
    args = parser.parse_args()
    
    if args.all:
        extract_all_subjects()
    elif args.subjects:
        extract_all_subjects(args.subjects)
    else:
        # Default: extract for subjects 1-4 (since 5 is already done)
        print("Extracting residuals for subjects 1-4...")
        subjects_to_process = [f'subject_{i:03d}' for i in range(1, 5)]
        extract_all_subjects(subjects_to_process)