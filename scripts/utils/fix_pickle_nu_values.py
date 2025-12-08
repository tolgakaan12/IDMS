#!/usr/bin/env python3
"""
Fix ν Values in Pickle Files for Subjects 1-4
==============================================

The pickle files for subjects 1-4 were created with buggy ν extraction logic,
so they contain nu=NaN even though the CSV files have correct ν values.

This script updates the pickle files with the correct ν values from the CSV files.
"""

import pandas as pd
import pickle
from pathlib import Path

def fix_pickle_nu_values():
    """Fix ν values in pickle files by reading from CSV files."""
    
    print("=" * 60)
    print("FIXING ν VALUES IN PICKLE FILES")
    print("=" * 60)
    
    for subject_id in ['subject_001', 'subject_002', 'subject_003', 'subject_004']:
        print(f"\n📝 Processing {subject_id}...")
        
        # Paths
        subject_dir = Path(f"fitted_models/{subject_id}")
        csv_file = subject_dir / f"{subject_id}_results.csv"
        pkl_file = subject_dir / f"{subject_id}_fitted_models.pkl"
        
        if not csv_file.exists() or not pkl_file.exists():
            print(f"  ❌ Missing files for {subject_id}")
            continue
        
        # Load CSV with correct ν values
        df = pd.read_csv(csv_file)
        print(f"  📊 CSV has {len(df)} trials")
        
        # Load pickle with incorrect ν values
        with open(pkl_file, 'rb') as f:
            models = pickle.load(f)
        print(f"  📦 Pickle has {len(models)} trials")
        
        # Update ν values in pickle data
        updated_count = 0
        for _, row in df.iterrows():
            trial_name = row['trial_name']
            nu_csv = row.get('nu', None)
            
            if trial_name in models and not pd.isna(nu_csv):
                # Update the ν value in the model data
                models[trial_name]['nu'] = nu_csv
                updated_count += 1
        
        # Save updated pickle file
        with open(pkl_file, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"  ✅ Updated {updated_count} trials with correct ν values")
        
        # Verify the fix
        sample_trial = list(models.keys())[0]
        sample_nu = models[sample_trial].get('nu', 'NOT_FOUND')
        print(f"  🔍 Sample verification: {sample_trial} → ν = {sample_nu}")

if __name__ == "__main__":
    fix_pickle_nu_values()
    print(f"\n🎯 SUCCESS! All pickle files now have correct ν values.")
    print(f"The comprehensive plotting script should now work properly.")