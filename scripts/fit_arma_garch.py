#!/usr/bin/env python3
"""
Fit ARMA-GARCH Models for All Subjects
======================================

Fit ARMA-GARCH-t models to residuals from all subjects (1-5).
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Use the local EnhancedARMAGARCH model for consistency
from arma_garch_residual_model import EnhancedARMAGARCH


def load_subject_residuals(subject_id: str, residuals_dir: str = None) -> tuple:
    """
    Load saved residuals for a specific subject.

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'subject_001')
    residuals_dir : str, optional
        Directory containing residuals. Default: residual_analysis/subject_residuals/{subject_id}

    Returns:
    --------
    residuals_dict, metadata_df
    """

    if residuals_dir is None:
        residuals_dir = f"residual_analysis/subject_residuals/{subject_id}"

    residuals_path = Path(residuals_dir)

    # Load residuals
    residuals_file = residuals_path / "trial_residuals.pkl"
    if not residuals_file.exists():
        raise FileNotFoundError(f"Residuals file not found: {residuals_file}")

    with open(residuals_file, "rb") as f:
        residuals_dict = pickle.load(f)

    # Load metadata
    metadata_file = residuals_path / "trial_residuals_metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata_df = pd.read_csv(metadata_file)

    print(f"✅ Loaded {len(residuals_dict)} trial residuals for {subject_id}")

    return residuals_dict, metadata_df


def fit_subject_arma_garch(subject_id: str, output_dir: str = None):
    """
    Fit ARMA-GARCH models to all trials of a subject.

    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'subject_001')
    output_dir : str, optional
        Output directory for fitted models
    """

    if output_dir is None:
        output_dir = f"residual_analysis_clean/fitted_models/{subject_id}"

    print(f"\n{'=' * 60}")
    print(f"FITTING ARMA-GARCH FOR {subject_id.upper()}")
    print(f"{'=' * 60}")

    # Load residuals
    try:
        residuals_dict, metadata_df = load_subject_residuals(subject_id)
    except FileNotFoundError as e:
        print(f"❌ Cannot fit models for {subject_id}: {e}")
        print(f"   Run extract_all_subjects_residuals.py first")
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Store results
    all_results = []
    failed_trials = []

    print(f"\nFitting ARMA-GARCH-t models with automatic order selection...")

    for trial_name in tqdm(residuals_dict.keys(), desc=f"Fitting {subject_id}"):
        try:
            # Get residuals
            residuals = residuals_dict[trial_name]

            # Fit model using EnhancedARMAGARCH (same as subject_005)
            model = EnhancedARMAGARCH(auto_select_orders=True)

            result = model.fit(residuals, distributions=["normal", "t"], verbose=False)

            if result is not None:
                # Extract metrics from EnhancedARMAGARCH result format (same as subject_005)
                best_dist = result.get("best_distribution", "normal")
                dist_comparison = result.get("distribution_comparison", {})

                if best_dist in dist_comparison:
                    metrics = dist_comparison[best_dist]
                    dist_params = result.get("distribution_params", {})

                    # Create result compatible with the plotting script expectations
                    flat_result = {
                        # Basic trial info
                        "trial_name": trial_name,
                        "subject": subject_id,
                        "trial": trial_name.split("/")[-1],
                        # Model results in expected format
                        "best_distribution": best_dist,
                        "bic": metrics.get("bic", np.nan),
                        "aic": metrics.get(
                            "bic", np.nan
                        ),  # Use BIC as AIC approximation
                        "log_likelihood": metrics.get("loglik", np.nan),
                        "nu": model.garch_models[best_dist].params.get("nu", np.nan)
                        if (best_dist == "t" and best_dist in model.garch_models)
                        else np.nan,
                        # Model parameters in expected format
                        "volatility_persistence": result.get(
                            "volatility_persistence", np.nan
                        ),
                        "volatility_halflife": result.get(
                            "volatility_halflife", np.nan
                        ),
                        "optimal_orders": result.get("optimal_orders", {}),
                        "garch_params": result.get("garch_params", {}),
                        "arma_params": result.get("arma_params", {}),
                        "distribution_params": result.get("distribution_params", {}),
                        # Add original residuals for plotting
                        "original_residuals": residuals,
                        # Add standardized residuals in the same format as subject_005
                        "standardized_residuals": model.standardized_residuals.get(
                            best_dist, np.array([])
                        ),
                    }

                    # Get split info from metadata
                    trial_meta = metadata_df[metadata_df["trial_name"] == trial_name]
                    if not trial_meta.empty:
                        flat_result["split"] = trial_meta.iloc[0]["split"]
                    else:
                        flat_result["split"] = "unknown"

                    all_results.append(flat_result)
                else:
                    failed_trials.append(trial_name)
            else:
                failed_trials.append(trial_name)

        except Exception as e:
            print(f"\n  Failed to fit {trial_name}: {e}")
            failed_trials.append(trial_name)

    # Convert to dictionary format like the working version
    results_dict = {}
    for result in all_results:
        trial_name = result["trial_name"]
        results_dict[trial_name] = result

    # Also create DataFrame for CSV export
    results_df = pd.DataFrame(all_results)

    # Save results
    if len(results_df) > 0:
        # Save as pickle in the format expected by the plotting script
        results_file = output_path / f"{subject_id}_fitted_models.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(results_dict, f)

        # Save as CSV (without full parameters which are dicts)
        csv_columns = [
            "trial_name",
            "subject",
            "trial",
            "split",
            "best_distribution",
            "bic",
            "log_likelihood",
            "volatility_persistence",
            "volatility_halflife",
            "nu",
        ]
        csv_df = results_df[csv_columns]
        csv_file = output_path / f"{subject_id}_results.csv"
        csv_df.to_csv(csv_file, index=False)

        # Save summary
        summary = {
            "subject": subject_id,
            "n_trials_total": len(residuals_dict),
            "n_trials_fitted": len(results_df),
            "n_trials_failed": len(failed_trials),
            "failed_trials": failed_trials,
            "auto_select_orders": True,
            "mean_bic": float(results_df["bic"].mean()),
            "mean_loglik": float(results_df["log_likelihood"].mean()),
            "mean_persistence": float(results_df["volatility_persistence"].mean()),
            "t_distribution_params": {
                "mean_nu": float(results_df["nu"].mean()),
                "std_nu": float(results_df["nu"].std()),
            },
        }

        summary_file = output_path / f"{subject_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")
        print(f"   Successful fits: {len(results_df)}/{len(residuals_dict)}")
        print(f"   Mean BIC: {summary['mean_bic']:.2f}")
        print(f"   Mean persistence: {summary['mean_persistence']:.3f}")
        print(f"   Mean ν (t-dist): {summary['t_distribution_params']['mean_nu']:.2f}")

        if failed_trials:
            print(f"   ⚠️  {len(failed_trials)} trials failed to fit")

    else:
        print(f"❌ No successful model fits for {subject_id}")

    return results_df


def fit_all_subjects(subjects: list = None):
    """
    Fit ARMA-GARCH models for multiple subjects.

    Parameters:
    -----------
    subjects : list, optional
        List of subject IDs. Default: all subjects (1-5)
    """

    if subjects is None:
        subjects = [f"subject_{i:03d}" for i in range(1, 6)]

    print(f"\n{'=' * 60}")
    print(f"FITTING ARMA-GARCH FOR {len(subjects)} SUBJECTS")
    print(f"{'=' * 60}")

    all_summaries = {}

    for subject_id in subjects:
        print(f"\nProcessing {subject_id}...")
        results_df = fit_subject_arma_garch(subject_id)

        if results_df is not None and len(results_df) > 0:
            all_summaries[subject_id] = {
                "n_trials": len(results_df),
                "mean_bic": float(results_df["bic"].mean()),
                "mean_loglik": float(results_df["log_likelihood"].mean()),
                "mean_persistence": float(results_df["volatility_persistence"].mean()),
                "mean_nu": float(results_df["nu"].mean()),
            }

    # Save combined summary
    if all_summaries:
        summary_path = Path(
            "residual_analysis_clean/fitted_models/all_subjects_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"ALL SUBJECTS COMPLETE")
        print(f"{'=' * 60}")

        # Print comparison table
        print(f"\nComparison across subjects:")
        print(
            f"{'Subject':<15} {'Trials':<10} {'Mean BIC':<12} {'Mean Persist':<12} {'Mean ν':<10}"
        )
        print("-" * 65)

        for subject, data in all_summaries.items():
            print(
                f"{subject:<15} {data['n_trials']:<10} {data['mean_bic']:<12.2f} {data['mean_persistence']:<12.3f} {data['mean_nu']:<10.2f}"
            )

    return all_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit ARMA-GARCH models for subjects")
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., subject_001 subject_002)",
    )
    parser.add_argument("--all", action="store_true", help="Process all subjects (1-5)")

    args = parser.parse_args()

    if args.all:
        fit_all_subjects()
    elif args.subjects:
        for subject in args.subjects:
            fit_subject_arma_garch(subject)
    else:
        # Default: fit for subjects 1-4
        print("Fitting ARMA-GARCH for subjects 1-4...")
        subjects_to_process = [f"subject_{i:03d}" for i in range(1, 5)]
        fit_all_subjects(subjects_to_process)
