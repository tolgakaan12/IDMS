#!/usr/bin/env python3
"""
Complete Multi-Subject ARMA-GARCH Workflow (Clean Implementation)
================================================================

Run the complete analysis for all subjects using the clean implementation:
1. Extract residuals from PyTorch models
2. Fit ARMA-GARCH-t models
3. Generate comprehensive analysis and visualizations
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import the individual workflow components
from extract_all_subjects_residuals import extract_all_subjects
from fit_all_subjects_arma_garch import fit_all_subjects


def run_complete_multi_subject_workflow(
    subjects: Optional[List[str]] = None, force_extract: bool = False
):
    """
    Run the complete multi-subject ARMA-GARCH workflow.

    Parameters:
    -----------
    subjects : list, optional
        List of subject IDs. Default: all subjects (1-5)
    force_extract : bool
        Force re-extraction even if residuals exist
    """

    if subjects is None:
        subjects = [f"subject_{i:03d}" for i in range(1, 6)]

    print("=" * 80)
    print("MULTI-SUBJECT ARMA-GARCH ANALYSIS (CLEAN IMPLEMENTATION)")
    print("=" * 80)
    print(f"Processing subjects: {', '.join(subjects)}")
    print(f"Model configuration: ARMA-GARCH-t with automatic order selection")
    print("=" * 80)

    # Step 1: Extract residuals for all subjects
    print("\n" + "=" * 60)
    print("STEP 1: RESIDUAL EXTRACTION")
    print("=" * 60)

    # Check if residuals already exist
    need_extraction = []

    for subject_id in subjects:
        residuals_path = Path(
            f"residual_analysis/subject_residuals/{subject_id}/trial_residuals.pkl"
        )
        if not residuals_path.exists() or force_extract:
            need_extraction.append(subject_id)
        else:
            print(f"✓ Residuals already exist for {subject_id}")

    if need_extraction:
        print(f"\nExtracting residuals for {len(need_extraction)} subjects...")
        extraction_results = extract_all_subjects(need_extraction)

        if not extraction_results:
            print("❌ Residual extraction failed. Cannot proceed.")
            return None

        print(
            f"✅ Successfully extracted residuals for {len(extraction_results)} subjects"
        )
    else:
        print("✓ All residuals already extracted")

    # Step 2: Fit ARMA-GARCH models for all subjects
    print("\n" + "=" * 60)
    print("STEP 2: ARMA-GARCH MODEL FITTING")
    print("=" * 60)

    print(
        f"Fitting ARMA-GARCH-t models with automatic order selection for all subjects..."
    )

    fitting_results = fit_all_subjects(subjects)

    if not fitting_results:
        print("❌ Model fitting failed. Cannot proceed.")
        return None

    print(f"✅ Successfully fitted models for {len(fitting_results)} subjects")

    # Step 3: Generate comprehensive analysis
    print("\n" + "=" * 60)
    print("STEP 3: COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    analysis_results = generate_multi_subject_analysis(subjects)

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Subjects processed: {len(subjects)}")

    # Load and display summary statistics
    summary_path = Path(
        "residual_analysis_clean/fitted_models/all_subjects_summary.json"
    )
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

        print(f"\nModel fit summary:")
        print(
            f"{'Subject':<15} {'Trials':<10} {'Mean AIC':<12} {'Mean BIC':<12} {'Mean ν':<10}"
        )
        print("-" * 65)

        for subject, data in summary.items():
            print(
                f"{subject:<15} {data['n_trials']:<10} {data['mean_aic']:<12.2f} "
                f"{data['mean_bic']:<12.2f} {data['mean_nu']:<10.2f}"
            )

    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"Residuals: residual_analysis/subject_residuals/")
    print(f"Fitted models: residual_analysis_clean/fitted_models/")
    print(f"Analysis plots: residual_analysis_clean/plots_multi_subject/")

    print(f"\n✨ Use plot_results_multi_subject.py for interactive analysis")

    return analysis_results


def generate_multi_subject_analysis(subjects: List[str]):
    """
    Generate comprehensive multi-subject analysis and comparisons.

    Parameters:
    -----------
    subjects : list
        List of subject IDs
    """

    print("Generating multi-subject analysis...")

    # Create output directory
    plots_dir = Path("residual_analysis_clean/plots_multi_subject")
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Load results for each subject
    for subject_id in subjects:
        results_path = Path(
            f"residual_analysis_clean/fitted_models/{subject_id}/{subject_id}_results.csv"
        )

        if results_path.exists():
            df = pd.read_csv(results_path)
            df["subject"] = subject_id
            all_results.append(df)
            print(f"  ✓ Loaded {len(df)} trials for {subject_id}")
        else:
            print(f"  ⚠️ No results found for {subject_id}: {results_path}")

    if not all_results:
        print("❌ No results found for any subject")
        return None

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save combined results
    combined_path = plots_dir / "all_subjects_combined_results.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"✅ Saved combined results to: {combined_path}")

    # Generate summary statistics
    summary_stats = {}

    for subject in subjects:
        subject_data = combined_df[combined_df["subject"] == subject]
        if len(subject_data) > 0:
            summary_stats[subject] = {
                "n_trials": len(subject_data),
                "mean_aic": float(subject_data["aic"].mean()),
                "std_aic": float(subject_data["aic"].std()),
                "mean_bic": float(subject_data["bic"].mean()),
                "std_bic": float(subject_data["bic"].std()),
                "mean_nu": float(subject_data["nu"].mean()),
                "std_nu": float(subject_data["nu"].std()),
                "mean_loglik": float(subject_data["log_likelihood"].mean()),
                "std_loglik": float(subject_data["log_likelihood"].std()),
            }

    # Save detailed summary
    summary_path = plots_dir / "multi_subject_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"✅ Saved analysis summary to: {summary_path}")

    # Generate comparison statistics
    print(f"\n📊 MULTI-SUBJECT COMPARISON:")
    print(f"Total trials across all subjects: {len(combined_df)}")
    print(f"Subjects with data: {len(summary_stats)}")

    # Overall statistics
    print(f"\nOverall statistics:")
    print(f"  AIC: {combined_df['aic'].mean():.2f} ± {combined_df['aic'].std():.2f}")
    print(f"  BIC: {combined_df['bic'].mean():.2f} ± {combined_df['bic'].std():.2f}")
    print(
        f"  ν (Student-t): {combined_df['nu'].mean():.2f} ± {combined_df['nu'].std():.2f}"
    )
    print(
        f"  Log-likelihood: {combined_df['log_likelihood'].mean():.2f} ± {combined_df['log_likelihood'].std():.2f}"
    )

    return {
        "combined_df": combined_df,
        "summary_stats": summary_stats,
        "plots_dir": plots_dir,
    }


def main():
    """Main entry point with command-line interface."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run complete multi-subject ARMA-GARCH workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for all subjects (1-5)
  python run_all_subjects_workflow.py --all

  # Run for specific subjects
  python run_all_subjects_workflow.py --subjects subject_001 subject_002 subject_005

  # Force re-extraction of residuals
  python run_all_subjects_workflow.py --all --force-extract
        """,
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., subject_001 subject_002)",
    )
    parser.add_argument("--all", action="store_true", help="Process all subjects (1-5)")
    parser.add_argument(
        "--force-extract", action="store_true", help="Force re-extraction of residuals"
    )

    args = parser.parse_args()

    # Determine subjects to process
    if args.all:
        subjects = [f"subject_{i:03d}" for i in range(1, 6)]
    elif args.subjects:
        subjects = args.subjects
    else:
        print("Please specify --all or --subjects")
        return

    # Run workflow
    run_complete_multi_subject_workflow(
        subjects=subjects, force_extract=args.force_extract
    )


if __name__ == "__main__":
    main()
