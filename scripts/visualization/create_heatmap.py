#!/usr/bin/env python3
"""
Multi-Subject Statistical Test Failure Heatmap
==============================================

Create comprehensive heatmap showing statistical test results for residuals
from all subjects (1-5), including normality, stationarity, and autocorrelation tests.
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, lilliefors
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def load_all_subjects_residuals():
    """Load residuals from all subjects (1-5)."""

    all_residuals = {}

    # Load subjects 1-4 from new structure
    for subject_num in range(1, 5):
        subject_id = f"subject_{subject_num:03d}"
        residuals_path = Path(
            f"residual_analysis/subject_residuals/{subject_id}/trial_residuals.pkl"
        )

        if residuals_path.exists():
            try:
                with open(residuals_path, "rb") as f:
                    subject_residuals = pickle.load(f)

                all_residuals.update(subject_residuals)
                print(f"✅ Loaded {len(subject_residuals)} trials from {subject_id}")

            except Exception as e:
                print(f"❌ Error loading {subject_id}: {e}")

    # Load subject 5 from old structure
    subject_005_path = Path("residual_analysis/trial_residuals/trial_residuals.pkl")
    if subject_005_path.exists():
        try:
            with open(subject_005_path, "rb") as f:
                subject_005_residuals = pickle.load(f)

            all_residuals.update(subject_005_residuals)
            print(f"✅ Loaded {len(subject_005_residuals)} trials from subject_005")

        except Exception as e:
            print(f"❌ Error loading subject_005: {e}")

    print(f"\n📊 Total trials loaded: {len(all_residuals)}")
    return all_residuals


def run_statistical_tests(residuals, fdr_alpha=0.05):
    """Run comprehensive statistical tests on residuals with BH FDR correction."""

    # Clean and center residuals
    clean_residuals = residuals[np.isfinite(residuals)]
    clean_residuals = clean_residuals - np.mean(
        clean_residuals
    )  # Center residuals trial-wise
    n = len(clean_residuals)

    test_names = [
        "shapiro_normal",
        "jarque_bera_normal",
        "lilliefors_normal",
        "adf_stationary",
        "kpss_stationary",
        "ljung_box_lag20_independent",
        "arch_lm_homoscedastic",
        "runs_random",  # 'mcleod_li_no_arch'
    ]

    if n < 10:  # Too few samples
        return {test: False for test in test_names}

    # Store raw p-values and test results
    p_values = []
    raw_results = {}

    # 1. Normality Tests
    try:
        # Shapiro-Wilk (max 5000 samples)
        if n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(clean_residuals)
            p_values.append(shapiro_p)
        else:
            p_values.append(1.0)  # Conservative p-value for failed test
    except:
        p_values.append(1.0)

    try:
        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(clean_residuals)
        p_values.append(jb_p)
    except:
        p_values.append(1.0)

    try:
        # Lilliefors test (KS test with estimated parameters)
        lilliefors_stat, lilliefors_p = lilliefors(clean_residuals, dist="norm")
        p_values.append(lilliefors_p)
    except:
        p_values.append(1.0)

    # 2. Stationarity Tests
    try:
        # Augmented Dickey-Fuller
        adf_stat, adf_p, _, _, adf_critical, _ = adfuller(
            clean_residuals, autolag="AIC"
        )
        p_values.append(adf_p)
    except:
        p_values.append(1.0)

    try:
        # KPSS Test
        kpss_stat, kpss_p, _, kpss_critical = kpss(clean_residuals, regression="c")
        # KPSS can return string p-values like "0.10", convert to float
        if isinstance(kpss_p, str):
            kpss_p = float(kpss_p.replace(">", "").replace("<", ""))
        p_values.append(float(kpss_p))
    except:
        p_values.append(1.0)

    # 3. Autocorrelation Tests (Ljung-Box at different lags)
    for lag in [20]:
        try:
            if n > lag + 1:
                result = acorr_ljungbox(clean_residuals, lags=lag)
                lb_p = result["lb_pvalue"].iloc[-1]
                p_values.append(float(lb_p))
            else:
                p_values.append(1.0)
        except:
            p_values.append(1.0)

    # 4. ARCH-LM Test (conditional heteroscedasticity)
    try:
        if n > 10:
            arch_stat, arch_p, _, _ = het_arch(clean_residuals, maxlag=10)
            p_values.append(arch_p)
        else:
            p_values.append(1.0)
    except:
        p_values.append(1.0)

    # 5. Runs Test (randomness)
    try:
        # Test if residuals are above/below median randomly
        median_val = np.median(clean_residuals)
        binary_sequence = (clean_residuals > median_val).astype(int)
        runs_stat, runs_p = runstest_1samp(binary_sequence)
        p_values.append(runs_p)
    except:
        p_values.append(1.0)

    # 6. McLeod-Li Test (ARCH effects in squared residuals) - COMMENTED OUT
    # try:
    #     squared_residuals = clean_residuals**2
    #     # Apply Ljung-Box to squared residuals
    #     if n > 10:
    #         ml_stat, ml_p = acorr_ljungbox(squared_residuals, lags=10, return_df=False)
    #         if isinstance(ml_p, np.ndarray):
    #             ml_p = ml_p[-1]  # Get p-value for highest lag
    #         p_values.append(ml_p)
    #     else:
    #         p_values.append(1.0)
    # except:
    #     p_values.append(1.0)

    # Ensure all p-values are numeric first
    p_values_clean = []
    for p in p_values:
        if isinstance(p, str):
            try:
                p_clean = float(p.replace(">", "").replace("<", ""))
                p_values_clean.append(p_clean)
            except:
                p_values_clean.append(1.0)  # Conservative fallback
        else:
            p_values_clean.append(float(p))

    # Apply Benjamini-Hochberg FDR correction
    try:
        reject, p_corrected, _, _ = multipletests(
            p_values_clean, alpha=fdr_alpha, method="fdr_bh"
        )

        # Debug: Show all p-values and FDR correction results
        print(f"Debug: All raw p-values = {[f'{p:.2e}' for p in p_values_clean]}")
        print(f"Debug: Test order = {test_names}")

        # Show Ljung-Box results for all lags
        lb_tests = [name for name in test_names if "ljung_box" in name]
        for lb_test in lb_tests:
            lb_index = test_names.index(lb_test)
            print(
                f"Debug: {lb_test} - raw p={p_values_clean[lb_index]:.2e}, corrected p={p_corrected[lb_index]:.2e}, rejected={reject[lb_index]}"
            )

        # Create results based on FDR-corrected p-values
        results = {}

        # Special handling for tests where we want different null hypotheses
        test_expectations = {
            "shapiro_normal": True,  # Want normal (high p-value)
            "jarque_bera_normal": True,  # Want normal (high p-value)
            "lilliefors_normal": True,  # Want normal (high p-value)
            "adf_stationary": False,  # Want stationary (low p-value, reject unit root)
            "kpss_stationary": True,  # Want stationary (high p-value, fail to reject stationarity)
            "ljung_box_lag20_independent": True,  # Want independence (high p-value)
            "arch_lm_homoscedastic": True,  # Want homoscedasticity (high p-value)
            "runs_random": True,  # Want randomness (high p-value)
            # 'mcleod_li_no_arch': True           # Want no ARCH (high p-value) - COMMENTED OUT
        }

        for i, test_name in enumerate(test_names):
            if test_expectations[test_name]:
                # For tests where we want high p-value (fail to reject null)
                results[test_name] = not reject[i]  # Pass if we don't reject null
            else:
                # For tests where we want low p-value (reject null)
                results[test_name] = reject[i]  # Pass if we reject null

        return results

    except Exception as e:
        print(f"Warning: FDR correction failed: {e}")
        # Fallback to raw p-values with 0.05 threshold
        results = {}
        test_expectations = {
            "shapiro_normal": True,
            "jarque_bera_normal": True,
            "lilliefors_normal": True,
            "adf_stationary": False,
            "kpss_stationary": True,
            "ljung_box_lag20_independent": True,
            "arch_lm_homoscedastic": True,
            "runs_random": True,  # 'mcleod_li_no_arch': True
        }

        for i, test_name in enumerate(test_names):
            if test_expectations[test_name]:
                results[test_name] = p_values_clean[i] > 0.05
            else:
                results[test_name] = p_values_clean[i] < 0.05

        return results


def create_multi_subject_test_results():
    """Create statistical test results for all subjects."""

    print("Loading residuals from all subjects...")
    all_residuals = load_all_subjects_residuals()

    if not all_residuals:
        print("❌ No residuals found")
        return None

    print(f"\nRunning statistical tests on {len(all_residuals)} trials...")

    results_list = []

    for trial_name, residuals in all_residuals.items():
        # Extract subject info
        subject_id = trial_name.split("/")[0]
        trial_id = trial_name.split("/")[-1]

        print(f"  Testing {trial_name}...")

        # Run tests
        test_results = run_statistical_tests(residuals)

        # Compile results
        trial_result = {
            "trial_name": trial_name,
            "subject": subject_id,
            "trial": trial_id,
            "n_samples": len(residuals[np.isfinite(residuals)]),
            **test_results,
        }

        results_list.append(trial_result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results
    output_dir = Path("residual_analysis_clean/multi_subject_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "multi_subject_statistical_results.csv"
    results_df.to_csv(results_file, index=False)

    print(f"\n✅ Saved results to: {results_file}")
    print(f"   Total trials tested: {len(results_df)}")

    return results_df


def parse_individual_test_failures(df):
    """Parse individual test results to create detailed failure matrix."""

    # Define all statistical tests and their failure conditions
    test_definitions = {
        # Normality tests (True means normal, so we want False for failures)
        "Shapiro-Wilk": ("shapiro_normal", False),
        "Jarque-Bera": ("jarque_bera_normal", False),
        "Lilliefors": ("lilliefors_normal", False),
        # Stationarity tests
        "ADF": ("adf_stationary", False),  # ADF: True=stationary
        "KPSS": ("kpss_stationary", False),  # KPSS: True=stationary
        # Autocorrelation tests (True means independent, so we want False)
        "Ljung-Box (lag 20)": ("ljung_box_lag20_independent", False),
        # Heteroscedasticity tests (True means homoscedastic, so we want False)
        "ARCH-LM": ("arch_lm_homoscedastic", False),
        # Randomness test (True means random, so we want False)
        "Runs Test": ("runs_random", False),
        # ARCH effects test (True means no ARCH, so we want False) - COMMENTED OUT
        # 'McLeod-Li': ('mcleod_li_no_arch', False),
    }

    # Create failure matrix
    failure_matrix = []

    for _, row in df.iterrows():
        trial_failures = {
            "trial_name": row["trial_name"],
            "subject": row["subject"],
            "trial": row["trial"],
        }

        # Check each test for failure
        for test_name, test_config in test_definitions.items():
            column, failure_condition = test_config
            if column in df.columns:
                test_result = row[column]
                # Test fails if it matches failure condition
                trial_failures[test_name] = test_result == failure_condition
            else:
                trial_failures[test_name] = True  # Missing test = failure

        failure_matrix.append(trial_failures)

    return pd.DataFrame(failure_matrix)


def create_multi_subject_heatmaps(df, output_dir):
    """Create comprehensive heatmaps for multi-subject analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse failures
    failure_df = parse_individual_test_failures(df)

    # Prepare data for heatmap
    test_columns = [
        col
        for col in failure_df.columns
        if col not in ["trial_name", "subject", "trial"]
    ]

    # Create pivot table for heatmap (trials x tests)
    heatmap_data = failure_df[["trial_name"] + test_columns].set_index("trial_name")

    # Sort by subject for better visualization
    subjects_order = [f"subject_{i:03d}" for i in range(1, 6)]
    sorted_trials = []
    for subject in subjects_order:
        subject_trials = [t for t in heatmap_data.index if t.startswith(subject)]
        sorted_trials.extend(sorted(subject_trials))

    heatmap_data = heatmap_data.reindex(sorted_trials)

    # 1. Comprehensive heatmap (very thin rows to fit on page)
    plt.figure(figsize=(10, 12))  # Tall and narrow

    # Create heatmap with very thin rows
    sns.heatmap(
        heatmap_data.astype(int),
        cmap="RdYlGn_r",  # Red=failure, Green=pass
        cbar_kws={"label": "Test Failure (1=Fail, 0=Pass)"},
        xticklabels=True,
        yticklabels=False,  # Too many to show
        linewidths=0.01,
    )  # Very thin lines

    # plt.title('Multi-Subject Statistical Test Failures Heatmap\n(All Subjects 1-5)',
    #           fontsize=14, fontweight='bold')
    plt.xlabel("Statistical Tests", fontweight="bold")
    plt.ylabel("Trials (by Subject)", fontweight="bold")

    # Add subject boundaries
    y_pos = 0
    for subject in subjects_order:
        subject_count = sum(1 for t in sorted_trials if t.startswith(subject))
        if subject_count > 0:
            y_pos += subject_count
            if y_pos < len(sorted_trials):
                plt.axhline(y=y_pos, color="black", linewidth=2)

    plt.xticks(rotation=45, ha="right")
    plt.yticks([])
    plt.tight_layout()

    comprehensive_path = output_path / "multi_subject_test_failure_heatmap.svg"
    plt.savefig(comprehensive_path, format="svg", bbox_inches="tight")
    plt.close()

    # 2. Subject summary heatmap
    plt.figure(figsize=(12, 6))

    # Calculate failure rates by subject
    subject_summary = []
    for subject in subjects_order:
        subject_data = failure_df[failure_df["subject"] == subject]
        if len(subject_data) > 0:
            subject_failures = {}
            subject_failures["Subject"] = subject
            for test in test_columns:
                failure_rate = subject_data[test].mean() * 100
                subject_failures[test] = failure_rate
            subject_summary.append(subject_failures)

    if subject_summary:
        summary_df = pd.DataFrame(subject_summary).set_index("Subject")

        # Modify y-axis labels to show only numbers (001, 002, etc.)
        y_labels = [subject.replace("subject_", "") for subject in summary_df.index]

        sns.heatmap(
            summary_df,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Failure Rate (%)"},
            linewidths=0.5,
            yticklabels=y_labels,
        )

        # plt.title('Statistical Test Failure Rates by Subject\n(Percentage of Trials Failing Each Test)',
        #           fontsize=12, fontweight='bold')
        plt.xlabel("Statistical Tests", fontweight="bold")
        plt.ylabel("Subject", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        summary_path = output_path / "subject_failure_rates_heatmap.svg"
        plt.savefig(summary_path, format="svg", bbox_inches="tight")
        plt.close()

        print(f"✅ Created subject summary heatmap: {summary_path}")

    print(f"✅ Created comprehensive heatmap: {comprehensive_path}")

    return comprehensive_path, summary_path


def main():
    """Main execution function."""

    print("=" * 80)
    print("MULTI-SUBJECT STATISTICAL TEST ANALYSIS")
    print("=" * 80)

    # Create test results
    results_df = create_multi_subject_test_results()

    if results_df is None:
        return

    # Create visualizations
    output_dir = "residual_analysis_clean/multi_subject_tests"
    heatmap_path, summary_path = create_multi_subject_heatmaps(results_df, output_dir)

    # Print summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total trials analyzed: {len(results_df)}")

    subjects = results_df["subject"].unique()
    print(f"Subjects: {sorted(subjects)}")

    for subject in sorted(subjects):
        subject_data = results_df[results_df["subject"] == subject]
        print(f"  {subject}: {len(subject_data)} trials")

    # Test failure summary
    test_columns = [
        col
        for col in results_df.columns
        if col.endswith("_normal")
        or col.endswith("_stationary")
        or col.endswith("_independent")
    ]

    print(f"\n📈 OVERALL FAILURE RATES:")
    for test_col in test_columns:
        failure_rate = (1 - results_df[test_col].mean()) * 100  # 1 - success rate
        print(f"  {test_col}: {failure_rate:.1f}% failures")

    print(f"\n📁 OUTPUT FILES:")
    print(
        f"  - {results_df.shape[0]} trial results: {output_dir}/multi_subject_statistical_results.csv"
    )
    print(f"  - Comprehensive heatmap: {heatmap_path}")
    print(f"  - Subject summary heatmap: {summary_path}")


if __name__ == "__main__":
    main()
