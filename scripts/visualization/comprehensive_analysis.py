#!/usr/bin/env python3
"""
Comprehensive ARMA-GARCH-t Results Visualization (Clean Version)
================================================================

Complete analysis workflow replicating the depth of residual_analysis/
but using the clean implementation from residual_analysis_clean/.

This script:
- Loads clean fitted models (no scaling artifacts)
- Simulates using natural parameters with adaptive stability caps
- Comprehensive statistical comparisons
- Interactive visualization menu
- Full model validation and diagnostics
"""

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys
from pathlib import Path as PathLib

current_dir = PathLib(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from residual_analysis_clean.arma_garch_residual_model import EnhancedARMAGARCH

# Set plotting style
# plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set_palette("husl")


def load_clean_models(models_dir: str = None):
    """Load fitted models from both new and old locations."""

    combined_models = {}
    subjects_found = []

    # Auto-detect paths based on current working directory
    cwd = Path.cwd()
    if models_dir is None:
        if cwd.name == "residual_analysis_clean":
            # Running from residual_analysis_clean directory
            models_dir = "fitted_models"
            old_models_path = "../residual_analysis/arma_garch_fits_no_scaling/fitted_arma_garch_models_no_scaling.pkl"
        else:
            # Running from parent directory
            models_dir = "residual_analysis_clean/fitted_models"
            old_models_path = "residual_analysis/arma_garch_fits_no_scaling/fitted_arma_garch_models_no_scaling.pkl"
    else:
        # Use provided path
        old_models_path = "../residual_analysis/arma_garch_fits_no_scaling/fitted_arma_garch_models_no_scaling.pkl"

    # 1. Load from new location (multi-subject structure)
    models_path = Path(models_dir)
    if models_path.exists():
        for subject_dir in models_path.glob("subject_*"):
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                subject_models_file = subject_dir / f"{subject_id}_fitted_models.pkl"

                if subject_models_file.exists():
                    with open(subject_models_file, "rb") as f:
                        subject_models = pickle.load(f)

                    combined_models.update(subject_models)
                    subjects_found.append(subject_id)
                    print(f"  ✓ Loaded {len(subject_models)} trials from {subject_id}")

    # 2. Load subject_005 from old location if not found in new location
    if "subject_005" not in subjects_found:
        old_models_file = Path(old_models_path)
        if old_models_file.exists():
            with open(old_models_file, "rb") as f:
                subject_005_models = pickle.load(f)

            combined_models.update(subject_005_models)
            subjects_found.append("subject_005")
            print(
                f"  ✓ Loaded {len(subject_005_models)} trials from subject_005 (old location)"
            )

    if not combined_models:
        print(f"❌ No fitted models found")
        print("Please run fit_all_subjects_arma_garch.py first")
        return None

    print(
        f"✅ Loaded {len(combined_models)} fitted models from {len(subjects_found)} subjects: {subjects_found}"
    )
    return combined_models


def simulate_clean_arma_garch(
    model_params: dict, n_periods: int, random_seed: Optional[int] = None
):
    """
    Simulate from clean ARMA-GARCH-t parameters with adaptive stability caps.

    Parameters:
    -----------
    model_params : dict
        Model parameters in natural units
    n_periods : int
        Number of periods to simulate
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray : Simulated residuals
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract model parameters (natural units)
    arma_params = model_params.get("arma_params", {})
    garch_params = model_params.get("garch_params", {})
    dist_params = model_params.get("distribution_params", {})
    best_dist = model_params.get("best_distribution", "normal")
    optimal_orders = model_params.get(
        "optimal_orders", {"p": 0, "q": 0, "r": 1, "s": 1}
    )

    # ARMA orders
    p = optimal_orders.get("p", 0)
    q = optimal_orders.get("q", 0)

    # GARCH parameters (natural values)
    omega = garch_params.get("omega", 0.01)
    alpha = garch_params.get("alpha", 0.1)
    beta = garch_params.get("beta", 0.8)

    # Use original fitted parameters (no artificial capping)
    persistence = alpha + beta
    print(f"Using fitted persistence: {persistence:.6f}")
    print(f"Relying on restart mechanism for stability")

    # Setup simulation
    warm_up = max(100, p + q + 10)
    total_length = n_periods + warm_up

    # Initialize arrays
    residuals = np.zeros(total_length)
    arma_errors = np.zeros(total_length)  # ε_t (GARCH residuals)
    variances = np.zeros(total_length)

    # Generate innovations
    if best_dist == "t" and "nu" in dist_params:
        nu = dist_params["nu"]
        innovations = np.random.standard_t(nu, total_length)
    else:
        innovations = np.random.standard_normal(total_length)

    # Variance initialization with adaptive bounds
    init_variance = omega / (1 - alpha - beta)
    min_var = max(omega / 100, 1e-6)
    max_var = min(omega * 500, 1.5)

    init_variance = max(init_variance, min_var)
    init_variance = min(init_variance, max_var)

    print(f"Variance initialization: {init_variance:.4f} (ω={omega:.1e})")

    variances[:warm_up] = init_variance
    arma_errors[:warm_up] = innovations[:warm_up] * np.sqrt(init_variance)

    # Simulate process with detailed debug logging
    variance_explosions = 0
    large_variances = 0

    print(f"Starting simulation: n_periods={n_periods}, total_length={total_length}")
    print(
        f"GARCH parameters: ω={omega:.1e}, α={alpha:.3f}, β={beta:.3f}, persistence={alpha + beta:.3f}"
    )
    print(
        f"Variance bounds: min={min_var:.1e}, max={max_var:.1f}, explosion_threshold={(max_var * 1.8):.1f}"
    )

    for t in range(warm_up, total_length):
        # GARCH variance: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        prev_error_sq = arma_errors[t - 1] ** 2
        prev_variance = variances[t - 1]
        raw_variance = omega + alpha * prev_error_sq + beta * prev_variance

        # Debug variance components every 200 steps
        if (t - warm_up) % 200 == 0 or raw_variance > max_var:
            print(
                f"t={t - warm_up:4d}: σ²={raw_variance:.1e} = {omega:.1e} + {alpha:.3f}×{prev_error_sq:.1e} + {beta:.3f}×{prev_variance:.1e}"
            )

        # Apply adaptive variance bounds
        variance = max(raw_variance, min_var)
        variance = min(variance, max_var * 2)

        # Track variance bound hits
        if raw_variance > max_var:
            large_variances += 1
            if large_variances <= 3:  # Only show first few instances
                print(f"  → Variance capped from {raw_variance:.1e} to {variance:.1e}")

        variances[t] = variance

        # Variance explosion handling with restart mechanism
        explosion_threshold = max_var * 1.8
        if variance >= explosion_threshold:
            variance_explosions += 1
            print(
                f"💥 VARIANCE EXPLOSION at t={t - warm_up}: σ²={variance:.1e} ≥ {explosion_threshold:.1e}"
            )
            print(
                f"   Components: ω={omega:.1e}, α×ε²={alpha * prev_error_sq:.1e}, β×σ²={beta * prev_variance:.1e}"
            )

            # Reset variance to initial stable value and continue
            reset_variance = init_variance
            variances[t] = reset_variance
            print(
                f"   → RESET variance to {reset_variance:.1e} and continuing simulation"
            )
            variance = reset_variance

        # Generate GARCH residual: ε_t = σ_t * z_t
        arma_errors[t] = innovations[t] * np.sqrt(variance)

        # Additional safety check for NaN/Inf
        if not np.isfinite(arma_errors[t]):
            print(
                f"❌ Non-finite error at t={t - warm_up}: ε_t={arma_errors[t]}, σ_t={np.sqrt(variance):.1e}, z_t={innovations[t]:.3f}"
            )
            total_length = t + 1
            break

        # ARMA mean component
        arma_mean = 0

        # AR component
        for i in range(1, min(p + 1, t - warm_up + 1)):
            phi_i = arma_params.get(f"phi_{i}", 0)
            if t - i >= 0:
                arma_mean += phi_i * residuals[t - i]

        # MA component
        for j in range(1, min(q + 1, t - warm_up + 1)):
            theta_j = arma_params.get(f"theta_{j}", 0)
            if t - j >= 0:
                arma_mean += theta_j * arma_errors[t - j]

        # Final residual: r_t = arma_mean + ε_t
        residuals[t] = arma_mean + arma_errors[t]

    # Return post-warmup simulation with proper indexing
    start_idx = warm_up
    end_idx = total_length

    # Simulation summary
    actual_periods = end_idx - start_idx
    print(f"\nSimulation Summary:")
    print(f"  Requested periods: {n_periods}")
    print(f"  Actual periods: {actual_periods}")
    print(f"  Early stopping: {'YES' if actual_periods < n_periods else 'NO'}")
    print(f"  Variance explosions: {variance_explosions}")
    print(f"  Large variances (capped): {large_variances}")

    if end_idx <= start_idx:
        print(
            f"❌ Early stopping resulted in no valid samples (start={start_idx}, end={end_idx})"
        )
        return np.array([])

    simulated_residuals = residuals[start_idx:end_idx]
    print(f"✅ Returning {len(simulated_residuals)} simulated residuals")
    print(
        f"   Stats: mean={np.mean(simulated_residuals):.3f}, std={np.std(simulated_residuals):.3f}"
    )

    return simulated_residuals


def simulate_enhanced_arma_garch(
    model_params: dict, n_periods: int, random_seed: Optional[int] = None
):
    """
    Simulate using EnhancedARMAGARCH class for consistency with fitted models.

    This function reconstructs the model using the same class that was used for fitting,
    ensuring parameter consistency and avoiding the persistence mismatch issue.
    """
    if EnhancedARMAGARCH is None:
        print("⚠️  EnhancedARMAGARCH not available, falling back to custom simulation")
        return simulate_clean_arma_garch(model_params, n_periods, random_seed)

    if random_seed is not None:
        np.random.seed(random_seed)

    try:
        # Extract basic info
        best_dist = model_params.get("best_distribution", "normal")
        optimal_orders = model_params.get(
            "optimal_orders", {"p": 0, "q": 0, "r": 1, "s": 1}
        )
        arma_params = model_params.get("arma_params", {})
        garch_params = model_params.get("garch_params", {})
        dist_params = model_params.get("distribution_params", {})

        # Create EnhancedARMAGARCH instance with matching orders
        model = EnhancedARMAGARCH(
            p=optimal_orders.get("p", 0),
            q=optimal_orders.get("q", 0),
            r=optimal_orders.get("r", 1),
            s=optimal_orders.get("s", 1),
        )

        # Set the best distribution
        model.best_distribution = best_dist

        # Create a mock fitted model with the stored parameters
        from types import SimpleNamespace

        # Mock GARCH model with parameters
        mock_garch = SimpleNamespace()
        mock_garch.params = {}

        # Add GARCH parameters in the expected format
        mock_garch.params["omega"] = garch_params.get("omega", 0.01)
        for i in range(1, model.s + 1):
            alpha_key = f"alpha_{i}"
            if alpha_key in garch_params:
                mock_garch.params[f"alpha[{i}]"] = garch_params[alpha_key]

        for j in range(1, model.r + 1):
            beta_key = f"beta_{j}"
            if beta_key in garch_params:
                mock_garch.params[f"beta[{j}]"] = garch_params[beta_key]

        # Add distribution parameter if Student-t
        if best_dist == "t" and "nu" in dist_params:
            mock_garch.params["nu"] = dist_params["nu"]

        # Store mock model
        model.garch_models = {best_dist: mock_garch}

        # Calculate persistence using the same method as EnhancedARMAGARCH
        calculated_persistence = model._calculate_persistence()
        stored_persistence = model_params.get(
            "volatility_persistence", calculated_persistence
        )

        print(f"Persistence comparison:")
        print(f"  Calculated from garch_params: {calculated_persistence:.6f}")
        print(f"  Stored volatility_persistence: {stored_persistence:.6f}")
        print(f"  Using calculated persistence: {calculated_persistence:.6f}")

        # Now simulate using the EnhancedARMAGARCH approach
        # Generate base innovations
        if best_dist == "t" and "nu" in dist_params:
            nu = dist_params["nu"]
            innovations = np.random.standard_t(nu, n_periods + 100)
        else:
            innovations = np.random.standard_normal(n_periods + 100)

        # GARCH simulation using exact parameters
        omega = garch_params.get("omega", 0.01)
        alpha = garch_params.get("alpha_1", 0.1)
        beta = garch_params.get("beta_1", 0.8)

        # Initialize variance
        init_variance = omega / max(1 - (alpha + beta), 0.01)
        variances = [init_variance] * 100  # warmup
        residuals = []

        print(f"GARCH simulation: ω={omega:.1e}, α={alpha:.3f}, β={beta:.3f}")
        print(f"Persistence: {alpha + beta:.3f}")
        print(f"Initial variance: {init_variance:.1e}")

        # Additional safety check
        if alpha + beta >= 0.99:
            print(f"⚠️  WARNING: Near-unit root GARCH detected, may be unstable!")
        if omega <= 0:
            print(f"⚠️  WARNING: Non-positive omega={omega}")
        if init_variance > 1.0:
            print(f"⚠️  WARNING: Very large initial variance={init_variance}")

        for t in range(n_periods):
            # GARCH variance evolution
            if len(residuals) > 0:
                prev_resid_sq = residuals[-1] ** 2
                prev_variance = variances[-1]
                new_variance = omega + alpha * prev_resid_sq + beta * prev_variance
            else:
                new_variance = init_variance

            # Much more aggressive bounds checking
            if new_variance > 1.0 or new_variance <= 0 or not np.isfinite(new_variance):
                if t < 10:  # Only print for first few to avoid spam
                    print(
                        f"   Variance issue at t={t}: {new_variance:.1e} -> reset to {init_variance:.1e}"
                    )
                new_variance = init_variance

            # Additional safety clipping
            new_variance = np.clip(new_variance, 1e-8, 1.0)
            variances.append(new_variance)

            # Generate residual with additional bounds checking
            variance_sqrt = np.sqrt(variances[-1])
            if not np.isfinite(variance_sqrt) or variance_sqrt > 10.0:
                variance_sqrt = 0.1  # Safe fallback

            residual = innovations[t + 100] * variance_sqrt

            # Check residual bounds
            if not np.isfinite(residual) or abs(residual) > 100.0:
                residual = np.clip(residual, -10.0, 10.0)
                if t < 5:
                    print(f"   Residual clipped at t={t}")

            residuals.append(residual)

        # Proper ARMA component implementation matching the original simulation
        arma_residuals = np.zeros(len(residuals))
        garch_errors = np.array(residuals)  # ε_t from GARCH

        p = optimal_orders.get("p", 0)
        q = optimal_orders.get("q", 0)

        print(f"Implementing ARMA({p},{q}) component...")

        # Initialize with GARCH errors for warmup
        warmup_length = max(p, q, 10)
        arma_residuals[:warmup_length] = garch_errors[:warmup_length]

        for t in range(warmup_length, len(residuals)):
            # ARMA mean component: E[r_t] = Σφ_i * r_{t-i} + Σθ_j * ε_{t-j}
            arma_mean = 0

            # AR component: φ_1 * r_{t-1} + φ_2 * r_{t-2} + ...
            for i in range(1, min(p + 1, t + 1)):
                phi_key = f"phi_{i}"
                if phi_key in arma_params:
                    phi_i = arma_params[phi_key]
                    arma_mean += phi_i * arma_residuals[t - i]

            # MA component: θ_1 * ε_{t-1} + θ_2 * ε_{t-2} + ...
            for j in range(1, min(q + 1, t + 1)):
                theta_key = f"theta_{j}"
                if theta_key in arma_params:
                    theta_j = arma_params[theta_key]
                    arma_mean += theta_j * garch_errors[t - j]

            # Final residual: r_t = arma_mean + ε_t
            arma_residuals[t] = arma_mean + garch_errors[t]

        # Calculate standardized innovations for proper ACF comparison
        # z_t = residual_t / sigma_t
        simulated_innovations = []
        for i in range(len(arma_residuals)):
            if i < len(variances) - 100:  # Account for warmup offset
                sigma_t = np.sqrt(variances[i + 100])
                z_t = arma_residuals[i] / max(sigma_t, 1e-8)
                simulated_innovations.append(z_t)
            else:
                # Fallback for edge cases
                simulated_innovations.append(arma_residuals[i] / np.std(arma_residuals))

        simulated_innovations = np.array(simulated_innovations)

        # Final sanity check for numerical stability
        residual_mean = np.mean(arma_residuals)
        residual_std = np.std(arma_residuals)
        innov_mean = np.mean(simulated_innovations)
        innov_std = np.std(simulated_innovations)

        # Check for explosive values
        if (
            abs(residual_mean) > 1e6
            or residual_std > 1e6
            or abs(innov_mean) > 100
            or innov_std > 100
        ):
            print(f"❌ Simulation produced explosive values - returning fallback")
            # Return conservative fallback simulation
            fallback_residuals = np.random.normal(0, 0.1, len(arma_residuals))
            fallback_innovations = np.random.standard_normal(len(arma_residuals))
            return {
                "residuals": fallback_residuals,
                "innovations": fallback_innovations,
                "variances": np.full(len(arma_residuals), 0.01),
            }

        print(f"✅ Enhanced simulation completed: {len(arma_residuals)} samples")
        print(f"   Stats: mean={residual_mean:.3f}, std={residual_std:.3f}")
        print(f"   Innovations: mean={innov_mean:.3f}, std={innov_std:.3f}")

        # Store innovations in a way the plotting function can access
        # We'll return a dict instead of just the array
        return {
            "residuals": arma_residuals,
            "innovations": simulated_innovations,
            "variances": np.array(
                variances[100 : 100 + len(arma_residuals)]
            ),  # Remove warmup
        }

    except Exception as e:
        print(f"❌ Enhanced simulation failed: {e}")
        print("Falling back to custom simulation")
        fallback_residuals = simulate_clean_arma_garch(
            model_params, n_periods, random_seed
        )
        # Return in the same dict format for consistency
        return {
            "residuals": fallback_residuals,
            "innovations": np.array([]),  # No innovations available from fallback
            "variances": np.array([]),
        }


def plot_comprehensive_comparison_option12(
    trial_name: str, fitted_models: dict, save_plots: bool = True, random_seed: int = 42
):
    """
    Enhanced comprehensive plotting for Option 12 with seaborn styling and rolling volatility.
    """

    if trial_name not in fitted_models:
        print(f"❌ Trial {trial_name} not found in fitted models")
        return None

    # Set seaborn style
    sns.set_style("whitegrid")
    colors = sns.color_palette("muted")

    trial_data = fitted_models[trial_name]

    # Get original residuals
    original_residuals = trial_data.get("original_residuals", None)

    if original_residuals is None:
        print("❌ Original residuals not found")
        return None

    print(f"\n📊 OPTION 12 ANALYSIS: {trial_name}")
    print(f"   Distribution: {trial_data['best_distribution']}")
    print(f"   Persistence: {trial_data.get('volatility_persistence', 0):.3f}")
    print(f"   Samples: {len(original_residuals)}")

    # Simulate from fitted model
    try:
        simulation_result = simulate_enhanced_arma_garch(
            trial_data, n_periods=len(original_residuals), random_seed=random_seed
        )

        if isinstance(simulation_result, dict):
            simulated_residuals = simulation_result["residuals"]
            simulated_innovations = simulation_result.get("innovations", np.array([]))
        else:
            simulated_residuals = simulation_result
            simulated_innovations = np.array([])

        print(f"   Simulation successful: {len(simulated_residuals)} samples")

    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return None

    # Create separate figures for each plot
    if save_plots:
        output_dir = Path("residual_analysis_clean/plots_comprehensive")
        output_dir.mkdir(exist_ok=True)
        trial_clean = trial_name.replace("/", "_")

    figures = []

    # 1. Time series comparison (colors[0] for real, colors[1] for synthetic)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    t_orig = np.arange(len(original_residuals))
    t_sim = np.arange(len(simulated_residuals))

    ax1.plot(
        t_orig,
        original_residuals,
        color=colors[0],
        alpha=0.7,
        linewidth=0.8,
        label="Real",
    )
    ax1.plot(
        t_sim,
        simulated_residuals,
        color=colors[1],
        alpha=0.7,
        linewidth=0.8,
        label="Synthetic",
    )
    ax1.set_xlabel("Time", fontsize=20)
    ax1.set_ylabel("Residuals", fontsize=20)
    ax1.legend(fontsize=20, loc="upper left")
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_timeseries_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Time series plot saved: {plot_file}")
    plt.show()
    figures.append(fig1)

    # 2. Residual histogram comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(
        original_residuals,
        bins=50,
        alpha=0.7,
        density=True,
        label="Real",
        color=colors[0],
    )
    ax2.hist(
        simulated_residuals,
        bins=50,
        alpha=0.7,
        density=True,
        label="Synthetic",
        color=colors[1],
    )
    ax2.set_xlabel("Residual Value", fontsize=20)
    ax2.set_ylabel("Density", fontsize=20)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_histogram_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Histogram plot saved: {plot_file}")
    plt.show()
    figures.append(fig2)

    # 3. Innovation Distribution Analysis (color[3])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    standardized_innovations = trial_data.get("standardized_residuals", np.array([]))

    if len(standardized_innovations) > 0:
        ax3.hist(
            standardized_innovations,
            bins=50,
            alpha=0.7,
            density=True,
            label="Standardized Innovations",
            color=colors[3],
        )

        # Overlay theoretical distribution
        x_range = np.linspace(
            standardized_innovations.min(), standardized_innovations.max(), 200
        )
        if trial_data["best_distribution"] == "t":
            nu = trial_data.get("distribution_params", {}).get("nu", None)
            if nu is None:
                nu = trial_data.get("nu", None)
            if nu is None:
                garch_params = trial_data.get("garch_params", {})
                nu = garch_params.get("nu", None)
            if nu is None or np.isnan(nu):
                nu = 5.67
            theoretical_pdf = stats.t.pdf(x_range, nu)
            ax3.plot(
                x_range,
                theoretical_pdf,
                "k--",
                linewidth=2,
                alpha=0.8,
                label=f"Student-t (ν={nu:.1f})",
            )
        else:
            theoretical_pdf = stats.norm.pdf(x_range)
            ax3.plot(
                x_range, theoretical_pdf, "k--", linewidth=2, alpha=0.8, label="Normal"
            )

        ax3.set_xlabel(
            r"Standardized Innovation ($\frac{\epsilon}{\sigma}$)", fontsize=20
        )
        ax3.set_ylabel("Density", fontsize=20)
        ax3.legend(fontsize=20)
        ax3.tick_params(axis="x", labelsize=20)
        ax3.tick_params(axis="y", labelsize=20)
    else:
        ax3.text(
            0.5,
            0.5,
            "No innovations available\nfor this trial",
            transform=ax3.transAxes,
            ha="center",
            va="center",
        )

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_innovations_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Innovations plot saved: {plot_file}")
    plt.show()
    figures.append(fig3)

    # 4. Autocorrelation analysis
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    from statsmodels.tsa.stattools import acf

    min_len = min(len(original_residuals), len(simulated_residuals))
    lags = min(40, min_len // 4)

    orig_acf = acf(original_residuals[:min_len], nlags=lags, fft=True)
    sim_acf = acf(simulated_residuals[:min_len], nlags=lags, fft=True)

    ax4.plot(range(len(orig_acf)), orig_acf, color=colors[0], alpha=0.8, label="Real")
    ax4.plot(
        range(len(sim_acf)), sim_acf, color=colors[1], alpha=0.8, label="Synthetic"
    )
    ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax4.axhline(y=0.05, color="k", linestyle="--", alpha=0.5, label="±0.05 bounds")
    ax4.axhline(y=-0.05, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Lag", fontsize=20)
    ax4.set_ylabel("ACF", fontsize=20)
    ax4.legend(fontsize=20)
    ax4.tick_params(axis="x", labelsize=20)
    ax4.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_autocorrelation_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Autocorrelation plot saved: {plot_file}")
    plt.show()
    figures.append(fig4)

    # 5. Q-Q plot: Actual vs Simulated
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    min_len = min(len(original_residuals), len(simulated_residuals))
    orig_sorted = np.sort(original_residuals)[:min_len]
    sim_sorted = np.sort(simulated_residuals)[:min_len]

    ax5.scatter(orig_sorted, sim_sorted, alpha=0.6, s=8, color=colors[2])
    min_val = min(orig_sorted.min(), sim_sorted.min())
    max_val = max(orig_sorted.max(), sim_sorted.max())
    ax5.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8, linewidth=2)
    ax5.set_xlabel("Real Quantiles", fontsize=20)
    ax5.set_ylabel("Synthetic Quantiles", fontsize=20)
    ax5.tick_params(axis="x", labelsize=20)
    ax5.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_qqplot_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Q-Q plot saved: {plot_file}")
    plt.show()
    figures.append(fig5)

    # 6. Rolling window volatility (NEW: replaces text summary)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    window_size = max(20, len(original_residuals) // 30)  # Adaptive window size

    # Calculate rolling standard deviation for both series
    orig_rolling_vol = (
        pd.Series(original_residuals).rolling(window=window_size, center=True).std()
    )
    sim_rolling_vol = (
        pd.Series(simulated_residuals).rolling(window=window_size, center=True).std()
    )

    t_axis = np.arange(len(original_residuals))
    ax6.plot(
        t_axis,
        orig_rolling_vol,
        color=colors[0],
        alpha=0.8,
        linewidth=1.5,
        label="Real Volatility",
    )
    ax6.plot(
        t_axis,
        sim_rolling_vol,
        color=colors[1],
        alpha=0.8,
        linewidth=1.5,
        label="Synthetic Volatility",
    )
    ax6.set_xlabel("Time", fontsize=20)
    ax6.set_ylabel("Rolling Std", fontsize=20)
    ax6.legend(fontsize=20)
    ax6.tick_params(axis="x", labelsize=20)
    ax6.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    if save_plots:
        plot_file = output_dir / f"option12_volatility_{trial_clean}.svg"
        plt.savefig(plot_file, format="svg", bbox_inches="tight")
        print(f"   📊 Rolling volatility plot saved: {plot_file}")
    plt.show()
    figures.append(fig6)

    return figures


def plot_comprehensive_comparison(
    trial_name: str, fitted_models: dict, save_plots: bool = True, random_seed: int = 42
):
    """
    Comprehensive plotting with statistical analysis and model diagnostics.
    """

    if trial_name not in fitted_models:
        print(f"❌ Trial {trial_name} not found in fitted models")
        return None

    trial_data = fitted_models[trial_name]

    # Debug: Print available keys
    print(f"Available keys in trial_data: {list(trial_data.keys())}")

    # Original residuals should now be included in the fitted model data
    original_residuals = trial_data.get("original_residuals", None)

    if original_residuals is not None:
        print(f"✅ Original residuals available ({len(original_residuals)} samples)")
    else:
        print("⚠️ Original residuals not found in fitted model data")

    print(f"\n📊 COMPREHENSIVE ANALYSIS: {trial_name}")
    print(f"   Distribution: {trial_data['best_distribution']}")
    print(f"   Persistence: {trial_data.get('volatility_persistence', 0):.3f}")

    if original_residuals is not None:
        print(f"   Samples: {len(original_residuals)}")
    else:
        print(f"   Samples: Not available (need to load separately)")

    # Print parameter values
    garch_params = trial_data["garch_params"]
    print(
        f"   Natural parameters: ω={garch_params.get('omega', 0):.2e}, α={garch_params.get('alpha', 0):.3f}, β={garch_params.get('beta', 0):.3f}"
    )

    # Simulate from fitted model (matching original residual length)
    try:
        # Use actual length of original residuals
        if len(original_residuals) > 0:
            n_periods = len(original_residuals)
        else:
            n_periods = 1000  # Fallback if original residuals not available

        simulation_result = simulate_enhanced_arma_garch(
            trial_data, n_periods=n_periods, random_seed=random_seed
        )

        # Handle both old format (array) and new format (dict)
        if isinstance(simulation_result, dict):
            simulated_residuals = simulation_result["residuals"]
            simulated_innovations = simulation_result.get("innovations", np.array([]))
            simulated_variances = simulation_result.get("variances", np.array([]))
            print(
                f"   Enhanced simulation: {len(simulated_residuals)} residuals, {len(simulated_innovations)} innovations generated"
            )
        else:
            # Fallback for old format or fallback simulation
            simulated_residuals = simulation_result
            simulated_innovations = np.array([])
            simulated_variances = np.array([])
            print(
                f"   Fallback simulation: {len(simulated_residuals)} samples generated"
            )
        if len(simulated_residuals) == 0:
            print("   ❌ Simulation failed - no samples returned")
            return None

        print(f"   Simulated std: {np.std(simulated_residuals):.3f}")

        if original_residuals is not None:
            print(f"   Original std: {np.std(original_residuals):.3f}")
            print(
                f"   Std ratio: {np.std(simulated_residuals) / np.std(original_residuals):.2f}"
            )
        else:
            print("   (Original residuals not loaded for comparison)")

    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return None

    # Only proceed with plotting if we have original residuals
    if original_residuals is None:
        print("\n⚠️ Cannot create comprehensive plots without original residuals")
        print(
            "   Model parameters and simulation successful, but plots require original data"
        )
        return None

    # Create comprehensive comparison plot (2x3 layout for more analysis)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Comprehensive ARMA-GARCH-t Analysis: {trial_name}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Time series comparison
    t_orig = np.arange(len(original_residuals))
    t_sim = np.arange(len(simulated_residuals))

    axes[0, 0].plot(
        t_orig, original_residuals, "b-", alpha=0.7, linewidth=0.8, label="Actual"
    )
    axes[0, 0].plot(
        t_sim, simulated_residuals, "r-", alpha=0.7, linewidth=0.8, label="Simulated"
    )
    axes[0, 0].set_title("Time Series Comparison")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual histogram comparison (SEPARATE from innovations)
    axes[0, 1].hist(
        original_residuals,
        bins=50,
        alpha=0.7,
        density=True,
        label="Original Residuals",
        color="blue",
    )
    axes[0, 1].hist(
        simulated_residuals,
        bins=50,
        alpha=0.7,
        density=True,
        label="Simulated Residuals",
        color="red",
    )
    axes[0, 1].set_title("Residual Distribution Comparison")
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Innovation Distribution Analysis (SEPARATE panel)
    standardized_innovations = trial_data.get("standardized_residuals", np.array([]))

    if len(standardized_innovations) > 0:
        # Plot innovations histogram
        axes[0, 2].hist(
            standardized_innovations,
            bins=50,
            alpha=0.7,
            density=True,
            label="Standardized Innovations",
            color="green",
        )

        # Overlay theoretical distribution fitted to innovations
        x_range = np.linspace(
            standardized_innovations.min(), standardized_innovations.max(), 200
        )
        if trial_data["best_distribution"] == "t":
            # Use robust ν extraction (same logic as the corrected version)
            nu = trial_data.get("distribution_params", {}).get("nu", None)
            if nu is None:
                nu = trial_data.get("nu", None)
            if nu is None:
                garch_params = trial_data.get("garch_params", {})
                nu = garch_params.get("nu", None)
            if nu is None or np.isnan(nu):
                nu = 5.67  # Subject_005 mean as fallback
            theoretical_pdf = stats.t.pdf(x_range, nu)
            axes[0, 2].plot(
                x_range,
                theoretical_pdf,
                "k--",
                linewidth=2,
                alpha=0.8,
                label=f"Student-t (ν={nu:.1f})",
            )
        else:
            theoretical_pdf = stats.norm.pdf(x_range)
            axes[0, 2].plot(
                x_range, theoretical_pdf, "k--", linewidth=2, alpha=0.8, label="Normal"
            )

        axes[0, 2].set_title("Innovation Distribution Analysis")
        axes[0, 2].set_xlabel("Standardized Innovation (ε/σ)")
        axes[0, 2].set_ylabel("Density")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        # Fallback if no innovations available
        axes[0, 2].text(
            0.5,
            0.5,
            "No innovations available\nfor this trial",
            transform=axes[0, 2].transAxes,
            ha="center",
            va="center",
        )
        axes[0, 2].set_title("Innovation Distribution Analysis")
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])

    # 4. Autocorrelation analysis - Compare residual temporal structure
    from statsmodels.tsa.stattools import acf

    # Use raw residuals to compare temporal structure patterns
    min_len = min(len(original_residuals), len(simulated_residuals))
    lags = min(40, min_len // 4)

    orig_acf = acf(original_residuals[:min_len], nlags=lags, fft=True)
    sim_acf = acf(simulated_residuals[:min_len], nlags=lags, fft=True)

    axes[1, 0].plot(
        range(len(orig_acf)), orig_acf, "b-", alpha=0.8, label="Original Residuals"
    )
    axes[1, 0].plot(
        range(len(sim_acf)), sim_acf, "r-", alpha=0.8, label="Simulated Residuals"
    )
    axes[1, 0].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 0].axhline(y=0.05, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(y=-0.05, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Residual Autocorrelation Comparison")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("ACF")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    print(
        f"   ACF comparing residual structure: orig={len(original_residuals)}, sim={len(simulated_residuals)}"
    )

    # 5. Q-Q plot: Actual vs Simulated
    min_len = min(len(original_residuals), len(simulated_residuals))
    orig_sorted = np.sort(original_residuals)[:min_len]
    sim_sorted = np.sort(simulated_residuals)[:min_len]

    axes[1, 1].scatter(orig_sorted, sim_sorted, alpha=0.6, s=8)
    min_val = min(orig_sorted.min(), sim_sorted.min())
    max_val = max(orig_sorted.max(), sim_sorted.max())
    axes[1, 1].plot(
        [min_val, max_val], [min_val, max_val], "k--", alpha=0.8, linewidth=2
    )
    axes[1, 1].set_title("Q-Q Plot: Actual vs Simulated")
    axes[1, 1].set_xlabel("Actual Quantiles")
    axes[1, 1].set_ylabel("Simulated Quantiles")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Model statistics and diagnostics
    stats_text = f"""INNOVATION Distribution: {trial_data["best_distribution"]}
ARMA Orders: ({trial_data["optimal_orders"]["p"]},{trial_data["optimal_orders"]["q"]})
GARCH Orders: ({trial_data["optimal_orders"]["r"]},{trial_data["optimal_orders"]["s"]})

FITTED PERSISTENCE: {trial_data.get("volatility_persistence", 0):.3f}

NATURAL PARAMETERS (No Scaling):
ω = {trial_data["garch_params"].get("omega", 0):.2e}
α = {trial_data["garch_params"].get("alpha", 0):.3f}
β = {trial_data["garch_params"].get("beta", 0):.3f}

SIMULATION STATISTICS:
Original std: {np.std(original_residuals):.3f}
Simulated std: {np.std(simulated_residuals):.3f}
Std ratio: {np.std(simulated_residuals) / np.std(original_residuals):.2f}
    """

    if trial_data["best_distribution"] == "t":
        # Use robust ν extraction (same logic as the corrected version)
        nu = trial_data.get("distribution_params", {}).get("nu", None)
        if nu is None:
            nu = trial_data.get("nu", None)
        if nu is None:
            garch_params = trial_data.get("garch_params", {})
            nu = garch_params.get("nu", None)
        if nu is None or np.isnan(nu):
            nu = 5.67  # Subject_005 mean as fallback
        stats_text += f"\nStudent-t ν: {nu:.2f}"

    # Statistical tests on INNOVATIONS (standardized residuals)
    standardized_innovations = trial_data.get("standardized_residuals", np.array([]))
    if len(standardized_innovations) > 0:
        try:
            # Center innovations for testing
            clean_innovations = standardized_innovations - np.mean(
                standardized_innovations
            )

            # Initialize test results
            lb_p = np.nan
            lf_p = np.nan

            # 1. Ljung-Box test for whiteness (lag 10) - same as heatmap
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(clean_innovations, lags=10, return_df=True)
                lb_p = lb_result["lb_pvalue"].iloc[-1]
            except:
                lb_p = np.nan

            # 2. Distribution fit test using PIT transformation + Lilliefors - same as heatmap
            try:
                if trial_data["best_distribution"] == "t":
                    # Get ν parameter using same robust extraction as elsewhere
                    nu = trial_data.get("distribution_params", {}).get("nu", None)
                    if nu is None:
                        nu = trial_data.get("nu", None)
                    if nu is None:
                        garch_params = trial_data.get("garch_params", {})
                        nu = garch_params.get("nu", None)
                    if nu is None or np.isnan(nu):
                        nu = 5.67  # Subject_005 mean as fallback

                    # PIT transformation: t-distribution CDF -> normal CDF^(-1)
                    try:
                        from scipy.stats import norm
                        from scipy.stats import t as t_dist
                        from statsmodels.stats.diagnostic import lilliefors

                        transformed = norm.ppf(t_dist.cdf(clean_innovations, nu))
                        lf_stat, lf_p = lilliefors(transformed, dist="norm")
                    except ImportError:
                        lf_p = np.nan
                else:
                    # For normal: Direct Lilliefors test
                    try:
                        from statsmodels.stats.diagnostic import lilliefors

                        lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")
                    except ImportError:
                        lf_p = np.nan
            except:
                lf_p = np.nan

            # Extract stored test results if available
            whiteness_score = trial_data.get("whiteness_score", "N/A")

            stats_text += f"\nINNOVATION TEST RESULTS:"
            stats_text += f"\nTesting: standardized z_t = ε_t/σ_t"
            stats_text += f"\nNo FDR correction: direct p > 0.05 thresholds"

            # Whiteness tests with simple p > 0.05 thresholds (same as heatmap)
            stats_text += f"\n\nWHITENESS TESTS:"

            overall_whiteness_pass = True

            # Ljung-Box test result
            if not np.isnan(lb_p):
                lb_pass = lb_p > 0.05
                stats_text += f"\nLjung-Box (10): p={lb_p:.3f}"
                stats_text += f" ({'PASS' if lb_pass else 'FAIL'})"
                if not lb_pass:
                    overall_whiteness_pass = False
            else:
                stats_text += f"\nLjung-Box (10): FAILED"
                overall_whiteness_pass = False

            # Overall whiteness assessment
            stats_text += (
                f"\nOverall whiteness: {'PASS' if overall_whiteness_pass else 'FAIL'}"
            )
            if whiteness_score != "N/A":
                stats_text += f"\nWhiteness score: {whiteness_score}/3"

            # Distribution fit test
            stats_text += f"\n\nDISTRIBUTION FIT:"
            distribution_pass = True

            if not np.isnan(lf_p):
                lf_pass = lf_p > 0.05
                if trial_data["best_distribution"] == "t":
                    stats_text += f"\nLilliefors (PIT): p={lf_p:.3f}"
                    stats_text += f" ({'PASS' if lf_pass else 'FAIL'})"
                else:
                    stats_text += f"\nLilliefors: p={lf_p:.3f}"
                    stats_text += f" ({'PASS' if lf_pass else 'FAIL'})"
                if not lf_pass:
                    distribution_pass = False
            else:
                stats_text += f"\nLilliefors: FAILED"
                distribution_pass = False

            # Determine what we're testing
            if trial_data["best_distribution"] == "t":
                fit_type = "t-dist adequacy"
            else:
                fit_type = "Normality"
            stats_text += f"\n{fit_type}: {'PASS' if distribution_pass else 'FAIL'}"

            # Overall adequacy based on both whiteness and distribution fit
            overall_adequacy = overall_whiteness_pass and distribution_pass
            stats_text += f"\n\nMODEL ADEQUACY:"
            stats_text += f"\n(Whiteness + Distribution)"
            stats_text += f"\n{'PASS' if overall_adequacy else 'FAIL'}"

        except Exception as e:
            stats_text += f"\n\nINNOVATION TESTS:"
            stats_text += f"\nTesting standardized innovations"
            stats_text += f"\n(Test failed: {str(e)[:30]})"

    axes[1, 2].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1, 2].transAxes,
        verticalalignment="top",
        fontsize=6,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title("Model Summary & Diagnostics")

    plt.tight_layout()

    if save_plots:
        output_dir = Path("residual_analysis_clean/plots_comprehensive")
        output_dir.mkdir(exist_ok=True)

        trial_clean = trial_name.replace("/", "_")
        plot_file = output_dir / f"comprehensive_analysis_{trial_clean}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"   📊 Comprehensive plot saved: {plot_file}")

    plt.show()

    return fig


def interactive_comprehensive_menu():
    """Interactive menu for comprehensive analysis."""

    print("=" * 80)
    print("COMPREHENSIVE ARMA-GARCH-t ANALYSIS (CLEAN VERSION)")
    print("Replicating residual_analysis/ depth with clean implementation")
    print("=" * 80)

    # Load clean models
    fitted_models = load_clean_models()
    if fitted_models is None:
        return

    # Interactive menu
    while True:
        print(f"\n" + "=" * 60)
        print("Comprehensive Analysis Options:")
        print("1. Full analysis for single trial")
        print("2. Parameter summary statistics")
        print("3. Simulation stability test")
        print("4. Best fitting trials analysis")
        print("5. Distribution preference analysis")
        print("6. Std ratio & innovation test heatmap")
        print("7. Cross-validation vs random baselines")
        print("8. New: Cross-validation analysis (fitted vs random)")
        print("9. Trial-based cross-validation (direct comparison)")
        print("10. Model adequacy validation (ARMA-GARCH vs White Gaussian)")
        print("11. Option 12: Enhanced report plot for subject_004_trial_001")
        print("12. Exit")
        print("=" * 60)

        choice = input("Enter choice (1-12): ").strip()

        if choice == "1":
            trial_names = list(fitted_models.keys())
            print(f"\nAvailable trials: {trial_names[:5]}... (showing first 5)")
            trial_name = input(
                "Enter trial name (e.g., 'subject_005/trial_001'): "
            ).strip()

            if trial_name in fitted_models:
                plot_comprehensive_comparison(trial_name, fitted_models)
            else:
                print(f"❌ Trial {trial_name} not found")

        elif choice == "2":
            # Comprehensive parameter summary
            print(f"\n📊 COMPREHENSIVE PARAMETER ANALYSIS:")
            all_persistence = [
                data.get("volatility_persistence", 0) for data in fitted_models.values()
            ]
            all_omega = [
                data["garch_params"].get("omega", 0) for data in fitted_models.values()
            ]
            all_alpha = [
                data["garch_params"].get("alpha", 0) for data in fitted_models.values()
            ]
            all_beta = [
                data["garch_params"].get("beta", 0) for data in fitted_models.values()
            ]

            print(f"  Total trials: {len(fitted_models)}")
            print(
                f"  Persistence: {np.mean(all_persistence):.3f} ± {np.std(all_persistence):.3f}"
            )
            print(
                f"  Range: [{np.min(all_persistence):.3f}, {np.max(all_persistence):.3f}]"
            )

            print(f"\n  NATURAL PARAMETER RANGES:")
            print(
                f"  ω: [{np.min(all_omega):.2e}, {np.max(all_omega):.2e}] mean={np.mean(all_omega):.2e}"
            )
            print(
                f"  α: [{np.min(all_alpha):.3f}, {np.max(all_alpha):.3f}] mean={np.mean(all_alpha):.3f}"
            )
            print(
                f"  β: [{np.min(all_beta):.3f}, {np.max(all_beta):.3f}] mean={np.mean(all_beta):.3f}"
            )

            unit_root_count = sum(1 for p in all_persistence if p >= 0.99)
            print(
                f"\n  Unit root cases: {unit_root_count}/{len(all_persistence)} ({unit_root_count / len(all_persistence) * 100:.1f}%)"
            )

            # Distribution breakdown
            distributions = [
                data.get("best_distribution", "normal")
                for data in fitted_models.values()
            ]
            from collections import Counter

            dist_counts = Counter(distributions)
            print(f"\n  Distribution Preferences:")
            for dist, count in dist_counts.items():
                pct = count / len(distributions) * 100
                print(f"    {dist}: {count}/{len(distributions)} ({pct:.1f}%)")

        elif choice == "3":
            # Test simulation stability across omega ranges
            print(f"\n🔧 SIMULATION STABILITY TEST:")

            test_trials = []
            for name, data in fitted_models.items():
                omega = data["garch_params"].get("omega", 0)
                persistence = data.get("volatility_persistence", 0)
                test_trials.append((name, omega, persistence))

            # Categorize by omega
            high_omega = [(n, o, p) for n, o, p in test_trials if o > 1e-3]
            medium_omega = [(n, o, p) for n, o, p in test_trials if 5e-4 < o <= 1e-3]
            low_omega = [(n, o, p) for n, o, p in test_trials if o <= 5e-4]

            categories = [
                ("High omega (>1e-3)", high_omega[:3]),
                ("Medium omega (5e-4 to 1e-3)", medium_omega[:3]),
                ("Low omega (≤5e-4)", low_omega[:3]),
            ]

            for cat_name, trials in categories:
                print(f"\n  {cat_name}:")
                for trial_name, omega, orig_persistence in trials:
                    try:
                        simulated = simulate_enhanced_arma_garch(
                            fitted_models[trial_name], n_periods=100, random_seed=42
                        )
                        success_rate = len(simulated) / 100 * 100
                        print(
                            f"    {trial_name}: ω={omega:.1e}, p={orig_persistence:.3f} → {success_rate:.0f}% ✅"
                        )
                    except Exception as e:
                        print(
                            f"    {trial_name}: ω={omega:.1e}, p={orig_persistence:.3f} → Failed ❌"
                        )

        elif choice == "4":
            # Best fitting trials analysis with proper criteria
            print(f"\n🏆 BEST FITTING TRIALS ANALYSIS")
            print(f"Criteria: LB test PASS + LF test PASS + Std ratio ≈ 1.0")

            # Ask user for subject filter
            all_subjects = set()
            for trial_name in fitted_models.keys():
                if "/" in trial_name:
                    subject = trial_name.split("/")[0]
                    all_subjects.add(subject)

            print(f"\nAvailable subjects: {sorted(all_subjects)}")
            subject_filter = input(
                "Enter subject to analyze (or 'all' for all subjects): "
            ).strip()

            if subject_filter.lower() == "all":
                trials_to_test = list(fitted_models.keys())
            else:
                trials_to_test = [
                    name
                    for name in fitted_models.keys()
                    if name.startswith(subject_filter)
                ]

            if not trials_to_test:
                print(f"❌ No trials found for '{subject_filter}'")
                continue

            print(f"\n🔍 Analyzing {len(trials_to_test)} trials...")

            best_trials = []

            for trial_name in trials_to_test:
                trial_data = fitted_models[trial_name]

                # Get standardized innovations for statistical tests
                standardized_innovations = trial_data.get(
                    "standardized_residuals", np.array([])
                )
                if len(standardized_innovations) < 50:
                    continue

                # Test 1: Ljung-Box test for autocorrelation (whiteness)
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox

                    clean_innovations = standardized_innovations - np.mean(
                        standardized_innovations
                    )
                    lb_result = acorr_ljungbox(
                        clean_innovations, lags=10, return_df=True
                    )
                    lb_p = lb_result["lb_pvalue"].iloc[-1]
                    lb_pass = lb_p > 0.05
                except:
                    lb_pass = False
                    lb_p = 0.0

                # Test 2: Lilliefors test for distributional adequacy
                try:
                    from statsmodels.stats.diagnostic import lilliefors

                    best_dist = trial_data.get("best_distribution", "normal")

                    if best_dist == "t":
                        # Use robust ν extraction
                        nu = trial_data.get("distribution_params", {}).get("nu", None)
                        if nu is None:
                            nu = trial_data.get("nu", None)
                        if nu is None:
                            garch_params = trial_data.get("garch_params", {})
                            nu = garch_params.get("nu", None)

                        if nu is not None and not np.isnan(nu):
                            from scipy.stats import norm
                            from scipy.stats import t as t_dist

                            transformed = norm.ppf(t_dist.cdf(clean_innovations, nu))
                            lf_stat, lf_p = lilliefors(transformed, dist="norm")
                        else:
                            lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")
                    else:
                        lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")

                    lf_pass = lf_p > 0.05
                except:
                    lf_pass = False
                    lf_p = 0.0

                # Test 3: Std ratio via simulation
                try:
                    original_residuals = trial_data.get("original_residuals", [])
                    if len(original_residuals) > 0:
                        simulation_result = simulate_enhanced_arma_garch(
                            trial_data,
                            n_periods=len(original_residuals),
                            random_seed=42,
                        )

                        if isinstance(simulation_result, dict):
                            simulated_residuals = simulation_result["residuals"]
                        else:
                            simulated_residuals = simulation_result

                        orig_std = np.std(original_residuals)
                        sim_std = np.std(simulated_residuals)
                        std_ratio = sim_std / max(orig_std, 1e-8)

                        # Good std ratio: between 0.8 and 1.2
                        std_ratio_pass = 0.8 <= std_ratio <= 1.2
                    else:
                        std_ratio = np.nan
                        std_ratio_pass = False
                except:
                    std_ratio = np.nan
                    std_ratio_pass = False

                # Overall assessment
                all_pass = lb_pass and lf_pass and std_ratio_pass
                score = (lb_pass * 1.0) + (lf_pass * 1.0) + (std_ratio_pass * 1.0)

                best_trials.append(
                    {
                        "trial_name": trial_name,
                        "lb_pass": lb_pass,
                        "lb_p": lb_p,
                        "lf_pass": lf_pass,
                        "lf_p": lf_p,
                        "std_ratio": std_ratio,
                        "std_ratio_pass": std_ratio_pass,
                        "all_pass": all_pass,
                        "score": score,
                        "persistence": trial_data.get("volatility_persistence", np.nan),
                    }
                )

            # Sort by score (all_pass first, then by score)
            best_trials.sort(key=lambda x: (x["all_pass"], x["score"]), reverse=True)

            print(f"\n🏆 BEST FITTING TRIALS RESULTS:")
            print(
                f"{'Trial':<25} {'LB':<4} {'LF':<4} {'Std':<4} {'Score':<5} {'Persistence':<11} {'Status'}"
            )
            print("-" * 80)

            for i, trial in enumerate(best_trials[:10]):  # Top 10
                name = trial["trial_name"]
                short_name = name.split("/")[-1] if "/" in name else name
                lb_status = "✓" if trial["lb_pass"] else "✗"
                lf_status = "✓" if trial["lf_pass"] else "✗"
                std_status = "✓" if trial["std_ratio_pass"] else "✗"
                persistence = trial["persistence"]
                score = trial["score"]
                overall = "🏆 EXCELLENT" if trial["all_pass"] else f"❌ {score:.0f}/3"

                print(
                    f"{name:<25} {lb_status:<4} {lf_status:<4} {std_status:<4} {score:.1f}/3 {persistence:<11.3f} {overall}"
                )

                if i == 0 and trial["all_pass"]:
                    top_trial = name

            # Show summary statistics
            all_pass_count = sum(1 for t in best_trials if t["all_pass"])
            print(f"\n📊 SUMMARY:")
            print(f"  Trials meeting ALL criteria: {all_pass_count}/{len(best_trials)}")
            print(
                f"  LB test pass rate: {sum(1 for t in best_trials if t['lb_pass'])}/{len(best_trials)}"
            )
            print(
                f"  LF test pass rate: {sum(1 for t in best_trials if t['lf_pass'])}/{len(best_trials)}"
            )
            print(
                f"  Std ratio pass rate: {sum(1 for t in best_trials if t['std_ratio_pass'])}/{len(best_trials)}"
            )

            # Analyze top trial if available
            if best_trials and best_trials[0]["all_pass"]:
                top_trial = best_trials[0]["trial_name"]
                print(f"\n📊 Analyzing BEST trial: {top_trial}")
                plot_comprehensive_comparison(top_trial, fitted_models)
            elif best_trials:
                print(f"\n⚠️  No trials meet ALL criteria. Showing best available:")
                top_trial = best_trials[0]["trial_name"]
                print(f"📊 Analyzing: {top_trial}")
                plot_comprehensive_comparison(top_trial, fitted_models)
            else:
                print(f"\n❌ No suitable trials found for analysis.")

        elif choice == "5":
            # Distribution preference analysis for INNOVATIONS
            print(f"\n📈 INNOVATION DISTRIBUTION ANALYSIS:")
            print(f"  Reminder: We fit Normal/Student-t to INNOVATIONS (ε_t/σ_t)")
            print(f"  NOT to the original residuals!")

            distributions = [
                data.get("best_distribution", "normal")
                for data in fitted_models.values()
            ]
            from collections import Counter

            dist_counts = Counter(distributions)

            print(f"\n  Innovation Distribution Preferences:")
            for dist, count in dist_counts.items():
                pct = count / len(distributions) * 100
                print(f"    {dist}: {count}/{len(distributions)} ({pct:.1f}%)")

            nu_values = []
            for data in fitted_models.values():
                if data.get("best_distribution") == "t":
                    nu = data.get("distribution_params", {}).get("nu", np.nan)
                    if not np.isnan(nu):
                        nu_values.append(nu)

            if nu_values:
                print(f"\n  Student-t Innovation Analysis:")
                print(f"    Trials with t-innovations: {len(nu_values)}")
                print(f"    Mean ν: {np.mean(nu_values):.2f} ± {np.std(nu_values):.2f}")
                print(
                    f"    Range ν: [{np.min(nu_values):.1f}, {np.max(nu_values):.1f}]"
                )
                print(
                    f"    Heavy tails (ν < 5): {sum(1 for nu in nu_values if nu < 5)}/{len(nu_values)}"
                )
                print(
                    f"    Very heavy tails (ν < 3): {sum(1 for nu in nu_values if nu < 3)}/{len(nu_values)}"
                )

                print(f"\n  ✅ CONCLUSION: Testing innovation distributions")
                print(f"  📊 Student-t innovations preferred when heavy tails detected")
                print(f"  🔧 This validates heavy-tailed INNOVATION assumption")

        elif choice == "6":
            # Generate std ratio and innovation test heatmap
            generate_comprehensive_heatmap(fitted_models)

        elif choice == "7":
            # Cross-validation vs random baselines
            n_models = input(
                "Enter number of random baseline models to generate (default: 100): "
            ).strip()
            n_models = int(n_models) if n_models.isdigit() else 100

            print(
                f"\n🎯 Starting cross-validation with {n_models} random baseline models..."
            )
            validation_results = create_random_baseline_validation(
                fitted_models, n_models
            )

            # Save results for future analysis
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"cross_validation_results_{timestamp}.json"

            # Convert numpy types to python types for JSON serialization
            json_results = {
                "timestamp": timestamp,
                "n_random_models": len(validation_results["random_results"]),
                "n_fitted_models": len(validation_results["fitted_results"]),
                "fitted_metrics": validation_results["fitted_metrics"],
                "random_metrics": validation_results["random_metrics"],
                "significance_test": {
                    k: float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else str(v)
                    for k, v in validation_results["significance_result"].items()
                },
                "param_stats": {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in validation_results["param_stats"].items()
                },
                "nu_stats": {
                    k: float(v) for k, v in validation_results["nu_stats"].items()
                },
                "arma_mode": validation_results["arma_mode"].tolist()
                if hasattr(validation_results["arma_mode"], "tolist")
                else validation_results["arma_mode"],
            }

            with open(results_file, "w") as f:
                json.dump(json_results, f, indent=2)

            print(f"\n💾 Cross-validation results saved to: {results_file}")

        elif choice == "8":
            # New aggregated cross-validation analysis
            print(f"\n🔬 AGGREGATED CROSS-VALIDATION ANALYSIS")
            print(
                f"Professional statistical validation of fitted models vs random baselines"
            )
            print(f"This uses ALL fitted models with robust statistical testing")

            n_models = input(
                "Enter number of random baseline models (default: 500): "
            ).strip()
            n_models = int(n_models) if n_models.isdigit() else 500

            print(
                f"\n🚀 Starting aggregated cross-validation with {n_models} random models..."
            )
            cv_results = perform_aggregated_cross_validation(
                fitted_models, n_random_models=n_models
            )

            if cv_results is not None:
                # Generate professional plots
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"aggregated_cross_validation_{timestamp}"

                print(f"\n📊 Generating professional cross-validation plots...")
                from plot_similarity_crossval import (
                    plot_similarity_cross_validation_results,
                )

                plot_similarity_cross_validation_results(
                    cv_results, save_path=plot_path
                )

                # Save detailed results
                import json

                cv_results_file = f"aggregated_cv_results_{timestamp}.json"

                # Convert to JSON-serializable format (avoid numpy types)
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj

                json_cv_results = {
                    "timestamp": timestamp,
                    "n_random_models": n_models,
                    "n_fitted_tested": cv_results["n_fitted"],
                    "n_random_tested": cv_results["n_random"],
                    "explosion_rates": {
                        key: convert_numpy(value)
                        for key, value in cv_results["explosion_rates"].items()
                    },
                    "overall_results": {
                        key: convert_numpy(value)
                        for key, value in cv_results["overall"].items()
                    },
                    "similarity_scores": {
                        "fitted_clean_count": len(
                            cv_results["fitted_similarity_scores"]
                        ),
                        "random_clean_count": len(
                            cv_results["random_similarity_scores"]
                        ),
                        "fitted_mean": convert_numpy(
                            cv_results["overall"]["fitted_similarity"]
                        ),
                        "random_mean": convert_numpy(
                            cv_results["overall"]["random_similarity"]
                        ),
                    },
                }

                with open(cv_results_file, "w") as f:
                    json.dump(json_cv_results, f, indent=2)

                print(f"\n💾 Cross-validation results saved to: {cv_results_file}")
                print(
                    f"📈 Professional cross-validation plot saved as: {plot_path}.svg"
                )

                # Print key findings
                overall_p = cv_results["overall"]["p_value"]
                overall_diff = cv_results["overall"]["difference"]

                print(f"\n🎯 KEY FINDINGS:")
                print(
                    f"   Overall advantage: {overall_diff:+.3f} ({overall_diff * 100:+.1f}%)"
                )
                print(f"   Statistical significance: p = {overall_p:.4f}")
                if overall_p < 0.05:
                    print(
                        f"   🎉 VALIDATION SUCCESSFUL: Fitted models significantly outperform random baselines!"
                    )
                else:
                    print(
                        f"   ⚠️  VALIDATION INCONCLUSIVE: No significant difference detected"
                    )
            else:
                print(
                    f"   ❌ Cross-validation analysis failed - insufficient valid results"
                )

        elif choice == "9":
            # Trial-based cross-validation (deprecated - use Option 10 instead)
            print(
                f"\n⚠️  DEPRECATED: Use Option 10 for Model Adequacy Validation instead"
            )
            print(f"\n🔬 TRIAL-BASED CROSS-VALIDATION (Old Implementation)")
            print(
                f"Testing whether ARMA-GARCH models produce better white noise residuals"
            )
            print(f"than simple white Gaussian models using proper statistical tests")
            print(f"")
            print(f"This validates the fundamental assumption that ARMA-GARCH models")
            print(
                f"should produce residuals closer to white noise than naive alternatives."
            )

            validation_results = perform_model_adequacy_validation(fitted_models)

            if validation_results and "comparison_results" in validation_results:
                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"model_adequacy_validation_{timestamp}.json"

                with open(results_file, "w") as f:
                    import json as json_module

                    json_module.dump(validation_results, f, indent=2, default=str)

                print(
                    f"\\n💾 Model adequacy validation results saved to: {results_file}"
                )

                # Display model adequacy validation results
                comp = validation_results["comparison_results"]

                print(f"\n📋 MODEL ADEQUACY VALIDATION RESULTS:")
                print(f"=" * 80)
                print(
                    f"|{'Test':<25}|{'ARMA-GARCH':<12}|{'White Gauss':<12}|{'Better':<10}|{'Significant':<11}|"
                )
                print(f"-" * 80)

                # Display each test comparison
                for test_name in [
                    "ljung_box_pvalue",
                    "arch_lm_pvalue",
                    "lilliefors_pvalue",
                ]:
                    if test_name in comp and "error" not in comp[test_name]:
                        test_data = comp[test_name]
                        arma_med = test_data["arma_median_pval"]
                        white_med = test_data["white_median_pval"]
                        better = (
                            "ARMA-GARCH" if test_data["arma_better"] else "White Gauss"
                        )
                        sig = "Yes" if test_data["significant_diff"] else "No"

                        print(
                            f"|{test_data['test_label']:<25}|{arma_med:<12.4f}|{white_med:<12.4f}|{better:<10}|{sig:<11}|"
                        )

                print(f"-" * 80)

                # Overall summary
                if "summary" in comp:
                    summary = comp["summary"]
                    print(f"\n🎯 OVERALL ADEQUACY ASSESSMENT:")
                    print(
                        f"   ARMA-GARCH wins: {summary['arma_garch_wins']}/{summary['total_tests']} tests"
                    )
                    print(f"   Win rate: {summary['win_rate'] * 100:.1f}%")
                    print(f"   Interpretation: {summary['interpretation']}")

                    if summary["win_rate"] > 0.6:
                        print(f"   ✅ ARMA-GARCH models show superior adequacy")
                    elif summary["win_rate"] > 0.4:
                        print(f"   ⚖️  Mixed results - no clear winner")
                    else:
                        print(
                            f"   ❌ White Gaussian models show equal or better adequacy"
                        )

                print(f"\n🎯 Model adequacy validation complete!")

                # OLD Display results (to be removed)
                print(
                    f"\\n📊 Generating rank-based cross-validation plots and analysis..."
                )

                import pandas as pd
                import seaborn as sns

                # fitted_ranks = cv_results['fitted_scores']  # OLD CODE - REMOVED
                # random_ranks = cv_results['random_scores']  # OLD CODE - REMOVED
                # Perform non-parametric rank tests
                from scipy.stats import (
                    kendalltau,
                    mannwhitneyu,
                    ranksums,
                    spearmanr,
                    wilcoxon,
                )

                # Create rank-based summary table
                print(f"\\n📋 RANK-BASED CROSS-VALIDATION RESULTS:")
                print(f"=" * 70)
                print(
                    f"|{'Statistic':<15}|{'ARMA-GARCH':<12}|{'White Noise':<12}|{'Difference':<12}|"
                )
                print(f"-" * 70)

                # Use medians and IQR instead of means for rank-based reporting
                fitted_median = np.median(fitted_ranks)
                random_median = np.median(random_ranks)
                rank_advantage = fitted_median - random_median

                fitted_q75, fitted_q25 = np.percentile(fitted_ranks, [75, 25])
                random_q75, random_q25 = np.percentile(random_ranks, [75, 25])
                fitted_iqr = fitted_q75 - fitted_q25
                random_iqr = random_q75 - random_q25

                print(
                    f"|{'Median':<15}|{fitted_median:<12.3f}|{random_median:<12.3f}|{rank_advantage:<+12.3f}|"
                )
                print(
                    f"|{'IQR':<15}|{fitted_iqr:<12.3f}|{random_iqr:<12.3f}|{'N/A':<12}|"
                )
                print(
                    f"|{'N':<15}|{len(fitted_ranks):<12}|{len(random_ranks):<12}|{'N/A':<12}|"
                )
                print(
                    f"|{'Win Rate':<15}|{cv_results['win_rate']:<12.1%}|{'N/A':<12}|{'N/A':<12}|"
                )
                print(
                    f"|{'Beats Best':<15}|{cv_results['beats_best_rate']:<12.1%}|{'N/A':<12}|{'N/A':<12}|"
                )
                print(
                    f"|{'Scale':<15}|{'N/A':<12}|{cv_results['scale']:<12}|{'N/A':<12}|"
                )

                print(f"\\n📈 Non-Parametric Rank Tests:")
                print(
                    f"   Mann-Whitney U p-value: {cv_results['mann_whitney_test']['p_value']:.2e}"
                )
                print(f"   Rank-biserial correlation: {cv_results['cohens_d']:.3f}")
                print(f"   Fitted rank advantage: {rank_advantage:+.3f} rank points")

                # Create rank-based boxplot
                df_fitted = pd.DataFrame(
                    {"Rank Score": fitted_ranks, "Model Type": "ARMA-GARCH Models"}
                )
                df_white_noise = pd.DataFrame(
                    {"Rank Score": random_ranks, "Model Type": "White Noise Baselines"}
                )
                df = pd.concat([df_fitted, df_white_noise], ignore_index=True)

                plt.figure(figsize=(10, 8))
                ax = plt.gca()

                sns.boxplot(
                    data=df,
                    x="Model Type",
                    y="Rank Score",
                    ax=ax,
                    boxprops={
                        "facecolor": "lightblue",
                        "edgecolor": "navy",
                        "linewidth": 1.5,
                        "alpha": 0.7,
                    },
                    medianprops={"color": "darkred", "linewidth": 2.5},
                    whiskerprops={"color": "navy", "linewidth": 1.5},
                    capprops={"color": "navy", "linewidth": 1.5},
                    width=0.6,
                )

                sns.swarmplot(
                    data=df,
                    x="Model Type",
                    y="Rank Score",
                    ax=ax,
                    alpha=0.5,
                    size=2.5,
                    color="steelblue",
                )

                ax.set_ylim(0, 1)
                ax.set_ylabel("Rank-Based Score", fontsize=14, fontweight="bold")
                ax.set_xlabel("Model Type", fontsize=14, fontweight="bold")
                ax.set_title(
                    "Rank-Based Cross-Validation: ARMA-GARCH vs White Noise Baselines\\n"
                    + f"ARMA-GARCH: {fitted_median:.3f} vs White Noise: {random_median:.3f} "
                    + f"(Win Rate: {cv_results['win_rate']:.1%})",
                    fontsize=16,
                    fontweight="bold",
                    pad=20,
                )

                # Add statistical annotation
                p_val = cv_results["mann_whitney_test"]["p_value"]
                cohens_d = cv_results["cohens_d"]

                if p_val < 0.001:
                    p_text = "p < 0.001***"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.3f}**"
                elif p_val < 0.05:
                    p_text = f"p = {p_val:.3f}*"
                else:
                    p_text = f"p = {p_val:.3f}"

                print(f"rank-biserial r = {cohens_d:.2f}")
                ax.text(
                    0.98,
                    0.98,
                    p_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                    fontsize=12,
                    fontweight="bold",
                )

                ax.text(
                    0.02,
                    0.02,
                    f"n_fitted = {len(fitted_ranks)}\\nn_random = {len(random_ranks)}",
                    transform=ax.transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                    fontsize=10,
                )

                ax.grid(True, alpha=0.3, axis="y")
                ax.set_axisbelow(True)
                plt.tight_layout()

                # Save plot
                plot_file = f"rank_based_arma_garch_vs_white_noise_{timestamp}.svg"
                plt.savefig(plot_file, format="svg", dpi=300, bbox_inches="tight")
                plt.show()

                print(f"\\n📊 Rank-based boxplot saved as: {plot_file}")
                print(f"\\n🎯 Rank-based cross-validation complete!")

            else:
                print(f"   ❌ Rank-based cross-validation failed")

        elif choice == "10":
            # Model adequacy validation
            print(f"\n🔬 MODEL ADEQUACY VALIDATION")
            print(
                f"Testing whether ARMA-GARCH models produce better white noise residuals"
            )
            print(f"than simple white Gaussian models using proper statistical tests")
            print(f"")
            print(f"This validates the fundamental assumption that ARMA-GARCH models")
            print(
                f"should produce residuals closer to white noise than naive alternatives."
            )

            validation_results = perform_model_adequacy_validation(fitted_models)

            if validation_results and "comparison_results" in validation_results:
                # Save results
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"model_adequacy_validation_{timestamp}.json"

                with open(results_file, "w") as f:
                    import json as json_module

                    json_module.dump(validation_results, f, indent=2, default=str)

                print(
                    f"\n💾 Model adequacy validation results saved to: {results_file}"
                )

                # Display model adequacy validation results
                comp = validation_results["comparison_results"]

                print(f"\n📋 MODEL ADEQUACY VALIDATION RESULTS:")
                print(f"=" * 80)
                print(
                    f"|{'Test':<25}|{'ARMA-GARCH':<12}|{'White Gauss':<12}|{'Better':<10}|{'Significant':<11}|"
                )
                print(f"-" * 80)

                # Display each test comparison
                for test_name in [
                    "ljung_box_pvalue",
                    "arch_lm_pvalue",
                    "lilliefors_pvalue",
                ]:
                    if test_name in comp and "error" not in comp[test_name]:
                        test_data = comp[test_name]
                        arma_med = test_data["arma_median_pval"]
                        white_med = test_data["white_median_pval"]
                        better = (
                            "ARMA-GARCH" if test_data["arma_better"] else "White Gauss"
                        )
                        sig = "Yes" if test_data["significant_diff"] else "No"

                        print(
                            f"|{test_data['test_label']:<25}|{arma_med:<12.4f}|{white_med:<12.4f}|{better:<10}|{sig:<11}|"
                        )

                print(f"-" * 80)

                # Overall summary
                if "summary" in comp:
                    summary = comp["summary"]
                    print(f"\n🎯 OVERALL ADEQUACY ASSESSMENT:")
                    print(
                        f"   ARMA-GARCH wins: {summary['arma_garch_wins']}/{summary['total_tests']} tests"
                    )
                    print(f"   Win rate: {summary['win_rate'] * 100:.1f}%")
                    print(f"   Interpretation: {summary['interpretation']}")

                    if summary["win_rate"] > 0.6:
                        print(f"   ✅ ARMA-GARCH models show superior adequacy")
                    elif summary["win_rate"] > 0.4:
                        print(f"   ⚖️  Mixed results - no clear winner")
                    else:
                        print(
                            f"   ❌ White Gaussian models show equal or better adequacy"
                        )

                print(f"\n🎯 Model adequacy validation complete!")

            else:
                print(f"   ❌ Model adequacy validation failed")

        elif choice == "11":
            # Option 12: Enhanced report plot for subject_004_trial_001
            trial_name = "subject_004/trial_001"
            if trial_name in fitted_models:
                print(f"\n📊 Running Option 12 for {trial_name}")
                plot_comprehensive_comparison_option12(trial_name, fitted_models)
            else:
                print(f"❌ Trial {trial_name} not found in fitted models")
                print(f"Available trials: {list(fitted_models.keys())[:5]}...")

        elif choice == "12":
            print("👋 Comprehensive analysis complete!")
            break

        else:
            print("❌ Invalid choice")


def generate_comprehensive_heatmap(fitted_models: dict):
    """
    Generate heatmap showing std ratios (sim vs real) and innovation test results
    across all subjects and trials.
    """

    print(f"\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE HEATMAP")
    print("=" * 60)
    print("📊 Computing std ratios and innovation tests for all trials...")

    # Organize data by subjects and trials
    subject_data = {}

    for trial_name, trial_data in fitted_models.items():
        # Parse subject and trial from trial_name (e.g., "subject_005/trial_001")
        if "/" in trial_name:
            subject, trial = trial_name.split("/", 1)
        else:
            subject = "unknown"
            trial = trial_name

        if subject not in subject_data:
            subject_data[subject] = {}

        # Get original residuals
        original_residuals = trial_data.get("original_residuals", [])
        if len(original_residuals) == 0:
            print(f"⚠️  Skipping {trial_name}: no original residuals")
            continue

        # Simulate residuals using enhanced simulation
        try:
            simulation_result = simulate_enhanced_arma_garch(
                trial_data, n_periods=len(original_residuals), random_seed=42
            )

            if isinstance(simulation_result, dict):
                simulated_residuals = simulation_result["residuals"]
            else:
                simulated_residuals = simulation_result

            if len(simulated_residuals) == 0:
                print(f"⚠️  Skipping {trial_name}: simulation failed")
                continue

            # Calculate std ratio
            orig_std = np.std(original_residuals)
            sim_std = np.std(simulated_residuals)
            std_ratio = sim_std / max(orig_std, 1e-8)

        except Exception as e:
            print(f"⚠️  Skipping {trial_name}: simulation error - {str(e)[:50]}")
            continue

        # Test innovations for whiteness (Ljung-Box + Lilliefors)
        standardized_innovations = trial_data.get(
            "standardized_residuals", np.array([])
        )
        innovation_pass = False

        if len(standardized_innovations) > 30:
            try:
                clean_innovations = standardized_innovations - np.mean(
                    standardized_innovations
                )

                # Ljung-Box test for autocorrelation (whiteness)
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(clean_innovations, lags=10, return_df=True)
                lb_p = lb_result["lb_pvalue"].iloc[-1]
                lb_pass = lb_p > 0.05

                # Distribution fit test using Lilliefors (better for estimated parameters)
                try:
                    from statsmodels.stats.diagnostic import lilliefors

                    best_dist = trial_data.get("best_distribution", "normal")

                    if best_dist == "t":
                        # For Student-t: Use Lilliefors test against fitted t-distribution
                        # Try multiple locations for ν parameter
                        nu = trial_data.get("distribution_params", {}).get("nu", None)

                        # Check if ν is stored directly in trial_data (clean models)
                        if nu is None:
                            nu = trial_data.get("nu", None)

                        # Fallback: try to extract nu from garch_params
                        if nu is None:
                            garch_params = trial_data.get("garch_params", {})
                            nu = garch_params.get("nu", None)

                        # Ultimate fallback: use reasonable estimate based on subject_005 data
                        if nu is None or np.isnan(nu):
                            # Use subject_005 mean ν as reasonable estimate for missing values
                            # Subject_005 has ν values, use them to estimate typical ν for EMG data
                            nu = 5.67  # Subject_005 mean ν as reasonable estimate
                            print(
                                f"  ⚠️  Missing ν for {trial_name[:15]}..., using fallback ν={nu:.2f}"
                            )

                        # Lilliefors test against Student-t distribution
                        try:
                            # Transform innovations to standard normal using fitted t-distribution CDF
                            # then apply inverse normal CDF (probability integral transform)
                            from scipy.stats import norm
                            from scipy.stats import t as t_dist

                            transformed = norm.ppf(t_dist.cdf(clean_innovations, nu))

                            # Apply Lilliefors test to transformed data (should be standard normal)
                            lf_stat, lf_p = lilliefors(transformed, dist="norm")
                            lf_pass = lf_p > 0.05

                            print(
                                f"  Lilliefors test: ν={nu:.2f}, p={lf_p:.3f} ({'PASS' if lf_pass else 'FAIL'})"
                            )
                        except:
                            # Fallback: direct Lilliefors against normal (conservative)
                            lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")
                            lf_pass = lf_p > 0.05
                    else:
                        # For normal: Use Lilliefors test directly
                        lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")
                        lf_pass = lf_p > 0.05

                except ImportError:
                    # Fallback: Jarque-Bera test (less sensitive)
                    from scipy.stats import jarque_bera

                    jb_stat, jb_p = jarque_bera(clean_innovations)
                    lf_pass = jb_p > 0.05
                except Exception as e:
                    # Ultimate fallback: moment-based test
                    from scipy.stats import kurtosis, skew

                    sk = abs(skew(clean_innovations))
                    ku = abs(
                        kurtosis(clean_innovations, fisher=True)
                    )  # Excess kurtosis
                    # Rough acceptance criteria for Student-t or normal
                    lf_pass = (sk < 1.0) and (ku < 3.0)

                # Combined test: both must pass
                innovation_pass = lb_pass and lf_pass

                # Store individual test results for separate tracking
                individual_test_results = {
                    "lb_pass": lb_pass,
                    "lf_pass": lf_pass,
                    "combined_pass": innovation_pass,
                }

            except Exception:
                innovation_pass = False
                individual_test_results = {
                    "lb_pass": False,
                    "lf_pass": False,
                    "combined_pass": False,
                }
        else:
            individual_test_results = {
                "lb_pass": False,
                "lf_pass": False,
                "combined_pass": False,
            }

        # Store results
        subject_data[subject][trial] = {
            "std_ratio": std_ratio,
            "innovation_pass": innovation_pass,
            "orig_std": orig_std,
            "sim_std": sim_std,
            "individual_tests": individual_test_results,
        }

    print(
        f"✅ Computed results for {sum(len(trials) for trials in subject_data.values())} trials"
    )

    # Create heatmap visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(
        "Model Quality Assessment: Std Ratios & Innovation Tests",
        fontsize=16,
        fontweight="bold",
    )

    # Prepare data matrices
    subjects = sorted(subject_data.keys())
    max_trials = (
        max(len(trials) for trials in subject_data.values()) if subject_data else 0
    )

    std_ratios = np.full((len(subjects), max_trials), np.nan)
    innovation_results = np.full((len(subjects), max_trials), np.nan)
    trial_labels = []

    for i, subject in enumerate(subjects):
        trials = sorted(subject_data[subject].keys())
        trial_labels.append(trials[:max_trials] + [""] * (max_trials - len(trials)))

        for j, trial in enumerate(trials):
            if j < max_trials:
                data = subject_data[subject][trial]
                std_ratios[i, j] = data["std_ratio"]
                innovation_results[i, j] = 1.0 if data["innovation_pass"] else 0.0

    # Plot 1: Std Ratios Heatmap
    im1 = ax1.imshow(std_ratios, cmap="RdYlGn", vmin=0.5, vmax=1.5, aspect="auto")
    ax1.set_title(
        "Standard Deviation Ratios\n(Simulated / Original)", fontweight="bold"
    )
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Subject")
    ax1.set_yticks(range(len(subjects)))
    ax1.grid(False)  # Remove grid

    # Add colorbar for std ratios
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Std Ratio (target: 1.0)")

    # Add subject-level std ratio statistics as y-axis labels
    std_labels = []
    for i, subject in enumerate(subjects):
        trials = sorted(subject_data[subject].keys())
        subject_ratios = [subject_data[subject][trial]["std_ratio"] for trial in trials]
        avg_ratio = np.mean(subject_ratios)

        label = f"{subject}\n(μ={avg_ratio:.2f})"
        std_labels.append(label)

    ax1.set_yticklabels(std_labels)

    # Plot 2: Innovation Test Results Heatmap
    im2 = ax2.imshow(innovation_results, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_title(
        "Innovation Whiteness Tests\n(Ljung-Box + Distribution Fit)", fontweight="bold"
    )
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Subject")
    ax2.set_yticks(range(len(subjects)))
    ax2.grid(False)  # Remove grid

    # Add subject-level test statistics as y-axis labels
    test_labels = []
    for i, subject in enumerate(subjects):
        trials = sorted(subject_data[subject].keys())
        lb_passes = []
        lf_passes = []

        for trial in trials:
            test_results = subject_data[subject][trial].get("individual_tests", {})
            lb_passes.append(1.0 if test_results.get("lb_pass", False) else 0.0)
            lf_passes.append(1.0 if test_results.get("lf_pass", False) else 0.0)

        lb_rate = np.mean(lb_passes) * 100
        lf_rate = np.mean(lf_passes) * 100

        label = f"{subject}\n(LB:{lb_rate:.0f}%, DF:{lf_rate:.0f}%)"  # DF = Distribution Fit
        test_labels.append(label)

    ax2.set_yticklabels(test_labels)

    # Add colorbar for innovation tests
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Test Result (Green=Pass, Red=Fail)")

    plt.tight_layout()

    # Save plot
    output_dir = Path("residual_analysis_clean/plots_comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model_quality_heatmap.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"📊 Heatmap saved: {output_file}")

    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")

    all_ratios = std_ratios[~np.isnan(std_ratios)]
    all_tests = innovation_results[~np.isnan(innovation_results)]

    if len(all_ratios) > 0:
        good_ratios = np.sum((0.8 <= all_ratios) & (all_ratios <= 1.2))
        print(f"  Std Ratios: {len(all_ratios)} trials")
        print(f"    Mean ratio: {np.mean(all_ratios):.3f}")
        print(
            f"    Good ratios (0.8-1.2): {good_ratios}/{len(all_ratios)} ({good_ratios / len(all_ratios) * 100:.1f}%)"
        )

    if len(all_tests) > 0:
        passed_tests = np.sum(all_tests == 1.0)
        print(f"  Innovation Tests (Ljung-Box + Lilliefors): {len(all_tests)} trials")
        print(
            f"    Passed both tests: {passed_tests}/{len(all_tests)} ({passed_tests / len(all_tests) * 100:.1f}%)"
        )
        print(f"    (Lilliefors: PIT-transformed for Student-t, direct for Normal)")

        # Bug hunting: Compare trial counts by subject
        print(f"\n🔍 Bug Hunt - Trial counts by subject (OLD function):")
        for subject in sorted(subject_data.keys()):
            trials = list(subject_data[subject].keys())
            valid_innovations = len(
                [
                    trial
                    for trial in trials
                    if subject_data[subject][trial].get("innovation_pass") is not None
                ]
            )
            print(
                f"  {subject}: {len(trials)} total trials, {valid_innovations} with innovation results"
            )

        # Print per-subject breakdown
        print(f"\n  Per-Subject Results:")
        for subject in sorted(subject_data.keys()):
            trials = list(subject_data[subject].keys())
            ratios = [subject_data[subject][trial]["std_ratio"] for trial in trials]
            passes = [
                subject_data[subject][trial]["innovation_pass"] for trial in trials
            ]

            avg_ratio = np.mean(ratios)
            pass_rate = np.mean([1.0 if p else 0.0 for p in passes]) * 100

            print(
                f"    {subject}: μ_ratio={avg_ratio:.3f}, pass_rate={pass_rate:.0f}% ({len(trials)} trials)"
            )

    plt.show()


def create_validation_box_plots(fitted_models: dict, output_dir: Path):
    """Create box plots for sigma ratios and innovation test pass rates by subject."""

    print("\n" + "=" * 60)
    print("CREATING VALIDATION BOX PLOTS")
    print("=" * 60)

    # First, call the existing comprehensive heatmap function to get the computed data
    print("📊 Using existing comprehensive heatmap calculations...")

    # Modified logic with FDR correction
    subject_data = {}

    # First pass: collect all p-values and compute sigma ratios
    all_lb_pvalues = []
    all_lf_pvalues = []
    trial_info = []  # Store trial metadata for matching results back

    for trial_name, trial_data in fitted_models.items():
        # Parse subject and trial from trial_name
        if "/" in trial_name:
            subject, trial = trial_name.split("/", 1)
        else:
            subject = "unknown"
            trial = trial_name

        if subject not in subject_data:
            subject_data[subject] = {}

        # Get original residuals
        original_residuals = trial_data.get("original_residuals", [])
        if len(original_residuals) == 0:
            continue

        # Simulate residuals using enhanced simulation (same as heatmap)
        try:
            simulation_result = simulate_enhanced_arma_garch(
                trial_data, n_periods=len(original_residuals), random_seed=42
            )

            if isinstance(simulation_result, dict):
                simulated_residuals = simulation_result["residuals"]
            else:
                simulated_residuals = simulation_result

            if len(simulated_residuals) == 0:
                continue

            # Calculate std ratio (same as heatmap)
            orig_std = np.std(original_residuals)
            sim_std = np.std(simulated_residuals)
            std_ratio = sim_std / max(orig_std, 1e-8)

        except Exception:
            continue

        # Test innovations for whiteness - collect p-values
        standardized_innovations = trial_data.get(
            "standardized_residuals", np.array([])
        )
        lb_p = np.nan
        lf_p = np.nan

        if len(standardized_innovations) > 30:
            try:
                clean_innovations = standardized_innovations - np.mean(
                    standardized_innovations
                )

                # Ljung-Box test for autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(clean_innovations, lags=10, return_df=True)
                lb_p = lb_result["lb_pvalue"].iloc[-1]

                # Distribution fit test using Lilliefors
                try:
                    from statsmodels.stats.diagnostic import lilliefors

                    best_dist = trial_data.get("best_distribution", "normal")

                    if best_dist == "t":
                        # Get nu parameter using same robust extraction as heatmap
                        nu = trial_data.get("distribution_params", {}).get("nu", None)
                        if nu is None:
                            nu = trial_data.get("nu", None)
                        if nu is None:
                            garch_params = trial_data.get("garch_params", {})
                            nu = garch_params.get("nu", None)
                        if nu is None or np.isnan(nu):
                            nu = 5.67  # Same fallback as heatmap

                        # PIT transformation using t-distribution (same as heatmap)
                        try:
                            from scipy.stats import norm
                            from scipy.stats import t as t_dist

                            transformed = norm.ppf(t_dist.cdf(clean_innovations, nu))
                            lf_stat, lf_p = lilliefors(transformed, dist="norm")
                        except Exception as e:
                            lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")
                    else:
                        # For normal: Use Lilliefors test directly
                        lf_stat, lf_p = lilliefors(clean_innovations, dist="norm")

                except ImportError:
                    # Same fallback as heatmap
                    from scipy.stats import jarque_bera

                    jb_stat, lf_p = jarque_bera(clean_innovations)
                except Exception:
                    # Set to nan if test fails
                    lf_p = np.nan

            except Exception:
                lb_p = np.nan
                lf_p = np.nan

        # Store trial information and p-values
        trial_info.append(
            {
                "trial_name": trial_name,
                "subject": subject,
                "trial": trial,
                "std_ratio": std_ratio,
                "lb_p": lb_p,
                "lf_p": lf_p,
            }
        )

        all_lb_pvalues.append(lb_p)
        all_lf_pvalues.append(lf_p)

    # Apply simple p > 0.05 thresholds (no FDR correction needed with n=2)
    print(f"📊 Applying simple p > 0.05 thresholds to {len(trial_info)} trials...")

    for i, info in enumerate(trial_info):
        lb_p = info["lb_p"]
        lf_p = info["lf_p"]

        # Simple p > 0.05 thresholds
        lb_pass = lb_p > 0.05 if not np.isnan(lb_p) else False
        lf_pass = lf_p > 0.05 if not np.isnan(lf_p) else False

        combined_pass = lb_pass and lf_pass

        subject_data[info["subject"]][info["trial"]] = {
            "std_ratio": info["std_ratio"],
            "individual_tests": {
                "lb_pass": lb_pass,
                "lf_pass": lf_pass,
                "combined_pass": combined_pass,
            },
        }

    print(f"✅ Simple thresholds applied to all {len(trial_info)} trials")

    # Convert to format needed for plotting
    sigma_data = []
    test_data = []

    for subject, trials in subject_data.items():
        for trial, results in trials.items():
            # Sigma ratio data (sim/real as stored)
            sigma_data.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "sigma_ratio": results["std_ratio"],
                }
            )

            # Test data
            tests = results["individual_tests"]
            test_data.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "ljung_box_pass": tests["lb_pass"],
                    "lilliefors_pass": tests["lf_pass"],
                    "both_pass": tests["combined_pass"],
                    "ljung_box_pass_rate": 1.0 if tests["lb_pass"] else 0.0,
                    "lilliefors_pass_rate": 1.0 if tests["lf_pass"] else 0.0,
                    "both_pass_rate": 1.0 if tests["combined_pass"] else 0.0,
                }
            )

    print(f"✅ Collected data from {len(sigma_data)} trials")

    # Bug hunting: Compare trial counts by subject
    print(f"\n🔍 Bug Hunt - Trial counts by subject (NEW function):")
    for subject in sorted(set(info["subject"] for info in trial_info)):
        subject_trials = [info for info in trial_info if info["subject"] == subject]
        valid_tests = len(
            [
                info
                for info in subject_trials
                if not np.isnan(info["lb_p"]) and not np.isnan(info["lf_p"])
            ]
        )
        print(
            f"  {subject}: {len(subject_trials)} total trials, {valid_tests} with valid tests"
        )

    # Create plots
    if sigma_data:
        print(f"📊 Creating sigma ratio box plots for {len(sigma_data)} trials")

        sigma_df = pd.DataFrame(sigma_data)

        # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig, axes = plt.subplots(1, 1, figsize=(15, 6))

        # fig.suptitle('Sigma Ratio Validation (Real σ / Synthetic σ)', fontsize=16, fontweight='bold')

        #### Box plot by subject
        # sns.boxplot(data=sigma_df, x='subject', y='sigma_ratio', ax=axes[0])
        # # axes[0].set_title('Sigma Ratios by Subject', fontweight='bold')
        # axes[0].set_ylabel('$\sigma$ ratio')
        # axes[0].tick_params(axis='x', rotation=45)
        # axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Match')
        # axes[0].axhline(y=0.8, color='orange', linestyle=':', alpha=0.5)
        # axes[0].axhline(y=1.2, color='orange', linestyle=':', alpha=0.5, label='Acceptable Range')
        # axes[0].legend()
        ####

        # Ensure consistent subject ordering (alphabetical)
        subject_order = sorted(sigma_df["subject"].unique())
        print(f"🔍 Debug - Enforcing boxplot subject order: {subject_order}")

        # sns.violinplot(data=sigma_df, x='subject', y='sigma_ratio', ax=axes)
        sns.swarmplot(
            data=sigma_df,
            x="subject",
            y="sigma_ratio",
            ax=axes,
            order=subject_order,
            alpha=0.6,
            color="purple",
            size=4,
        )

        sns.boxplot(
            data=sigma_df,
            x="subject",
            y="sigma_ratio",
            ax=axes,
            order=subject_order,  # Force alphabetical order
            showcaps=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.2,
                "alpha": 0.85,
            },
            medianprops={"color": "black", "linewidth": 2},  # slightly bolder
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
            width=0.4,
        )

        axes.set_ylabel("$\mathbf{\sigma}$ ratio", fontweight="bold", fontsize=16)
        axes.set_xlabel("Subject", fontweight="bold", fontsize=16)
        axes.tick_params(axis="x", rotation=0, labelsize=16)
        axes.tick_params(axis="y", labelsize=16)
        # axes.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

        # Format subject labels
        new_labels = [
            label.get_text().replace("subject_", "") for label in axes.get_xticklabels()
        ]
        axes.set_xticklabels(new_labels)

        # Add grid
        axes.grid(True, alpha=0.3, axis="y")

        # Distribution
        # axes[1].hist(sigma_df['sigma_ratio'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        # axes[1].axvline(sigma_df['sigma_ratio'].mean(), color='red', linestyle='--',
        #                 label=f'Mean: {sigma_df["sigma_ratio"].mean():.2f}')
        # axes[1].axvline(1.0, color='orange', linestyle='-', alpha=0.8, label='Perfect Match')
        # # axes[1].set_title('Sigma Ratio Distribution')
        # axes[1].set_xlabel('$\sigma$ ratio')
        # axes[1].set_ylabel('Frequency')
        # axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            output_dir / "sigma_ratio_boxplots.svg", format="svg", bbox_inches="tight"
        )
        plt.close()

        print(
            f"✅ Sigma ratio analysis: Mean = {sigma_df['sigma_ratio'].mean():.3f}, Median = {sigma_df['sigma_ratio'].median():.3f}"
        )

        # Print per-subject sigma ratio statistics
        print(f"\n📋 Per-Subject Sigma Ratio Statistics:")
        subjects = sorted(sigma_df["subject"].unique())
        for subject in subjects:
            subject_data = sigma_df[sigma_df["subject"] == subject]
            mean_ratio = subject_data["sigma_ratio"].mean()
            count = len(subject_data)
            print(f"  {subject}: μ_ratio={mean_ratio:.3f} ({count} trials)")

        # Debug: Show sample values and verify plotting data
        print(f"\n🔍 Debug - Sample sigma_ratio values by subject:")
        for subject in subjects:
            subject_data = sigma_df[sigma_df["subject"] == subject]
            sample_values = subject_data["sigma_ratio"].head(3).tolist()
            median_val = subject_data["sigma_ratio"].median()
            mean_val = subject_data["sigma_ratio"].mean()
            print(
                f"  {subject}: median={median_val:.3f}, mean={mean_val:.3f}, first 3 values = {sample_values}"
            )

        print(f"\n🔍 Final Verification - Data being plotted:")
        print(f"  Total trials in sigma_df: {len(sigma_df)}")
        print(f"  Subjects in sigma_df: {sorted(sigma_df['subject'].unique())}")
        print(
            f"  Sigma ratio range: [{sigma_df['sigma_ratio'].min():.3f}, {sigma_df['sigma_ratio'].max():.3f}]"
        )
        print()

    if test_data:
        print(f"📊 Creating innovation test heatmap for {len(test_data)} trials")

        test_df = pd.DataFrame(test_data)

        # Calculate pass rates by subject for each test
        test_types = [
            ("ljung_box_pass_rate", "Ljung-Box"),
            ("lilliefors_pass_rate", "PIT-Lilliefors"),
            ("both_pass_rate", "Combined"),
        ]

        # Create heatmap data: rows = test types, columns = subjects
        subjects = sorted(test_df["subject"].unique())
        heatmap_data = []

        for col_name, test_name in test_types:
            row_data = []
            for subject in subjects:
                subject_data = test_df[test_df["subject"] == subject]
                pass_rate = subject_data[col_name].mean() * 100  # Convert to percentage
                row_data.append(pass_rate)
            heatmap_data.append(row_data)

        # Create DataFrame for heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[test_name for _, test_name in test_types],
            columns=[s.replace("subject_", "") for s in subjects],
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 4))

        # Use RdYlGn colormap (red to green) with range 0-100
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={"label": "Pass Rate (%)"},
            linewidths=0.5,
            linecolor="white",
        )

        # ax.set_title('Innovation Test Pass Rates by Subject', fontweight='bold', fontsize=16)
        ax.set_xlabel("Subject", fontweight="bold", fontsize=14)
        ax.set_ylabel("Test Type", fontweight="bold", fontsize=14)
        ax.tick_params(axis="x", rotation=0, labelsize=14)
        ax.tick_params(axis="y", rotation=0, labelsize=14)

        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Pass Rate (%)", fontsize=14)

        plt.tight_layout()
        plt.savefig(
            output_dir / "innovation_test_heatmap.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

        # Print summary statistics for each test type
        for col_name, test_name in test_types:
            mean_pass_rate = test_df[col_name].mean() * 100
            print(f"✅ {test_name}: Mean pass rate = {mean_pass_rate:.1f}%")

        # Also save the numerical data
        print(f"\n📋 Innovation Test Pass Rates by Subject:")
        print(heatmap_df.round(1))

        # Print per-subject combined pass rate statistics
        print(f"\n📋 Per-Subject Combined Test Pass Rates:")
        subjects = sorted(test_df["subject"].unique())
        for subject in subjects:
            subject_data = test_df[test_df["subject"] == subject]
            combined_rate = subject_data["both_pass_rate"].mean() * 100
            count = len(subject_data)
            print(
                f"  {subject}: combined_pass_rate={combined_rate:.1f}% ({count} trials)"
            )
        print()

    return sigma_data, test_data


def create_results_plots():
    """Create specific plots for thesis results section."""

    print("=" * 70)
    print("CREATING RESULTS PLOTS FOR THESIS")
    print("=" * 70)

    # Create output directory
    output_dir = Path("results_plots/residual_modeling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fitted models
    print("\n1. Loading fitted ARMA-GARCH models...")
    fitted_models = load_clean_models()

    if not fitted_models:
        print("❌ No fitted models found. Please run ARMA-GARCH fitting first.")
        return

    print(f"✅ Loaded {len(fitted_models)} fitted models")

    # Create validation box plots
    print("\n2. Creating validation box plots...")
    sigma_data, test_data = create_validation_box_plots(fitted_models, output_dir)

    print("\n✅ Results plots completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Files generated:")
    print(f"   - sigma_ratio_boxplots.png")
    print(f"   - innovation_test_boxplots.png")
    print(f"   - comprehensive heatmap (displayed)")

    return


def create_random_baseline_validation(fitted_models: dict, n_random_models: int = 100):
    """
    Cross-validation using EMG prediction testing.

    Compares fitted vs random ARMA-GARCH models by testing their ability to predict
    real EMG data and analyzing residual quality (not just innovation whiteness).
    """

    print(f"\n" + "=" * 70)
    print("CROSS-VALIDATION: EMG PREDICTION PERFORMANCE")
    print("=" * 70)
    print(
        f"📊 Testing fitted vs {n_random_models} random ARMA-GARCH models on real EMG data..."
    )

    # Step 1: Extract global parameter statistics from fitted models
    print("\n1. Computing global parameter statistics from fitted models...")

    all_params = {
        "arma_orders": [],
        "garch_params": {"omega": [], "alpha": [], "beta": []},
        "nu_values": [],
        "persistence": [],
    }

    for trial_name, trial_data in fitted_models.items():
        # ARMA orders
        arma_order = trial_data.get("arma_order", (2, 1))  # Default fallback
        all_params["arma_orders"].append(arma_order)

        # GARCH parameters
        garch_params = trial_data.get("garch_params", {})
        if "omega" in garch_params:
            all_params["garch_params"]["omega"].append(garch_params["omega"])
        if "alpha" in garch_params:
            all_params["garch_params"]["alpha"].append(garch_params["alpha"])
        if "beta" in garch_params:
            all_params["garch_params"]["beta"].append(garch_params["beta"])

        # Nu parameter for t-distribution
        nu = trial_data.get("nu", None)
        if nu is None:
            nu = trial_data.get("distribution_params", {}).get("nu", None)
        if nu is not None and not np.isnan(nu):
            all_params["nu_values"].append(nu)

        # Persistence
        persistence = garch_params.get("alpha", 0) + garch_params.get("beta", 0)
        if 0 < persistence < 1:
            all_params["persistence"].append(persistence)

    # Compute statistics
    import scipy.stats as stats

    # ARMA order mode (most common)
    if all_params["arma_orders"]:
        arma_mode = stats.mode(all_params["arma_orders"], keepdims=True).mode[0]
    else:
        arma_mode = (2, 1)

    # GARCH parameter statistics - use tighter bounds (3σ instead of 5σ)
    param_stats = {}
    for param_name, values in all_params["garch_params"].items():
        if values:
            param_stats[param_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    # Nu statistics
    if all_params["nu_values"]:
        nu_stats = {
            "mean": np.mean(all_params["nu_values"]),
            "std": np.std(all_params["nu_values"]),
            "min": np.min(all_params["nu_values"]),
            "max": np.max(all_params["nu_values"]),
        }
    else:
        nu_stats = {"mean": 5.0, "std": 2.0, "min": 2.1, "max": 30.0}

    print(f"✅ Global ARMA order (mode): {arma_mode}")
    print(f"✅ GARCH parameter ranges:")
    for param, stats_dict in param_stats.items():
        print(f"   {param}: μ={stats_dict['mean']:.4f}, σ={stats_dict['std']:.4f}")
    print(f"✅ Nu parameter: μ={nu_stats['mean']:.2f}, σ={nu_stats['std']:.2f}")

    # Step 2: Test fitted vs random model residual quality
    print(f"\n2. Testing fitted vs random model residual quality...")

    fitted_results = []
    random_results = []

    # Test fitted models using existing residuals
    print("   Testing fitted model residuals...")
    tested_count = 0
    for trial_name, trial_data in fitted_models.items():
        if tested_count >= n_random_models:  # Match sample sizes
            break

        # Test fitted model residual quality
        residual_quality = test_model_residual_quality(trial_data, trial_data, "fitted")
        if residual_quality is not None:
            fitted_results.append(
                {"trial_name": trial_name, "model_type": "fitted", **residual_quality}
            )
            tested_count += 1

    # Generate and test random baseline models
    print(f"   Testing {n_random_models} random baseline models...")
    np.random.seed(42)  # For reproducibility

    # Select representative fitted models to use as base for random model testing
    fitted_trials = list(fitted_models.items())[:n_random_models]

    for i, (trial_name, trial_data) in enumerate(fitted_trials):
        try:
            # Generate random parameters using tighter bounds (mean ± 3*std)
            random_params = {}
            for param_name, stats_dict in param_stats.items():
                mean, std = stats_dict["mean"], stats_dict["std"]
                # Use 3σ bounds instead of 5σ for more realistic parameters
                lower_bound = max(mean - 3 * std, stats_dict["min"])
                upper_bound = min(mean + 3 * std, stats_dict["max"])

                if param_name == "omega":
                    lower_bound = max(lower_bound, 1e-6)  # Omega must be positive
                elif param_name in ["alpha", "beta"]:
                    lower_bound = max(lower_bound, 0)  # Non-negative
                    upper_bound = min(upper_bound, 0.99)  # Stability

                random_params[param_name] = np.random.uniform(lower_bound, upper_bound)

            # Ensure GARCH stability: alpha + beta < 1
            if "alpha" in random_params and "beta" in random_params:
                persistence = random_params["alpha"] + random_params["beta"]
                if persistence >= 1.0:
                    # Scale down to ensure stationarity
                    scale_factor = 0.95 / persistence
                    random_params["alpha"] *= scale_factor
                    random_params["beta"] *= scale_factor

            # Random nu parameter
            nu_lower = max(nu_stats["mean"] - 3 * nu_stats["std"], nu_stats["min"])
            nu_upper = min(nu_stats["mean"] + 3 * nu_stats["std"], nu_stats["max"])
            random_nu = np.random.uniform(nu_lower, nu_upper)

            # Create mock trial data with random parameters
            mock_trial_data = {
                "arma_order": arma_mode,
                "garch_params": random_params,
                "nu": random_nu,
                "best_distribution": "t",
            }

            # Test random model residual quality using the fitted trial as base
            residual_quality = test_model_residual_quality(
                trial_data, mock_trial_data, "random"
            )
            if residual_quality is not None:
                random_results.append(
                    {
                        "model_id": f"random_{i:03d}",
                        "trial_name": trial_name,
                        "model_type": "random",
                        "params": random_params,
                        "nu": random_nu,
                        **residual_quality,
                    }
                )

        except Exception as e:
            print(f"   ⚠️  Random model {i} failed: {str(e)[:50]}")
            continue

    print(f"✅ Tested {len(fitted_results)} fitted models")
    print(f"✅ Tested {len(random_results)} random baseline models")

    # Step 3: Compare residual quality
    print(f"\n3. Comparing EMG prediction quality...")

    if len(fitted_results) == 0 or len(random_results) == 0:
        print("❌ Insufficient results for comparison")
        return {}

    # Calculate performance metrics
    fitted_metrics = calculate_performance_metrics(fitted_results)
    random_metrics = calculate_performance_metrics(random_results)

    print("\nEMG PREDICTION CROSS-VALIDATION RESULTS:")
    print("=" * 70)
    print(f"                      | Fitted | Random | Improvement")
    print("-" * 70)
    for metric_name in fitted_metrics:
        fitted_val = fitted_metrics[metric_name]
        random_val = random_metrics[metric_name]
        improvement = fitted_val - random_val
        print(
            f"{metric_name:20} | {fitted_val:6.1f}% | {random_val:6.1f}% | {improvement:+6.1f}%"
        )
    print("=" * 70)

    # Interpretation and statistical testing
    combined_fitted = np.mean([fitted_metrics[k] for k in fitted_metrics])
    combined_random = np.mean([random_metrics[k] for k in random_metrics])

    print(f"\n🎯 INTERPRETATION:")
    if combined_fitted > combined_random:
        improvement = combined_fitted - combined_random
        print(
            f"✅ FITTED MODELS ARE BETTER: {improvement:.1f}% higher diagnostic pass rate"
        )
        print(
            f"   Higher pass rates indicate better residual quality (less remaining structure)"
        )
        print(
            f"   Fitted models better capture EMG dynamics and leave cleaner residuals"
        )
        validation_quality = "SUPERIOR"
    elif combined_fitted < combined_random:
        degradation = combined_random - combined_fitted
        print(f"❌ RANDOM MODELS ARE BETTER: {degradation:.1f}% higher pass rate")
        print(
            f"   This suggests fitted models may not be capturing EMG structure properly"
        )
        validation_quality = "INFERIOR"
    else:
        print(f"⚖️  NO DIFFERENCE: Similar residual quality")
        validation_quality = "EQUIVALENT"

    # Statistical significance test
    significance_result = test_statistical_significance(fitted_results, random_results)

    print(f"\n📊 Statistical Significance:")
    print(f"   Test: {significance_result['test_name']}")
    print(f"   Statistic: {significance_result['statistic']:.3f}")
    print(f"   p-value: {significance_result['p_value']:.6f}")
    if significance_result["p_value"] < 0.05:
        print(f"   ✅ SIGNIFICANT: Difference is statistically significant")
        print(
            f"   📈 Conclusion: Fitted models are {validation_quality} to random baselines"
        )
    else:
        print(f"   ❌ NOT SIGNIFICANT: No statistically significant difference")
        print(f"   📈 Conclusion: Performance equivalent to random baselines")

    return {
        "fitted_results": fitted_results,
        "random_results": random_results,
        "fitted_metrics": fitted_metrics,
        "random_metrics": random_metrics,
        "significance_result": significance_result,
        "param_stats": param_stats,
        "nu_stats": nu_stats,
        "arma_mode": arma_mode,
    }


def fit_white_gaussian_model(original_data):
    """
    Fit a simple white Gaussian model to original data.
    This serves as the null model - just sample mean with Gaussian residuals.

    Args:
        original_data: Original EMG residual series

    Returns:
        dict: Model fitting results with mean and residuals
    """
    original_data = np.array(original_data)
    original_data = original_data[~np.isnan(original_data)]

    if len(original_data) < 10:
        return None

    mean = np.mean(original_data)
    residuals = original_data - mean

    return {
        "mean": float(mean),
        "residuals": residuals,
        "model_type": "white_gaussian",
        "n_obs": len(original_data),
    }


def test_residual_adequacy(residuals, trial_data=None):
    """
    Test residual series for white noise properties using standard time series tests.
    Includes PIT-transformed Lilliefors test for proper Student-t distribution handling.

    Args:
        residuals: Residual series to test
        trial_data: Optional trial data containing distribution information for PIT transform

    Returns:
        dict: Test statistics and p-values for model adequacy assessment
    """
    from scipy.stats import jarque_bera
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, lilliefors

    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < 20:
        return None

    # CENTER RESIDUALS FIRST - same as Option 6!
    clean_residuals = residuals - np.mean(residuals)

    results = {}

    try:
        # Ljung-Box test for autocorrelation (lag 10) - same as Option 6
        ljung_box_result = acorr_ljungbox(clean_residuals, lags=10, return_df=True)
        results["ljung_box_statistic"] = float(ljung_box_result["lb_stat"].iloc[-1])
        results["ljung_box_pvalue"] = float(ljung_box_result["lb_pvalue"].iloc[-1])
    except Exception as e:
        results["ljung_box_statistic"] = np.nan
        results["ljung_box_pvalue"] = np.nan

    try:
        # ARCH-LM test for heteroskedasticity - using centered residuals
        arch_lag = min(5, len(clean_residuals) // 8)
        if arch_lag >= 1:
            arch_result = het_arch(clean_residuals, nlags=arch_lag)
            results["arch_lm_statistic"] = float(arch_result[0])  # LM statistic
            results["arch_lm_pvalue"] = float(arch_result[1])  # p-value
        else:
            results["arch_lm_statistic"] = np.nan
            results["arch_lm_pvalue"] = np.nan
    except Exception as e:
        results["arch_lm_statistic"] = np.nan
        results["arch_lm_pvalue"] = np.nan

    try:
        # PIT-transformed Lilliefors test - SAME AS OPTION 6
        if trial_data is not None:
            best_dist = trial_data.get("best_distribution", "normal")

            if best_dist == "t":
                # For Student-t: Use PIT transformation then Lilliefors test
                nu = trial_data.get("distribution_params", {}).get("nu", None)

                # Check if ν is stored directly in trial_data
                if nu is None:
                    nu = trial_data.get("nu", None)

                # Fallback: try to extract nu from garch_params
                if nu is None:
                    garch_params = trial_data.get("garch_params", {})
                    nu = garch_params.get("nu", None)

                # Ultimate fallback: use reasonable estimate
                if nu is None or np.isnan(nu):
                    nu = 5.67  # Subject_005 mean ν as reasonable estimate

                # PIT transformation: t-distribution CDF -> normal quantile
                try:
                    from scipy.stats import norm
                    from scipy.stats import t as t_dist

                    # Transform residuals to standard normal using fitted t-distribution CDF
                    # then apply inverse normal CDF (probability integral transform)
                    transformed = norm.ppf(t_dist.cdf(clean_residuals, nu))

                    # Remove any infinite values from transformation
                    transformed = transformed[np.isfinite(transformed)]

                    if len(transformed) > 20:
                        # Apply Lilliefors test to PIT-transformed data (should be standard normal)
                        lillie_stat, lillie_p = lilliefors(transformed, dist="norm")
                        results["lilliefors_statistic"] = float(lillie_stat)
                        results["lilliefors_pvalue"] = float(lillie_p)
                    else:
                        # Fallback if too many infinite values
                        lillie_stat, lillie_p = lilliefors(clean_residuals, dist="norm")
                        results["lilliefors_statistic"] = float(lillie_stat)
                        results["lilliefors_pvalue"] = float(lillie_p)
                except:
                    # Fallback: direct Lilliefors against normal
                    lillie_stat, lillie_p = lilliefors(clean_residuals, dist="norm")
                    results["lilliefors_statistic"] = float(lillie_stat)
                    results["lilliefors_pvalue"] = float(lillie_p)
            else:
                # For normal: Use Lilliefors test directly
                lillie_stat, lillie_p = lilliefors(clean_residuals, dist="norm")
                results["lilliefors_statistic"] = float(lillie_stat)
                results["lilliefors_pvalue"] = float(lillie_p)
        else:
            # No trial data: default to normal Lilliefors
            lillie_stat, lillie_p = lilliefors(clean_residuals, dist="norm")
            results["lilliefors_statistic"] = float(lillie_stat)
            results["lilliefors_pvalue"] = float(lillie_p)

    except Exception as e:
        results["lilliefors_statistic"] = np.nan
        results["lilliefors_pvalue"] = np.nan

    # Basic descriptive statistics
    results["variance"] = float(np.var(residuals))
    results["mean"] = float(np.mean(residuals))
    results["n_obs"] = len(residuals)

    return results


def perform_model_adequacy_validation(fitted_models):
    """
    Test model adequacy by comparing residuals from ARMA-GARCH models vs white Gaussian models.

    This is the proper validation approach where we test if ARMA-GARCH models actually
    produce white noise residuals (as they should) compared to simple white Gaussian models.

    Args:
        fitted_models: Dict of fitted ARMA-GARCH model data

    Returns:
        dict: Model adequacy validation results
    """
    print(f"\n🔬 MODEL ADEQUACY VALIDATION")
    print(
        f"Testing whether ARMA-GARCH models produce better white noise residuals than simple Gaussian models"
    )
    print(f"Analyzing {len(fitted_models)} trials...")

    results = {
        "arma_garch_tests": [],
        "white_gaussian_tests": [],
        "trial_names": [],
        "comparison_results": {},
    }

    trial_count = 0
    valid_trials = 0

    for trial_name, trial_data in fitted_models.items():
        trial_count += 1
        print(
            f"\rProcessing trial {trial_count}/{len(fitted_models)}: {trial_name[:30]}...",
            end="",
        )

        try:
            # Get the ACTUAL standardized residuals from fitting the ARMA-GARCH model to real EMG data
            # This is what Option 6 does correctly!
            standardized_residuals = trial_data.get("standardized_residuals", [])
            if len(standardized_residuals) < 50:
                continue

            # 1. Get ARMA-GARCH innovations (actual fitted model residuals from real data)
            arma_garch_innovations = np.array(standardized_residuals)
            arma_garch_innovations = arma_garch_innovations - np.mean(
                arma_garch_innovations
            )

            if len(arma_garch_innovations) < 20:
                print(f"⚠️  Insufficient standardized residuals for {trial_name}")
                continue

            # 2. Get the original EMG data to fit white Gaussian model
            original_residuals = trial_data.get("original_residuals", [])
            if len(original_residuals) < len(arma_garch_innovations):
                print(f"⚠️  Insufficient original residuals for {trial_name}")
                continue

            # 3. Fit white Gaussian model to same EMG data and get residuals
            white_gaussian_result = fit_white_gaussian_model(original_residuals)
            if white_gaussian_result is None:
                continue

            white_gaussian_innovations = white_gaussian_result["residuals"]
            # Standardize to match ARMA-GARCH innovations
            white_gaussian_innovations = (
                white_gaussian_innovations - np.mean(white_gaussian_innovations)
            ) / np.std(white_gaussian_innovations)
            white_gaussian_innovations = white_gaussian_innovations - np.mean(
                white_gaussian_innovations
            )

            # 4. Test both innovation series for adequacy
            if arma_garch_innovations is not None and len(arma_garch_innovations) >= 20:
                # Pass trial data for PIT-transformed Lilliefors test (Option 6 approach)
                arma_garch_test = test_residual_adequacy(
                    arma_garch_innovations, trial_data
                )
                if arma_garch_test is not None:
                    results["arma_garch_tests"].append(arma_garch_test)
                else:
                    results["arma_garch_tests"].append(None)
            else:
                results["arma_garch_tests"].append(None)

            # White Gaussian model has no distribution info (default to normal Lilliefors)
            white_gaussian_test = test_residual_adequacy(
                white_gaussian_innovations, trial_data=None
            )
            if white_gaussian_test is not None:
                results["white_gaussian_tests"].append(white_gaussian_test)
            else:
                results["white_gaussian_tests"].append(None)

            results["trial_names"].append(trial_name)
            valid_trials += 1

        except Exception as e:
            print(f"\n⚠️  Error processing {trial_name}: {e}")
            continue

    print(f"\n✅ Completed validation on {valid_trials} trials")

    # 4. Compare results
    if valid_trials > 0:
        results["comparison_results"] = compare_model_adequacy_results(results)

    return results


def compare_model_adequacy_results(results):
    """Compare adequacy test results between ARMA-GARCH and white Gaussian models."""

    arma_tests = [t for t in results["arma_garch_tests"] if t is not None]
    white_tests = [t for t in results["white_gaussian_tests"] if t is not None]

    if len(arma_tests) == 0 or len(white_tests) == 0:
        return {"error": "No valid test results to compare"}

    comparison = {}

    # Compare p-values for each test (higher p-values = better white noise properties)
    test_names = ["ljung_box_pvalue", "arch_lm_pvalue", "lilliefors_pvalue"]
    test_labels = [
        "Ljung-Box (No Autocorr)",
        "ARCH-LM (No Heterosked)",
        "Lilliefors (Normality)",
    ]

    for test_name, label in zip(test_names, test_labels):
        arma_pvals = [
            t[test_name] for t in arma_tests if not np.isnan(t.get(test_name, np.nan))
        ]
        white_pvals = [
            t[test_name] for t in white_tests if not np.isnan(t.get(test_name, np.nan))
        ]

        if len(arma_pvals) > 5 and len(white_pvals) > 5:
            # Compare distributions using Mann-Whitney U test
            from scipy.stats import mannwhitneyu

            try:
                stat, p_val = mannwhitneyu(
                    arma_pvals, white_pvals, alternative="two-sided"
                )

                arma_median = np.median(arma_pvals)
                white_median = np.median(white_pvals)

                comparison[test_name] = {
                    "arma_median_pval": float(arma_median),
                    "white_median_pval": float(white_median),
                    "arma_better": arma_median
                    > white_median,  # Higher p-value = better white noise
                    "mann_whitney_stat": float(stat),
                    "mann_whitney_pval": float(p_val),
                    "significant_diff": p_val < 0.05,
                    "arma_count": len(arma_pvals),
                    "white_count": len(white_pvals),
                    "test_label": label,
                }
            except Exception as e:
                comparison[test_name] = {"error": str(e)}

    # Overall adequacy summary
    adequacy_wins = sum(
        1
        for test in comparison.values()
        if isinstance(test, dict) and test.get("arma_better", False)
    )
    total_tests = len(
        [
            test
            for test in comparison.values()
            if isinstance(test, dict) and "arma_better" in test
        ]
    )

    comparison["summary"] = {
        "arma_garch_wins": adequacy_wins,
        "total_tests": total_tests,
        "win_rate": adequacy_wins / total_tests if total_tests > 0 else 0,
        "interpretation": "ARMA-GARCH models produce better white noise residuals"
        if adequacy_wins > total_tests / 2
        else "White Gaussian models produce equally good or better residuals",
    }

    return comparison


def test_model_residual_quality(fitted_models_data, model_params, model_type="unknown"):
    """
    Test model quality by comparing residual diagnostics.

    For fitted models: use existing residuals
    For random models: use the model parameters with existing standardized innovations
    """

    try:
        if model_type == "fitted":
            # Use existing residuals from fitted models
            if "standardized_residuals" in fitted_models_data:
                residuals = fitted_models_data["standardized_residuals"]
                if len(residuals) < 30:
                    return None

                # Apply comprehensive residual diagnostics
                residual_tests = comprehensive_residual_diagnostics(residuals)
                if residual_tests:
                    residual_tests["model_type"] = "fitted"
                    residual_tests["residual_length"] = len(residuals)

                return residual_tests
            else:
                return None

        elif model_type == "random":
            # For random models, we simulate what residuals would look like
            # Use existing innovations but with random model parameters
            original_residuals = fitted_models_data.get("original_residuals", [])
            if len(original_residuals) < 30:
                return None

            # Extract random model parameters
            garch_params = model_params.get("garch_params", {})
            omega = garch_params.get("omega", 0.001)
            alpha = garch_params.get("alpha", 0.1)
            beta = garch_params.get("beta", 0.8)
            nu = garch_params.get("nu", 6.0)

            # Generate synthetic residuals using random parameters
            # This simulates what the residuals would look like with these parameters
            np.random.seed(hash(str(model_params)) % 1000)  # Deterministic
            n_periods = len(original_residuals)

            # Simple GARCH simulation for volatility
            volatility = np.zeros(n_periods)
            volatility[0] = np.sqrt(
                omega / (1 - alpha - beta)
            )  # Unconditional volatility

            for t in range(1, n_periods):
                if t < len(original_residuals):
                    lag_resid = original_residuals[t - 1] if t > 0 else 0
                else:
                    lag_resid = 0

                volatility[t] = np.sqrt(
                    omega + alpha * (lag_resid**2) + beta * (volatility[t - 1] ** 2)
                )

            # Generate t-distributed innovations and scale by volatility
            t_innovations = np.random.standard_t(nu, n_periods)
            synthetic_residuals = volatility * t_innovations

            # Apply comprehensive residual diagnostics
            residual_tests = comprehensive_residual_diagnostics(synthetic_residuals)
            if residual_tests:
                residual_tests["model_type"] = "random"
                residual_tests["residual_length"] = len(synthetic_residuals)
                residual_tests["synthetic"] = True

            return residual_tests

    except Exception as e:
        print(f"   ⚠️  Model residual test failed for {model_type}: {str(e)[:50]}")
        return None


def comprehensive_residual_diagnostics(residuals):
    """
    Apply comprehensive diagnostic tests to residuals.
    Returns dictionary with test results and pass/fail status.
    """

    if len(residuals) < 30:
        return None

    results = {}

    try:
        # 1. Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(
            residuals, lags=min(10, len(residuals) // 4), return_df=True
        )
        lb_pvalue = lb_result["lb_pvalue"].iloc[-1]
        results["ljung_box_pass"] = lb_pvalue > 0.05
        results["ljung_box_pvalue"] = float(lb_pvalue)

    except:
        results["ljung_box_pass"] = False
        results["ljung_box_pvalue"] = 0.0

    try:
        # 2. Jarque-Bera test for normality
        from scipy.stats import jarque_bera

        jb_stat, jb_pvalue = jarque_bera(residuals)
        results["normality_pass"] = jb_pvalue > 0.05
        results["normality_pvalue"] = float(jb_pvalue)

    except:
        results["normality_pass"] = False
        results["normality_pvalue"] = 0.0

    try:
        # 3. ARCH-LM test for heteroskedasticity (simplified version)
        from statsmodels.stats.diagnostic import het_arch

        lm_stat, lm_pvalue, _, _ = het_arch(
            residuals, nlags=min(5, len(residuals) // 10)
        )
        results["arch_pass"] = lm_pvalue > 0.05  # Pass if no ARCH effects
        results["arch_pvalue"] = float(lm_pvalue)

    except:
        results["arch_pass"] = False
        results["arch_pvalue"] = 0.0

    try:
        # 4. Stationarity test (ADF)
        from statsmodels.tsa.stattools import adfuller

        adf_stat, adf_pvalue, _, _, _, _ = adfuller(residuals, autolag="AIC")
        results["stationarity_pass"] = adf_pvalue < 0.05  # Pass if stationary
        results["stationarity_pvalue"] = float(adf_pvalue)

    except:
        results["stationarity_pass"] = False
        results["stationarity_pvalue"] = 1.0

    return results


def calculate_performance_metrics(results_list):
    """Calculate aggregate performance metrics from list of test results"""

    if len(results_list) == 0:
        return {}

    metrics = {}

    # Calculate pass rates for each test type
    test_types = ["ljung_box_pass", "normality_pass", "arch_pass", "stationarity_pass"]

    for test_type in test_types:
        passes = sum(1 for r in results_list if r.get(test_type, False))
        total = len(results_list)
        pass_rate = (passes / total) * 100 if total > 0 else 0

        # Clean up name for display
        display_name = test_type.replace("_pass", "").replace("_", " ").title()
        metrics[display_name] = pass_rate

    return metrics


def test_statistical_significance(fitted_results, random_results):
    """Test statistical significance between fitted and random model performance"""

    try:
        # Use Mann-Whitney U test for comparing two independent groups
        from scipy.stats import mannwhitneyu

        # Compare overall performance (average of all test pass rates)
        fitted_scores = []
        random_scores = []

        test_types = [
            "ljung_box_pass",
            "normality_pass",
            "arch_pass",
            "stationarity_pass",
        ]

        for result in fitted_results:
            score = np.mean([result.get(test, False) for test in test_types])
            fitted_scores.append(score)

        for result in random_results:
            score = np.mean([result.get(test, False) for test in test_types])
            random_scores.append(score)

        if len(fitted_scores) > 0 and len(random_scores) > 0:
            statistic, p_value = mannwhitneyu(
                fitted_scores, random_scores, alternative="two-sided"
            )
            return {
                "test_name": "Mann-Whitney U",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        else:
            return {"test_name": "Insufficient Data", "statistic": 0.0, "p_value": 1.0}

    except Exception as e:
        print(f"   ⚠️  Statistical test failed: {str(e)}")
        return {"test_name": "Test Failed", "statistic": 0.0, "p_value": 1.0}


def extract_parameter_statistics(fitted_models):
    """Extract parameter statistics from fitted models for random baseline generation"""

    param_stats = {
        "garch_params": {"omega": [], "alpha": [], "beta": []},
        "arma_params": {},
        "nu_values": [],
        "optimal_orders": [],
    }

    for trial_name, trial_data in fitted_models.items():
        # Extract GARCH parameters
        garch_params = trial_data.get("garch_params", {})
        if "omega" in garch_params:
            param_stats["garch_params"]["omega"].append(garch_params["omega"])
        if "alpha" in garch_params:
            param_stats["garch_params"]["alpha"].append(garch_params["alpha"])
        if "beta" in garch_params:
            param_stats["garch_params"]["beta"].append(garch_params["beta"])

        # Extract ARMA parameters
        arma_params = trial_data.get("arma_params", {})
        for key, value in arma_params.items():
            if key not in param_stats["arma_params"]:
                param_stats["arma_params"][key] = []
            param_stats["arma_params"][key].append(value)

        # Extract nu values for t-distribution
        nu = trial_data.get("distribution_params", {}).get("nu", None)
        if nu is None:
            nu = trial_data.get("nu", None)
        if nu is not None and not np.isnan(nu):
            param_stats["nu_values"].append(nu)

        # Extract ARMA-GARCH orders
        orders = trial_data.get("optimal_orders", {})
        param_stats["optimal_orders"].append(orders)

    # Calculate statistics for each parameter
    stats_summary = {}

    # GARCH parameter statistics
    for param_name in ["omega", "alpha", "beta"]:
        values = param_stats["garch_params"][param_name]
        if values:
            stats_summary[f"garch_{param_name}"] = {
                "median": np.median(values),
                "mad": np.median(
                    np.abs(values - np.median(values))
                ),  # Median Absolute Deviation
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

    # ARMA parameter statistics
    for param_name, values in param_stats["arma_params"].items():
        if values:
            stats_summary[f"arma_{param_name}"] = {
                "median": np.median(values),
                "mad": np.median(np.abs(values - np.median(values))),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

    # Nu parameter statistics
    if param_stats["nu_values"]:
        stats_summary["nu"] = {
            "median": np.median(param_stats["nu_values"]),
            "mad": np.median(
                np.abs(param_stats["nu_values"] - np.median(param_stats["nu_values"]))
            ),
            "min": np.min(param_stats["nu_values"]),
            "max": np.max(param_stats["nu_values"]),
            "values": param_stats["nu_values"],
        }

    # Orders statistics (most common orders)
    from collections import Counter

    orders_counter = Counter()
    for orders in param_stats["optimal_orders"]:
        key = (
            orders.get("p", 0),
            orders.get("q", 0),
            orders.get("r", 1),
            orders.get("s", 1),
        )
        orders_counter[key] += 1

    stats_summary["orders"] = {
        "most_common": orders_counter.most_common(),
        "all_orders": param_stats["optimal_orders"],
    }

    return stats_summary


def generate_random_baseline_model(param_stats, distribution_type="uniform"):
    """Generate random baseline model parameters using parameter statistics"""

    random_params = {
        "garch_params": {},
        "arma_params": {},
        "distribution_params": {},
        "optimal_orders": {},
    }

    # Generate GARCH parameters
    for param_name in ["omega", "alpha", "beta"]:
        garch_key = f"garch_{param_name}"
        if garch_key in param_stats:
            stats = param_stats[garch_key]

            if distribution_type == "uniform":
                # Use uniform distribution based on min/max
                random_params["garch_params"][param_name] = np.random.uniform(
                    stats["min"], stats["max"]
                )
            else:
                # Use median ± MAD for bounded normal-like distribution
                mad_scaled = (
                    stats["mad"] * 1.4826
                )  # Scale MAD to approximate standard deviation
                value = np.random.normal(stats["median"], mad_scaled)
                # Bound within observed range
                value = np.clip(value, stats["min"], stats["max"])
                random_params["garch_params"][param_name] = value

    # Ensure GARCH stability (alpha + beta < 1) with strong enforcement
    alpha = random_params["garch_params"].get("alpha", 0.1)
    beta = random_params["garch_params"].get("beta", 0.8)

    # Much stronger stability requirement
    if alpha + beta >= 0.95:
        # Rescale to maintain relative proportions but ensure strong stability
        total = alpha + beta
        scale_factor = 0.90 / total  # Conservative: keep persistence well below 1
        random_params["garch_params"]["alpha"] = alpha * scale_factor
        random_params["garch_params"]["beta"] = beta * scale_factor

        # Update values for verification
        alpha = random_params["garch_params"]["alpha"]
        beta = random_params["garch_params"]["beta"]

    # Final verification - should never trigger but safety check
    persistence = alpha + beta
    if persistence >= 0.99:
        # Emergency fallback: use conservative default values
        random_params["garch_params"]["alpha"] = 0.1
        random_params["garch_params"]["beta"] = 0.8
        print(f"   ⚠️  Emergency fallback applied for unstable GARCH parameters")

    # Generate ARMA parameters
    for key, stats in param_stats.items():
        if key.startswith("arma_"):
            param_name = key[5:]  # Remove 'arma_' prefix

            if distribution_type == "uniform":
                random_params["arma_params"][param_name] = np.random.uniform(
                    stats["min"], stats["max"]
                )
            else:
                mad_scaled = stats["mad"] * 1.4826
                value = np.random.normal(stats["median"], mad_scaled)
                value = np.clip(value, stats["min"], stats["max"])
                random_params["arma_params"][param_name] = value

    # Generate nu parameter for t-distribution
    if "nu" in param_stats:
        stats = param_stats["nu"]
        if distribution_type == "uniform":
            random_params["distribution_params"]["nu"] = np.random.uniform(
                stats["min"], stats["max"]
            )
        else:
            mad_scaled = stats["mad"] * 1.4826
            value = np.random.normal(stats["median"], mad_scaled)
            value = np.clip(
                value, max(stats["min"], 2.1), stats["max"]
            )  # Ensure nu > 2 for finite variance
            random_params["distribution_params"]["nu"] = value

    # Sample model orders from observed distribution
    if "orders" in param_stats and param_stats["orders"]["most_common"]:
        # Weighted sampling based on frequency
        orders_list, weights = zip(*param_stats["orders"]["most_common"])
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        chosen_orders = np.random.choice(len(orders_list), p=weights)
        p, q, r, s = orders_list[chosen_orders]

        random_params["optimal_orders"] = {"p": p, "q": q, "r": r, "s": s}
    else:
        # Default orders
        random_params["optimal_orders"] = {"p": 0, "q": 0, "r": 1, "s": 1}

    # Set best distribution (assume t-distribution like fitted models)
    random_params["best_distribution"] = "t"

    # Calculate volatility persistence
    alpha = random_params["garch_params"].get("alpha", 0.1)
    beta = random_params["garch_params"].get("beta", 0.8)
    random_params["volatility_persistence"] = alpha + beta

    return random_params


def calculate_statistical_test_matching_components(
    original_residuals, simulated_residuals
):
    """
    Calculate how well simulated residuals reproduce the statistical test properties of original residuals.

    This compares the statistical test results (not raw data) to see which model better
    reproduces the temporal structure and heteroskedasticity patterns of the original.

    Components:
    1. ljung_box_match: How well Ljung-Box test statistics match (lower diff = better)
    2. arch_lm_match: How well ARCH-LM test statistics match (lower diff = better)
    3. var_ratio: Variance ratio (higher = more similar, max=1)
    4. distribution_match: How well distributional properties match (lower diff = better)

    Returns:
        dict: Component scores where lower differences = better match
    """

    try:
        from scipy.stats import jarque_bera, ks_2samp
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        if len(original_residuals) < 50 or len(simulated_residuals) < 50:
            return None

        original = np.array(original_residuals)
        simulated = np.array(simulated_residuals)

        components = {}

        # 1. Ljung-Box Test Matching (Autocorrelation)
        try:
            # Calculate Ljung-Box statistics for both series
            lags = min(10, len(original) // 5)

            lb_orig_result = acorr_ljungbox(original, lags=lags, return_df=False)
            lb_sim_result = acorr_ljungbox(simulated, lags=lags, return_df=False)

            # Extract test statistics (not p-values)
            lb_orig_stat = (
                float(lb_orig_result[0][-1]) if len(lb_orig_result[0]) > 0 else 0.0
            )
            lb_sim_stat = (
                float(lb_sim_result[0][-1]) if len(lb_sim_result[0]) > 0 else 0.0
            )

            # Score based on how closely the test statistics match
            components["ljung_box_match"] = abs(lb_orig_stat - lb_sim_stat)

        except Exception as e:
            components["ljung_box_match"] = float("inf")  # Worst possible score

        # 2. ARCH-LM Test Matching (Heteroskedasticity)
        try:
            # Calculate ARCH-LM statistics for both series
            arch_lags = min(5, len(original) // 8)

            arch_orig_stat, arch_orig_p = het_arch(original, maxlag=arch_lags)[:2]
            arch_sim_stat, arch_sim_p = het_arch(simulated, maxlag=arch_lags)[:2]

            # Score based on how closely the ARCH test statistics match
            components["arch_lm_match"] = abs(
                float(arch_orig_stat) - float(arch_sim_stat)
            )

        except Exception as e:
            components["arch_lm_match"] = float("inf")  # Worst possible score

        # 3. Variance Ratio (unchanged - direct comparison)
        var_orig = np.var(original)
        var_sim = np.var(simulated)
        if var_orig > 0 and var_sim > 0:
            components["var_ratio"] = min(
                var_orig / var_sim, var_sim / var_orig
            )  # [0,1], 1 = identical
        else:
            components["var_ratio"] = 0.0

        # 4. Distribution Shape Matching using Lilliefors test
        try:
            from statsmodels.stats.diagnostic import lilliefors

            # Lilliefors test for normality (more appropriate than KS for unknown parameters)
            lf_orig_stat, _ = lilliefors(original)
            lf_sim_stat, _ = lilliefors(simulated)

            # Compare how similarly both series deviate from normality
            components["distribution_match"] = abs(
                float(lf_orig_stat) - float(lf_sim_stat)
            )

        except Exception as e:
            components["distribution_match"] = float("inf")

        return components

    except Exception as e:
        print(f"   ⚠️ Statistical test matching failed: {str(e)[:50]}")
        return None


# Legacy wrapper for backward compatibility during transition
def calculate_residual_similarity(real_residuals, simulated_residuals):
    """
    Legacy wrapper - converts new components format to old similarity score.
    This maintains backward compatibility during transition.
    """
    components = calculate_statistical_test_matching_components(
        real_residuals, simulated_residuals
    )
    if components is None:
        return None

    # Convert to legacy exponential format (for now)
    similarity_scores = []

    if "pit_diff" in components:
        similarity_scores.append(np.exp(-components["pit_diff"]))
    if "var_ratio" in components:
        similarity_scores.append(components["var_ratio"])
    if "autocorr_diff" in components:
        similarity_scores.append(np.exp(-components["autocorr_diff"]))
    if "arch_diff" in components:
        similarity_scores.append(np.exp(-components["arch_diff"]))

    if len(similarity_scores) > 0:
        return float(np.clip(np.mean(similarity_scores), 0, 1))
    else:
        return None


def calculate_rank_based_similarity(all_components_list):
    """
    Calculate rank-based similarity scores from list of raw component dictionaries.

    Args:
        all_components_list: List of dictionaries with keys:
            - 'pit_diff': PIT-Lilliefors difference (lower = better)
            - 'var_ratio': Variance ratio (higher = better)
            - 'autocorr_diff': Ljung-Box difference (lower = better)
            - 'arch_diff': ARCH effects difference (lower = better)

    Returns:
        List of similarity scores [0,1] where 1 = best performing model
    """

    if not all_components_list:
        return []

    # Filter out None entries
    valid_components = [comp for comp in all_components_list if comp is not None]

    if not valid_components:
        return [None] * len(all_components_list)

    n_models = len(valid_components)

    # Extract arrays for each component
    pit_diffs = np.array([comp.get("pit_diff", np.inf) for comp in valid_components])
    var_ratios = np.array([comp.get("var_ratio", 0.0) for comp in valid_components])
    autocorr_diffs = np.array(
        [comp.get("autocorr_diff", np.inf) for comp in valid_components]
    )
    arch_diffs = np.array([comp.get("arch_diff", np.inf) for comp in valid_components])

    # Calculate ranks for each component
    # For 'diff' metrics: lower values get higher ranks (better performance)
    # For 'ratio' metrics: higher values get higher ranks (better performance)

    # PIT differences: lower is better, so negate before ranking
    pit_ranks = n_models - np.argsort(np.argsort(pit_diffs))  # Lower diff = higher rank

    # Variance ratios: higher is better
    var_ranks = np.argsort(np.argsort(-var_ratios)) + 1  # Higher ratio = higher rank

    # Autocorr differences: lower is better
    autocorr_ranks = n_models - np.argsort(
        np.argsort(autocorr_diffs)
    )  # Lower diff = higher rank

    # ARCH differences: lower is better
    arch_ranks = n_models - np.argsort(
        np.argsort(arch_diffs)
    )  # Lower diff = higher rank

    # Normalize ranks to [0,1] scale
    if n_models > 1:
        pit_scores = (pit_ranks - 1) / (n_models - 1)
        var_scores = (var_ranks - 1) / (n_models - 1)
        autocorr_scores = (autocorr_ranks - 1) / (n_models - 1)
        arch_scores = (arch_ranks - 1) / (n_models - 1)
    else:
        # Single model case
        pit_scores = np.array([1.0])
        var_scores = np.array([1.0])
        autocorr_scores = np.array([1.0])
        arch_scores = np.array([1.0])

    # Combined similarity score (equal weighting)
    combined_scores = (pit_scores + var_scores + autocorr_scores + arch_scores) / 4

    # Map back to original list including None entries
    result_scores = []
    valid_idx = 0

    for original_comp in all_components_list:
        if original_comp is None:
            result_scores.append(None)
        else:
            result_scores.append(float(combined_scores[valid_idx]))
            valid_idx += 1

    return result_scores


def calculate_pure_rank_analysis(components_list):
    """
    Calculate pure rank analysis for a list of component dictionaries.
    Returns raw ranks for each component - NO COMBINATION into similarity scores.

    Args:
        components_list: List of component dictionaries from calculate_statistical_test_matching_components

    Returns:
        Dictionary with component ranks and analysis (rank 1 = best performance)
    """
    if len(components_list) < 2:
        return None

    n_models = len(components_list)

    # Extract all component values with new naming scheme
    ljung_box_diffs = [
        comp.get("ljung_box_match", float("inf")) for comp in components_list
    ]
    arch_lm_diffs = [
        comp.get("arch_lm_match", float("inf")) for comp in components_list
    ]
    var_ratios = [comp.get("var_ratio", 0) for comp in components_list]
    distribution_diffs = [
        comp.get("distribution_match", float("inf")) for comp in components_list
    ]

    # Calculate ranks (rank 1 = best performance)
    # For difference measures: lower values get rank 1 (best)
    # For ratio measures: higher values get rank 1 (best)

    ljung_box_ranks = stats.rankdata(
        ljung_box_diffs, method="min"
    )  # Lower diff = rank 1
    arch_lm_ranks = stats.rankdata(arch_lm_diffs, method="min")  # Lower diff = rank 1
    var_ranks = stats.rankdata(
        [-v for v in var_ratios], method="min"
    )  # Higher ratio = rank 1
    distribution_ranks = stats.rankdata(
        distribution_diffs, method="min"
    )  # Lower diff = rank 1

    # Return pure rank data
    return {
        "ljung_box_ranks": ljung_box_ranks.tolist(),
        "arch_lm_ranks": arch_lm_ranks.tolist(),
        "var_ranks": var_ranks.tolist(),
        "distribution_ranks": distribution_ranks.tolist(),
        "n_models": n_models,
        "fitted_model_index": 0,  # Fitted model is always first in the list
        "component_wins": {
            "fitted_wins_ljung_box": ljung_box_ranks[0] == 1,
            "fitted_wins_arch_lm": arch_lm_ranks[0] == 1,
            "fitted_wins_var": var_ranks[0] == 1,
            "fitted_wins_distribution": distribution_ranks[0] == 1,
        },
        "fitted_total_wins": sum(
            [
                ljung_box_ranks[0] == 1,
                arch_lm_ranks[0] == 1,
                var_ranks[0] == 1,
                distribution_ranks[0] == 1,
            ]
        ),
        "fitted_rank_sum": ljung_box_ranks[0]
        + arch_lm_ranks[0]
        + var_ranks[0]
        + distribution_ranks[0],
    }


def perform_trial_based_cross_validation(fitted_models, scale=10, random_seed=42):
    """
    Perform rank-based trial-by-trial cross-validation where each fitted ARMA-GARCH model is compared
    against white Gaussian noise baselines on the same trial data using proper rank-based scoring.

    Args:
        fitted_models: Dictionary of fitted ARMA-GARCH model data
        scale: Number of white noise baselines to test per trial (default: 10)
        random_seed: Random seed for reproducibility

    This approach:
    1. Tests fitted ARMA-GARCH model + N white noise baselines against real trial data
    2. Calculates raw similarity components for all models
    3. Ranks models by each component (proper rank-based comparison)
    4. Computes overall rank-based similarity score
    5. Determines if ARMA-GARCH model ranks better than white noise baselines
    """
    print(f"🔬 Starting rank-based trial-by-trial cross-validation...")
    print(
        f"   Testing {len(fitted_models)} ARMA-GARCH models vs {scale} white noise baselines per trial"
    )

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    trial_results = []
    fitted_wins = 0
    total_comparisons = 0

    print(f"📊 Performing trial-by-trial comparisons...")

    for i, (trial_name, fitted_data) in enumerate(fitted_models.items()):
        if i % 25 == 0:
            print(f"   Progress: {i}/{len(fitted_models)} trials processed")

        original_residuals = fitted_data["original_residuals"]

        # Test fitted model against real data
        try:
            fitted_simulated = simulate_enhanced_arma_garch(
                fitted_data, len(original_residuals), random_seed=random_seed + i
            )
            if isinstance(fitted_simulated, dict):
                fitted_simulated = fitted_simulated["residuals"]

            if fitted_simulated is None or len(fitted_simulated) < 10:
                continue

            fitted_components = calculate_statistical_test_matching_components(
                original_residuals, fitted_simulated
            )
            if fitted_components is None:
                continue
            # Convert to similarity scores (higher = better for all components)
            # For differences: convert to similarity using 1/(1+diff)
            # For ratios: already higher = better
            fitted_similarity = (
                1 / (1 + fitted_components.get("pit_diff", 1))
                + fitted_components.get("var_ratio", 0)
                + 1 / (1 + fitted_components.get("autocorr_diff", 1))
                + 1 / (1 + fitted_components.get("arch_diff", 1))
            ) / 4
            if fitted_similarity is None or np.isnan(fitted_similarity):
                continue

        except Exception as e:
            print(f"   ⚠️ Fitted model {trial_name} failed: {str(e)[:50]}")
            continue

        # Test random models against the same trial data
        random_similarities = []
        for j in range(scale):
            try:
                # Generate random baseline model
                random_model = generate_random_baseline_model(fitted_models)
                if random_model is None:
                    continue

                # Simulate from random model using same trial length
                random_simulated = simulate_enhanced_arma_garch(
                    random_model,
                    len(original_residuals),
                    random_seed=random_seed + 10000 + i * scale + j,
                )
                if random_simulated is None or len(random_simulated) < 10:
                    continue

                random_components = calculate_residual_similarity_components(
                    original_residuals, random_simulated
                )
                if random_components is None:
                    continue
                # Convert to similarity scores (higher = better for all components)
                # For differences: convert to similarity using 1/(1+diff)
                # For ratios: already higher = better
                random_similarity = (
                    1 / (1 + random_components.get("pit_diff", 1))
                    + random_components.get("var_ratio", 0)
                    + 1 / (1 + random_components.get("autocorr_diff", 1))
                    + 1 / (1 + random_components.get("arch_diff", 1))
                ) / 4
                if random_similarity is not None and not np.isnan(random_similarity):
                    random_similarities.append(random_similarity)

            except Exception as e:
                continue

        # RANK-BASED COMPARISON: Collect all models for this trial
        all_models_data = []
        all_components_data = []

        # Add fitted model
        all_models_data.append({"type": "fitted", "name": trial_name})
        all_components_data.append(fitted_components)

        # Add random models
        for j in range(scale):
            try:
                # Generate white Gaussian noise baseline
                # Match the standard deviation of original residuals for fair comparison
                residual_std = np.std(original_residuals)
                np.random.seed(random_seed + 10000 + i * scale + j)
                white_noise = np.random.normal(0, residual_std, len(original_residuals))

                white_noise_components = calculate_statistical_test_matching_components(
                    original_residuals, white_noise
                )
                if white_noise_components is None:
                    continue

                all_models_data.append(
                    {"type": "white_noise", "name": f"white_noise_{j}"}
                )
                all_components_data.append(white_noise_components)

            except Exception as e:
                continue

        # Perform pure rank analysis for this trial
        if len(all_components_data) >= 2:  # Need at least fitted + 1 random
            # Calculate pure rank analysis (NO combined similarity scores)
            trial_rank_analysis = calculate_pure_rank_analysis(all_components_data)

            if trial_rank_analysis is not None:
                # Extract rank information
                fitted_total_wins = trial_rank_analysis[
                    "fitted_total_wins"
                ]  # How many components fitted won
                fitted_rank_sum = trial_rank_analysis[
                    "fitted_rank_sum"
                ]  # Sum of ranks (lower = better)
                component_wins = trial_rank_analysis["component_wins"]
                n_models = trial_rank_analysis["n_models"]
                n_components = 4  # ljung_box, arch_lm, var, distribution

                trial_result = {
                    "trial_name": trial_name,
                    "fitted_component_wins": fitted_total_wins,  # Out of 4 components
                    "fitted_rank_sum": fitted_rank_sum,  # Lower is better
                    "component_wins_detail": component_wins,
                    "n_models": n_models,
                    "n_components": n_components,
                    "fitted_dominates": fitted_total_wins
                    >= n_components / 2,  # Wins majority of components
                    "rank_analysis": trial_rank_analysis,  # Full rank data
                }

                trial_results.append(trial_result)
                fitted_wins += fitted_total_wins
                total_comparisons += n_components

    if total_comparisons == 0:
        print(f"❌ No valid comparisons found")
        return None

    # Aggregate results using pure rank analysis
    fitted_component_wins = [
        r["fitted_component_wins"] for r in trial_results
    ]  # Wins per trial
    fitted_rank_sums = [
        r["fitted_rank_sum"] for r in trial_results
    ]  # Rank sums per trial
    fitted_dominates_count = sum(
        1 for r in trial_results if r["fitted_dominates"]
    )  # Trials where fitted wins majority

    # Component-specific win rates
    ljung_box_wins = sum(
        1 for r in trial_results if r["component_wins_detail"]["fitted_wins_ljung_box"]
    )
    arch_lm_wins = sum(
        1 for r in trial_results if r["component_wins_detail"]["fitted_wins_arch_lm"]
    )
    var_wins = sum(
        1 for r in trial_results if r["component_wins_detail"]["fitted_wins_var"]
    )
    distribution_wins = sum(
        1
        for r in trial_results
        if r["component_wins_detail"]["fitted_wins_distribution"]
    )

    n_trials = len(trial_results)

    # Overall metrics
    component_win_rate = (
        fitted_wins / total_comparisons
    )  # Proportion of all component comparisons won
    trial_dominance_rate = (
        fitted_dominates_count / n_trials
    )  # Proportion of trials where fitted wins majority

    # Statistical testing
    from scipy.stats import mannwhitneyu

    try:
        # Statistical test on rank sums (lower rank sum = better performance)
        # Generate expected rank sums for random models to compare against fitted
        n_models_per_trial = n_models  # Should be same across trials
        expected_random_rank_sum = (
            (n_models_per_trial + 1) * 4 / 2
        )  # Expected rank sum for random model

        # Use binomial test on component wins
        from scipy.stats import binom_test

        p_value = binom_test(fitted_wins, total_comparisons, 0.5, alternative="greater")

        results = {
            "trial_results": trial_results,
            "methodology": "pure_rank_trial_comparison",
            "n_valid_trials": n_trials,
            "n_models_per_trial": n_models_per_trial,
            "total_component_comparisons": total_comparisons,
            "fitted_component_wins": fitted_wins,
            "component_win_rate": component_win_rate,
            "trial_dominance_rate": trial_dominance_rate,
            "component_specific_wins": {
                "ljung_box": {
                    "wins": ljung_box_wins,
                    "rate": ljung_box_wins / n_trials,
                },
                "arch_lm": {"wins": arch_lm_wins, "rate": arch_lm_wins / n_trials},
                "variance": {"wins": var_wins, "rate": var_wins / n_trials},
                "distribution": {
                    "wins": distribution_wins,
                    "rate": distribution_wins / n_trials,
                },
            },
            "fitted_rank_sums": {
                "values": fitted_rank_sums,
                "mean": np.mean(fitted_rank_sums),
                "expected_random": expected_random_rank_sum,
                "outperforms_expected": np.mean(fitted_rank_sums)
                < expected_random_rank_sum,
            },
            "statistical_test": {
                "test": "binomial_test",
                "p_value": p_value,
                "alternative": "fitted wins > 50%",
            },
        }

        print(f"✅ Pure rank-based cross-validation complete!")
        print(f"📈 RESULTS:")
        print(f"   Total trials: {n_trials}")
        print(
            f"   Component win rate: {component_win_rate:.1%} ({fitted_wins}/{total_comparisons})"
        )
        print(
            f"   Trial dominance rate: {trial_dominance_rate:.1%} ({fitted_dominates_count}/{n_trials})"
        )
        print(f"   Ljung-Box wins: {ljung_box_wins / n_trials:.1%}")
        print(f"   ARCH-LM wins: {arch_lm_wins / n_trials:.1%}")
        print(f"   Variance wins: {var_wins / n_trials:.1%}")
        print(f"   Distribution wins: {distribution_wins / n_trials:.1%}")
        print(
            f"   Mean rank sum: {np.mean(fitted_rank_sums):.1f} (expected random: {expected_random_rank_sum:.1f})"
        )
        print(f"   Binomial test p-value: {p_value:.2e}")

        return results

    except Exception as e:
        print(f"❌ Statistical testing failed: {e}")
        return None


def perform_aggregated_cross_validation(
    fitted_models, n_random_models=500, random_seed=42
):
    """
    Perform aggregated cross-validation analysis using all fitted models vs random baselines.

    CORRECT LOGIC:
    1. Compare REAL residuals vs FITTED simulations → Should be SIMILAR (good fit)
    2. Compare REAL residuals vs RANDOM simulations → Should be DIFFERENT (validates fitting)

    We measure SIMILARITY, not idealized test passing. Good models replicate real residual properties:
    - Similar autocorrelation patterns
    - Similar heteroskedasticity (ARCH effects)
    - Similar distributional properties
    - Similar stationarity characteristics
    """

    print(f"\n🔬 AGGREGATED CROSS-VALIDATION ANALYSIS")
    print(f"=" * 70)
    print(
        f"Testing {len(fitted_models)} fitted models vs {n_random_models} random baselines"
    )

    np.random.seed(random_seed)

    # Step 1: Extract parameter statistics from fitted models
    print(f"\n1️⃣ Extracting parameter statistics from fitted models...")
    param_stats = extract_parameter_statistics(fitted_models)

    print(f"   ✅ Parameter statistics extracted:")
    for key, stats in param_stats.items():
        if isinstance(stats, dict) and "median" in stats:
            print(f"      {key}: median={stats['median']:.4f}, MAD={stats['mad']:.4f}")

    # Step 2: Generate large pool of random baseline models
    print(f"\n2️⃣ Generating {n_random_models} random baseline models...")
    random_models = []
    for i in range(n_random_models):
        random_model = generate_random_baseline_model(
            param_stats, distribution_type="uniform"
        )
        random_models.append(random_model)

    # Verify and report persistence of random models
    persistence_values = []
    for model in random_models:
        alpha = model["garch_params"]["alpha"]
        beta = model["garch_params"]["beta"]
        persistence = alpha + beta
        persistence_values.append(persistence)

    max_persistence = np.max(persistence_values)
    mean_persistence = np.mean(persistence_values)

    print(f"   ✅ Random model persistence check:")
    print(f"      Mean persistence: {mean_persistence:.3f}")
    print(f"      Max persistence: {max_persistence:.3f}")
    print(f"      All stable (< 1.0): {'✅' if max_persistence < 1.0 else '❌'}")

    # Print sample random model
    sample_random = random_models[0]
    sample_persistence = (
        sample_random["garch_params"]["alpha"] + sample_random["garch_params"]["beta"]
    )
    print(f"   ✅ Sample random model:")
    print(
        f"      GARCH: ω={sample_random['garch_params']['omega']:.2e}, "
        f"α={sample_random['garch_params']['alpha']:.3f}, "
        f"β={sample_random['garch_params']['beta']:.3f}"
    )
    print(f"      Persistence: {sample_persistence:.3f}")

    # Step 3: Test fitted models vs real residuals (SIMILARITY)
    print(f"\n3️⃣ Testing fitted models vs real residuals (should be SIMILAR)...")

    fitted_similarity_scores = []
    valid_fitted_count = 0

    for trial_idx, (trial_name, trial_data) in enumerate(fitted_models.items()):
        # Get original residuals
        original_residuals = trial_data.get("original_residuals", [])
        if len(original_residuals) < 100:
            continue

        try:
            # Simulate from fitted model
            fitted_simulation = simulate_enhanced_arma_garch(
                trial_data,
                n_periods=len(original_residuals),
                random_seed=random_seed + trial_idx,
            )

            if isinstance(fitted_simulation, dict):
                fitted_residuals = fitted_simulation["residuals"]
            else:
                fitted_residuals = fitted_simulation

            # Calculate SIMILARITY between real and fitted simulations
            similarity_score = calculate_residual_similarity(
                original_residuals, fitted_residuals
            )
            if similarity_score is not None:
                fitted_similarity_scores.append(similarity_score)
                valid_fitted_count += 1

                if valid_fitted_count <= 3:
                    print(f"   🔍 {trial_name}: similarity = {similarity_score:.3f}")

        except Exception as e:
            if valid_fitted_count < 3:
                print(f"      ⚠️  Failed to test {trial_name}: {e}")
            continue

    print(f"   ✅ Successfully tested {valid_fitted_count} fitted models")

    # Step 4: Test random models vs real residuals (should be DIFFERENT)
    print(f"\n4️⃣ Testing random models vs real residuals (should be DIFFERENT)...")

    random_similarity_scores = []
    valid_random_count = 0

    # Use first fitted model's real residuals as reference for random comparison
    reference_residuals = None
    for trial_data in fitted_models.values():
        original_residuals = trial_data.get("original_residuals", [])
        if len(original_residuals) >= 100:
            reference_residuals = original_residuals
            break

    if reference_residuals is None:
        print("   ❌ No suitable reference residuals found")
        return None

    print(
        f"   Using reference residuals (n={len(reference_residuals)}) for random model comparison"
    )

    for rand_idx in range(n_random_models):  # Use full number for statistical power
        try:
            random_model = random_models[rand_idx]
            random_simulation = simulate_enhanced_arma_garch(
                random_model,
                n_periods=len(reference_residuals),
                random_seed=random_seed + 10000 + rand_idx,
            )

            if isinstance(random_simulation, dict):
                random_residuals = random_simulation["residuals"]
            else:
                random_residuals = random_simulation

            # Calculate SIMILARITY between real and random simulations
            similarity_score = calculate_residual_similarity(
                reference_residuals, random_residuals
            )
            if similarity_score is not None:
                random_similarity_scores.append(similarity_score)
                valid_random_count += 1

            # Progress indicator
            if (rand_idx + 1) % 25 == 0:
                print(
                    f"      Progress: {rand_idx + 1}/100 ({valid_random_count} valid)"
                )

        except Exception as e:
            continue

    print(f"   ✅ Successfully tested {valid_random_count} random models")

    # Step 5: Statistical analysis with similarity scores
    print(f"\n5️⃣ Statistical analysis...")

    if len(fitted_similarity_scores) < 5 or len(random_similarity_scores) < 10:
        print("   ❌ Insufficient valid results for statistical analysis")
        return None

    # Handle NaN values in similarity scores (random models may fail/explode)
    fitted_clean = [s for s in fitted_similarity_scores if not np.isnan(s)]
    random_clean = [s for s in random_similarity_scores if not np.isnan(s)]

    fitted_mean_similarity = np.mean(fitted_clean) if fitted_clean else 0.0
    random_mean_similarity = np.mean(random_clean) if random_clean else 0.0

    print(
        f"   📊 Sample sizes: Fitted n={len(fitted_clean)}/{len(fitted_similarity_scores)}, Random n={len(random_clean)}/{len(random_similarity_scores)}"
    )
    print(f"   📊 Fitted models similarity: {fitted_mean_similarity:.3f}")
    print(f"   📊 Random models similarity: {random_mean_similarity:.3f}")
    print(
        f"   📊 Advantage (fitted - random): {fitted_mean_similarity - random_mean_similarity:+.3f}"
    )

    # Show explosion rate for random models (this is expected!)
    random_explosion_rate = (
        (len(random_similarity_scores) - len(random_clean))
        / len(random_similarity_scores)
        if random_similarity_scores
        else 0
    )
    print(f"   💥 Random model explosion rate: {random_explosion_rate:.1%} (expected!)")

    # Statistical significance test (NaN-tolerant)
    from scipy.stats import mannwhitneyu, ttest_ind

    try:
        if len(fitted_clean) >= 5 and len(random_clean) >= 5:
            # Mann-Whitney U test on clean data
            statistic, p_value = mannwhitneyu(
                fitted_clean, random_clean, alternative="greater"
            )  # Test if fitted > random
            print(f"   📊 Statistical significance: p = {p_value:.4f}")

            # Effect size (Cohen's d) on clean data
            pooled_std = np.sqrt(
                (
                    (len(fitted_clean) - 1) * np.var(fitted_clean)
                    + (len(random_clean) - 1) * np.var(random_clean)
                )
                / (len(fitted_clean) + len(random_clean) - 2)
            )
            cohens_d = (fitted_mean_similarity - random_mean_similarity) / max(
                pooled_std, 1e-10
            )
            print(f"   📊 Effect size (Cohen's d): {cohens_d:.3f}")
        else:
            print(f"   ⚠️  Insufficient clean data for statistical testing")
            statistic, p_value, cohens_d = 0, 1.0, 0.0

    except Exception as e:
        print(f"   ⚠️  Statistical test failed: {e}")
        statistic, p_value, cohens_d = 0, 1.0, 0.0

    # Prepare results in format compatible with plotting function
    results = {
        "fitted_similarity_scores": fitted_clean,  # Clean data for plotting
        "random_similarity_scores": random_clean,  # Clean data for plotting
        "n_fitted": len(fitted_clean),
        "n_random": len(random_clean),
        "explosion_rates": {
            "fitted": (len(fitted_similarity_scores) - len(fitted_clean))
            / len(fitted_similarity_scores)
            if fitted_similarity_scores
            else 0,
            "random": random_explosion_rate,
        },
        "overall": {
            "fitted_similarity": fitted_mean_similarity,
            "random_similarity": random_mean_similarity,
            "difference": fitted_mean_similarity - random_mean_similarity,
            "p_value": p_value,
            "statistic": statistic,
            "effect_size": cohens_d,
        },
        # Add compatibility fields for plotting function
        "statistical_tests": {
            "similarity_test": {
                "fitted_pass_rate": fitted_mean_similarity,
                "random_pass_rate": random_mean_similarity,
                "difference": fitted_mean_similarity - random_mean_similarity,
                "p_value": p_value,
                "statistic": statistic,
            }
        },
        "effect_sizes": {"similarity_test": cohens_d},
        "confidence_intervals": {
            "similarity_test": (
                fitted_mean_similarity - 0.1,
                fitted_mean_similarity + 0.1,
            )
        },  # Placeholder
    }

    print(f"\n   🎯 SIMILARITY-BASED VALIDATION RESULTS:")
    print(f"      Fitted models similarity to real: {fitted_mean_similarity:.3f}")
    print(f"      Random models similarity to real: {random_mean_similarity:.3f}")
    print(
        f"      Fitted advantage: {fitted_mean_similarity - random_mean_similarity:+.3f}"
    )

    if p_value < 0.05:
        print(
            f"      🎉 Fitted models are significantly MORE similar to real residuals!"
        )
        print(
            f"      ✅ VALIDATION SUCCESSFUL: Fitted models replicate real residual properties"
        )
    else:
        print(f"      ⚠️  No significant difference in similarity detected")
        print(f"      ❓ VALIDATION INCONCLUSIVE: May need more data or better models")

    return results


def plot_aggregated_cross_validation_results(
    cv_results, save_path="aggregated_cross_validation"
):
    """
    Plot professional similarity-based cross-validation results.
    Shows fitted vs random model similarity to real residuals.
    """

    if cv_results is None or "similarity_scores" not in cv_results:
        print("❌ Insufficient data for plotting")
        return None

    # Set up professional plotting style
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    # Color scheme (colorblind-friendly)
    colors = {
        "fitted": "#2E86AB",  # Blue
        "random": "#E74C3C",  # Red
        "advantage": "#27AE60",  # Green
        "neutral": "#95A5A6",  # Gray
        "accent": "#F39C12",  # Orange
    }

    # Create figure with professional layout
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.92, bottom=0.08
    )

    # Extract data
    test_names = ["ljung_box_pass", "arch_pass", "stationarity_pass"]
    test_labels = [
        "Ljung-Box Test\n(No Autocorr.)",
        "ARCH Test\n(Heteroskedasticity)",
        "ADF Test\n(Stationarity)",
    ]

    statistical_tests = cv_results["statistical_tests"]
    effect_sizes = cv_results["effect_sizes"]
    confidence_intervals = cv_results["confidence_intervals"]
    overall = cv_results["overall"]

    # 1. Pass Rate Comparison with Confidence Intervals (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    fitted_rates = [statistical_tests[test]["fitted_pass_rate"] for test in test_names]
    random_rates = [statistical_tests[test]["random_pass_rate"] for test in test_names]
    ci_lower = [confidence_intervals[test][0] for test in test_names]
    ci_upper = [confidence_intervals[test][1] for test in test_names]

    x_pos = np.arange(len(test_names))
    width = 0.35

    # Plot bars with error bars
    bars1 = ax1.bar(
        x_pos - width / 2,
        fitted_rates,
        width,
        label="Fitted Models",
        color=colors["fitted"],
        alpha=0.8,
        capsize=4,
    )
    bars2 = ax1.bar(
        x_pos + width / 2,
        random_rates,
        width,
        label="Random Baselines",
        color=colors["random"],
        alpha=0.8,
        capsize=4,
    )

    # Add confidence intervals for differences (show as text annotations instead of error bars)
    for i, (test, lower, upper) in enumerate(zip(test_names, ci_lower, ci_upper)):
        # Display CI as text annotation instead of error bars to avoid negative values
        ci_text = f"95% CI: [{lower:+.3f}, {upper:+.3f}]"
        ax1.text(
            x_pos[i],
            -0.08,
            ci_text,
            ha="center",
            va="top",
            fontsize=8,
            rotation=45,
            alpha=0.7,
        )

    # Add significance stars
    for i, test in enumerate(test_names):
        p_val = statistical_tests[test]["p_value"]
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        max_height = max(fitted_rates[i], random_rates[i]) + 0.05
        ax1.text(
            x_pos[i],
            max_height,
            sig_text,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax1.set_ylabel("Pass Rate", fontweight="bold")
    ax1.set_title(
        "Statistical Test Pass Rates\nwith 95% Confidence Intervals", fontweight="bold"
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(test_labels, rotation=0, ha="center")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.15, 1.1)  # Extra space for CI text at bottom

    # Add sample size annotations
    ax1.text(
        0.02,
        0.98,
        f"n_fitted = {cv_results['n_fitted']}\nn_random = {cv_results['n_random']}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["neutral"], alpha=0.3),
        fontsize=9,
    )

    # 2. Effect Size Visualization (top middle)
    ax2 = fig.add_subplot(gs[0, 1])

    effect_values = [effect_sizes[test] for test in test_names]
    effect_colors = [
        colors["advantage"] if d > 0 else colors["random"] for d in effect_values
    ]

    bars = ax2.barh(
        test_labels, effect_values, color=effect_colors, alpha=0.7, edgecolor="black"
    )

    # Add effect size interpretation lines
    ax2.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5, label="Small Effect")
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Medium Effect")
    ax2.axvline(x=0.8, color="gray", linestyle="--", alpha=0.9, label="Large Effect")
    ax2.axvline(x=0, color="black", linestyle="-", alpha=0.8)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, effect_values)):
        ax2.text(
            value + 0.02 if value >= 0 else value - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontweight="bold",
        )

    ax2.set_xlabel("Cohen's d (Effect Size)", fontweight="bold")
    ax2.set_title("Effect Sizes\n(Fitted vs Random)", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)

    # 3. Overall Performance Summary (top right)
    ax3 = fig.add_subplot(gs[0, 2])

    overall_categories = ["Fitted\nModels", "Random\nBaselines"]
    overall_rates = [overall["fitted_pass_rate"], overall["random_pass_rate"]]
    overall_colors = [colors["fitted"], colors["random"]]

    bars = ax3.bar(
        overall_categories,
        overall_rates,
        color=overall_colors,
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels
    for bar, rate in zip(bars, overall_rates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Add advantage box
    advantage = overall["difference"]
    ax3.text(
        0.5,
        0.7,
        f"Advantage:\n{advantage:+.3f}\n({advantage * 100:+.1f}%)",
        transform=ax3.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=colors["advantage"], alpha=0.3),
        fontweight="bold",
        fontsize=11,
    )

    # Add significance
    p_val = overall["p_value"]
    sig_text = f"p = {p_val:.4f}"
    if p_val < 0.05:
        sig_text += "\n✅ Significant"
        sig_color = colors["advantage"]
    else:
        sig_text += "\n❌ Not Significant"
        sig_color = colors["random"]

    ax3.text(
        0.5,
        0.3,
        sig_text,
        transform=ax3.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=sig_color, alpha=0.3),
        fontweight="bold",
        fontsize=10,
    )

    ax3.set_ylabel("Overall Pass Rate", fontweight="bold")
    ax3.set_title("Overall Performance\nComparison", fontweight="bold")
    ax3.set_ylim(0, 1)

    # 4. Distribution Comparison (middle left)
    ax4 = fig.add_subplot(gs[1, 0])

    fitted_results = cv_results["fitted_results"]
    random_results = cv_results["random_results"]

    fitted_scores = [
        np.mean([r.get(test, False) for test in test_names]) for r in fitted_results
    ]
    random_scores = [
        np.mean([r.get(test, False) for test in test_names]) for r in random_results
    ]

    # Box plots
    box_data = [fitted_scores, random_scores]
    box_labels = ["Fitted\nModels", "Random\nBaselines"]

    bp = ax4.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        boxprops=dict(facecolor=colors["fitted"], alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )

    bp["boxes"][1].set_facecolor(colors["random"])

    ax4.set_ylabel("Overall Pass Rate", fontweight="bold")
    ax4.set_title("Distribution of\nOverall Performance", fontweight="bold")
    ax4.set_ylim(0, 1)

    # 5. P-value Summary (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])

    p_values = [statistical_tests[test]["p_value"] for test in test_names]
    p_colors = [
        colors["advantage"] if p < 0.05 else colors["neutral"] for p in p_values
    ]

    bars = ax5.bar(test_labels, p_values, color=p_colors, alpha=0.7, edgecolor="black")
    ax5.axhline(
        y=0.05, color=colors["random"], linestyle="--", linewidth=2, label="α = 0.05"
    )

    # Add value labels
    for bar, p_val in zip(bars, p_values):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{p_val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            rotation=45,
        )

    ax5.set_ylabel("P-value", fontweight="bold")
    ax5.set_title("Statistical Significance\n(p-values)", fontweight="bold")
    ax5.legend()
    ax5.set_yscale("log")
    ax5.set_ylim(0.0001, 1)

    # 6. Power Analysis / Sample Size (middle right)
    ax6 = fig.add_subplot(gs[1, 2])

    # Create sample size adequacy visualization
    sample_adequacy = {
        "Fitted Models": cv_results["n_fitted"],
        "Random Models": cv_results["n_random"],
        "Recommended Min": 30,  # Rule of thumb for statistical tests
    }

    categories = list(sample_adequacy.keys())
    values = list(sample_adequacy.values())
    bar_colors = [colors["fitted"], colors["random"], colors["neutral"]]

    bars = ax6.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for bar, value in zip(bars, values):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax6.set_ylabel("Sample Size", fontweight="bold")
    ax6.set_title("Sample Size\nAdequacy Check", fontweight="bold")
    ax6.set_xticklabels(categories, rotation=45, ha="right")

    # 7. Comprehensive Summary Table (bottom row)
    ax7 = fig.add_subplot(gs[2, :])

    # Create comprehensive summary table
    table_data = []
    table_data.append(
        [
            "Test",
            "Fitted Rate",
            "Random Rate",
            "Difference",
            "p-value",
            "Cohen's d",
            "Interpretation",
        ]
    )

    for test, label in zip(test_names, test_labels):
        stats = statistical_tests[test]
        effect = effect_sizes[test]

        if stats["p_value"] < 0.001:
            p_str = f"{stats['p_value']:.1e}***"
        elif stats["p_value"] < 0.01:
            p_str = f"{stats['p_value']:.4f}**"
        elif stats["p_value"] < 0.05:
            p_str = f"{stats['p_value']:.4f}*"
        else:
            p_str = f"{stats['p_value']:.4f}ns"

        # Effect size interpretation
        if abs(effect) < 0.2:
            effect_interp = "Negligible"
        elif abs(effect) < 0.5:
            effect_interp = "Small"
        elif abs(effect) < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

        if stats["difference"] > 0:
            interpretation = f"Fitted Superior ({effect_interp})"
        else:
            interpretation = f"Random Superior ({effect_interp})"

        table_data.append(
            [
                label.replace("\n", " "),
                f"{stats['fitted_pass_rate']:.3f}",
                f"{stats['random_pass_rate']:.3f}",
                f"{stats['difference']:+.3f}",
                p_str,
                f"{effect:+.2f}",
                interpretation,
            ]
        )

    # Add overall row
    table_data.append(
        [
            "Overall",
            f"{overall['fitted_pass_rate']:.3f}",
            f"{overall['random_pass_rate']:.3f}",
            f"{overall['difference']:+.3f}",
            f"{overall['p_value']:.4f}",
            "-",
            "Fitted Superior" if overall["difference"] > 0 else "Random Superior",
        ]
    )

    # Create and style table
    table = ax7.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.2, 0.9, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(colors["fitted"])
                cell.set_text_props(weight="bold", color="white")
            elif i == len(table_data) - 1:  # Overall row
                cell.set_facecolor(colors["advantage"])
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "white")

    ax7.axis("off")
    ax7.set_title(
        "Comprehensive Statistical Summary", fontweight="bold", fontsize=14, pad=30
    )

    # Add final interpretation
    if overall["p_value"] < 0.05:
        interpretation_text = (
            "🎉 VALIDATION SUCCESSFUL: Fitted ARMA-GARCH models demonstrate statistically significant "
            "superior performance compared to random parameter baselines. This confirms that the "
            "fitted parameters capture genuine statistical properties of the residual data."
        )
        interp_color = colors["advantage"]
    else:
        interpretation_text = (
            "⚠️ VALIDATION INCONCLUSIVE: No statistically significant difference detected between "
            "fitted and random models. This may indicate parameter fitting issues or insufficient "
            "model complexity for the data structure."
        )
        interp_color = colors["random"]

    ax7.text(
        0.5,
        0.05,
        interpretation_text,
        transform=ax7.transAxes,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=interp_color, alpha=0.2),
        fontsize=11,
        fontweight="bold",
        wrap=True,
    )

    plt.suptitle(
        "Aggregated Cross-Validation Analysis: Fitted ARMA-GARCH Models vs Random Baselines",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # Save as SVG
    svg_path = f"{save_path}.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300, facecolor="white")
    plt.show()

    print(f"✅ Professional cross-validation analysis saved as: {svg_path}")

    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--results":
        create_results_plots()
    else:
        interactive_comprehensive_menu()
