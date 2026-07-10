#!/usr/bin/env python3
"""
Enhanced ARMA-GARCH Model for Residual Analysis
===============================================

Implements ARMA-GARCH models with automatic order selection and
multiple distribution options for modeling elbow trajectory residuals.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedARMAGARCH:
    """
    Enhanced ARMA-GARCH model with automatic order selection and diagnostics.
    """
    
    def __init__(self, auto_select_orders: bool = True,
                 max_p: int = 3, max_q: int = 3,
                 max_r: int = 2, max_s: int = 2):
        """
        Initialize ARMA-GARCH model.
        
        Args:
            auto_select_orders: Whether to automatically select model orders using BIC
            max_p, max_q: Maximum ARMA orders
            max_r, max_s: Maximum GARCH orders
        """
        self.auto_select_orders = auto_select_orders
        self.max_p = max_p
        self.max_q = max_q
        self.max_r = max_r
        self.max_s = max_s
        
        self.optimal_orders = None
        self.arma_model = None
        self.garch_models = {}
        self.standardized_residuals = {}
        self.fitted = False
        
    def select_optimal_orders(self, residuals: np.ndarray, verbose: bool = True) -> Dict:
        """
        Select optimal ARMA and GARCH orders using BIC.
        """
        best_bic = np.inf
        best_orders = {'p': 1, 'q': 1, 'r': 1, 's': 1}
        
        # Center residuals
        residuals_centered = residuals - np.mean(residuals)
        
        if verbose:
            print("Selecting optimal orders using BIC...")
        
        # Try different ARMA orders
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                    
                try:
                    # Fit ARMA model
                    model = ARIMA(residuals_centered, order=(p, 0, q))
                    fitted = model.fit(method='yule_walker' if p > 0 else 'innovations_mle')
                    
                    # Get ARMA residuals for GARCH fitting
                    arma_resid = fitted.resid
                    
                    # Try different GARCH orders on ARMA residuals
                    for r in range(1, self.max_r + 1):
                        for s in range(1, self.max_s + 1):
                            try:
                                # Fit GARCH model
                                garch = arch_model(arma_resid, vol='Garch', p=r, q=s,
                                                 dist='normal', rescale=False)
                                garch_fit = garch.fit(disp='off', show_warning=False)
                                
                                # Calculate combined BIC
                                total_bic = fitted.bic + garch_fit.bic
                                
                                if total_bic < best_bic:
                                    best_bic = total_bic
                                    best_orders = {'p': p, 'q': q, 'r': r, 's': s}
                                    
                            except:
                                continue
                                
                except:
                    continue
        
        if verbose:
            print(f"  Optimal orders: ARMA({best_orders['p']},{best_orders['q']}) "
                  f"GARCH({best_orders['r']},{best_orders['s']}) with BIC={best_bic:.2f}")
        
        return best_orders
    
    def fit(self, residuals: np.ndarray, 
            distributions: List[str] = ['normal', 't'],
            verbose: bool = True) -> Dict:
        """
        Fit ARMA-GARCH model with multiple distributions.
        
        Args:
            residuals: Residual time series
            distributions: List of distributions to try ['normal', 't', 'ged']
            verbose: Whether to print fitting progress
            
        Returns:
            Dictionary with fitting results
        """
        # Center residuals
        self.residuals = residuals
        self.residuals_centered = residuals - np.mean(residuals)
        
        # Select optimal orders if requested
        if self.auto_select_orders:
            self.optimal_orders = self.select_optimal_orders(residuals, verbose)
        else:
            self.optimal_orders = {'p': 1, 'q': 1, 'r': 1, 's': 1}
        
        # Fit ARMA model
        if verbose:
            print(f"\nFitting ARMA({self.optimal_orders['p']},{self.optimal_orders['q']}) model...")
        
        if self.optimal_orders['p'] > 0 or self.optimal_orders['q'] > 0:
            arma = ARIMA(self.residuals_centered, 
                        order=(self.optimal_orders['p'], 0, self.optimal_orders['q']))
            self.arma_model = arma.fit(method='yule_walker' if self.optimal_orders['p'] > 0 
                                      else 'innovations_mle')
            arma_residuals = self.arma_model.resid
        else:
            arma_residuals = self.residuals_centered
            self.arma_model = None
        
        # Fit GARCH models with different distributions
        if verbose:
            print(f"Fitting GARCH({self.optimal_orders['r']},{self.optimal_orders['s']}) "
                  f"with distributions: {distributions}")
        
        best_bic = np.inf
        best_dist = None
        
        for dist in distributions:
            try:
                # Map distribution names
                arch_dist = dist
                if dist == 'normal':
                    arch_dist = 'normal'
                elif dist == 't':
                    arch_dist = 't'
                elif dist == 'ged':
                    arch_dist = 'ged'
                
                # Fit GARCH model
                garch = arch_model(arma_residuals, 
                                 vol='Garch',
                                 p=self.optimal_orders['r'], 
                                 q=self.optimal_orders['s'],
                                 dist=arch_dist,
                                 rescale=False)
                
                garch_fit = garch.fit(disp='off', show_warning=False)
                self.garch_models[dist] = garch_fit
                
                # Extract standardized residuals
                self.standardized_residuals[dist] = garch_fit.std_resid
                
                # Track best model
                if garch_fit.bic < best_bic:
                    best_bic = garch_fit.bic
                    best_dist = dist
                    
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not fit {dist} distribution: {e}")
                continue
        
        self.best_distribution = best_dist
        self.fitted = True
        
        # Prepare results
        results = {
            'optimal_orders': self.optimal_orders,
            'best_distribution': best_dist,
            'arma_params': self._extract_arma_params(),
            'garch_params': self._extract_garch_params(best_dist),
            'distribution_comparison': self._compare_distributions(),
            'volatility_persistence': self._calculate_volatility_persistence(best_dist),
            'volatility_halflife': self._calculate_volatility_halflife(best_dist)
        }
        
        if verbose:
            print(f"  Best distribution: {best_dist} (BIC={best_bic:.2f})")
            print(f"  Volatility persistence: {results['volatility_persistence']:.4f}")
        
        return results
    
    def _extract_arma_params(self) -> Dict:
        """Extract ARMA parameters."""
        if self.arma_model is None:
            return {}
            
        params = {}
        
        # AR parameters
        for i in range(self.optimal_orders['p']):
            param_name = f'ar.L{i+1}'
            if param_name in self.arma_model.params:
                params[f'phi_{i+1}'] = self.arma_model.params[param_name]
        
        # MA parameters
        for i in range(self.optimal_orders['q']):
            param_name = f'ma.L{i+1}'
            if param_name in self.arma_model.params:
                params[f'theta_{i+1}'] = self.arma_model.params[param_name]
        
        return params
    
    def _extract_garch_params(self, dist: str) -> Dict:
        """Extract GARCH parameters for given distribution."""
        if dist not in self.garch_models:
            return {}
            
        model = self.garch_models[dist]
        params = {
            'omega': model.params.get('omega', np.nan),
            'alpha': model.params.get('alpha[1]', np.nan),
            'beta': model.params.get('beta[1]', np.nan)
        }
        
        # Add higher order parameters if present
        for i in range(2, self.optimal_orders['r'] + 1):
            if f'alpha[{i}]' in model.params:
                params[f'alpha_{i}'] = model.params[f'alpha[{i}]']
                
        for i in range(2, self.optimal_orders['s'] + 1):
            if f'beta[{i}]' in model.params:
                params[f'beta_{i}'] = model.params[f'beta[{i}]']
        
        return params
    
    def _calculate_volatility_persistence(self, dist: str) -> float:
        """Calculate volatility persistence (sum of alpha + beta)."""
        if dist not in self.garch_models:
            return np.nan
            
        model = self.garch_models[dist]
        persistence = 0
        
        # Sum all alpha parameters
        for i in range(1, self.optimal_orders['r'] + 1):
            param_name = f'alpha[{i}]' if i > 1 else 'alpha[1]'
            if param_name in model.params:
                persistence += model.params[param_name]
        
        # Sum all beta parameters
        for i in range(1, self.optimal_orders['s'] + 1):
            param_name = f'beta[{i}]' if i > 1 else 'beta[1]'
            if param_name in model.params:
                persistence += model.params[param_name]
        
        return persistence
    
    def _calculate_volatility_halflife(self, dist: str) -> float:
        """Calculate half-life of volatility shocks."""
        persistence = self._calculate_volatility_persistence(dist)
        
        if np.isnan(persistence) or persistence <= 0 or persistence >= 1:
            return np.nan
            
        # Half-life formula: log(0.5) / log(persistence)
        halflife = np.log(0.5) / np.log(persistence)
        return halflife
    
    def _compare_distributions(self) -> Dict:
        """Compare different distribution fits."""
        comparison = {}
        
        for dist, model in self.garch_models.items():
            comparison[dist] = {
                'bic': model.bic,
                'aic': model.aic,
                'loglikelihood': model.loglikelihood
            }
            
            # Add distribution-specific parameters
            if dist == 't' and 'nu' in model.params:
                comparison[dist]['nu'] = model.params['nu']
            elif dist == 'ged' and 'nu' in model.params:
                comparison[dist]['nu'] = model.params['nu']
        
        return comparison
    
    def test_innovation_whiteness(self, verbose: bool = True) -> Dict:
        """
        Test if standardized residuals are white noise.
        """
        if not self.fitted or self.best_distribution not in self.standardized_residuals:
            raise ValueError("Model must be fitted first")
        
        std_resid = self.standardized_residuals[self.best_distribution]
        
        results = {
            'distribution': self.best_distribution,
            'whiteness_score': 0,
            'is_white_noise': False,
            'tests': {}
        }
        
        # Ljung-Box test for autocorrelation
        lb_result = acorr_ljungbox(std_resid, lags=20, return_df=True)
        lb_pass = (lb_result['lb_pvalue'] > 0.05).sum() / len(lb_result)
        results['tests']['ljung_box'] = {
            'pass_rate': lb_pass,
            'is_white': lb_pass > 0.8
        }
        if lb_pass > 0.8:
            results['whiteness_score'] += 1
        
        # Test for squared residuals (volatility clustering)
        std_resid_sq = std_resid ** 2
        lb_sq_result = acorr_ljungbox(std_resid_sq, lags=20, return_df=True)
        lb_sq_pass = (lb_sq_result['lb_pvalue'] > 0.05).sum() / len(lb_sq_result)
        results['tests']['ljung_box_squared'] = {
            'pass_rate': lb_sq_pass,
            'no_clustering': lb_sq_pass > 0.8
        }
        if lb_sq_pass > 0.8:
            results['whiteness_score'] += 1
        
        # Check if residuals are centered around zero
        t_stat, p_value = stats.ttest_1samp(std_resid, 0)
        results['tests']['zero_mean'] = {
            'p_value': p_value,
            'is_zero_mean': p_value > 0.05
        }
        if p_value > 0.05:
            results['whiteness_score'] += 1
        
        # Overall assessment
        results['whiteness_score'] = results['whiteness_score'] / 3.0
        results['is_white_noise'] = results['whiteness_score'] >= 0.66
        
        if verbose:
            print(f"\nInnovation Whiteness Tests ({self.best_distribution}):")
            print(f"  Ljung-Box pass rate: {lb_pass:.1%}")
            print(f"  Ljung-Box (squared) pass rate: {lb_sq_pass:.1%}")
            print(f"  Zero mean test p-value: {p_value:.4f}")
            print(f"  Overall whiteness score: {results['whiteness_score']:.2f}/1.0")
            print(f"  Is white noise: {results['is_white_noise']}")
        
        return results
    
    def test_distribution_goodness_of_fit(self, verbose: bool = True) -> Dict:
        """
        Test goodness of fit for the selected distribution.
        """
        if not self.fitted or self.best_distribution not in self.standardized_residuals:
            raise ValueError("Model must be fitted first")
        
        std_resid = self.standardized_residuals[self.best_distribution]
        
        results = {
            'distribution': self.best_distribution,
            'fit_score': 0,
            'good_fit': False,
            'tests': {}
        }
        
        # Normality test for standardized residuals
        if self.best_distribution == 'normal':
            # Should be N(0,1)
            ks_stat, ks_p = stats.kstest(std_resid, 'norm')
            results['tests']['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'pass': ks_p > 0.05
            }
            if ks_p > 0.05:
                results['fit_score'] += 1
                
        elif self.best_distribution == 't':
            # Test against t-distribution
            model = self.garch_models[self.best_distribution]
            if 'nu' in model.params:
                nu = model.params['nu']
                # Standardize to t-distribution
                ks_stat, ks_p = stats.kstest(std_resid, lambda x: stats.t.cdf(x, nu))
                results['tests']['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'pass': ks_p > 0.05,
                    'nu': nu
                }
                if ks_p > 0.05:
                    results['fit_score'] += 1
        
        # Q-Q plot correlation test
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(std_resid)))
        empirical_quantiles = np.sort(std_resid)
        qq_corr = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
        
        results['tests']['qq_correlation'] = {
            'correlation': qq_corr,
            'good_fit': qq_corr > 0.95
        }
        if qq_corr > 0.95:
            results['fit_score'] += 1
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(std_resid)
        results['tests']['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'pass': jb_p > 0.05
        }
        if jb_p > 0.05:
            results['fit_score'] += 1
        
        # Overall assessment
        results['fit_score'] = results['fit_score'] / 3.0
        results['good_fit'] = results['fit_score'] >= 0.66
        
        if verbose:
            print(f"\nDistribution Goodness-of-Fit ({self.best_distribution}):")
            for test_name, test_results in results['tests'].items():
                if 'p_value' in test_results:
                    print(f"  {test_name}: p-value={test_results['p_value']:.4f}, "
                          f"pass={test_results.get('pass', test_results.get('good_fit', False))}")
            print(f"  Overall fit score: {results['fit_score']:.2f}/1.0")
            print(f"  Good fit: {results['good_fit']}")
        
        return results
    
    def simulate(self, n_periods: int, n_simulations: int = 1000, 
                random_seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate future values using the fitted ARMA-GARCH model.
        
        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of simulated values (n_simulations, n_periods)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get the best GARCH model
        garch_model = self.garch_models[self.best_distribution]
        
        # Simulate from GARCH model
        simulations = garch_model.forecast(horizon=n_periods, 
                                          method='simulation',
                                          simulations=n_simulations)
        
        # Extract simulated values
        # Handle different return formats from arch package
        if hasattr(simulations.simulations, 'values'):
            # Pandas DataFrame format
            simulated_values = simulations.simulations.values[-1, :, :].T
        else:
            # NumPy array format
            simulated_values = simulations.simulations[-1, :, :].T
        
        # Add back ARMA effects if present
        if self.arma_model is not None:
            # This is simplified - proper ARMA simulation would be more complex
            arma_forecast = self.arma_model.forecast(steps=n_periods)
            simulated_values += arma_forecast.values.reshape(-1, 1)
        
        # Add back mean
        simulated_values += np.mean(self.residuals)
        
        return simulated_values