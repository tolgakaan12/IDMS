"""
Enhanced ARMA-GARCH Residual Model with Innovation Testing
=========================================================
Features:
- Automatic order selection via BIC grid search
- Multiple distribution support (Normal, Student-t, GED, Skewed-t)
- Innovation quality testing and diagnostics
- Distribution goodness-of-fit testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.sandbox.stats.runs import runstest_1samp
from typing import Tuple, Dict, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')


class EnhancedARMAGARCH:
    """
    Enhanced ARMA(p,q)-GARCH(r,s) model with automatic order selection and innovation testing.
    
    Features:
    - Automatic order selection via BIC grid search
    - Multiple distribution support (Normal, Student-t, GED, Skewed-t)
    - Innovation quality testing and diagnostics
    - Distribution goodness-of-fit testing
    """
    
    def __init__(self, p: Optional[int] = None, q: Optional[int] = None, 
                 r: Optional[int] = None, s: Optional[int] = None,
                 auto_select_orders: bool = True):
        # Model orders (None means auto-select)
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.auto_select_orders = auto_select_orders
        
        # Model components
        self.arma_model = None
        self.garch_models = {}  # Store models for different distributions
        self.fitted = False
        
        # Data storage
        self.residuals = None
        self.standardized_residuals = {}  # Per distribution
        self.distribution_results = {}
        self.best_distribution = None
        
        # Parameters
        self.scaling_factor = 1.0  # No scaling needed - work with natural parameters
        
        # Results storage
        self.order_selection_results = {}
        self.innovation_tests = {}
        self.goodness_of_fit_tests = {}

        
    def fit(self, residuals: np.ndarray, distributions: List[str] = ['normal', 't'], 
            verbose: bool = True) -> Dict:
        """
        Fit ARMA-GARCH model with automatic order selection and multiple distributions.
        
        Parameters:
        -----------
        residuals : np.ndarray
            Time series of residuals to model
        distributions : List[str]
            Distributions to test ['normal', 't', 'ged', 'skewt']
        verbose : bool
            Print progress information
        """
        # Center the residuals
        self.residuals = residuals - np.mean(residuals)
        
        # Scale for numerical stability
        scaled_residuals = self.residuals * self.scaling_factor
        
        if verbose:
            print(f"Enhanced ARMA-GARCH Model Fitting")
            print(f"="*50)
            print(f"Residual length: {len(self.residuals)}")
            print(f"Residual mean: {np.mean(self.residuals):.6f}")
            print(f"Residual std: {np.std(self.residuals):.6f}")
            print(f"Auto-select orders: {self.auto_select_orders}")
            print(f"Distributions to test: {distributions}")
        
        # Step 1: Select optimal ARMA order
        if self.auto_select_orders and (self.p is None or self.q is None):
            if verbose:
                print(f"\nStep 1: Selecting optimal ARMA order...")
            optimal_p, optimal_q, arma_results = self.select_optimal_arma_order(
                scaled_residuals, verbose=verbose)
            self.p, self.q = optimal_p, optimal_q
            self.order_selection_results['arma'] = arma_results
        
        # Step 2: Fit ARMA model
        if verbose:
            print(f"\nStep 2: Fitting ARMA({self.p},{self.q}) model...")
        
        try:
            self.arma_model = ARIMA(
                scaled_residuals,
                order=(self.p, 0, self.q),
                trend='n',
                enforce_invertibility=True,
                enforce_stationarity=True
            ).fit()
            
            arma_residuals = self.arma_model.resid
            
        except Exception as e:
            if verbose:
                print(f"ARMA fitting failed: {e}")
            raise
        
        # Step 3: Select optimal GARCH order
        if self.auto_select_orders and (self.r is None or self.s is None):
            if verbose:
                print(f"\nStep 3: Selecting optimal GARCH order...")
            optimal_r, optimal_s, garch_results = self.select_optimal_garch_order(
                arma_residuals, verbose=verbose)
            self.r, self.s = optimal_r, optimal_s
            self.order_selection_results['garch'] = garch_results
        
        # Step 4: Fit GARCH models with different distributions
        if verbose:
            print(f"\nStep 4: Fitting GARCH({self.r},{self.s}) models...")
        
        for dist in distributions:
            if verbose:
                print(f"  Fitting with {dist} distribution...")
            
            try:
                garch_model = arch_model(
                    arma_residuals,
                    mean='Zero',
                    vol='GARCH',
                    p=self.s,
                    q=self.r,
                    dist=dist
                ).fit(disp='off')
                
                self.garch_models[dist] = garch_model
                
                # Extract standardized residuals
                std_resid = garch_model.std_resid
                std_resid_clean = std_resid[~np.isnan(std_resid)]
                self.standardized_residuals[dist] = std_resid_clean
                
                # Store results
                self.distribution_results[dist] = {
                    'aic': garch_model.aic,
                    'bic': garch_model.bic,
                    'loglikelihood': garch_model.loglikelihood,
                    'converged': garch_model.convergence_flag
                }
                
                if verbose:
                    print(f"    {dist}: AIC={garch_model.aic:.2f}, BIC={garch_model.bic:.2f}")
                
            except Exception as e:
                if verbose:
                    print(f"    {dist}: Failed - {e}")
                continue
        
        # Step 5: Select best distribution
        if self.distribution_results:
            self.best_distribution = min(self.distribution_results.keys(), 
                                       key=lambda x: self.distribution_results[x]['bic'])
            if verbose:
                print(f"\nBest distribution by BIC: {self.best_distribution}")
        
        self.fitted = True
        
        # Compile results
        results = {
            'optimal_orders': {'p': self.p, 'q': self.q, 'r': self.r, 's': self.s},
            'order_selection': self.order_selection_results,
            'distribution_comparison': self.distribution_results,
            'best_distribution': self.best_distribution,
            'arma_params': self._extract_arma_params(),
            'garch_params': self._extract_garch_params(),
            'volatility_persistence': self._calculate_persistence(),
            'volatility_halflife': self._calculate_halflife()
        }
        
        if verbose:
            self._print_enhanced_summary(results)
        
        return results
    
    def select_optimal_arma_order(self, residuals: np.ndarray, 
                                 p_max: int = 4, q_max: int = 3, 
                                 verbose: bool = True) -> Tuple[int, int, Dict]:
        """
        Select optimal ARMA order using BIC grid search.
        
        Returns:
        --------
        optimal_p, optimal_q, results_dict
        """
        if verbose:
            print(f"    BIC grid search: p=[0,{p_max}], q=[0,{q_max}]")
        
        best_bic = np.inf
        best_p, best_q = 1, 1
        results = {}
        
        for p in range(0, p_max + 1):
            for q in range(0, q_max + 1):
                if p == 0 and q == 0:  # Skip (0,0) model
                    continue
                    
                try:
                    model = ARIMA(residuals, order=(p, 0, q), trend='n',
                                enforce_invertibility=True, 
                                enforce_stationarity=True).fit()
                    
                    bic = model.bic
                    results[(p, q)] = {'bic': bic, 'aic': model.aic}
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_p, best_q = p, q
                    
                    if verbose and (p + q) <= 3:  # Only print for small orders
                        print(f"      ARMA({p},{q}): BIC={bic:.2f}")
                        
                except:
                    results[(p, q)] = {'bic': np.inf, 'aic': np.inf}
                    continue
        
        if verbose:
            print(f"    → Optimal: ARMA({best_p},{best_q}) with BIC={best_bic:.2f}")
        
        return best_p, best_q, results
    
    def select_optimal_garch_order(self, arma_residuals: np.ndarray,
                                  r_max: int = 1, s_max: int = 2,
                                #   r_max: int = 2, s_max: int = 2,
                                  verbose: bool = True) -> Tuple[int, int, Dict]:
        """
        Select optimal GARCH order using BIC grid search.
        
        Returns:
        --------
        optimal_r, optimal_s, results_dict
        """
        if verbose:
            print(f"    BIC grid search: r=[1,{r_max}], s=[1,{s_max}]")
        
        best_bic = np.inf
        best_r, best_s = 1, 1
        results = {}
        
        for r in range(1, r_max + 1):
            for s in range(1, s_max + 1):
                try:
                    model = arch_model(arma_residuals, mean='Zero', vol='GARCH',
                                     p=s, q=r, dist='normal').fit(disp='off')
                    
                    bic = model.bic
                    results[(r, s)] = {'bic': bic, 'aic': model.aic}
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_r, best_s = r, s
                    
                    if verbose:
                        print(f"      GARCH({r},{s}): BIC={bic:.2f}")
                        
                except:
                    results[(r, s)] = {'bic': np.inf, 'aic': np.inf}
                    continue
        
        if verbose:
            print(f"    → Optimal: GARCH({best_r},{best_s}) with BIC={best_bic:.2f}")
        
        return best_r, best_s, results
    
    def simulate(self, n_samples: int, distribution: Optional[str] = None, 
                 random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate from fitted ARMA-GARCH model.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to simulate
        distribution : str, optional
            Distribution to use for simulation. If None, uses best distribution.
        random_state : int, optional
            Random seed for reproducibility
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        dist = distribution or self.best_distribution
        if dist not in self.garch_models:
            raise ValueError(f"Distribution {dist} not available")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Extract parameters for the specified distribution
        arma_params = self._extract_arma_params()
        garch_params = self._extract_garch_params(dist)
        garch_model = self.garch_models[dist]
        
        # Get initial conditions from fitted model (in scaled space)
        eps_scaled = self.arma_model.resid
        
        # Initial values for GARCH recursion
        e_prev = eps_scaled[-1]  # Last ARMA residual (scaled)
        
        # Get last conditional volatility squared
        if hasattr(garch_model, 'conditional_volatility'):
            s2_prev = garch_model.conditional_volatility[-1]**2
        else:
            persistence = self._calculate_persistence()
            s2_prev = garch_params['omega'] / (1 - persistence) if persistence < 1 else garch_params['omega']
        
        # Initialize arrays (all in scaled space)
        y_scaled = np.zeros(n_samples)
        epsilon_scaled = np.zeros(n_samples)
        sigma2 = np.zeros(n_samples)
        
        # Generate innovations based on distribution
        if dist == 't':
            nu = garch_model.params.get('nu', 5)
            z = np.random.standard_t(nu, size=n_samples)
            # Scale to unit variance
            z_std = z * np.sqrt((nu - 2) / nu) if nu > 2 else z
        elif dist == 'normal':
            z_std = np.random.standard_normal(n_samples)
        elif dist == 'ged':
            # For GED, use normal as approximation (can be improved)
            z_std = np.random.standard_normal(n_samples)
        else:
            # Default to normal
            z_std = np.random.standard_normal(n_samples)
        
        # Get initial y values for AR recursion (from scaled data)
        scaled_residuals = self.residuals * self.scaling_factor
        if self.p > 0:
            y_hist = list(scaled_residuals[-self.p:])
        else:
            y_hist = []
            
        # Get initial epsilon values for MA recursion (from ARMA residuals)
        if self.q > 0:
            e_hist = list(eps_scaled[-self.q:])
        else:
            e_hist = []
        
        # Simulate process (all in scaled space)
        for t in range(n_samples):
            # GARCH variance equation (in scaled space)
            sigma2[t] = garch_params['omega'] + \
                        garch_params.get('alpha', 0.0) * e_prev**2 + \
                        garch_params.get('beta', 0.0) * s2_prev
            
            # Innovation (scaled space)
            sigma_t = np.sqrt(sigma2[t])
            epsilon_scaled[t] = sigma_t * z_std[t]
            
            # ARMA process (scaled space)
            ar = 0
            if self.p > 0 and len(y_hist) >= self.p:
                ar_coefs = [arma_params.get(f'phi_{i}', 0.0) for i in range(1, self.p+1)]
                ar = np.dot(ar_coefs, y_hist[-self.p:][::-1])
            
            ma = 0
            if self.q > 0 and len(e_hist) >= self.q:
                ma_coefs = [arma_params.get(f'theta_{j}', 0.0) for j in range(1, self.q+1)]
                ma = np.dot(ma_coefs, e_hist[-self.q:][::-1])
            
            y_scaled[t] = ar + ma + epsilon_scaled[t]
            
            # Update history (all in scaled space)
            if self.p > 0:
                y_hist.append(y_scaled[t])
                if len(y_hist) > self.p:
                    y_hist.pop(0)
                    
            if self.q > 0:
                e_hist.append(epsilon_scaled[t])
                if len(e_hist) > self.q:
                    e_hist.pop(0)
            
            # Update for next iteration - USE THE ACTUAL RESIDUAL
            e_prev = epsilon_scaled[t]
            s2_prev = sigma2[t]
        
        # Scale back to original scale
        y_original = y_scaled / self.scaling_factor
        
        return y_original
    
    def test_innovation_whiteness(self, distribution: Optional[str] = None, 
                                 verbose: bool = True) -> Dict:
        """
        Test if standardized innovations are white noise (no patterns).
        
        Parameters:
        -----------
        distribution : str, optional
            Distribution to test. If None, uses best distribution.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        dist = distribution or self.best_distribution
        if dist not in self.standardized_residuals:
            raise ValueError(f"Distribution {dist} not available")
        
        std_resid = self.standardized_residuals[dist]
        results = {'distribution': dist}
        
        if verbose:
            print(f"\nTesting innovation whiteness for {dist} distribution:")
            print(f"Standardized residuals: n={len(std_resid)}")
            print(f"Mean: {np.mean(std_resid):.6f}")
            print(f"Std: {np.std(std_resid):.6f}")
        
        # Test 1: No autocorrelation (Ljung-Box)
        if len(std_resid) > 20:
            lb_result = acorr_ljungbox(std_resid, lags=20, return_df=True)
            lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
            results['ljung_box_pvalue'] = lb_pvalue
            results['no_autocorr'] = lb_pvalue > 0.05
            
            if verbose:
                status = "PASS" if lb_pvalue > 0.05 else "FAIL"
                print(f"  Ljung-Box (no autocorr): p={lb_pvalue:.4f} [{status}]")
        
        # Test 2: No ARCH effects
        try:
            arch_stat, arch_p, _, _ = het_arch(std_resid, maxlag=5)
            results['arch_lm_pvalue'] = arch_p
            results['no_arch'] = arch_p > 0.05
            
            if verbose:
                status = "PASS" if arch_p > 0.05 else "FAIL"
                print(f"  ARCH-LM (no volatility): p={arch_p:.4f} [{status}]")
        except:
            results['arch_lm_pvalue'] = None
            results['no_arch'] = None
        
        # Test 3: Runs test for randomness
        try:
            runs_stat, runs_p = runstest_1samp(std_resid)
            results['runs_pvalue'] = runs_p
            results['random'] = runs_p > 0.05
            
            if verbose:
                status = "PASS" if runs_p > 0.05 else "FAIL"
                print(f"  Runs test (randomness): p={runs_p:.4f} [{status}]")
        except:
            results['runs_pvalue'] = None
            results['random'] = None
        
        # Overall assessment
        tests_passed = sum([
            results.get('no_autocorr', False),
            results.get('no_arch', False), 
            results.get('random', False)
        ])
        results['whiteness_score'] = tests_passed
        results['is_white_noise'] = tests_passed >= 2
        
        if verbose:
            print(f"  → Whiteness score: {tests_passed}/3 tests passed")
            print(f"  → Assessment: {'WHITE NOISE' if results['is_white_noise'] else 'NOT WHITE NOISE'}")
        
        self.innovation_tests[dist] = results
        return results
    
    def test_distribution_goodness_of_fit(self, distribution: Optional[str] = None,
                                        verbose: bool = True) -> Dict:
        """
        Test goodness-of-fit of standardized innovations to assumed distribution.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        dist = distribution or self.best_distribution
        if dist not in self.standardized_residuals:
            raise ValueError(f"Distribution {dist} not available")
        
        std_resid = self.standardized_residuals[dist]
        garch_model = self.garch_models[dist]
        results = {'distribution': dist}
        
        if verbose:
            print(f"\nTesting goodness-of-fit for {dist} distribution:")
        
        if dist == 't':
            # Student-t distribution
            nu = garch_model.params.get('nu', 5)
            results['nu'] = nu
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(std_resid, lambda x: stats.t.cdf(x, df=nu))
            results['ks_statistic'] = ks_stat
            results['ks_pvalue'] = ks_p
            results['ks_accepted'] = ks_p > 0.05
            
            # Anderson-Darling (using normal as approximation)
            ad_stat, ad_crit, ad_sig = stats.anderson(std_resid, dist='norm')
            results['ad_statistic'] = ad_stat
            results['ad_critical_5pct'] = ad_crit[2]
            results['ad_accepted'] = ad_stat <= ad_crit[2]
            
            if verbose:
                print(f"  Student-t parameters: ν={nu:.2f}")
                ks_status = "ACCEPT" if ks_p > 0.05 else "REJECT"
                ad_status = "ACCEPT" if ad_stat <= ad_crit[2] else "REJECT"
                print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f} [{ks_status}]")
                print(f"  AD test: stat={ad_stat:.4f}, crit={ad_crit[2]:.4f} [{ad_status}]")
        
        elif dist == 'normal':
            # Normal distribution
            ks_stat, ks_p = stats.kstest(std_resid, 'norm')
            results['ks_statistic'] = ks_stat
            results['ks_pvalue'] = ks_p
            results['ks_accepted'] = ks_p > 0.05
            
            # Shapiro-Wilk test (if sample size allows)
            if len(std_resid) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(std_resid)
                results['shapiro_statistic'] = shapiro_stat
                results['shapiro_pvalue'] = shapiro_p
                results['shapiro_accepted'] = shapiro_p > 0.05
            
            if verbose:
                ks_status = "ACCEPT" if ks_p > 0.05 else "REJECT"
                print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f} [{ks_status}]")
                if 'shapiro_pvalue' in results:
                    sw_status = "ACCEPT" if results['shapiro_pvalue'] > 0.05 else "REJECT"
                    print(f"  Shapiro test: stat={results['shapiro_statistic']:.4f}, p={results['shapiro_pvalue']:.4f} [{sw_status}]")
        
        # Overall fit assessment
        fit_tests = []
        if 'ks_accepted' in results:
            fit_tests.append(results['ks_accepted'])
        if 'ad_accepted' in results:
            fit_tests.append(results['ad_accepted'])
        if 'shapiro_accepted' in results:
            fit_tests.append(results['shapiro_accepted'])
        
        results['fit_score'] = sum(fit_tests) if fit_tests else 0
        results['good_fit'] = results['fit_score'] >= 1
        
        if verbose:
            print(f"  → Fit score: {results['fit_score']}/{len(fit_tests)} tests accepted")
            print(f"  → Assessment: {'GOOD FIT' if results['good_fit'] else 'POOR FIT'}")
        
        self.goodness_of_fit_tests[dist] = results
        return results


    def _extract_arma_params(self) -> Dict[str, float]:
        """Extract ARMA parameters from fitted model."""
        params = {}
        
        # The ARMA model parameters
        if hasattr(self.arma_model, 'arparams') and len(self.arma_model.arparams) > 0:
            for i, param in enumerate(self.arma_model.arparams):
                params[f'phi_{i+1}'] = float(param)
        
        if hasattr(self.arma_model, 'maparams') and len(self.arma_model.maparams) > 0:
            for j, param in enumerate(self.arma_model.maparams):
                params[f'theta_{j+1}'] = float(param)
        
        # Ensure we have all required parameters
        for i in range(1, self.p + 1):
            if f'phi_{i}' not in params:
                params[f'phi_{i}'] = 0.0
                
        for j in range(1, self.q + 1):
            if f'theta_{j}' not in params:
                params[f'theta_{j}'] = 0.0
        
        return params
    
    def _extract_garch_params(self, distribution: Optional[str] = None) -> Dict[str, float]:
        """Extract GARCH parameters from fitted model."""
        dist = distribution or self.best_distribution
        if dist not in self.garch_models:
            return {}
            
        garch_model = self.garch_models[dist]
        params = {}
        
        # Get omega (constant term)
        params['omega'] = garch_model.params['omega']
        
        # Get ARCH parameters (alpha)
        for i in range(1, self.s + 1):
            param_name = f'alpha[{i}]'
            if param_name in garch_model.params:
                params[f'alpha_{i}'] = garch_model.params[param_name]
        
        # Get GARCH parameters (beta)
        for j in range(1, self.r + 1):
            param_name = f'beta[{j}]'
            if param_name in garch_model.params:
                params[f'beta_{j}'] = garch_model.params[param_name]
        
        # For backward compatibility, also store alpha[1] and beta[1] as 'alpha' and 'beta'
        if 'alpha[1]' in garch_model.params:
            params['alpha'] = garch_model.params['alpha[1]']
        if 'beta[1]' in garch_model.params:
            params['beta'] = garch_model.params['beta[1]']
            
        return params
    
    def _calculate_persistence(self) -> float:
        """Calculate volatility persistence (sum of all α and β)."""
        params = self._extract_garch_params()
        
        # Sum all ARCH coefficients
        alpha_sum = sum(params.get(f'alpha_{i}', 0.0) for i in range(1, self.s + 1))
        
        # Sum all GARCH coefficients
        beta_sum = sum(params.get(f'beta_{j}', 0.0) for j in range(1, self.r + 1))
        
        return alpha_sum + beta_sum
    
    def _calculate_halflife(self) -> float:
        """Calculate volatility half-life in samples."""
        persistence = self._calculate_persistence()
        if persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(persistence)
    
    def _print_enhanced_summary(self, results: Dict):
        """Print enhanced model summary."""
        print("\n" + "="*70)
        print("Enhanced ARMA-GARCH Model Summary")
        print("="*70)
        
        # Model specification
        orders = results['optimal_orders']
        best_dist = results['best_distribution']
        print(f"\nOptimal Model: ARMA({orders['p']},{orders['q']})-GARCH({orders['r']},{orders['s']})-{best_dist}")
        
        # Order selection results
        if 'order_selection' in results and results['order_selection']:
            print(f"\nOrder Selection:")
            if 'arma' in results['order_selection']:
                arma_results = results['order_selection']['arma']
                best_arma_bic = min([r['bic'] for r in arma_results.values() if np.isfinite(r['bic'])])
                print(f"  ARMA: Best BIC = {best_arma_bic:.2f}")
            if 'garch' in results['order_selection']:
                garch_results = results['order_selection']['garch']
                best_garch_bic = min([r['bic'] for r in garch_results.values() if np.isfinite(r['bic'])])
                print(f"  GARCH: Best BIC = {best_garch_bic:.2f}")
        
        # Distribution comparison
        if 'distribution_comparison' in results:
            print(f"\nDistribution Comparison:")
            dist_results = results['distribution_comparison']
            for dist, metrics in sorted(dist_results.items(), key=lambda x: x[1]['bic']):
                if metrics.get('converged', False):
                    print(f"  {dist:<8}: BIC={metrics['bic']:.2f}, AIC={metrics['aic']:.2f}")
            print(f"  → Best: {best_dist}")
        
        # Model parameters
        if best_dist and best_dist in self.garch_models:
            garch_model = self.garch_models[best_dist]
            print(f"\nModel Parameters:")
            
            # ARMA parameters
            arma_params = results['arma_params']
            if self.p > 0:
                print(f"  AR coefficients:")
                for i in range(1, self.p + 1):
                    if f'phi_{i}' in arma_params:
                        print(f"    φ_{i}: {arma_params[f'phi_{i}']:8.4f}")
            
            if self.q > 0:
                print(f"  MA coefficients:")
                for j in range(1, self.q + 1):
                    if f'theta_{j}' in arma_params:
                        print(f"    θ_{j}: {arma_params[f'theta_{j}']:8.4f}")
            
            # GARCH parameters
            garch_params = results['garch_params']
            print(f"  GARCH parameters:")
            print(f"    ω: {garch_params.get('omega', 0):8.6f}")
            if 'alpha' in garch_params:
                print(f"    α: {garch_params['alpha']:8.4f}")
            if 'beta' in garch_params:
                print(f"    β: {garch_params['beta']:8.4f}")
            
            # Distribution parameters
            if best_dist == 't' and 'nu' in garch_model.params:
                print(f"    ν: {garch_model.params['nu']:8.2f}")
            
            # Volatility dynamics
            print(f"\nVolatility Dynamics:")
            print(f"  Persistence (α+β): {results['volatility_persistence']:.4f}")
            halflife = results['volatility_halflife']
            if np.isfinite(halflife):
                print(f"  Half-life: {halflife:.1f} samples")
            else:
                print(f"  Half-life: ∞ (unit root in volatility)")
    
    def _print_summary(self, results: Dict):
        """Print basic model summary (backward compatibility)."""
        print("\n" + "="*60)
        print("ARMA-GARCH Model Summary")
        print("="*60)
        
        best_dist = results.get('best_distribution', 't')
        if best_dist in self.garch_models and 'nu' in self.garch_models[best_dist].params:
            nu = self.garch_models[best_dist].params['nu']
            print(f"\nModel: ARMA({self.p},{self.q})-GARCH({self.r},{self.s})-{best_dist}(ν={nu:.2f})")
        else:
            print(f"\nModel: ARMA({self.p},{self.q})-GARCH({self.r},{self.s})-{best_dist}")
        
        print("\nARMA Parameters:")
        arma_params = results['arma_params']
        
        # Print AR parameters
        if self.p > 0:
            print("  AR coefficients:")
            for i in range(1, self.p + 1):
                if f'phi_{i}' in arma_params:
                    print(f"    φ_{i}: {arma_params[f'phi_{i}']:8.4f}")
        
        # Print MA parameters
        if self.q > 0:
            print("  MA coefficients:")
            for j in range(1, self.q + 1):
                if f'theta_{j}' in arma_params:
                    print(f"    θ_{j}: {arma_params[f'theta_{j}']:8.4f}")
        
        print("\nGARCH Parameters:")
        garch_params = results['garch_params']
        print(f"  ω (omega): {garch_params['omega']:8.4f}")
        
        # Print ARCH parameters
        if self.s > 0:
            for i in range(1, self.s + 1):
                if f'alpha_{i}' in garch_params:
                    print(f"  α_{i}: {garch_params[f'alpha_{i}']:8.4f}")
        
        # Print GARCH parameters
        if self.r > 0:
            for j in range(1, self.r + 1):
                if f'beta_{j}' in garch_params:
                    print(f"  β_{j}: {garch_params[f'beta_{j}']:8.4f}")
        
        print(f"\nStudent-t ν: {results['student_t_nu']:.3f}")
        print(f"Volatility persistence (α+β): {results['volatility_persistence']:.4f}")
        print(f"Volatility half-life: {results['volatility_halflife']:.1f} samples")
        print(f"\nAIC: {results['aic']:.1f}")
        print(f"BIC: {results['bic']:.1f}")


# Test function to verify the fix
def test_scaling_fix():
    """Test that the scaling is handled correctly"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load example data
    print("Testing ARMA-GARCH model with scaling fix...")
    
    # Generate synthetic test data with known properties
    np.random.seed(42)
    n = 1000
    
    # Simple AR(1)-GARCH(1,1) process
    true_residuals = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = 1e-6  # Small variance (typical of residuals)
    
    for t in range(1, n):
        sigma2[t] = 1e-7 + 0.15 * true_residuals[t-1]**2 + 0.8 * sigma2[t-1]
        true_residuals[t] = 0.5 * true_residuals[t-1] + np.sqrt(sigma2[t]) * np.random.standard_t(5)
    
    # Fit enhanced model
    model = EnhancedARMAGARCH(p=1, q=0, r=1, s=1, auto_select_orders=False)
    results = model.fit(true_residuals, distributions=['normal', 't'], verbose=True)
    
    # Simulate
    simulated = model.simulate(n, random_state=123)
    
    # Compare statistics
    print(f"\nOriginal data stats:")
    print(f"  Mean: {np.mean(true_residuals):.6f}")
    print(f"  Std:  {np.std(true_residuals):.6f}")
    print(f"  Min:  {np.min(true_residuals):.6f}")
    print(f"  Max:  {np.max(true_residuals):.6f}")
    
    print(f"\nSimulated data stats:")
    print(f"  Mean: {np.mean(simulated):.6f}")
    print(f"  Std:  {np.std(simulated):.6f}")
    print(f"  Min:  {np.min(simulated):.6f}")
    print(f"  Max:  {np.max(simulated):.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    axes[0].plot(true_residuals[:500], label='Original', alpha=0.7)
    axes[0].plot(simulated[:500], label='Simulated', alpha=0.7)
    axes[0].set_title('Time Series Comparison (first 500 points)')
    axes[0].legend()
    
    # Density
    axes[1].hist(true_residuals, bins=50, alpha=0.5, density=True, label='Original')
    axes[1].hist(simulated, bins=50, alpha=0.5, density=True, label='Simulated')
    axes[1].set_title('Density Comparison')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, true_residuals, simulated


if __name__ == "__main__":
    # Run test
    model, true_data, sim_data = test_scaling_fix()