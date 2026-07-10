"""
Uncertainty-Enabled Trajectory Layers
====================================

Implements Jacobian-based uncertainty propagation for elbow trajectory
prediction using the SAME integrated trajectory equation as
`TrajEstimator` in `model_selection/modular_architectures.py`.

Parameters and equation (Method 2 – integrated form):
- Parameter order: [vd, c1, a0]
- Definitions: c2 = a0 − λ·c1, c3 = c2 − λ·c1 (= a0 − 2λ·c1)
- Trajectory: y(t) = vd·t + [ c3 + (λ·c1 + λ·c2·t − c2)·exp(λ·t) ] / λ²

These layers compute the trajectory and its analytical Jacobians with
respect to [vd, c1, a0], so parameter log-variances can be propagated to
trajectory-space uncertainties.

Key Features:
- JacobianTrajEstimator: Trajectory + analytical Jacobians (fixed λ)
- AdaptJacobianTrajEstimator: Trainable eigenvalue λ with stability clipping
- UncertaintyTrajEstimator: Wraps mean/log-var params and propagates to traj space

Date: 2025-08-25
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Tuple


class JacobianTrajEstimator(Layer):
    """
    Trajectory estimator with analytical Jacobians (fixed λ).

    - Inputs (batch, 3, 1): [vd, c1, a0]
    - Outputs:
      - trajectories: (batch, n_times)
      - jacobians:   (batch, n_times, 3) for [vd, c1, a0]
    - Uses integrated equation y(t) = vd·t + [ c3 + (λ·c1 + λ·c2·t − c2)·exp(λ·t) ] / λ²
    """
    
    def __init__(self, eig: float = -2.0, time_samples: Tuple = (1/3, 2/3, 1), **kwargs):
        """
        Initialize JacobianTrajEstimator.
        
        Args:
            eig: Eigenvalue λ for trajectory dynamics
            time_samples: Time points for trajectory evaluation
            **kwargs: Additional layer arguments
        """
        super(JacobianTrajEstimator, self).__init__(**kwargs)
        self.eig_init = eig
        self.time_samples = time_samples
        
        # Convert time samples to tensor for computation
        self.T = tf.constant(time_samples, dtype=tf.float32)[None, None, ...]  # (1, 1, n_times)
        self.eig = tf.constant(eig, dtype=tf.float32)
        
    @property
    def eigsq(self):
        """Squared eigenvalue for trajectory computation."""
        return self.eig * self.eig
    
    def call(self, inputs, **kwargs):
        """
        Forward pass using TrajEstimator integrated equation with params [vd, c1, a0].

        Returns trajectory (batch, n_times) and Jacobians (batch, n_times, 3).
        """
        batch_size = tf.shape(inputs)[0]

        # Extract parameters: (batch, 3, 1) -> individual parameters
        vd = inputs[:, 0, :, None]  # (batch, 1, 1)
        c1 = inputs[:, 1, :, None]  # (batch, 1, 1)
        a0 = inputs[:, 2, :, None]  # (batch, 1, 1)

        # Use bounded eigenvalue for stability, mirroring TrajEstimator
        lam = tf.clip_by_value(self.eig, -4.0, -0.5)
        lam_sq = lam * lam
        exp_lam_t = tf.exp(lam * self.T)  # (1, 1, n_times)

        # c2, c3 and integrated form
        c2 = a0 - lam * c1
        c3 = c2 - lam * c1
        numerator = c3 + (lam * c1 + lam * c2 * self.T - c2) * exp_lam_t
        integrated = vd * self.T + numerator / lam_sq  # (batch, 1, n_times)

        trajectories = tf.squeeze(integrated, axis=1)  # (batch, n_times)

        # Analytical Jacobians ∂y/∂[vd, c1, a0]
        jacobians = self._compute_jacobians(batch_size, lam, exp_lam_t)
        return trajectories, jacobians
    
    def _compute_jacobians(self, batch_size, lam, exp_lam_t):
        """Analytical Jacobians ∂y/∂[vd, c1, a0] for the integrated form.

        With λ treated constant w.r.t. params:
          ∂y/∂vd = T
          ∂y/∂c1 = (-2/λ) + ((2/λ) - T)·exp(λT)
          ∂y/∂a0 = (1/λ²) + ((T/λ) - (1/λ²))·exp(λT)
        """
        T = self.T
        lam_sq = lam * lam

        dy_dvd = T
        dy_dc1 = (-2.0 / lam) + ((2.0 / lam) - T) * exp_lam_t
        dy_da0 = (1.0 / lam_sq) + ((T / lam) - (1.0 / lam_sq)) * exp_lam_t

        dy_dvd_b = tf.tile(dy_dvd, [batch_size, 1, 1])
        dy_dc1_b = tf.tile(dy_dc1, [batch_size, 1, 1])
        dy_da0_b = tf.tile(dy_da0, [batch_size, 1, 1])

        jacobians = tf.stack([
            tf.squeeze(dy_dvd_b, axis=1),
            tf.squeeze(dy_dc1_b, axis=1),
            tf.squeeze(dy_da0_b, axis=1)
        ], axis=-1)
        return jacobians
    
    def propagate_uncertainty(self, param_log_var, jacobians):
        """
        Propagate parameter uncertainties to trajectory space using Jacobians.
        
        Formula: σ²_traj = J @ Σ_params @ J^T
        
        Args:
            param_log_var: (batch, 3) log variances of parameters
            jacobians: (batch, n_times, 3) Jacobians
            
        Returns:
            traj_log_var: (batch, n_times) trajectory log variances
        """
        # Convert log variance to variance
        param_var = tf.exp(param_log_var)  # (batch, 3)
        
        # Expand parameter variances for broadcasting
        param_var_expanded = param_var[:, None, :]  # (batch, 1, 3)
        
        # Compute J @ Σ @ J^T = sum_i (J_i^2 * σ²_i) for diagonal covariance
        J_squared = tf.square(jacobians)  # (batch, n_times, 3)
        traj_var = tf.reduce_sum(J_squared * param_var_expanded, axis=-1)  # (batch, n_times)
        
        # Convert back to log variance with numerical stability
        traj_var = tf.maximum(traj_var, 1e-8)
        traj_log_var = tf.math.log(traj_var)
        
        return traj_log_var


class AdaptJacobianTrajEstimator(JacobianTrajEstimator):
    """Adaptive trajectory estimator with trainable eigenvalue λ (clipped)."""
    
    def __init__(self, eig=-2.0, time_samples=(1/3, 2/3, 1), **kwargs):
        """
        Initialize AdaptJacobianTrajEstimator.
        
        Args:
            eig: Initial eigenvalue
            time_samples: Time points for trajectory evaluation
            **kwargs: Additional layer arguments
        """
        # Call parent Layer.__init__ with only layer kwargs
        super().__init__(**kwargs)
        self.eig_init = eig
        self.time_samples = time_samples
        
        # Convert time samples to tensor for computation
        self.T = tf.constant(self.time_samples, dtype=tf.float32)[None, None, ...]  # (1, 1, n_times)
        
    def build(self, input_shape):
        """Build layer with trainable eigenvalue."""
        super().build(input_shape)
        
        # Create trainable eigenvalue with stability constraints
        self.eig = self.add_weight(
            name='adaptive_eig',
            shape=[1],
            trainable=True,
            initializer=tf.keras.initializers.constant(self.eig_init)
        )
        
    @property  
    def eigsq(self):
        """Constrained squared eigenvalue for numerical stability."""
        # Constrain eigenvalue to prevent instability: λ ∈ [-6.0, -0.1]
        constrained_eig = tf.clip_by_value(self.eig, -6.0, -0.1)
        return constrained_eig * constrained_eig
        
    def call(self, inputs, **kwargs):
        """Forward pass using integrated equation with trainable λ and params [vd, c1, a0]."""
        batch_size = tf.shape(inputs)[0]

        # Constrain eigenvalue for stability
        lam = tf.clip_by_value(self.eig, -6.0, -0.1)

        # Extract parameters
        vd = inputs[:, 0, :, None]
        c1 = inputs[:, 1, :, None]
        a0 = inputs[:, 2, :, None]

        lam_sq = lam * lam
        exp_lam_t = tf.exp(lam * self.T)

        c2 = a0 - lam * c1
        c3 = c2 - lam * c1

        numerator = c3 + (lam * c1 + lam * c2 * self.T - c2) * exp_lam_t
        integrated = vd * self.T + numerator / lam_sq

        trajectories = tf.squeeze(integrated, axis=1)
        jacobians = self._compute_jacobians(batch_size, lam, exp_lam_t)
        return trajectories, jacobians


class UncertaintyTrajEstimator(Layer):
    """
    Complete uncertainty-enabled trajectory estimator.
    
    Combines parameter prediction (mean + log variance) with trajectory generation
    and uncertainty propagation in a single layer.
    """
    
    def __init__(self, eig: float = -2.0, time_samples: Tuple = (1/3, 2/3, 1), 
                 trainable_eig: bool = True, **kwargs):
        """
        Initialize UncertaintyTrajEstimator.
        
        Args:
            eig: Initial eigenvalue
            time_samples: Time points for trajectory evaluation
            trainable_eig: Whether eigenvalue should be trainable
            **kwargs: Additional layer arguments
        """
        super(UncertaintyTrajEstimator, self).__init__(**kwargs)
        
        # Choose trajectory estimator type
        if trainable_eig:
            self.traj_estimator = AdaptJacobianTrajEstimator(
                eig=eig, time_samples=time_samples, name='adapt_traj_estimator'
            )
        else:
            self.traj_estimator = JacobianTrajEstimator(
                eig=eig, time_samples=time_samples, name='jacobian_traj_estimator' 
            )
            
    def call(self, inputs, **kwargs):
        """Forward pass with uncertainty; inputs parameterize [vd, c1, a0]."""
        mean_params, log_var_params = inputs
        
        # Generate trajectory and compute Jacobians
        mean_traj, jacobians = self.traj_estimator(mean_params)
        
        # Propagate parameter uncertainties to trajectory space
        traj_log_var = self.traj_estimator.propagate_uncertainty(log_var_params, jacobians)
        
        # Stack mean and log variance for loss computation
        output = tf.stack([mean_traj, traj_log_var], axis=-1)  # (batch, n_times, 2)
        
        return output


# Register custom layers for model loading/saving
custom_uncertainty_layers = {
    'JacobianTrajEstimator': JacobianTrajEstimator,
    'AdaptJacobianTrajEstimator': AdaptJacobianTrajEstimator,
    'UncertaintyTrajEstimator': UncertaintyTrajEstimator
}


def test_jacobian_traj_estimator():
    """Test the JacobianTrajEstimator implementation."""
    print("Testing JacobianTrajEstimator...")
    
    # Test parameters
    batch_size = 4
    n_times = 10
    time_samples = np.linspace(0.05, 0.55, n_times)  # 50ms delay, 500ms horizon
    
    # Create estimator
    estimator = JacobianTrajEstimator(eig=-3.0, time_samples=time_samples)
    
    # Test input: (batch, 3, 1) parameters
    test_params = tf.random.normal((batch_size, 3, 1))
    
    # Forward pass
    trajectories, jacobians = estimator(test_params)
    
    print(f"✅ Input shape: {test_params.shape}")
    print(f"✅ Trajectory shape: {trajectories.shape}")
    print(f"✅ Jacobians shape: {jacobians.shape}")
    
    # Test uncertainty propagation
    param_log_var = tf.random.normal((batch_size, 3)) - 2.0  # Small log variances
    traj_log_var = estimator.propagate_uncertainty(param_log_var, jacobians)
    
    print(f"✅ Parameter log var shape: {param_log_var.shape}")
    print(f"✅ Trajectory log var shape: {traj_log_var.shape}")
    
    # Test adaptive version
    adaptive_estimator = AdaptJacobianTrajEstimator(eig=-3.0, time_samples=time_samples)
    adaptive_estimator.build(test_params.shape)
    
    traj_adaptive, jac_adaptive = adaptive_estimator(test_params)
    print(f"✅ Adaptive trajectory shape: {traj_adaptive.shape}")
    print(f"✅ Trainable eigenvalue: {adaptive_estimator.eig.numpy()}")
    
    print("✅ JacobianTrajEstimator tests passed!")


if __name__ == "__main__":
    test_jacobian_traj_estimator()
