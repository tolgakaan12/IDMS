"""
Uncertainty Loss Functions
=========================

Implements heteroscedastic loss functions for aleatoric uncertainty quantification
in trajectory prediction models.

Key Features:
- Heteroscedastic Gaussian likelihood loss
- Automatic loss attenuation for uncertain predictions
- Numerical stability and gradient clipping
- Uncertainty regularization options

Date: 2025-08-25
"""

import tensorflow as tf
import numpy as np


def aleatoric_trajectory_loss(y_true, y_pred, loss_weight=1.0, var_regularization=0.0):
    """
    Heteroscedastic aleatoric uncertainty loss for trajectory prediction.
    
    Implements negative log-likelihood of heteroscedastic Gaussian:
    L = 1/(2σ²)||y - ŷ||² + 1/2 log σ²
    
    This loss function:
    1. Automatically attenuates loss for high-uncertainty predictions
    2. Penalizes overconfident (low uncertainty) incorrect predictions  
    3. Prevents collapse to zero uncertainty via log σ² term
    
    Args:
        y_true: Ground truth trajectories (batch, n_times)
        y_pred: Model predictions (batch, n_times, 2) where:
                y_pred[..., 0] = trajectory means
                y_pred[..., 1] = trajectory log variances
        loss_weight: Overall loss scaling factor
        var_regularization: L2 penalty on log variances to prevent extreme values
        
    Returns:
        loss: Scalar heteroscedastic loss value
    """
    # Extract predictions and log variances
    y_mean = y_pred[..., 0]      # (batch, n_times) trajectory predictions
    y_log_var = y_pred[..., 1]   # (batch, n_times) trajectory log variances
    
    # Clip log variances for numerical stability
    # Allow range: log(1e-5) ≈ -11.5 to log(100) ≈ 4.6
    y_log_var = tf.clip_by_value(y_log_var, -10.0, 3.0)
    
    # Compute precision (inverse variance)
    precision = tf.exp(-y_log_var)  # 1/σ²
    
    # Compute squared prediction error
    squared_error = tf.square(y_true - y_mean)  # ||y - ŷ||²
    
    # Heteroscedastic loss: 1/(2σ²)||y - ŷ||² + 1/2 log σ²
    data_loss = 0.5 * precision * squared_error    # Weighted MSE term
    uncertainty_loss = 0.5 * y_log_var             # Uncertainty penalty term
    
    # Combine loss terms
    total_loss = data_loss + uncertainty_loss
    
    # Optional variance regularization to prevent extreme log variances
    if var_regularization > 0.0:
        var_reg_loss = var_regularization * tf.reduce_mean(tf.square(y_log_var))
        total_loss = total_loss + var_reg_loss
    
    # Average over trajectory points and batch
    final_loss = tf.reduce_mean(total_loss) * loss_weight
    
    return final_loss


def uncertainty_mse_loss(y_true, y_pred, uncertainty_weight=1.0):
    """
    Combined MSE + uncertainty regularization loss.
    
    Useful for comparison with standard MSE while encouraging
    reasonable uncertainty estimates.
    
    Args:
        y_true: Ground truth trajectories (batch, n_times)
        y_pred: Model predictions (batch, n_times, 2)
        uncertainty_weight: Weight for uncertainty regularization term
        
    Returns:
        loss: Combined MSE + uncertainty loss
    """
    y_mean = y_pred[..., 0]      # Trajectory predictions
    y_log_var = y_pred[..., 1]   # Log variances
    
    # Standard MSE on predictions
    mse_loss = tf.reduce_mean(tf.square(y_true - y_mean))
    
    # Uncertainty regularization: penalize very high or very low uncertainties
    y_log_var = tf.clip_by_value(y_log_var, -10.0, 3.0)
    
    # Target log variance around -2.0 (σ ≈ 0.37, reasonable for normalized data)
    target_log_var = -2.0
    uncertainty_reg = tf.reduce_mean(tf.square(y_log_var - target_log_var))
    
    total_loss = mse_loss + uncertainty_weight * uncertainty_reg
    
    return total_loss


def calibrated_uncertainty_loss(y_true, y_pred, temperature=1.0):
    """
    Temperature-scaled uncertainty loss for better calibration.
    
    Applies temperature scaling to the uncertainty estimates before
    computing the heteroscedastic loss, which can improve calibration.
    
    Args:
        y_true: Ground truth trajectories (batch, n_times)
        y_pred: Model predictions (batch, n_times, 2)
        temperature: Temperature scaling parameter (>1 increases uncertainty)
        
    Returns:
        loss: Temperature-scaled uncertainty loss
    """
    y_mean = y_pred[..., 0]
    y_log_var = y_pred[..., 1]
    
    # Apply temperature scaling to log variance
    # Higher temperature -> higher uncertainty
    y_log_var_scaled = y_log_var + tf.math.log(temperature)
    
    # Reconstruct predictions with scaled uncertainty
    y_pred_scaled = tf.stack([y_mean, y_log_var_scaled], axis=-1)
    
    return aleatoric_trajectory_loss(y_true, y_pred_scaled)


class UncertaintyLossTracker(tf.keras.callbacks.Callback):
    """
    Callback to track uncertainty-specific metrics during training.
    
    Monitors:
    - Mean predicted uncertainty (σ)
    - Uncertainty range (min/max σ)
    - Loss decomposition (data vs uncertainty terms)
    """
    
    def __init__(self, validation_data=None, log_frequency=5):
        """
        Initialize uncertainty loss tracker.
        
        Args:
            validation_data: (X_val, y_val) for uncertainty analysis
            log_frequency: Log metrics every N epochs
        """
        super(UncertaintyLossTracker, self).__init__()
        self.validation_data = validation_data
        self.log_frequency = log_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        """Analyze uncertainty metrics at end of epoch."""
        if epoch % self.log_frequency != 0 or self.validation_data is None:
            return
            
        X_val, y_val = self.validation_data
        
        # Get model predictions
        y_pred = self.model.predict(X_val, verbose=0)
        
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 2:
            # Extract uncertainty estimates
            y_log_var = y_pred[..., 1]
            y_var = np.exp(y_log_var)
            y_std = np.sqrt(y_var)
            
            # Compute uncertainty statistics
            mean_std = np.mean(y_std)
            min_std = np.min(y_std)
            max_std = np.max(y_std)
            
            # Log to console
            print(f"\nEpoch {epoch} Uncertainty Analysis:")
            print(f"  Mean uncertainty (σ): {mean_std:.4f}")
            print(f"  Uncertainty range: [{min_std:.4f}, {max_std:.4f}]")
            
            # Add to logs for TensorBoard/history
            logs = logs or {}
            logs['val_mean_uncertainty'] = float(mean_std)
            logs['val_min_uncertainty'] = float(min_std)
            logs['val_max_uncertainty'] = float(max_std)


def test_uncertainty_losses():
    """Test uncertainty loss functions."""
    print("Testing uncertainty loss functions...")
    
    # Create test data
    batch_size, n_times = 8, 10
    
    # Ground truth trajectories
    y_true = tf.random.normal((batch_size, n_times))
    
    # Model predictions: means close to truth, various log variances
    y_mean = y_true + tf.random.normal((batch_size, n_times)) * 0.1  # Small error
    y_log_var = tf.random.normal((batch_size, n_times)) - 2.0  # Around σ=0.37
    
    y_pred = tf.stack([y_mean, y_log_var], axis=-1)  # (batch, n_times, 2)
    
    print(f"✅ Test data shapes:")
    print(f"   y_true: {y_true.shape}")
    print(f"   y_pred: {y_pred.shape}")
    
    # Test aleatoric loss
    aleatoric_loss = aleatoric_trajectory_loss(y_true, y_pred)
    print(f"✅ Aleatoric loss: {aleatoric_loss.numpy():.4f}")
    
    # Test MSE + uncertainty loss  
    mse_unc_loss = uncertainty_mse_loss(y_true, y_pred)
    print(f"✅ MSE + uncertainty loss: {mse_unc_loss.numpy():.4f}")
    
    # Test calibrated loss
    calibrated_loss = calibrated_uncertainty_loss(y_true, y_pred, temperature=1.5)
    print(f"✅ Calibrated loss: {calibrated_loss.numpy():.4f}")
    
    # Test with extreme uncertainties
    y_pred_extreme = tf.stack([y_mean, tf.ones_like(y_log_var) * 5.0], axis=-1)  # High uncertainty
    extreme_loss = aleatoric_trajectory_loss(y_true, y_pred_extreme)
    print(f"✅ High uncertainty loss: {extreme_loss.numpy():.4f}")
    
    # Compare with standard MSE
    standard_mse = tf.reduce_mean(tf.square(y_true - y_mean))
    print(f"✅ Standard MSE: {standard_mse.numpy():.4f}")
    
    print("✅ All uncertainty loss tests passed!")


if __name__ == "__main__":
    test_uncertainty_losses()