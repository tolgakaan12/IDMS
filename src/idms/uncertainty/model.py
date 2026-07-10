"""
Elbow Trajectory Uncertainty Model
=================================

Complete uncertainty-enabled model for elbow trajectory prediction, adapted from the working
uncertainty implementation. Uses Jacobian-based parameter-to-trajectory uncertainty propagation.

Key Features:
- Dual-head architecture: mean + log variance parameters
- JacobianTrajEstimator for uncertainty propagation
- Heteroscedastic aleatoric loss function
- Compatible with existing elbow training pipeline

Date: 2025-08-25
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Dropout, Reshape, SpatialDropout2D, 
    Conv2D, LayerNormalization, BatchNormalization, Activation,
    DepthwiseConv2D, InputLayer
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

from idms.uncertainty.blocks import encode_layers, decode_layers, add_seq_layers
from idms.uncertainty.losses import aleatoric_trajectory_loss


class ElbowJacobianTrajEstimator(tf.keras.layers.Layer):
    """
    Elbow trajectory estimator with analytical Jacobian computation for uncertainty propagation.
    
    Adapted from working uncertainty model but specialized for 1D elbow angle prediction.
    """

    def __init__(self, eig=-2., time_samples=(1/3, 2/3, 1), **kwargs):
        super(ElbowJacobianTrajEstimator, self).__init__(**kwargs)
        self.time_samples = time_samples
        self.T = tf.constant(time_samples, dtype=tf.float32)[None, None, ...]  # (1, 1, n_times)
        self.eig = tf.constant(eig, dtype=tf.float32)
        self.eigsq = self.eig * self.eig
        self.exp_precalc = tf.exp(self.eig * self.T)  # (1, 1, n_times)

    def call(self, inputs, **kwargs):
        """
        Compute trajectory and Jacobians for 1D elbow angle.
        
        Args:
            inputs: (batch, 3, 1) parameter tensor [θ_∞, A, B]
            
        Returns:
            trajectories: (batch, n_times) trajectory points
            jacobians: (batch, n_times, 3) analytical Jacobians
        """
        batch_size = tf.shape(inputs)[0]
        
        # Extract parameters: (batch, 3, 1) -> individual parameters
        theta_inf = inputs[:, 0, :, None]  # (batch, 1, 1) - final angle
        A = inputs[:, 1, :, None]          # (batch, 1, 1) - amplitude  
        B = inputs[:, 2, :, None]          # (batch, 1, 1) - damping term

        # Compute trajectory using physics equation: θ(t) = θ_∞ + A·exp(λ·t) + B·t·exp(λ·t)
        term1 = theta_inf  # Final equilibrium angle
        term2 = A * self.exp_precalc  # Exponential decay term
        term3 = B * self.T * self.exp_precalc  # Damped oscillation term
        
        trajectories = term1 + term2 + term3  # (batch, 1, n_times)
        trajectories = tf.squeeze(trajectories, axis=1)  # (batch, n_times)

        # Compute Jacobians
        jacobians = self._compute_jacobians(batch_size)
        return trajectories, jacobians

    def _compute_jacobians(self, batch_size):
        """Compute analytical Jacobians ∂trajectory/∂parameters."""
        # ∂θ(t)/∂θ_∞ = 1 (constant for all time points)
        dtraj_dtheta_inf = tf.tile(tf.ones_like(self.T), [batch_size, 1, 1])

        # ∂θ(t)/∂A = exp(λ·t)
        dtraj_dA = tf.tile(self.exp_precalc, [batch_size, 1, 1])

        # ∂θ(t)/∂B = t·exp(λ·t)  
        dtraj_dB = tf.tile(self.T * self.exp_precalc, [batch_size, 1, 1])

        # Stack Jacobians: (batch, 1, n_times) -> (batch, n_times, 3)
        jacobians = tf.stack([
            tf.squeeze(dtraj_dtheta_inf, axis=1),  # (batch, n_times)
            tf.squeeze(dtraj_dA, axis=1),          # (batch, n_times)  
            tf.squeeze(dtraj_dB, axis=1)           # (batch, n_times)
        ], axis=-1)  # (batch, n_times, 3)

        return jacobians


class ElbowAdaptJacobianTrajEstimator(ElbowJacobianTrajEstimator):
    """Adaptive version with trainable eigenvalue."""

    def __init__(self, eig=-2., time_samples=(1/3, 2/3, 1), **kwargs):
        # Initialize base layer without calling parent __init__ to avoid conflicts
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.eig_init = eig
        self.time_samples = time_samples
        self.T = tf.constant(time_samples, dtype=tf.float32)[None, None, ...]

    def build(self, input_shape):
        super().build(input_shape)
        # Create trainable eigenvalue
        self.eig = self.add_weight(
            name='adaptive_eig',
            shape=[1],
            trainable=True,
            initializer=tf.keras.initializers.constant(self.eig_init)
        )

    def call(self, inputs, **kwargs):
        """Forward pass with adaptive eigenvalue."""
        batch_size = tf.shape(inputs)[0]
        
        # Constrain eigenvalue for stability
        constrained_eig = tf.clip_by_value(self.eig, -6.0, -0.1)
        
        # Extract parameters
        theta_inf = inputs[:, 0, :, None]
        A = inputs[:, 1, :, None]
        B = inputs[:, 2, :, None]

        # Compute trajectory with current eigenvalue
        exp_eig_t = tf.exp(constrained_eig * self.T)
        term1 = theta_inf
        term2 = A * exp_eig_t
        term3 = B * self.T * exp_eig_t
        
        trajectories = tf.squeeze(term1 + term2 + term3, axis=1)

        # Compute Jacobians with current eigenvalue
        jacobians = self._compute_adaptive_jacobians(batch_size, exp_eig_t)
        return trajectories, jacobians

    def _compute_adaptive_jacobians(self, batch_size, exp_eig_t):
        """Compute Jacobians with current adaptive eigenvalue."""
        dtraj_dtheta_inf = tf.tile(tf.ones_like(self.T), [batch_size, 1, 1])
        dtraj_dA = tf.tile(exp_eig_t, [batch_size, 1, 1])
        dtraj_dB = tf.tile(self.T * exp_eig_t, [batch_size, 1, 1])

        jacobians = tf.stack([
            tf.squeeze(dtraj_dtheta_inf, axis=1),
            tf.squeeze(dtraj_dA, axis=1),
            tf.squeeze(dtraj_dB, axis=1)
        ], axis=-1)

        return jacobians


class ElbowUncertaintyModel(Model):
    """
    Complete uncertainty-enabled elbow trajectory model.
    
    Architecture: EMG → TCN Encoder → Dual Heads (mean, log_var) → Trajectory + Uncertainty
    """

    def __init__(self, model_params=None, name='elbow_uncertainty_model', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_params = model_params or {}
        if model_params is not None:
            self._build_model_components(model_params)

    def _build_model_components(self, model_params):
        """Build encoder and trajectory estimator components."""
        # Build encoder
        self.encoder = self._build_encoder(model_params)

        # Two heads: mean and log variance (3 parameters each)
        self.mean_head = Dense(3, activation='linear', name='mean_params')
        self.log_var_head = Dense(3, activation='linear', name='log_var_params')

        # Initialize log variance head with small negative values
        # Will be set after model is built

        # Trajectory estimator with Jacobians
        time_samples = model_params.get('time_samples', np.linspace(0.05, 0.55, 10))
        eig = model_params.get('eig', -3.0)
        trainable_eig = model_params.get('trainable_eig', True)
        
        if trainable_eig:
            self.traj_estimator = ElbowAdaptJacobianTrajEstimator(
                eig=eig, time_samples=time_samples, name='adapt_traj_estimator'
            )
        else:
            self.traj_estimator = ElbowJacobianTrajEstimator(
                eig=eig, time_samples=time_samples, name='traj_estimator'
            )

        # Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse")

    def _build_encoder(self, params):
        """Build TCN encoder matching existing elbow architecture."""
        encoder = tf.keras.Sequential(name='encoder')

        # Input reshaping
        if len(params['input_shape']) < 3:
            encoder.add(Reshape((1, *params['input_shape']), input_shape=params['input_shape']))
        else:
            encoder.add(InputLayer(input_shape=params['input_shape']))

        # TCN encoding layers
        e_layers = encode_layers(
            depth_mul_in=params['depth_mul_in'],
            krnl_in=params['krnl_in'],
            pad=params['pad'],
            strides=params.get('strides', ((1, 1), (1, 1), (1, 1))),
            dil=params['dil'],
            mpool=params.get('mpool', ((0, 0), (0, 0), (1, 64))),
            acts=params['acts'],
            l_norm=params.get('l_norm', False),
            conv_drp=params.get('conv_drp', True),
            drp=params['drp']
        )

        for layer in e_layers:
            encoder.add(layer)

        # Optional feature convolution
        if params.get('feature_conv') is not None:
            encoder.add(Conv2D(
                kernel_size=(1, 1),
                filters=params['feature_conv'],
                activation=params['acts'][0],
                padding='same',
                kernel_regularizer=l2(params.get('l2_reg', 0.001))
            ))

        encoder.add(SpatialDropout2D(params['drp'] / 4))
        encoder.add(Flatten())

        # Dense layers
        d_layers = decode_layers(
            dense=params['dense'],
            acts=params['acts'],
            b_norm=params.get('b_norm', False),
            dense_drp=params.get('dense_drp', True),
            drp=params['drp']
        )

        for layer in d_layers:
            encoder.add(layer)

        return encoder

    def call(self, inputs, training=None):
        """Forward pass with uncertainty quantification."""
        # Extract features using encoder
        features = self.encoder(inputs, training=training)

        # Get mean and log variance parameters
        mean_params = tf.reshape(self.mean_head(features), (-1, 3, 1))  # (batch, 3, 1)
        log_var_params = self.log_var_head(features)  # (batch, 3)

        # Generate trajectory and Jacobians
        mean_traj, jacobians = self.traj_estimator(mean_params)

        # Propagate uncertainty: σ²_traj = J @ Σ_params @ J^T
        traj_log_var = self._propagate_uncertainty(log_var_params, jacobians)

        # Stack mean and log variance for loss computation
        return tf.stack([mean_traj, traj_log_var], axis=-1)  # (batch, n_times, 2)

    def _propagate_uncertainty(self, param_log_var, jacobians):
        """Propagate parameter uncertainties using Jacobians."""
        # Convert log variance to variance
        param_var = tf.exp(param_log_var)  # (batch, 3)
        param_var_expanded = param_var[:, None, :]  # (batch, 1, 3)

        # Compute trajectory variance: sum_i (J_i^2 * σ²_i) for diagonal covariance
        J_squared = tf.square(jacobians)  # (batch, n_times, 3)
        traj_var = tf.reduce_sum(J_squared * param_var_expanded, axis=-1)  # (batch, n_times)

        # Convert back to log variance with numerical stability
        traj_var = tf.maximum(traj_var, 1e-8)
        return tf.math.log(traj_var)

    def train_step(self, data):
        """Custom training step with heteroscedastic loss."""
        x, y = data
        y = tf.cast(y, tf.float32)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = aleatoric_trajectory_loss(y, y_pred)
            if self.losses:
                loss += tf.add_n(self.losses)

        # Apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        y_mean = tf.cast(y_pred[..., 0], tf.float32)
        mse = tf.reduce_mean(tf.square(y - y_mean))
        self.mse_tracker.update_state(mse)

        return {"loss": self.loss_tracker.result(), "mse": self.mse_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker]

    def initialize_log_var_head(self, init_log_var=-3.0):
        """Initialize log variance head after model is built."""
        if hasattr(self.log_var_head, 'bias') and self.log_var_head.bias is not None:
            self.log_var_head.bias.assign(tf.fill(self.log_var_head.bias.shape, init_log_var))


def create_elbow_uncertainty_model(
    input_shape=(1000, 4),
    trajectory_samples=10,
    trajectory_horizon=0.5,
    depth_mul_in=(3, 3, 2),
    krnl_in=((1, 15), (1, 3), (1, 3)),
    pad='same',
    dil=((1, 1), (1, 15), (1, 45)),
    mpool=((0, 0), (0, 0), (1, 64)),
    dense=(512, 128, 128),  # Match experiment_015 architecture
    acts=('silu', 'silu', 'silu'),
    drp=0.2,
    l2_reg=0.001,
    learning_rate=0.002,
    weight_decay=0.01,
    initial_eig=-3.0,
    trainable_eig=True,
    uncertainty_init_log_var=-3.0
):
    """
    Create uncertainty-enabled elbow trajectory model.
    
    Returns:
        tf.keras.Model: Compiled uncertainty model
    """
    # Model parameters
    model_params = {
        'input_shape': input_shape,
        'depth_mul_in': depth_mul_in,
        'krnl_in': krnl_in,
        'pad': pad,
        'dil': dil,
        'mpool': mpool,
        'dense': dense,
        'acts': acts,
        'drp': drp,
        'l2_reg': l2_reg,
        'dense_drp': True,
        'conv_drp': True,
        'time_samples': np.linspace(trajectory_horizon/trajectory_samples, trajectory_horizon, trajectory_samples),
        'eig': initial_eig,
        'trainable_eig': trainable_eig
    }

    # Create model
    model = ElbowUncertaintyModel(model_params=model_params)

    # Build model with dummy input
    dummy_input = tf.zeros((1, *input_shape))
    _ = model(dummy_input)

    # Initialize log variance head
    model.initialize_log_var_head(uncertainty_init_log_var)

    # Compile model
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0),
        loss=aleatoric_trajectory_loss,
        metrics=['mse']
    )

    return model


def test_elbow_uncertainty_model():
    """Test the elbow uncertainty model."""
    print("🔬 Testing Elbow Uncertainty Model")
    print("=" * 50)

    # Test parameters
    input_shape = (1000, 4)
    trajectory_samples = 10
    trajectory_horizon = 0.5

    print(f"Input shape: {input_shape}")
    print(f"Trajectory: {trajectory_samples} samples over {trajectory_horizon}s")

    try:
        # Create model
        model = create_elbow_uncertainty_model(
            input_shape=input_shape,
            trajectory_samples=trajectory_samples,
            trajectory_horizon=trajectory_horizon,
            dense=(512, 128, 128),  # Match experiment_015
            learning_rate=0.002,
            weight_decay=0.01
        )

        print(f"✅ Model created successfully")
        print(f"✅ Total parameters: {model.count_params():,}")

        # Test forward pass
        batch_size = 4
        test_input = tf.random.normal((batch_size, *input_shape))
        predictions = model.predict(test_input, verbose=0)

        print(f"✅ Forward pass: input {test_input.shape} → output {predictions.shape}")

        # Analyze predictions
        if len(predictions.shape) == 3 and predictions.shape[-1] == 2:
            mean_pred = predictions[..., 0]  # Trajectory predictions
            log_var_pred = predictions[..., 1]  # Log variances
            std_pred = np.sqrt(np.exp(log_var_pred))  # Standard deviations

            print(f"✅ Mean prediction range: [{mean_pred.min():.4f}, {mean_pred.max():.4f}]")
            print(f"✅ Uncertainty (σ) range: [{std_pred.min():.4f}, {std_pred.max():.4f}]")
            print(f"✅ Average uncertainty: {np.mean(std_pred):.4f}")

        print("✅ Elbow uncertainty model test completed!")
        return model

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model = test_elbow_uncertainty_model()