"""
Elbow Trajectory Architectures - Method 2 Implementation
=======================================================

New trajectory-based models for elbow angle prediction using Method 2 approach:
- Network outputs 3 trajectory parameters [v0, c1, a0]
- ElbowTrajEstimator expands to smooth trajectory
- Loss calculated in trajectory space
- Based on existing elbow architectures but modified for trajectory prediction

Date: 2025-08-23
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, DepthwiseConv2D, Reshape, Input, SpatialDropout2D, Conv2D, LayerNormalization, InputLayer, BatchNormalization, Activation
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

# Import existing modular components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_selection.modular_architectures import encode_layers, decode_layers, add_seq_layers, TrajEstimator, AdaptTrajEstimator
from model_selection.uncertainty_trajectory_layers import JacobianTrajEstimator, AdaptJacobianTrajEstimator, UncertaintyTrajEstimator
from model_selection.uncertainty_losses import aleatoric_trajectory_loss
from model_selection.elbow_uncertainty_model import create_elbow_uncertainty_model


def elbow_trajectory_single_head_model(input_shape, trajectory_samples=20, trajectory_horizon=2.0,
                                     depth_mul_in=(3, 3, 2),
                                     krnl_in=((1, 15), (1, 3), (1, 3)),
                                     pad='same', strides=((1, 1), (1, 1), (1, 1)),
                                     dil=((1, 1), (1, 15), (1, 45)),
                                     mpool=((0, 0), (0, 0), (1, 64)),
                                     dense=(512, 128, 64),
                                     acts=('silu', 'silu', 'silu'),
                                     feature_conv=None,
                                     b_norm=False, l_norm=False,
                                     dense_drp=True, conv_drp=True, drp=0.1,
                                     l2_reg=0.001, learning_rate=0.002,
                                     weight_decay=0.01,
                                     initial_eig=-2.0, trainable_eig=True):
    """
    Method 2: Single-head trajectory model for elbow angle prediction.
    
    Architecture:
    EMG[500,4] → TCN → Dense → 3 params → ElbowTrajEstimator → trajectory[samples]
    
    Args:
        input_shape: Input EMG shape (time_steps, n_channels)
        trajectory_samples: Number of trajectory points to generate
        trajectory_horizon: Time horizon in seconds for trajectory
        ... (other args same as original elbow model)
        initial_eig: Initial value for learnable eigenvalue parameter
        trainable_eig: Whether to make eig learnable
        
    Returns:
        tf.keras.Model: Compiled trajectory prediction model
    """
    model = Sequential(name='Elbow_Trajectory_Single_Head_Model')
    
    # Input layer - reshape for convolution if needed
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))
    
    # TCN Encoder layers (same as original elbow model)
    e_layers = encode_layers(depth_mul_in=depth_mul_in, krnl_in=krnl_in, pad=pad,
                             strides=strides, dil=dil, mpool=mpool, acts=acts, 
                             l_norm=l_norm, conv_drp=conv_drp, drp=drp)
    
    model = add_seq_layers(model, e_layers)
    
    # Optional feature convolution
    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, 
                         activation=acts[0], padding='same',
                         kernel_regularizer=l2(l2_reg)))
        model.add(SpatialDropout2D(drp / 4))
    
    # Flatten for dense layers
    model.add(Flatten())
    
    # Dense decoder layers (same as original)
    d_layers = decode_layers(dense=dense, acts=acts, b_norm=b_norm, 
                             dense_drp=dense_drp, drp=drp)
    model = add_seq_layers(model, d_layers)
    
    # Output layer for 3 trajectory parameters [vd, c1, a0] for 1D elbow angle
    model.add(Dense(3, activation='linear', name='trajectory_params',
                    kernel_regularizer=l2(l2_reg)))
    
    # Reshape for TrajEstimator: (batch, 3) -> (batch, 3, 1) for 1D 
    model.add(Reshape((3, 1), name='reshape_params'))
    
    # TrajEstimator layer - expands parameters to trajectory using IDMS method
    # Skip t=0 like IDMS: predict FUTURE trajectory only  
    time_samples = np.linspace(trajectory_horizon/trajectory_samples, trajectory_horizon, trajectory_samples)
    if trainable_eig:
        model.add(AdaptTrajEstimator(eig=initial_eig, time_samples=time_samples, 
                                   name='trajectory_estimator'))
    else:
        model.add(TrajEstimator(eig=initial_eig, time_samples=time_samples,
                              name='trajectory_estimator'))
    
    # Squeeze the last dimension for 1D output: (batch, 10, 1) -> (batch, 10)
    model.add(Reshape((trajectory_samples,), name='squeeze_1d'))
    
    # Compile with MSE loss in trajectory space
    model.compile(
        loss=MeanSquaredError(),
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0),
        metrics=['mse', 'mae']
    )
    
    return model


def elbow_trajectory_dual_head_model(input_shape, trajectory_samples=20, trajectory_horizon=2.0,
                                   depth_mul_in=(3, 3, 2),
                                   krnl_in=((1, 15), (1, 3), (1, 3)),
                                   pad='same', strides=((1, 1), (1, 1), (1, 1)),
                                   dil=((1, 1), (1, 15), (1, 45)),
                                   mpool=((0, 0), (0, 0), (1, 64)),
                                   dense=(512, 128, 64),
                                   acts=('silu', 'silu', 'silu'),
                                   feature_conv=None,
                                   b_norm=False, l_norm=False,
                                   dense_drp=True, conv_drp=True, drp=0.1,
                                   l2_reg=0.001, learning_rate=0.002,
                                   weight_decay=0.01,
                                   initial_eig=-2.0, trainable_eig=True,
                                   angle_head_size=64, velocity_head_size=64):
    """
    Method 2: Dual-head trajectory model with separate parameter prediction for angle and velocity.
    
    Architecture:
    EMG → TCN → Shared Dense → [Angle Head → 3 params → AngleTraj]
                              [Velocity Head → 3 params → VelTraj]
    """
    # Input layer
    if len(input_shape) < 3:
        inputs = Input(shape=input_shape)
        reshaped = Reshape((1, *input_shape))(inputs)
    else:
        inputs = Input(shape=input_shape)
        reshaped = inputs
    
    # TCN Encoder (shared)
    x = reshaped
    for i, (depth_mul, kernel, stride, dilation) in enumerate(zip(depth_mul_in, krnl_in, strides, dil)):
        x = DepthwiseConv2D(
            kernel_size=kernel,
            depth_multiplier=depth_mul,
            activation=acts[0],
            padding=pad,
            strides=stride,
            dilation_rate=dilation,
            name=f'depthwise_conv_{i}'
        )(x)
        
        if mpool[i][0]:
            x = tf.keras.layers.MaxPooling2D(pool_size=mpool[i], name=f'maxpool_{i}')(x)
        
        if l_norm:
            x = LayerNormalization(axis=-1, name=f'layernorm_{i}')(x)
        
        if conv_drp:
            x = SpatialDropout2D(drp / 4, name=f'spatial_dropout_{i}')(x)
    
    # Optional feature convolution
    if feature_conv is not None:
        x = Conv2D(kernel_size=(1, 1), filters=feature_conv, 
                   activation=acts[0], padding='same',
                   kernel_regularizer=l2(l2_reg), name='feature_conv')(x)
        x = SpatialDropout2D(drp / 4, name='feature_dropout')(x)
    
    # Flatten
    x = Flatten(name='flatten')(x)
    
    # Shared dense layers
    for i, units in enumerate(dense[:-1]):
        if b_norm:
            x = Dense(units, kernel_regularizer=l2(l2_reg), name=f'dense_{i}')(x)
            x = Activation(acts[1], name=f'activation_{i}')(x)
            x = BatchNormalization(name=f'batchnorm_{i}')(x)
        else:
            x = Dense(units, activation=acts[1], kernel_regularizer=l2(l2_reg), name=f'dense_{i}')(x)
        
        if dense_drp:
            x = Dropout(drp, name=f'dropout_{i}')(x)
    
    # Final shared dense layer
    shared_features = Dense(dense[-1], activation=acts[1], 
                           kernel_regularizer=l2(l2_reg), name='shared_features')(x)
    
    # Angle Trajectory Head
    angle_head = Dense(angle_head_size, activation=acts[1], kernel_regularizer=l2(l2_reg), name='angle_dense')(shared_features)
    angle_head = Dropout(drp/2, name='angle_dropout')(angle_head)
    angle_params = Dense(3, activation='linear', kernel_regularizer=l2(l2_reg), name='angle_trajectory_params')(angle_head)
    angle_params_reshaped = Reshape((3, 1), name='angle_reshape')(angle_params)
    
    # Velocity Trajectory Head  
    velocity_head = Dense(velocity_head_size, activation=acts[1], kernel_regularizer=l2(l2_reg), name='velocity_dense')(shared_features)
    velocity_head = Dropout(drp/2, name='velocity_dropout')(velocity_head)
    velocity_params = Dense(3, activation='linear', kernel_regularizer=l2(l2_reg), name='velocity_trajectory_params')(velocity_head)
    velocity_params_reshaped = Reshape((3, 1), name='velocity_reshape')(velocity_params)
    
    # Trajectory Estimators - Skip t=0 like IDMS: predict FUTURE trajectory only
    time_samples = np.linspace(trajectory_horizon/trajectory_samples, trajectory_horizon, trajectory_samples)
    
    if trainable_eig:
        angle_trajectory = AdaptTrajEstimator(eig=initial_eig, time_samples=time_samples,
                                            name='angle_trajectory_estimator')(angle_params_reshaped)
        velocity_trajectory = AdaptTrajEstimator(eig=initial_eig, time_samples=time_samples,
                                               name='velocity_trajectory_estimator')(velocity_params_reshaped)
    else:
        angle_trajectory = TrajEstimator(eig=initial_eig, time_samples=time_samples,
                                       name='angle_trajectory_estimator')(angle_params_reshaped)
        velocity_trajectory = TrajEstimator(eig=initial_eig, time_samples=time_samples,
                                          name='velocity_trajectory_estimator')(velocity_params_reshaped)
    
    # Create model with dual trajectory outputs
    model = Model(inputs=inputs, outputs=[angle_trajectory, velocity_trajectory], 
                  name='Elbow_Trajectory_Dual_Head_Model')
    
    # Compile with separate losses
    model.compile(
        loss={
            'angle_trajectory_estimator': 'mse',
            'velocity_trajectory_estimator': 'mse'
        },
        loss_weights={
            'angle_trajectory_estimator': 1.0,
            'velocity_trajectory_estimator': 1.0
        },
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0),
        metrics={
            'angle_trajectory_estimator': ['mse', 'mae'],
            'velocity_trajectory_estimator': ['mse', 'mae']
        }
    )
    
    return model


def elbow_trajectory_uncertainty_model(input_shape, trajectory_samples=20, trajectory_horizon=2.0,
                                     depth_mul_in=(3, 3, 2),
                                     krnl_in=((1, 15), (1, 3), (1, 3)),
                                     pad='same', strides=((1, 1), (1, 1), (1, 1)),
                                     dil=((1, 1), (1, 15), (1, 45)),
                                     mpool=((0, 0), (0, 0), (1, 64)),
                                     dense=(512, 128, 64),
                                     acts=('silu', 'silu', 'silu'),
                                     feature_conv=None,
                                     b_norm=False, l_norm=False,
                                     dense_drp=True, conv_drp=True, drp=0.1,
                                     l2_reg=0.001, learning_rate=0.002,
                                     weight_decay=0.01,
                                     initial_eig=-2.0, trainable_eig=True,
                                     uncertainty_init_log_var=-3.0):
    """
    Uncertainty-enabled single-head trajectory model for elbow angle prediction.
    
    Architecture:
    EMG[500,4] → TCN → Dense → [3 mean params, 3 log var params] → UncertaintyTrajEstimator → trajectory[samples, 2]
    
    The model outputs both trajectory predictions and uncertainty estimates using
    Jacobian-based parameter-to-trajectory space uncertainty propagation.
    
    Args:
        input_shape: Input EMG shape (time_steps, n_channels)
        trajectory_samples: Number of trajectory points to generate
        trajectory_horizon: Time horizon in seconds for trajectory
        uncertainty_init_log_var: Initial log variance for uncertainty parameters
        ... (other args same as original elbow model)
        
    Returns:
        tf.keras.Model: Compiled uncertainty trajectory prediction model
    """
    # Input layer - use functional API for more complex architecture
    if len(input_shape) < 3:
        inputs = Input(shape=input_shape)
        reshaped = Reshape((1, *input_shape))(inputs)
    else:
        inputs = Input(shape=input_shape)
        reshaped = inputs
    
    # TCN Encoder layers (same as original elbow model)
    x = reshaped
    
    # Build encoder layers manually for functional API
    for i, (depth_mul, kernel, stride, dilation) in enumerate(zip(depth_mul_in, krnl_in, strides, dil)):
        x = DepthwiseConv2D(
            kernel_size=kernel,
            depth_multiplier=depth_mul,
            activation=acts[0],
            padding=pad,
            strides=stride,
            dilation_rate=dilation,
            name=f'depthwise_conv_{i}'
        )(x)
        
        if mpool[i][0]:
            x = tf.keras.layers.MaxPooling2D(pool_size=mpool[i], name=f'maxpool_{i}')(x)
        
        if l_norm:
            x = LayerNormalization(axis=-1, name=f'layernorm_{i}')(x)
        
        if conv_drp:
            x = SpatialDropout2D(drp / 4, name=f'spatial_dropout_{i}')(x)
    
    # Optional feature convolution
    if feature_conv is not None:
        x = Conv2D(kernel_size=(1, 1), filters=feature_conv, 
                   activation=acts[0], padding='same',
                   kernel_regularizer=l2(l2_reg), name='feature_conv')(x)
        x = SpatialDropout2D(drp / 4, name='feature_dropout')(x)
    
    # Flatten for dense layers
    x = Flatten(name='flatten')(x)
    
    # Shared dense layers (same as original)
    for i, units in enumerate(dense):
        if b_norm:
            x = Dense(units, kernel_regularizer=l2(l2_reg), name=f'dense_{i}')(x)
            x = Activation(acts[1], name=f'activation_{i}')(x)
            x = BatchNormalization(name=f'batchnorm_{i}')(x)
        else:
            x = Dense(units, activation=acts[1], kernel_regularizer=l2(l2_reg), name=f'dense_{i}')(x)
        
        if dense_drp:
            x = Dropout(drp, name=f'dropout_{i}')(x)
    
    # Dual heads: mean parameters and log variance parameters
    mean_head = Dense(3, activation='linear', name='mean_trajectory_params',
                      kernel_regularizer=l2(l2_reg))(x)
    
    log_var_head = Dense(3, activation='linear', name='log_var_trajectory_params',
                         kernel_regularizer=l2(l2_reg))(x)
    
    # Reshape mean parameters for TrajEstimator: (batch, 3) -> (batch, 3, 1)
    mean_params_reshaped = Reshape((3, 1), name='reshape_mean_params')(mean_head)
    
    # Generate trajectory and uncertainty using specialized layer
    time_samples = np.linspace(trajectory_horizon/trajectory_samples, trajectory_horizon, trajectory_samples)
    
    if trainable_eig:
        traj_estimator = AdaptJacobianTrajEstimator(
            eig=initial_eig, time_samples=time_samples, name='adapt_jacobian_traj_estimator'
        )
    else:
        traj_estimator = JacobianTrajEstimator(
            eig=initial_eig, time_samples=time_samples, name='jacobian_traj_estimator'
        )
    
    # Forward pass: compute trajectories and Jacobians
    mean_trajectory, jacobians = traj_estimator(mean_params_reshaped)
    
    # Propagate parameter uncertainties to trajectory space
    traj_log_var = traj_estimator.propagate_uncertainty(log_var_head, jacobians)
    
    # Stack mean and log variance for heteroscedastic loss
    output = tf.stack([mean_trajectory, traj_log_var], axis=-1, name='uncertainty_output')
    
    # Create model
    model = Model(inputs=inputs, outputs=output, name='Elbow_Trajectory_Uncertainty_Model')
    
    # Initialize log variance head with small negative values for stability
    def init_log_var_bias(layer):
        if hasattr(layer, 'name') and layer.name == 'log_var_trajectory_params':
            # Initialize bias to uncertainty_init_log_var for stable training
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(tf.fill(layer.bias.shape, uncertainty_init_log_var))
    
    # Apply initialization after model creation
    model.built = True  # Mark as built to access layers
    for layer in model.layers:
        init_log_var_bias(layer)
    
    # Compile with heteroscedastic uncertainty loss
    model.compile(
        loss=aleatoric_trajectory_loss,
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0),
        metrics=['mse']  # Track MSE of predictions (first output channel)
    )
    
    return model


def test_trajectory_models():
    """Test the trajectory-based elbow models."""
    print("Testing Elbow Trajectory Models")
    print("=" * 50)
    
    # Test input shape (4 EMG channels, 500 time steps)
    input_shape = (500, 4)
    trajectory_samples = 20
    trajectory_horizon = 2.0
    
    print(f"Input shape: {input_shape}")
    print(f"Trajectory: {trajectory_samples} samples over {trajectory_horizon}s")
    
    # Test single-head trajectory model
    print("\n1. Testing Single-Head Trajectory Model")
    try:
        model_single = elbow_trajectory_single_head_model(
            input_shape=input_shape,
            trajectory_samples=trajectory_samples,
            trajectory_horizon=trajectory_horizon,
            dense=(256, 128, 64),
            learning_rate=0.002
        )
        
        print(f"   Model created successfully")
        print(f"   Total parameters: {model_single.count_params():,}")
        print(f"   Output shape: {model_single.output_shape}")
        
        # Test forward pass
        batch_size = 8
        sample_input = np.random.randn(batch_size, *input_shape)
        pred_single = model_single.predict(sample_input, verbose=0)
        print(f"   Forward pass: input {sample_input.shape} → output {pred_single.shape}")
        print(f"   Trajectory range: [{pred_single.min():.3f}, {pred_single.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test dual-head trajectory model
    print("\n2. Testing Dual-Head Trajectory Model")
    try:
        model_dual = elbow_trajectory_dual_head_model(
            input_shape=input_shape,
            trajectory_samples=trajectory_samples,
            trajectory_horizon=trajectory_horizon,
            dense=(256, 128, 64),
            learning_rate=0.002
        )
        
        print(f"   Model created successfully")
        print(f"   Total parameters: {model_dual.count_params():,}")
        print(f"   Output shapes: {[output.shape for output in model_dual.output]}")
        
        # Test forward pass
        pred_dual = model_dual.predict(sample_input, verbose=0)
        print(f"   Forward pass: input {sample_input.shape}")
        print(f"   Angle trajectory: {pred_dual[0].shape}, range [{pred_dual[0].min():.3f}, {pred_dual[0].max():.3f}]")
        print(f"   Velocity trajectory: {pred_dual[1].shape}, range [{pred_dual[1].min():.3f}, {pred_dual[1].max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test uncertainty-enabled trajectory model
    print("\n3. Testing Uncertainty Trajectory Model")
    try:
        model_uncertainty = elbow_trajectory_uncertainty_model(
            input_shape=input_shape,
            trajectory_samples=trajectory_samples,
            trajectory_horizon=trajectory_horizon,
            dense=(256, 128, 64),
            learning_rate=0.002,
            weight_decay=0.01,
            uncertainty_init_log_var=-3.0
        )
        
        print(f"   Model created successfully")
        print(f"   Total parameters: {model_uncertainty.count_params():,}")
        print(f"   Output shape: {model_uncertainty.output_shape}")
        
        # Test forward pass
        pred_uncertainty = model_uncertainty.predict(sample_input, verbose=0)
        print(f"   Forward pass: input {sample_input.shape} → output {pred_uncertainty.shape}")
        
        # Analyze uncertainty predictions
        if len(pred_uncertainty.shape) == 3 and pred_uncertainty.shape[-1] == 2:
            mean_pred = pred_uncertainty[..., 0]  # Trajectory predictions
            log_var_pred = pred_uncertainty[..., 1]  # Log variances
            
            std_pred = np.sqrt(np.exp(log_var_pred))  # Standard deviations
            
            print(f"   Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
            print(f"   Uncertainty (σ) range: [{std_pred.min():.3f}, {std_pred.max():.3f}]")
            print(f"   Average uncertainty: {np.mean(std_pred):.3f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ All trajectory model tests completed!")


if __name__ == "__main__":
    test_trajectory_models()