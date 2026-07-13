"""
TCANet adapted for IDMS Trajectory Prediction
Adapted from: Zhao, W., Lu, H., Zhang, B. et al. TCANet: a temporal convolutional attention network 
for motor imagery EEG decoding. Cogn Neurodyn 19, 91 (2025).

Key adaptations:
1. Multi-Scale CNN → Multi-Scale EMG feature extraction
2. TCN + Attention → Temporal dynamics modeling  
3. Classification head → Trajectory parameter regression (v0, c1, a0)
4. Add TrajEstimator layer for IDMS trajectory generation
"""

import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat


class MSEMGNet(nn.Module):
    """Multi-Scale EMG Feature Extractor (adapted from MSCNet)"""
    
    def __init__(self, 
                 f1=16, 
                 pooling_size=8,  # Smaller pooling for EMG (2000Hz vs EEG)
                 dropout_rate=0.5, 
                 number_channel=4):  # 4 EMG channels
        super().__init__()
        
        # Multi-scale temporal convolutions for EMG
        # Scale 1: High frequency (125 samples ≈ 62.5ms at 2000Hz)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 125), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        
        # Scale 2: Medium frequency (62 samples ≈ 31ms at 2000Hz)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 62), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), 
            nn.Dropout(dropout_rate),
        )        
        
        # Scale 3: Low frequency (31 samples ≈ 15.5ms at 2000Hz)
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 31), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.cnn1(x)  # High frequency features
        x2 = self.cnn2(x)  # Medium frequency features  
        x3 = self.cnn3(x)  # Low frequency features
        
        # Concatenate multi-scale features along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 3*f1, 1, time_reduced)
        x = self.projection(x)  # (batch, time_reduced, 3*f1)
        return x    


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution to ensure no information leakage from future timesteps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding manually to ensure causality
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (padding, 0))  # Only pad on the left (causal padding)
        return super().forward(x)


class TCNBlock(nn.Module):
    """TCN Block with Proper Padding for Causal Convolution (adapted for EMG)"""

    def __init__(
        self,
        input_dimension: int,
        depth: int,
        kernel_size: int,
        filters: int,
        drop_prob: float,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.activation = activation()
        self.drop_prob = drop_prob
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        self.downsample = (
            nn.Conv1d(input_dimension, filters, kernel_size=1, bias=False)
            if input_dimension != filters
            else None
        )

        for i in range(depth):
            dilation = 2**i
            conv_block = nn.Sequential(
                CausalConv1d(
                    in_channels=input_dimension if i == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.drop_prob),
                CausalConv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.drop_prob),
            )
            self.layers.append(conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, input_dimension)
        x = x.permute(0, 2, 1)  # (batch_size, input_dimension, time_steps)

        res = x if self.downsample is None else self.downsample(x)
        for layer in self.layers:
            out = layer(x)
            out = out + res  # Residual connection
            out = self.activation(out)
            res = out  # Update residual
            x = out  # Update input for next layer

        out = out.permute(0, 2, 1)  # (batch_size, time_steps, filters)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for temporal feature integration"""
    
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    """Residual connection with layer normalization"""
    
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    """Single Transformer Encoder Block"""
    
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5):
        super().__init__(
            ResidualAdd(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                emb_size, 
                drop_p
            ),
        )    


class TransformerEncoder(nn.Sequential):
    """Multi-layer Transformer Encoder"""
    
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class PyTorchTrajEstimator(nn.Module):
    """
    PyTorch implementation of IDMS Trajectory Estimator
    Converts 3 parameters (v0, c1, a0) into smooth trajectory points
    """
    
    def __init__(self, eig=-2.0, time_samples=None):
        super().__init__()
        if time_samples is None:
            # Default: 10 points from 0.05s to 0.55s (0.05s delay + 0.5s horizon)
            time_samples = np.linspace(0.05, 0.55, 10)
        
        self.time_samples = time_samples
        self.register_buffer('T', torch.tensor(time_samples, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))
        self.eig = eig
        
        # Precompute constants
        self.eigsq = self.eig * self.eig
        self.register_buffer('exp_precalc', torch.exp(self.eig * self.T))
        
    def forward(self, inputs):
        """
        Convert trajectory parameters to trajectory points
        
        Args:
            inputs: (batch, 3, 1) - [v0, c1, a0] parameters
            
        Returns:
            trajectory: (batch, n_points) - trajectory points
        """
        # Extract parameters (same as TensorFlow version)
        vd = inputs[:, 0:1, :]  # Desired velocity (v0)
        c1 = inputs[:, 1:2, :]  # Motion parameter  
        a0 = inputs[:, 2:3, :]  # Initial acceleration
        
        # Apply bounds to prevent division by zero (same as TF version)
        bounded_eig = torch.clamp(torch.tensor(self.eig), -4.0, -0.5)
        bounded_eigsq = bounded_eig * bounded_eig
        bounded_exp = torch.exp(bounded_eig * self.T)
        
        # Calculate trajectory coefficients
        c2 = a0 - bounded_eig * c1
        c3 = c2 - bounded_eig * c1

        # Trajectory integration (same mathematical formula as TF version)
        integrated = vd * self.T + \
                     (c3 + (bounded_eig*c1 + bounded_eig*c2*self.T - c2)*bounded_exp)/bounded_eigsq
        
        # integrated shape: (batch, n_points, 1) -> (batch, n_points)
        return integrated.squeeze(-1)  # (batch, n_points)


class TCANet_IDMS(nn.Module):
    """TCANet adapted for IDMS Trajectory Prediction"""
    
    def __init__(
        self,
        # Signal related parameters (adapted for EMG)
        n_chans=4,              # 4 EMG channels
        n_times=1000,           # EMG window size (0.5s at 2000Hz)
        
        # Model parameters
        activation: nn.Module = nn.ELU,
        f1: int = 16,           # Base number of filters
        pooling_size: int = 8,   # Pooling size (smaller for EMG)
        drop_prob: float = 0.5,
        
        # TCN parameters
        tcn_depth: int = 3,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        
        # Transformer parameters
        transformer_heads: int = 4,
        transformer_depth: int = 2,
        
        # Trajectory parameters
        trajectory_points: int = 10,
        trajectory_horizon: float = 0.25,  # 500ms horizon
        trajectory_delay: float = 0.05,   # 50ms delay
    ):
        super().__init__()
        
        self.n_times = n_times
        self.n_chans = n_chans
        self.trajectory_points = trajectory_points
        
        # Multi-Scale EMG Feature Extractor (adapted from MSCNet)
        self.ms_emg_net = MSEMGNet(
            f1=f1,
            number_channel=n_chans,
            dropout_rate=drop_prob,
            pooling_size=pooling_size
        )

        # TCN Block for temporal dynamics
        tcn_input_dim = 3 * f1  # 3 scales * f1 filters each
        self.tcn_block = TCNBlock(
            input_dimension=tcn_input_dim,
            depth=tcn_depth,
            kernel_size=tcn_kernel_size,
            filters=tcn_filters,
            drop_prob=0.25,
            activation=activation,
        )
        
        # Transformer for attention-based feature integration
        self.transformer = TransformerEncoder(
            heads=transformer_heads, 
            depth=transformer_depth, 
            emb_size=tcn_filters
        )
        
        self.drop = nn.Dropout(0.25)
        
        # Calculate flattened dimension after pooling
        time_reduced = n_times // pooling_size  # e.g., 1000 // 8 = 125
        
        # Trajectory parameter regression head (instead of classification)
        self.trajectory_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_filters * time_reduced, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),  # 3 trajectory parameters: v0, c1, a0
        )
        
        # IDMS Trajectory Estimator 
        time_samples = np.linspace(
            trajectory_delay, 
            trajectory_delay + trajectory_horizon, 
            trajectory_points
        )
        self.trajectory_estimator = PyTorchTrajEstimator(
            eig=-2.0,
            time_samples=time_samples
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TCANet-IDMS
        
        Args:
            x: (batch_size, 1, n_chans, n_times) - EMG windows
            
        Returns:
            trajectory: (batch_size, trajectory_points) - predicted trajectory
        """
        # Multi-scale EMG feature extraction
        x = self.ms_emg_net(x)  # (batch, time_reduced, 3*f1)
        
        # Temporal convolutional processing
        x = self.tcn_block(x)  # (batch, time_reduced, tcn_filters)
        
        # Transformer attention
        sa = self.transformer(x)  # (batch, time_reduced, tcn_filters)
        x = self.drop(sa + x)  # Residual connection
        
        # Trajectory parameter regression
        params = self.trajectory_params(x)  # (batch, 3)
        params = params.unsqueeze(-1)  # (batch, 3, 1)
        
        # Generate trajectory from parameters
        trajectory = self.trajectory_estimator(params)  # (batch, trajectory_points)
        
        return trajectory


def create_tcanet_idms_model(
    window_size=1000,
    n_channels=4, 
    trajectory_points=10,
    trajectory_horizon=0.25,
    trajectory_delay=0.05
):
    """
    Factory function to create TCANet-IDMS model
    
    Args:
        window_size: EMG window size in samples
        n_channels: Number of EMG channels  
        trajectory_points: Number of trajectory output points
        trajectory_horizon: Trajectory prediction horizon (seconds)
        trajectory_delay: Delay before trajectory starts (seconds)
        
    Returns:
        TCANet_IDMS model
    """
    model = TCANet_IDMS(
        n_chans=n_channels,
        n_times=window_size,
        trajectory_points=trajectory_points,
        trajectory_horizon=trajectory_horizon,
        trajectory_delay=trajectory_delay,
        
        # Architecture hyperparameters (tunable)
        f1=16,
        pooling_size=8,
        drop_prob=0.5,
        tcn_depth=3,
        tcn_filters=32,
        transformer_heads=4,
        transformer_depth=2,
    )
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test the adapted model
    model = create_tcanet_idms_model(
        window_size=1000,
        n_channels=4,
        trajectory_points=10
    )
    
    # Test with dummy data
    batch_size = 16
    dummy_emg = torch.randn(batch_size, 1, 4, 1000)  # (batch, 1, channels, time)
    
    with torch.no_grad():
        trajectory_pred = model(dummy_emg)
        print(f"Input shape: {dummy_emg.shape}")
        print(f"Output trajectory shape: {trajectory_pred.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    print("✓ TCANet-IDMS model created successfully!")