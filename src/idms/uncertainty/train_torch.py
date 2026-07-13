"""Training loop for the PyTorch aleatoric uncertainty model (Contribution 3).

Trains AleatoricUncertaintyModel on the same EMG->trajectory data as the C1
estimator, but with the heteroscedastic NLL loss. Replaces the TensorFlow
train_elbow_trajectory_predictor.py (model_type='uncertainty'). Pure torch.
"""
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from idms.uncertainty.model_torch import AleatoricUncertaintyModel
from idms.uncertainty.losses_torch import aleatoric_trajectory_loss


def _epoch(model, loader, device, optimizer=None, var_regularization=0.0, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            loss = aleatoric_trajectory_loss(y, model(x), var_regularization=var_regularization)
        if train:
            optimizer.zero_grad()
            loss.backward()
            # NLL is numerically fragile early in training (Seitzer et al. 2022) -> clip.
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def fit_uncertainty_model(
    model: AleatoricUncertaintyModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 20,
    weight_decay: float = 1e-2,
    var_regularization: float = 0.0,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """Train with AdamW + NLL loss + gradient clipping + early stopping. Restores best-val weights."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10, min_lr=1e-7)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        train_loss = _epoch(model, train_loader, device, optimizer, var_regularization, grad_clip)
        history["train_loss"].append(train_loss)
        if val_loader is None:
            continue
        val_loss = _epoch(model, val_loader, device)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main(dataset_path: str = "data/idms_ready_dataset.h5"):
    """Train an uncertainty model end-to-end on the elbow dataset."""
    from idms.estimator.data.torch_dataset import create_idms_dataloaders

    train_loader, val_loader, _ = create_idms_dataloaders(dataset_path=dataset_path)
    model = AleatoricUncertaintyModel()
    history = fit_uncertainty_model(model, train_loader, val_loader)
    print(f"Done. best val NLL: {min(history['val_loss']):.4f}")
    return model, history
