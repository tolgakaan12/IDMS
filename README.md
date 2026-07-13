# sEMG → Elbow Trajectory Intent Estimation

Core code for the MSc thesis **"Residual Modelling and Uncertainty-Aware
sEMG-to-Trajectory Intent Estimation"** (T. K. Celebi, 2025).

Predicts near-future **elbow-angle trajectories** from 4-channel surface EMG, then
(2) models the estimator's residuals with ARMA-GARCH-t, and (3) adds predictive
uncertainty. The intent-estimator result is **replication-verified**: the trained
checkpoint reproduces test **R² = 0.7982814312** exactly.

## Install

```bash
mamba env create -f environment.yml   # or conda
mamba activate idms
pip install -e .            # runtime (PyTorch, statsmodels, arch, ...)
pip install -e ".[dev]"     # + pytest
```

The package installs as `idms`; import from anywhere (`from idms.estimator.models.tcanet import ...`).

## Repository structure

```
src/idms/
├── common/        metrics.py · config.py (DataConfig) · save_load_util.py
├── data/          generator.py (windowing, numpy — no TF) · preproc.py
├── estimator/     C1 — PyTorch
│   ├── models/tcanet.py          TCANet + differentiable IDMS trajectory layer
│   ├── data/torch_dataset.py     torch Dataset/DataLoaders over the generator
│   └── training/train.py         TCANetIDMSTrainer + main()
├── residuals/     C2 — arma_garch.py (EnhancedARMAGARCH) · cache.py · trial_stats.py
└── uncertainty/   C3 — PyTorch (model_torch · losses_torch · jacobian_torch · train_torch)
scripts/           thin entrypoints (run from repo root)
tests/             pytest suite
```

## The three contributions & how to run them

Everything below assumes the dataset is at `data/idms_ready_dataset.h5` (see **Data**).

**1. Intent estimator (PyTorch).**
```bash
python scripts/train_estimator.py                    # train TCANet
python scripts/compare_training_strategies.py        # Table 4.1: pretrain→finetune vs direct
python scripts/calculate_baseline_r2.py              # naive baseline (Table 4.2)
```

**2. Residual modelling (ARMA-GARCH-t).**
```bash
python scripts/extract_pytorch_residuals.py          # residuals from a trained model
python scripts/fit_arma_garch_all_trials.py          # fit ARMA-GARCH-t across trials
python scripts/pytorch_residual_statistical_tests.py # diagnostic test battery
python scripts/create_synthetic_residual_validation.py
```

**3. Uncertainty (aleatoric + MC-dropout, PyTorch).**
```bash
python scripts/train_uncertainty.py --dataset data/idms_ready_dataset.h5
```
> The model predicts a mean and log-variance for each of [vd, c1, a0], propagates the
> parameter variance to trajectory space via the analytical Jacobian (aleatoric), and
> uses MC-dropout at inference (epistemic). Uses the correct [vd, c1, a0]
> parameterization (`idms.uncertainty.model_torch.AleatoricUncertaintyModel`).

## Data

The pipeline reads a single HDF5 file, `data/idms_ready_dataset.h5`, with layout:

```
/subjects/<subject_id>/<trial_id>/emg_data/<channel>   # 1-D array per channel, 2 kHz
                                          /...          # + elbow angle
```
Default EMG channels: `biceps, triceps, bra, ecu`. The canonical windowing/split
constants live in `idms.common.config.DataConfig` (window 1000, stride 50,
delay 0.05 s, horizon 0.25 s, 10 points, test_ratio 0.05, seed 42).

The dataset and trained checkpoints are **not committed** (large / private) — see
`data/README.md`. A documented schema spec + a synthetic-example generator are
planned so the repo can be run without the private data.

## Replicating the thesis R²

With a trained checkpoint (`best_model.pt`) and the dataset:
```python
import torch
from idms.estimator.models.tcanet import create_tcanet_idms_model
from idms.estimator.data.torch_dataset import PyTorchIDMSDataset
# build the 'test' split with the model_config from the checkpoint, load state_dict
# (strict=True), forward-pass, r2_score on flattened trajectory points -> 0.7982814312
```
See `CORE_CODE_INVENTORY.md` for the exact checkpoint/config/paths that reproduce it.

## Testing

```bash
pytest              # metrics equivalence, TCANet forward+param-count, ARMA-GARCH fit
```

## Frameworks

The entire project is **PyTorch** — no TensorFlow. The data generator is pure numpy/h5py.

## Status

- ✅ All three contributions (estimator, residuals, uncertainty): PyTorch, tested (25 pytest tests).
- ✅ C1 replication-locked (test R² = 0.7982814312, re-verified after every refactor).
- ✅ C3 uncertainty uses the correct [vd, c1, a0] Jacobian (analytical == autograd), with
  aleatoric + MC-dropout epistemic uncertainty.
- ⏳ Onboarding for external data: schema spec, synthetic example, `--dataset` CLI args (planned).
