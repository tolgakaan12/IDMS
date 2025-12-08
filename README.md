# Intent Driven Motion Synthesis and Residual Analysis

Predicting elbow joint trajectories from sEMG (surface electromyography) signals using deep learning and statistical residual analysis.

## Overview

This project implements a TCANet-IDMS model that:
- Extracts features from 4-channel EMG signals using multi-scale CNNs
- Captures temporal dynamics with causal dilated convolutions
- Predicts trajectory parameters using transformer attention
- Analyses prediction residuals with ARMA-GARCH-t models

## Project Structure

```
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # TCANet-IDMS neural network
│   ├── training/           # Training utilities
│   ├── residual_analysis/  # ARMA-GARCH statistical models
│   └── evaluation/         # Metrics
├── scripts/                # Experiment and visualization scripts
├── data/                   # Raw and processed datasets
└── outputs/                # Generated figures and results
```

## Setup

```bash
conda env create -f env.yml
conda activate IDMS
```

## Usage

### Run Experiments
```bash
python scripts/run_experiments.py
```

### Extract Residuals
```bash
python scripts/extract_residuals.py --model-path <path-to-model>
```

### Fit ARMA-GARCH Models
```bash
python scripts/fit_arma_garch.py --subject subject_001
```

### Complete Workflow
```bash
python scripts/run_residual_workflow.py
```

## Model Architecture

![Model Architecture](docs/nnoverview.png)

- **Input**: EMG window (4 channels, 1000 samples at 2000Hz)
- **Feature Extraction**: 3-scale CNN (125, 62, 31 sample kernels)
- **Temporal Processing**: TCN with exponential dilation
- **Attention**: Multi-head transformer encoder
- **Output**: 10 trajectory points (0.25s horizon)

## Data

- 5 subjects with EMG recordings
- 4 EMG channels: biceps, triceps, brachioradialis, extensor carpi ulnaris
- HDF5 format: `data/idms_ready_dataset.h5`

## Dependencies

- PyTorch 2.8
- NumPy, Pandas, SciPy
- StatsModels, ARCH
- Matplotlib, Seaborn

See `env.yml` for complete environment specification.
