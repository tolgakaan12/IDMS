# sEMG-to-Elbow Trajectory Intent Estimation — Thesis Core Code

Clean, self-contained extraction of the code behind the MSc thesis
**"Residual Modelling and Uncertainty-Aware sEMG-to-Trajectory Intent Estimation"** (T. K. Celebi, 2025).

This repo is the **core code that produced the thesis results**, pulled from the working copies and
organised so it runs. The intent-estimator result is **replication-verified**: loading the trained
checkpoint against the dataset reproduces the recorded test **R² = 0.7982814312** exactly.

## The three contributions

1. **Intent estimator (PyTorch)** — a TCANet (multi-scale EMG conv → TCN → multi-head attention → FC head)
   that predicts IDMS trajectory parameters `[v0, c1, a0]` and passes them through a differentiable
   critically-damped trajectory layer to produce a 10-point elbow-angle trajectory. R² = 0.54–0.81.
2. **Residual modelling** — extract the estimator's residuals and fit **ARMA(p,q)-GARCH(r,s)** with
   Student-t innovations (`EnhancedARMAGARCH`), with BIC order selection, a statistical-test battery, and
   synthetic-vs-baseline validation.
3. **Uncertainty (TensorFlow prototype)** — aleatoric heteroscedastic NLL propagated via the analytical
   Jacobian of the trajectory layer, plus epistemic MC-dropout. *Note: this contribution is a separate
   TensorFlow lineage (see "Frameworks" below).*

## Layout

```
.
├── data_gen/                 # dataset generator + preprocessing (Keras Sequence)
│   ├── idms_trajectory_datagenerator.py
│   └── preproc.py
├── pytorch_models/           # Contribution 1 (PyTorch)
│   ├── tcanet_idms.py        #   TCANet + differentiable IDMS trajectory layer
│   ├── train_tcanet_idms.py  #   training
│   └── pytorch_data_adapter.py
├── residual_analysis/        # Contribution 2
│   └── arma_garch_residual_model.py   # EnhancedARMAGARCH (the model used for results)
├── model_selection/          # Contribution 3 (TensorFlow)
│   ├── elbow_uncertainty_model.py
│   ├── uncertainty_losses.py
│   ├── uncertainty_trajectory_layers.py   # [vd,c1,a0] Jacobian
│   ├── elbow_trajectory_architectures.py
│   └── modular_architectures.py
├── utility/save_load_util.py # raw-EMG loading (imported lazily by preproc)
│
├── compare_training_strategies.py    # Table 4.1 / Fig 4.1: general vs fine-tuned vs subject-specific
├── calculate_baseline_r2.py          # baselines (Table 4.2)
├── test_best_model.py, verify_r2_calculation.py
├── extract_pytorch_residuals.py      # C2: residuals from the trained model
├── save_individual_trial_residuals.py
├── fit_arma_garch_all_trials.py      # C2: batch fit across all trials
├── pytorch_residual_statistical_tests.py, trial_level_statistical_tests.py
├── create_synthetic_residual_validation.py
├── train_elbow_trajectory_predictor.py   # C3 trainer (model_type='uncertainty')
├── data/                     # place idms_ready_dataset.h5 here (not committed)
└── CORE_CODE_INVENTORY.md    # provenance: where each file came from + replication recipe
```

## Pipeline / how to run

Run scripts from the repo root (they import the local packages).

```bash
# 1. Train the intent estimator
python pytorch_models/train_tcanet_idms.py
# 2. Cross-validation / fine-tuning comparison (Table 4.1)
python compare_training_strategies.py
# 3. Extract residuals from the trained model, then fit ARMA-GARCH-t across trials
python extract_pytorch_residuals.py
python fit_arma_garch_all_trials.py
# 4. Uncertainty model (TensorFlow)
python train_elbow_trajectory_predictor.py   # uses model_type='uncertainty'
```

## Replicating the thesis R² (verified)

- Model: `pytorch_models/tcanet_idms.py` (`create_tcanet_idms_model`)
- Checkpoint: a `best_model.pt` from `tcanet_idms_20250829_203210/` (see CORE_CODE_INVENTORY.md)
- Dataset: `data/idms_ready_dataset.h5`
- Test split is deterministic (`subjects=['subject_003'], stride=50, test_ratio=0.05, val_ratio=0.2, seed=42`).
- Load with `torch.load(..., weights_only=False)` (checkpoint stores `model_config`/`history`).

## Frameworks

The estimator (C1) and residual modelling (C2) are **PyTorch/Python**. The uncertainty model (C3) is a
**TensorFlow** prototype. Additionally, `data_gen/` subclasses `keras.utils.Sequence`, so **TensorFlow is
required even for the PyTorch pipeline**. Both frameworks are in `environment.yml`.

## Notes / known issues
- C3 (`elbow_uncertainty_model.py`) currently wires the older `[θ∞,A,B]` Jacobian rather than the
  thesis-correct `[vd,c1,a0]` form in `uncertainty_trajectory_layers.py`; its plots use synthetic demo data.
  (A PyTorch port of C3 is a planned follow-up.)
- The dataset (`idms_ready_dataset.h5`, ~649 MB) and model checkpoints (`*.pt`) are not committed — see
  `data/README.md`.
- `EnhancedARMAGARCH` is the model used for the published results — distinct from the `CleanARMAGARCH`
  re-derivation in the earlier `thesis_code_for_github` release.
