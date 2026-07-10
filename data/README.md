# Data

Place the elbow dataset here:

- `idms_ready_dataset.h5` (~649 MB) — synchronized 4-channel sEMG + elbow-angle trajectories.

The dataset and trained checkpoints are **not committed** (see repo `.gitignore`). This project's copy of
the dataset lives at `report/idmss/data/idms_ready_dataset.h5`; the data-preparation pipeline that builds it
(from raw sEMG + Vicon mocap) is in `masters/exp_analysis/` (`create_h5_dataset.py`, `extract_emg_trials.py`,
`emg_mocap_sync.py`). See `../CORE_CODE_INVENTORY.md`.
