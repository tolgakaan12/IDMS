# Data

Place the elbow dataset here:

- `idms_ready_dataset.h5` — synchronized 4-channel sEMG + elbow-angle trajectories.

The dataset and trained checkpoints are private and not committed (see repo
`.gitignore`). The dataset is built from raw sEMG + Vicon motion capture by a
separate data-preparation pipeline; the HDF5 layout it produces is documented in
the top-level `README.md` (**Data**).
