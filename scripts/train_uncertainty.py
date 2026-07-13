#!/usr/bin/env python3
"""Train the aleatoric uncertainty model (Contribution 3, PyTorch).

Thin entrypoint over idms.uncertainty.train_torch.main().
"""
import argparse

from idms.uncertainty.train_torch import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PyTorch uncertainty model")
    parser.add_argument("--dataset", default="data/idms_ready_dataset.h5", help="Path to idms_ready_dataset.h5")
    args = parser.parse_args()
    main(dataset_path=args.dataset)
