"""
Predictions of a trained model on the test simulations, in physical units and on the original mesh nodes
(the transformation keeps the identity of every node, so no interpolation back to the physical mesh is needed).

Writes <run>/predictions.npz (one array (N, n_targets) per test simulation, keyed by simulation name) and prints the
NRMSE over all test nodes: sqrt(mean((pred - ref)^2)) / (max(ref) - min(ref)) for each predicted variable.

usage: python scripts/predict.py --run runs/exp_<time>_<id> [--data-dir /path/to/transformed/dataset]
"""
import argparse
import os

import numpy as np
import torch
import yaml

from dgp.data.dataset import CFDPointcloudDataset
from dgp.models import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run folder (config.yaml and best_model.pt)")
    ap.add_argument("--data-dir", default=None, help="dataset folder, if different from the one used for training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(os.path.join(a.run, "config.yaml")) as f:
        config = yaml.safe_load(f)
    if a.data_dir:
        config["data"]["base_folder"] = a.data_dir
    device = torch.device(a.device)
    model = build_model(config).to(device)
    state = torch.load(os.path.join(a.run, "best_model.pt"), map_location=device, weights_only=False)
    state.pop("_metadata", None)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}             # weights saved from torch.compile
    model.load_state_dict(state, strict=False)
    model.eval()

    test = CFDPointcloudDataset(config["data"]["base_folder"], config, is_train=False)
    key = [k for k in config["normalization"] if k.endswith("_input_mean")][0][:-len("_input_mean")]
    channels = test.target_channels
    mean = np.array([float(config["normalization"][f"{key}_target_{c}_mean"]) for c in channels])
    std = np.array([float(config["normalization"][f"{key}_target_{c}_std"]) for c in channels])

    preds, refs, out = [], [], {}
    with torch.no_grad():
        for i, name in enumerate(test.mesh_names):
            x, y = test.get_intact_mesh(i)
            x = x.to(device)
            p = model(x) if config["model"]["architecture"] == "siren" else model(x.unsqueeze(0)).squeeze(0)
            p = p.float().cpu().numpy() * std + mean
            out[name] = p
            preds.append(p)
            refs.append(y.numpy() * std + mean)
    np.savez(os.path.join(a.run, "predictions.npz"), **out)
    P, Y = np.vstack(preds), np.vstack(refs)
    nrmse = np.sqrt(((P - Y) ** 2).mean(0)) / (Y.max(0) - Y.min(0))
    print(f"{len(out)} test simulations -> {os.path.join(a.run, 'predictions.npz')}")
    print("NRMSE over all test nodes:", {f"channel {c}": float(v) for c, v in zip(channels, nrmse)})


if __name__ == "__main__":
    main()
