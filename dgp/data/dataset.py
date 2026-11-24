import json
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

COORDS = {"ori": "X_original", "tgeom": "X_Tgeom", "tbal": "X_Tbal"}


def mesh_features(data, dataset_name):
    """Per-node input features of one simulation: [X_original, X_Tgeom, X_Tbal, params (, sdf for VKI-LS59)]."""
    N = data["X_original"].shape[0]
    cols = [data["X_original"], data["X_Tgeom"], data["X_Tbal"], np.tile(data["params"], (N, 1))]
    if dataset_name == "vki_ls59":
        cols.append(data["sdf"].reshape(N, 1))
    return np.concatenate(cols, axis=1)


def norm_key(config):
    if config["data"].get("dataset_name", "airfrans") == "airfrans":
        return f"airfrans_{config['data'].get('dataset_version', 'full')}"
    return "vki_ls59"


def load_norm_stats(config):
    """Normalisation statistics written by scripts/compute_norm_stats.py into the dataset folder, in the
    keys read by the dataset and the metrics: <key>_input_mean / _std, <key>_target_<c>_mean / _std."""
    key = norm_key(config)
    with open(os.path.join(config["data"]["base_folder"], "norm_stats.json")) as f:
        s = json.load(f)[key]
    norm = {f"{key}_input_mean": s["input_mean"], f"{key}_input_std": s["input_std"]}
    for c, (m, sd) in enumerate(zip(s["target_mean"], s["target_std"])):
        norm[f"{key}_target_{c}_mean"], norm[f"{key}_target_{c}_std"] = m, sd
    return norm


class CFDPointcloudDataset(Dataset):
    """
    Loads the transformed simulations (one .pkl per simulation, listed in manifest.json), normalises them and
    serves them either as fixed-size chunks of shuffled points (INR / SIREN) or as intact meshes (GINO, Transolver++).
    Input features per node: the global parameters (6 for AirfRANS; 9 + the local SDF for VKI-LS59) followed by
    the two coordinates of the chosen representation (coordinate_type: ori, tgeom or tbal).
    """
    def __init__(self, data_path, config, is_train=True):
        super().__init__()
        self.config = config
        self.architecture = config['model']['architecture']
        self.N = config['model'].get('N', 8192)
        self.target_channel = config['data']['target_channel']
        self.target_channels = list(self.target_channel) if isinstance(self.target_channel, (list, tuple)) else [self.target_channel]
        self.dataset_name = config['data'].get('dataset_name', 'airfrans')
        self.dataset_version = config['data'].get('dataset_version', 'full')
        self.coordinate_type = config['data'].get('coordinate_type', 'ori')
        split_str = "train" if is_train else "test"

        self.is_dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        print(f"[Rank {self.rank}] Loading {split_str} data...")

        key = norm_key(config)
        in_mean = np.array(config['normalization'][f'{key}_input_mean'], dtype=np.float32)
        in_std = np.array(config['normalization'][f'{key}_input_std'], dtype=np.float32)
        target_mean = np.array([float(config['normalization'][f'{key}_target_{c}_mean']) for c in self.target_channels], dtype=np.float32)
        target_std = np.array([float(config['normalization'][f'{key}_target_{c}_std']) for c in self.target_channels], dtype=np.float32)
        if self.coordinate_type not in COORDS:
            raise ValueError(f"Unknown coordinate_type: {self.coordinate_type}")
        cx = 2 * list(COORDS).index(self.coordinate_type)
        n_params = 6 if self.dataset_name == "airfrans" else 10                 # VKI-LS59: 9 parameters + SDF

        with open(os.path.join(data_path, 'manifest.json'), 'r') as f:
            manifest = json.load(f)
        if self.dataset_name == "airfrans":
            version = "full" if split_str == "test" else self.dataset_version  # scarce and full share the test set
            file_list = manifest[f"{version}_{split_str}"]
        else:
            file_list = manifest[f"{split_str}_samples"]

        inputs_list, targets_list, val_coords_list = [], [], []
        self.mesh_lengths, self.mesh_names = [], []
        for filename in file_list:
            self.mesh_names.append(filename.replace('.pkl', ''))
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            with open(os.path.join(data_path, filename), 'rb') as f:
                data = pickle.load(f)
            self.mesh_lengths.append(data['X_original'].shape[0])

            raw_mesh = mesh_features(data, self.dataset_name).astype(np.float32)
            raw_mesh -= in_mean
            raw_mesh /= (in_std + 1e-8)
            mesh_targets = data['fields'][:, self.target_channels].astype(np.float32)
            mesh_targets -= target_mean
            mesh_targets /= (target_std + 1e-8)
            if not is_train:
                val_coords_list.append(data['X_original'].astype(np.float32))

            # [global parameters, x, y]
            inputs_list.append(np.concatenate([raw_mesh[:, 6:6 + n_params], raw_mesh[:, cx:cx + 2]], axis=1))
            targets_list.append(mesh_targets)

        if not is_train:
            self.val_original_coords = torch.from_numpy(np.vstack(val_coords_list))

        if self.architecture in ["gino", "transolver"]:
            inputs_list = [torch.from_numpy(arr).pin_memory() for arr in inputs_list]
            targets_list = [torch.from_numpy(arr).pin_memory() for arr in targets_list]
            if self.is_dist and is_train:
                remainder = len(inputs_list) % self.world_size
                if remainder != 0:                                   # same number of meshes on every rank
                    pad_count = self.world_size - remainder
                    inputs_list.extend([inputs_list[-1]] * pad_count)
                    targets_list.extend([targets_list[-1]] * pad_count)
                per_rank = len(inputs_list) // self.world_size
                self.inputs = inputs_list[self.rank * per_rank:(self.rank + 1) * per_rank]
                self.targets = targets_list[self.rank * per_rank:(self.rank + 1) * per_rank]
            else:
                self.inputs, self.targets = inputs_list, targets_list
            self.num_items = len(self.inputs)
            print(f"[Rank {self.rank}] {split_str.capitalize()} dataset ready: {self.num_items} intact meshes.")

        elif self.architecture == "siren":
            raw_inputs, raw_targets = np.vstack(inputs_list), np.vstack(targets_list)
            del inputs_list, targets_list
            if is_train:
                # scramble all points before sharding, so that every GPU gets a uniform mix of all meshes
                rng = np.random.default_rng(seed=self.config['system'].get('seed', 42))
                perm = rng.permutation(raw_inputs.shape[0])
                raw_inputs, raw_targets = raw_inputs[perm], raw_targets[perm]
            if self.is_dist and is_train:
                per_rank = raw_inputs.shape[0] // self.world_size
                raw_inputs = raw_inputs[self.rank * per_rank:(self.rank + 1) * per_rank]
                raw_targets = raw_targets[self.rank * per_rank:(self.rank + 1) * per_rank]
            remainder = raw_inputs.shape[0] % self.N
            if remainder > 0:
                raw_inputs, raw_targets = raw_inputs[:-remainder], raw_targets[:-remainder]
            self.num_items = raw_inputs.shape[0] // self.N
            # fixed chunks of N points
            self.inputs = torch.from_numpy(raw_inputs).view(self.num_items, self.N, -1).pin_memory()
            self.targets = torch.from_numpy(raw_targets).view(self.num_items, self.N, -1).pin_memory()
            print(f"[Rank {self.rank}] {split_str.capitalize()} dataset ready: {self.num_items} chunks of size {self.N}.")
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def __len__(self):
        return self.num_items

    def __getitem__(self, indices):
        return self.inputs[indices], self.targets[indices]

    def get_intact_mesh(self, mesh_idx):
        if self.architecture in ["gino", "transolver"]:
            return self.inputs[mesh_idx], self.targets[mesh_idx]
        start_idx = sum(self.mesh_lengths[:mesh_idx])
        end_idx = start_idx + self.mesh_lengths[mesh_idx]
        flat_inputs = self.inputs.view(-1, self.inputs.shape[-1])
        flat_targets = self.targets.view(-1, self.targets.shape[-1])
        return flat_inputs[start_idx:end_idx], flat_targets[start_idx:end_idx]

    def get_original_coords(self, mesh_idx):
        if hasattr(self, 'val_original_coords'):
            start_idx = sum(self.mesh_lengths[:mesh_idx])
            return self.val_original_coords[start_idx:start_idx + self.mesh_lengths[mesh_idx]]
        return None

    def reshuffle(self):
        """In-place reshuffling of the points between the chunks (INR), on the device that holds them."""
        if self.architecture == "siren":
            flat_inputs = self.inputs.view(-1, self.inputs.shape[-1])
            flat_targets = self.targets.view(-1, self.targets.shape[-1])
            perm = torch.randperm(flat_inputs.size(0), device=flat_inputs.device)
            self.inputs = flat_inputs[perm].view(self.num_items, self.N, -1)
            self.targets = flat_targets[perm].view(self.num_items, self.N, -1)
