"""
Phase 1: Common Mesh Interpolation for MMGP Baseline.
Interpolates the coordinate fields and flow fields of all train/test samples 
onto the reference topology of Sample 0.
"""

import os
import sys
import json
import gc
import time
import pickle
import numpy as np
from functools import partial
import multiprocessing
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from tqdm import tqdm




import argparse

_ap = argparse.ArgumentParser(description='MMGP baseline, phase 1: interpolation of the T_geom coordinates and fields onto a common mesh')
_ap.add_argument('--data-dir', required=True, help='transformed dataset (scripts/transform_airfrans.py)')
_ap.add_argument('--work-dir', default='mmgp_airfrans', help='folder for the intermediate arrays, models and results')
_ap.add_argument('--task', choices=['full', 'scarce'], default='full', help='AirfRANS training set')
_ap.add_argument('--workers', type=int, default=16)
_args = _ap.parse_args()
DATASET_DIR = _args.data_dir
MANIFEST_PATH = os.path.join(DATASET_DIR, 'manifest.json')
OUTPUT_DIR = os.path.join(_args.work_dir, 'preprocessed')
NUM_WORKERS = _args.workers


def interpolate_sample_worker(filename_tag, dataset_type, X_ref, dataset_dir):
    """
    Worker function to interpolate a single mesh sample onto the reference mesh coordinates.
    """
    try:
        # Robust file search hierarchy
        possible_paths = [
            os.path.join(dataset_dir, dataset_type, f"{filename_tag}morphed.pkl"), # In subfolder, with morphed
            os.path.join(dataset_dir, dataset_type, f"{filename_tag}.pkl"),        # In subfolder, no morphed
            os.path.join(dataset_dir, f"{filename_tag}morphed.pkl"),               # Flat folder, with morphed
            os.path.join(dataset_dir, f"{filename_tag}.pkl")                       # Flat folder, no morphed
        ]
        
        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break
                
        if file_path is None:
            raise FileNotFoundError(f"Could not find {filename_tag} in {dataset_dir}")
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        x_src = data['X_Tgeom']        # Source morphed coordinates
        x_orig = data['X_original']    # Physical coordinates (Z field)
        fields = data['fields']        # Physical fields (u, v, p)
        params = data['params']        # Physical parameters (e.g., angle of attack, velocity)
        
        # 1. Interpolate Physical Coordinate Field (Shape Data)
        lin_interp_coords = LinearNDInterpolator(x_src, x_orig)
        interp_coords = lin_interp_coords(X_ref)
        
        # Fill boundary NaNs if any
        if np.isnan(interp_coords).any():
            nn_interp_coords = NearestNDInterpolator(x_src, x_orig)
            nan_mask = np.isnan(interp_coords)
            interp_coords[nan_mask] = nn_interp_coords(X_ref)[nan_mask]
            del nn_interp_coords

        del lin_interp_coords

        # 2. Interpolate Flow Fields (Target Data)
        lin_interp_fields = LinearNDInterpolator(x_src, fields)
        interp_fields = lin_interp_fields(X_ref)
        
        # Fill boundary NaNs if any
        if np.isnan(interp_fields).any():
            nn_interp_fields = NearestNDInterpolator(x_src, fields)
            nan_mask = np.isnan(interp_fields)
            interp_fields[nan_mask] = nn_interp_fields(X_ref)[nan_mask]
            del nn_interp_fields

        del lin_interp_fields

        # Explicit cleanup to prevent memory accumulation
        del data, x_src, x_orig, fields
        gc.collect()

        return {
            'coords': interp_coords.astype(np.float32),
            'fields': interp_fields.astype(np.float32),
            'params': np.asarray(params, dtype=np.float32),
            'tag': filename_tag
        }

    except Exception as e:
        print(f"Error processing {filename_tag}: {e}")
        return None


def run_phase1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    train_tags = manifest[f'{_args.task}_train']
    test_tags = manifest['full_test']
    
    # 1. Setup Common Reference Mesh (Sample 0 of full_train)
    ref_tag = train_tags[0]
    
    # Robust search for reference file
    ref_file = None
    for p in [os.path.join(DATASET_DIR, 'full_train', f"{ref_tag}morphed.pkl"),
              os.path.join(DATASET_DIR, 'full_train', f"{ref_tag}.pkl"),
              os.path.join(DATASET_DIR, f"{ref_tag}morphed.pkl"),
              os.path.join(DATASET_DIR, f"{ref_tag}.pkl")]:
        if os.path.exists(p):
            ref_file = p
            break
            
    if ref_file is None:
        raise FileNotFoundError(f"Could not find reference mesh {ref_tag}")
        
    print(f"Setting reference common mesh: {ref_tag}")
    with open(ref_file, 'rb') as f:
        ref_data = pickle.load(f)
        
    X_ref = ref_data['X_Tgeom'].astype(np.float32)
    num_common_nodes = len(X_ref)
    print(f"Common reference mesh has {num_common_nodes} nodes.")
    
    # Save reference mesh coordinates for inverse interpolation in Phase 3
    ref_save_path = os.path.join(OUTPUT_DIR, 'reference_mesh_coords.npy')
    np.save(ref_save_path, X_ref)
    print(f"Saved reference mesh coordinates to {ref_save_path}\n")

    # 2. Process Training and Test Splits
    for split_name, split_tags in [('train', train_tags), ('test', test_tags)]:
        dataset_type = 'full_train' if split_name == 'train' else 'full_test'
        print(f"--- Processing {split_name.upper()} split ({len(split_tags)} samples) on {NUM_WORKERS} cores ---")
        
        worker_func = partial(
            interpolate_sample_worker,
            dataset_type=dataset_type,
            X_ref=X_ref,
            dataset_dir=DATASET_DIR
        )
        
        t0 = time.time()
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(worker_func, split_tags),
                total=len(split_tags),
                desc=f"Interpolating {split_name}"
            ))
            
        elapsed = time.time() - t0
        print(f"{split_name.capitalize()} interpolation completed in {elapsed:.2f} seconds.")

        # Filter out failed samples if any
        valid_results = [r for r in results if r is not None]
        if len(valid_results) != len(split_tags):
            print(f"Warning: {len(split_tags) - len(valid_results)} samples failed during interpolation.")

        # Aggregate arrays
        print(f"Stacking {split_name} matrices...")
        coords_matrix = np.stack([r['coords'] for r in valid_results], axis=0)
        fields_matrix = np.stack([r['fields'] for r in valid_results], axis=0)
        params_matrix = np.stack([r['params'] for r in valid_results], axis=0)
        tags_list = [r['tag'] for r in valid_results]

        # Save arrays to disk
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_coords.npy"), coords_matrix)
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_fields.npy"), fields_matrix)
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_params.npy"), params_matrix)
        with open(os.path.join(OUTPUT_DIR, f"{split_name}_tags.json"), 'w') as f:
            json.dump(tags_list, f, indent=4)

        print(f"Saved {split_name} arrays:")
        print(f"  Coords: {coords_matrix.shape}")
        print(f"  Fields: {fields_matrix.shape}")
        print(f"  Params: {params_matrix.shape}\n")

    print("Phase 1 completed successfully! Ready for Phase 2 PCA & GP training.")


if __name__ == '__main__':
    run_phase1()