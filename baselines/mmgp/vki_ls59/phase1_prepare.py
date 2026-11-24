"""
Phase 1: Data Preparation for VKI-LS59 (MMGP Baseline).
Extracts, stacks, and saves the original coordinates, fields, 
and scalar parameters without requiring spatial interpolation.
"""

import os
import json
import pickle
import numpy as np
from functools import partial
import multiprocessing
from tqdm import tqdm



import argparse

_ap = argparse.ArgumentParser(description='MMGP baseline, phase 1: collection of the (structured, node-aligned) coordinates and fields')
_ap.add_argument('--data-dir', required=True, help='transformed dataset (scripts/transform_vki_ls59.py)')
_ap.add_argument('--work-dir', default='mmgp_vki_ls59', help='folder for the intermediate arrays, models and results')
_ap.add_argument('--workers', type=int, default=16)
_args = _ap.parse_args()
DATASET_DIR = _args.data_dir
MANIFEST_PATH = os.path.join(DATASET_DIR, 'manifest.json')
OUTPUT_DIR = os.path.join(_args.work_dir, 'preprocessed')
NUM_WORKERS = _args.workers


def extract_sample_worker(filename_tag, dataset_dir):
    """Worker function to load and extract arrays from a single mesh sample."""
    try:
        # Sanitize the tag in case it already contains '.pkl'
        clean_tag = filename_tag.replace('.pkl', '')
        
        # Robust file search hierarchy for flat folder
        possible_paths = [
            os.path.join(dataset_dir, f"{clean_tag}morphed.pkl"),
            os.path.join(dataset_dir, f"{clean_tag}.pkl")
        ]
        
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if file_path is None:
            raise FileNotFoundError(f"Could not find {clean_tag} in {dataset_dir}")
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        x_orig = data['X_original']    # Physical coordinates (Shape Data)
        fields = data['fields']        # Physical fields (mach, nut)
        
        # 'params' should contain angle_in and mach_out.
        params = np.asarray(data['params'], dtype=np.float32).flatten()
        
        # Explicit cleanup
        del data
        
        return {
            'coords': x_orig.astype(np.float32),
            'fields': fields.astype(np.float32),
            'params': params,
            'tag': clean_tag
        }

    except Exception as e:
        print(f"Error processing {filename_tag}: {e}")
        return None

def run_phase1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    # Updated keys for VKI-LS59
    train_tags = manifest['train_samples']
    test_tags = manifest['test_samples']

    # Process Training and Test Splits
    for split_name, split_tags in [('train', train_tags), ('test', test_tags)]:
        print(f"--- Processing {split_name.upper()} split ({len(split_tags)} samples) on {NUM_WORKERS} cores ---")
        
        worker_func = partial(
            extract_sample_worker,
            dataset_dir=DATASET_DIR
        )
        
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(worker_func, split_tags),
                total=len(split_tags),
                desc=f"Extracting {split_name}"
            ))

        valid_results = [r for r in results if r is not None]
        if len(valid_results) != len(split_tags):
            print(f"Warning: {len(split_tags) - len(valid_results)} samples failed during extraction.")

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