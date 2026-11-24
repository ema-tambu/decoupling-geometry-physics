"""
Phase 3: MMGP Inference and Evaluation.
Uses the trained GP to predict the field embeddings, applies inverse PCA, 
reverse-interpolates back to the original test topologies, and exports the 
predictions and standalone plots.
"""

import os
import gc
import json
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from tqdm import tqdm
import matplotlib.patches as patches




import argparse

_ap = argparse.ArgumentParser(description='MMGP baseline, phase 3: prediction of the test simulations and interpolation back to their meshes')
_ap.add_argument('--data-dir', required=True, help='transformed dataset (scripts/transform_airfrans.py)')
_ap.add_argument('--work-dir', default='mmgp_airfrans', help='folder for the intermediate arrays, models and results')
_ap.add_argument('--workers', type=int, default=16)
_args = _ap.parse_args()
DATASET_DIR = _args.data_dir
PREPROC_DIR = os.path.join(_args.work_dir, 'preprocessed')
MODELS_DIR = os.path.join(_args.work_dir, 'models')
OUTPUT_DIR = os.path.join(_args.work_dir, 'results')
NUM_WORKERS = _args.workers


def plot_field_grid_mmgp(x_orig, x_tgt, y_true, y_pred, sample, output_dir, model_tag="MMGP Baseline"):
    """
    Generates a publication-ready 3x3 grid comparing GT, Prediction, and Error.
    Dynamically extracts the airfoil boundary polygon from the morphed coordinate space.
    """
    # 1. Dynamically extract the airfoil polygon
    # In the morphed space (x_tgt), the airfoil boundary is exactly at radius r=0.5
    radii = np.linalg.norm(x_tgt, axis=1)
    is_airfoil = np.isclose(radii, 0.5, atol=1e-3)
    X_airfoil = x_orig[is_airfoil]
    
    # Sort the airfoil points by angle to form a clean, non-intersecting polygon
    angles = np.arctan2(X_airfoil[:, 1], X_airfoil[:, 0])
    X_airfoil = X_airfoil[np.argsort(angles)]
    
    # 2. Setup Figure
    labels = ["u-velocity", "v-velocity", "pressure"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f"{sample} - {model_tag}", fontweight="bold", fontsize=18)
    
    for j, title in enumerate(["Ground Truth", "Prediction", "NRMSE"]):
        axes[0, j].set_title(title, fontweight="bold", fontsize=14)

    X = x_orig[:, 0]
    Y = x_orig[:, 1]

    for i in range(3):
        # Calculate Point-wise Normalized RMSE
        norm_factor = np.max(y_true[:, i]) - np.min(y_true[:, i])
        if norm_factor < 1e-12: norm_factor = 1.0
        err = np.abs(y_true[:, i] - y_pred[:, i]) / norm_factor
        
        v_min = min(y_true[:, i].min(), y_pred[:, i].min())
        v_max = max(y_true[:, i].max(), y_pred[:, i].max())
        
        specs = [
            dict(data=y_true[:, i], cmap="turbo", levels=np.linspace(v_min, v_max, 128), extend="both"),
            dict(data=y_pred[:, i], cmap="turbo", levels=np.linspace(v_min, v_max, 128), extend="both"),
            dict(data=err, cmap="inferno", levels=np.linspace(0, 0.02, 128), extend="max"),
        ]
        
        ims = [] 
        for j, spec in enumerate(specs):
            ax = axes[i, j]
            # tricontourf interpolates the unstructured point cloud
            im = ax.tricontourf(X, Y, spec["data"], levels=spec["levels"], cmap=spec["cmap"], extend=spec["extend"])
            ims.append(im)
            
            # Mask the inner airfoil hole with a white polygon
            # poly = patches.Polygon(X_airfoil, closed=True, facecolor="white", edgecolor="black", linewidth=0.5, zorder=10)
            # ax.add_patch(poly)
            
            # Zoom in closely around the airfoil
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        # --- NATIVE COLORBAR MANAGEMENT ---
        # 1. Shared colorbar for GT and Pred
        cbar_main = fig.colorbar(
            ims[1], 
            ax=[axes[i, 0], axes[i, 1]], 
            orientation="vertical", 
            shrink=0.85, 
            aspect=20, 
            pad=0.03
        )
        
        # 2. Individual colorbar for the error
        cbar_err = fig.colorbar(
            ims[2], 
            ax=axes[i, 2], 
            orientation="vertical", 
            shrink=0.85, 
            aspect=20, 
            pad=0.03
        )
        cbar_err.set_ticks([0, 0.01, 0.02])
        cbar_err.set_ticklabels(["0%", "1%", "2%"])

        # Add Field Label on the far left
        axes[i, 0].text(-0.08, 0.5, labels[i], transform=axes[i, 0].transAxes, rotation=90, 
                        va="center", ha="right", fontsize=13, fontweight="bold")

    output_path = os.path.join(output_dir, f"{sample}_{model_tag}_fields.png")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def evaluate_sample_worker(sample_tuple, X_ref, dataset_dir, output_dir):
    """
    Reverse-interpolates the GP prediction from the common mesh back to the original mesh.
    Generates plots for flagged samples.
    """
    idx, tag, pred_field_on_ref, make_plot = sample_tuple
    try:
        # Robust file search
        possible_paths = [
            os.path.join(dataset_dir, 'full_test', f"{tag}morphed.pkl"),
            os.path.join(dataset_dir, 'full_test', f"{tag}.pkl"),
            os.path.join(dataset_dir, f"{tag}morphed.pkl"),
            os.path.join(dataset_dir, f"{tag}.pkl")
        ]
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if file_path is None:
            raise FileNotFoundError(f"Could not find {tag}")
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        x_tgt = data['X_Tgeom']        # Morphed coordinates of this specific sample
        x_orig = data['X_original']    # Physical coordinates
        y_true = data['fields']        # True CFD fields (u, v, p)
        
        # Reverse Interpolation: From Common Mesh (X_ref) -> Sample's Morphed Mesh (x_tgt)
        lin_interp = LinearNDInterpolator(X_ref, pred_field_on_ref)
        y_pred = lin_interp(x_tgt)
        
        # Handle boundary NaNs
        if np.isnan(y_pred).any():
            nn_interp = NearestNDInterpolator(X_ref, pred_field_on_ref)
            nan_mask = np.isnan(y_pred)
            y_pred[nan_mask] = nn_interp(x_tgt)[nan_mask]
            del nn_interp
            
        del lin_interp

        # Generate plot if flagged
        if make_plot:
            # We pass the full y_true and y_pred (all 3 fields), plus x_tgt to extract the polygon
            plot_field_grid_mmgp(x_orig, x_tgt, y_true, y_pred, tag, output_dir)

        del data, x_tgt, x_orig, y_true
        gc.collect()

        return {
            'tag': tag,
            'y_pred': y_pred.astype(np.float32)
        }

    except Exception as e:
        print(f"Error processing {tag}: {e}")
        return None


def run_phase3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading models and preprocessed data...")
    X_ref = np.load(os.path.join(PREPROC_DIR, 'reference_mesh_coords.npy'))
    test_coords = np.load(os.path.join(PREPROC_DIR, 'test_coords.npy'))
    test_params = np.load(os.path.join(PREPROC_DIR, 'test_params.npy'))
    with open(os.path.join(PREPROC_DIR, 'test_tags.json'), 'r') as f:
        test_tags = json.load(f)
        
    with open(os.path.join(MODELS_DIR, 'pca_coords.pkl'), 'rb') as f:
        pca_coords = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'pca_fields.pkl'), 'rb') as f:
        pca_fields = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'gp_model.pkl'), 'rb') as f:
        gp = pickle.load(f)

    # 1. Forward Pass (Inference)
    print("\nExecuting GP Inference...")
    test_coords_flat = test_coords.reshape(test_coords.shape[0], -1)
    Z_hat_test = pca_coords.transform(test_coords_flat)
    
    physics_params_test = test_params[:, :2]
    X_test = np.hstack([physics_params_test, Z_hat_test])
    X_test_scaled = scaler_X.transform(X_test)
    
    U_hat_pred = gp.predict(X_test_scaled)
    
    # Inverse PCA to get fields on the common mesh
    pred_fields_flat = pca_fields.inverse_transform(U_hat_pred)
    pred_fields_common = pred_fields_flat.reshape(test_coords.shape[0], X_ref.shape[0], 3)
    
    print("Inference complete. Beginning reverse-interpolation onto original test topologies...")

    # 2. Reverse Interpolation & Plotting
    # Flag the first 3 samples to generate standalone plots
    sample_tuples = [(i, test_tags[i], pred_fields_common[i], i < 3) for i in range(len(test_tags))]
    
    worker_func = partial(evaluate_sample_worker, X_ref=X_ref, dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR)
    
    t0 = time.time()
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(worker_func, sample_tuples),
            total=len(sample_tuples),
            desc="Evaluating Test Set"
        ))
    
    print(f"Reverse interpolation completed in {time.time() - t0:.2f} seconds.")
    
    # 3. Aggregate and Save Predictions
    valid_results = [r for r in results if r is not None]
    
    print("\nSaving test predictions for your inference script...")
    predictions_dict = {r['tag']: r['y_pred'] for r in valid_results}
    
    out_path = os.path.join(OUTPUT_DIR, 'mmgp_test_predictions.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(predictions_dict, f)
        
    print(f"Predictions successfully saved to {out_path}!")
    print(f"Standalone plots for the first 3 samples saved in {OUTPUT_DIR}")

if __name__ == '__main__':
    run_phase3()