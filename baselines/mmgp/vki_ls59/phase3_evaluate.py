"""
Phase 3: MMGP Inference and Evaluation for VKI-LS59.
Uses the trained GP to predict the field embeddings, applies inverse PCA, 
calculates RRMSE directly on the native topology, exports the predictions,
and generates publication-ready 2x3 structured grid plots.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import argparse

_ap = argparse.ArgumentParser(description='MMGP baseline, phase 3: prediction of the test simulations')
_ap.add_argument('--data-dir', required=True, help='transformed dataset (scripts/transform_vki_ls59.py)')
_ap.add_argument('--work-dir', default='mmgp_vki_ls59', help='folder for the intermediate arrays, models and results')
_ap.add_argument('--workers', type=int, default=16)
_args = _ap.parse_args()
DATASET_DIR = _args.data_dir
PREPROC_DIR = os.path.join(_args.work_dir, 'preprocessed')
MODELS_DIR = os.path.join(_args.work_dir, 'models')
OUTPUT_DIR = os.path.join(_args.work_dir, 'results')
NUM_WORKERS = _args.workers


def plot_2x3_fields(X: np.ndarray, Y: np.ndarray, 
                    y_true_mach: np.ndarray, y_pred_mach: np.ndarray, err_mach: np.ndarray,
                    y_true_nut: np.ndarray, y_pred_nut: np.ndarray, err_nut: np.ndarray,
                    sample_idx: str, model_name: str, out_dir: str):
    
    # ==========================================
    # 🎛️ LAYOUT TUNING DIALS 🎛️
    # ==========================================
    zoom_factor = 1.75
    FIG_SIZE = (8*zoom_factor, 5*zoom_factor)
    W_SPACE = 0.1
    H_SPACE = 0.0
    TOP_MARGIN = 0.85
    # ==========================================

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE)
    fig.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE, top=TOP_MARGIN)
    
    # Optional: uncomment to add global title
    # fig.suptitle(f"{model_name} - Sample {sample_idx}", fontweight="bold", fontsize=16)
    
    X_grid = X.reshape(301, 121)
    Y_grid = Y.reshape(301, 121)
    
    col_titles = ["Ground Truth", "Prediction", "NRMSE"]
    row_labels = ["Mach Number", r"Turbulent Viscosity ($\nu_t$)"]
    
    data_rows = [
        (y_true_mach, y_pred_mach, err_mach),
        (y_true_nut, y_pred_nut, err_nut)
    ]
    
    # Set Column Titles just on the top row
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=14)
    
    for i in range(2):
        true_f, pred_f, err_f = data_rows[i]
        
        v_min = true_f.min()
        v_max = true_f.max()
        if v_min == v_max: v_max += 1e-5 

        # Define plotting specs
        specs = [
            dict(data=true_f.reshape(301, 121), cmap="turbo",   vmin=v_min, vmax=v_max, extend="both"),
            dict(data=pred_f.reshape(301, 121), cmap="turbo",   vmin=v_min, vmax=v_max, extend="both"),
            dict(data=err_f.reshape(301, 121),  cmap="inferno", vmin=0,     vmax=0.02,  extend="max"),
        ]
        
        ims = [] 
        for j, spec in enumerate(specs):
            ax = axes[i, j]
            
            im = ax.pcolormesh(X_grid, Y_grid, spec["data"], 
                               cmap=spec["cmap"], 
                               vmin=spec["vmin"], 
                               vmax=spec["vmax"], 
                               shading='gouraud')
            ims.append(im)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # --- NATIVE COLORBAR MANAGEMENT ---
        # Shared colorbar for Column 0 (GT) and Column 1 (Pred)
        cbar_main = fig.colorbar(
            ims[1], 
            ax=[axes[i, 0], axes[i, 1]], 
            orientation="vertical", 
            shrink=0.85, 
            aspect=20, 
            pad=0.03,
            extend=specs[1]["extend"]
        )
        cbar_main.ax.tick_params(labelsize=11)
        
        # Individual colorbar for the error (Column 2)
        cbar_err = fig.colorbar(
            ims[2], 
            ax=axes[i, 2], 
            orientation="vertical", 
            shrink=0.85, 
            aspect=20, 
            pad=0.03,
            extend=specs[2]["extend"]
        )
        cbar_err.set_ticks([0, 0.01, 0.02])
        cbar_err.set_ticklabels(["0%", "1%", "2%"])
        cbar_err.ax.tick_params(labelsize=11)

        # Add rotated row labels
        axes[i, 0].text(-0.08, 0.5, row_labels[i], transform=axes[i, 0].transAxes, rotation=90, 
                        va="center", ha="right", fontsize=14, fontweight="bold")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{model_name}_{sample_idx}_fields.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run_phase3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading models and preprocessed data...")
    test_coords = np.load(os.path.join(PREPROC_DIR, 'test_coords.npy'))
    test_fields = np.load(os.path.join(PREPROC_DIR, 'test_fields.npy'))
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
    
    # Slicing the exact same params as Phase 2
    IDX_ANGLE_IN = 7
    IDX_MACH_OUT = 8
    physics_params_test = test_params[:, [IDX_ANGLE_IN, IDX_MACH_OUT]]
    
    X_test = np.hstack([physics_params_test, Z_hat_test])
    X_test_scaled = scaler_X.transform(X_test)
    
    U_hat_pred = gp.predict(X_test_scaled)
    
    # Inverse PCA to get fields directly on the native topology
    pred_fields_flat = pca_fields.inverse_transform(U_hat_pred)
    pred_fields = pred_fields_flat.reshape(test_coords.shape[0], 36421, 2)
    
    print("Inference complete. Calculating RRMSE metrics...")

    # 2. Metric Calculation (MMGP RRMSE Formula)
    err_fractions = np.zeros((test_coords.shape[0], 2))
    
    for i in range(test_coords.shape[0]):
        for k in range(2): # Field 0: Mach, Field 1: nut
            u_ref = test_fields[i, :, k]
            u_pred = pred_fields[i, :, k]
            
            mse = np.mean((u_ref - u_pred)**2)
            max_val_sq = np.max(np.abs(u_ref))**2
            
            if max_val_sq < 1e-12:
                err_fractions[i, k] = 0.0
            else:
                err_fractions[i, k] = mse / max_val_sq
                
    rrmse_mach = np.sqrt(np.mean(err_fractions[:, 0]))
    rrmse_nut = np.sqrt(np.mean(err_fractions[:, 1]))
    
    print("\n" + "="*50)
    print("MMGP BASELINE RESULTS (VKI-LS59 RRMSE)")
    print("="*50)
    print(f"Mach (RRMSE) : {rrmse_mach:.4e}")
    print(f"Nut  (RRMSE) : {rrmse_nut:.4e}")
    print("="*50)

    # 3. Generate Standalone Plots
    print("\nGenerating publication plots for the first 3 test samples...")
    for i in tqdm(range(min(3, len(test_tags))), desc="Plotting"):
        
        y_true = test_fields[i]
        y_pred = pred_fields[i]
        
        # Calculate NRMSE arrays for plotting
        norm_mach = np.max(y_true[:, 0]) - np.min(y_true[:, 0])
        norm_mach = norm_mach if norm_mach > 1e-12 else 1.0
        err_mach = np.abs(y_true[:, 0] - y_pred[:, 0]) / norm_mach
        
        norm_nut = np.max(y_true[:, 1]) - np.min(y_true[:, 1])
        norm_nut = norm_nut if norm_nut > 1e-12 else 1.0
        err_nut = np.abs(y_true[:, 1] - y_pred[:, 1]) / norm_nut

        plot_2x3_fields(
            X=test_coords[i, :, 0], 
            Y=test_coords[i, :, 1], 
            y_true_mach=y_true[:, 0], 
            y_pred_mach=y_pred[:, 0], 
            err_mach=err_mach,
            y_true_nut=y_true[:, 1], 
            y_pred_nut=y_pred[:, 1], 
            err_nut=err_nut,
            sample_idx=test_tags[i], 
            model_name="MMGP Baseline", 
            out_dir=OUTPUT_DIR
        )

    # 4. Save Predictions for your Inference Script
    print("\nSaving test predictions...")
    predictions_dict = {test_tags[i]: pred_fields[i] for i in range(len(test_tags))}
    
    out_path = os.path.join(OUTPUT_DIR, 'mmgp_test_predictions.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(predictions_dict, f)
        
    print(f"Predictions successfully saved to {out_path}!")

if __name__ == '__main__':
    run_phase3()