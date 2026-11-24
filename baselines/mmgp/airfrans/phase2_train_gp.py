"""
Phase 2: Dimensionality Reduction and GP Training.
Fits PCA on the common mesh coordinates and fields, and trains a multi-output 
Gaussian Process to predict the field embeddings from the parameters and shape embeddings.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


import argparse

_ap = argparse.ArgumentParser(description='MMGP baseline, phase 2: PCA of coordinates and fields, Gaussian-process regression')
_ap.add_argument('--work-dir', default='mmgp_airfrans', help='folder for the intermediate arrays, models and results')
_args = _ap.parse_args()
INPUT_DIR = os.path.join(_args.work_dir, 'preprocessed')
OUTPUT_DIR = os.path.join(_args.work_dir, 'models')


def run_phase2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading preprocessed arrays from Phase 1...")
    coords = np.load(os.path.join(INPUT_DIR, 'train_coords.npy'))   # (800, 180283, 2)
    fields = np.load(os.path.join(INPUT_DIR, 'train_fields.npy'))   # (800, 180283, 3)
    params = np.load(os.path.join(INPUT_DIR, 'train_params.npy'))   # (800, 6)

    N_samples = coords.shape[0]
    
    # 1. Flatten the spatial dimensions for PCA
    print("\nFlattening spatial dimensions...")
    coords_flat = coords.reshape(N_samples, -1)  # (800, 360566)
    fields_flat = fields.reshape(N_samples, -1)  # (800, 540849)

    # 2. Shape Embedding (Coordinates PCA)
    print("Fitting PCA on Coordinates (Target: 32 components)...")
    pca_coords = PCA(n_components=32, random_state=42)
    Z_hat = pca_coords.fit_transform(coords_flat)
    print(f"Shape embedding Z_hat variance explained: {np.sum(pca_coords.explained_variance_ratio_):.4f}")

    # 3. Field Embedding (Flow Fields PCA)
    print("Fitting PCA on Flow Fields (Target: 64 components)...")
    pca_fields = PCA(n_components=64, random_state=42)
    U_hat = pca_fields.fit_transform(fields_flat)
    print(f"Field embedding U_hat variance explained: {np.sum(pca_fields.explained_variance_ratio_):.4f}")

    # 4. Prepare GP Inputs (Strictly 2 physical params + 32 shape features)
    print("\nPreparing GP inputs (Dropping NACA parameters to match MMGP strict methodology)...")
    
    # Slice the params array to strictly keep only 'v_in' and 'aoa' (the first 2 columns)
    physics_params = params[:, :2] 
    
    X_train = np.hstack([physics_params, Z_hat]) # (800, 2 + 32) = (800, 34)
    
    # Scale inputs for stable GP optimization
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # 5. Define and Train the Gaussian Process
    print("Configuring Anisotropic Matern-5/2 Gaussian Process...")
    # Anisotropic ARD kernel: length_scale is an array of size equal to number of features
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X_train_scaled.shape[1]), 
        length_scale_bounds=(1e-2, 1e2), 
        nu=2.5
    )
    
    # Scikit-Learn's GPR natively supports multi-output (U_hat is 800x64)
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=5, # Multiple restarts to avoid local minima during L-BFGS
        normalize_y=True,       # Ensures a zero-mean assumption for the GP
        random_state=42
    )

    print(f"Training GP on {X_train_scaled.shape[0]} samples (Input dim: {X_train_scaled.shape[1]}, Output dim: {U_hat.shape[1]})...")
    print("This may take a few minutes as L-BFGS optimizes the kernel hyperparameters.")
    gp.fit(X_train_scaled, U_hat)
    
    print("\nGP Training Complete!")
    print(f"Optimized Kernel: {gp.kernel_}")

    # 6. Save all artifacts for Phase 3
    print("\nSaving models to disk...")
    with open(os.path.join(OUTPUT_DIR, 'pca_coords.pkl'), 'wb') as f:
        pickle.dump(pca_coords, f)
        
    with open(os.path.join(OUTPUT_DIR, 'pca_fields.pkl'), 'wb') as f:
        pickle.dump(pca_fields, f)
        
    with open(os.path.join(OUTPUT_DIR, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
        
    with open(os.path.join(OUTPUT_DIR, 'gp_model.pkl'), 'wb') as f:
        pickle.dump(gp, f)

    print("Phase 2 complete! Ready for Phase 3 Evaluation.")

if __name__ == '__main__':
    run_phase2()