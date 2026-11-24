# MMGP baseline

Our implementation of Mesh Morphing with Gaussian Processes (MMGP; Casenave et al., NeurIPS 2023) on the transformed
datasets of this repository (`scripts/transform_airfrans.py`, `scripts/transform_vki_ls59.py`).

1. **Common mesh.** AirfRANS: the coordinates after `T_geom` and the fields of every simulation are interpolated
   (piecewise linear, nearest neighbour outside the convex hull) onto the morphed mesh of the first training
   simulation. VKI-LS59: all meshes share the same structured grid, so the nodes correspond directly.
2. **Reduction and regression.** PCA of the mesh coordinates (shape embedding, 32 modes) and of the fields (64 modes);
   a multi-output Gaussian process (anisotropic Matérn-5/2 times a constant kernel, standardised inputs, 5 restarts
   of the marginal-likelihood optimisation) maps the operating parameters and the shape embedding to the field
   embedding.
3. **Prediction.** Field embeddings of the test simulations, inverse PCA and interpolation back to their own meshes.

```bash
# AirfRANS (--task full: 800 training simulations, --task scarce: 200)
python baselines/mmgp/airfrans/phase1_interpolate.py --data-dir data/airfrans --work-dir mmgp_airfrans_full --task full
python baselines/mmgp/airfrans/phase2_train_gp.py --work-dir mmgp_airfrans_full
python baselines/mmgp/airfrans/phase3_evaluate.py --data-dir data/airfrans --work-dir mmgp_airfrans_full

# VKI-LS59
python baselines/mmgp/vki_ls59/phase1_prepare.py --data-dir data/vki_ls59 --work-dir mmgp_vki_ls59
python baselines/mmgp/vki_ls59/phase2_train_gp.py --work-dir mmgp_vki_ls59
python baselines/mmgp/vki_ls59/phase3_evaluate.py --data-dir data/vki_ls59 --work-dir mmgp_vki_ls59
```

Phase 1 of AirfRANS keeps all training simulations on the common mesh in memory (about 800 x 180,000 nodes); use
`--workers` to set the number of parallel processes.
