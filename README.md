# Decoupling Geometry and Physics: Information-Preserving Transformations for Lightweight Operator Learning

[![TMLR](https://img.shields.io/badge/TMLR-2026-blue)](https://openreview.net/forum?id=SemYm9xelS)

Official code of the TMLR paper *Decoupling Geometry and Physics: Information-Preserving Transformations for
Lightweight Operator Learning*.

![Overview](assets/overview.png)

## 1. Overview

Shape-dependent CFD meshes are mapped onto a common reference domain by an explicit, one-to-one transformation
`T = T_bal ∘ T_geom`, before any learning:

- **`T_geom` (geometric alignment):** radial-basis-function (RBF) mesh morphing of the boundary displacement towards a
  user-defined target configuration (a circle for airfoils, a rectangle for turbine blades), followed by a monotone
  correction that puts the morphed boundaries exactly on their targets;
- **`T_bal` (density balancing):** a strictly increasing empirical CDF of each coordinate, pooled over the training
  meshes, which expands the finely resolved regions (boundary layers, wakes) and compresses the far field.

Only the coordinates are mapped: every mesh node keeps its values, so predictions made in the reference domain are
mapped back to the physical mesh without interpolation. The transformation is computed in double precision and is
deterministic. Any model can then be trained on the transformed coordinates; the paper uses a compact
shift-modulated SIREN (INR) and compares it with GINO, Transolver++, MMGP and MARIO.

```
dgp/                      Python package
  morphing/               RBF morphing (kernels, regularised fit, displacement and det J), strictly increasing eCDF
  datasets/               airfrans.py, vki_ls59.py: boundary targets, T_geom, boundary corrections
  models/                 shift-modulated SIREN (INR), GINO and Transolver++ wrappers
  data/, utils/           dataset loader, configuration, logging, metrics
scripts/                  transform_airfrans.py, transform_vki_ls59.py, compute_norm_stats.py, train.py, predict.py
configs/                  hyper-parameters of the INR, GINO and Transolver++ for both datasets
baselines/mmgp/           our implementation of the MMGP baseline
notebooks/                transform_one_airfoil.ipynb: T_geom and T_bal on one AirfRANS simulation
```

## 2. Installation

```bash
git clone https://github.com/ema-tambu/decoupling-geometry-physics.git
cd decoupling-geometry-physics
conda create -n dgp python=3.11
conda activate dgp
pip install -e ".[all]"
```

The INR configurations use the LAMB optimizer of [NVIDIA Apex](https://github.com/NVIDIA/apex) (`FusedLAMB`),
which has to be installed separately with its CUDA extensions. Without Apex, `scripts/train.py` stops when LAMB is
requested, unless `ALLOW_LAMB_FALLBACK=1` is set (AdamW is then used instead).

Tested with Python 3.10 / 3.11, NumPy 2, JAX 0.10 (float64), PyVista 0.48, PyTorch 2.12, neuraloperator 2.0 and
scikit-learn 1.7 / 1.8.

## 3. Data

- **AirfRANS** (Bonnet et al., NeurIPS 2022): download and unzip
  [Dataset.zip](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip) (about 9.5 GB, 1000 simulations); the
  folder contains `manifest.json` and one sub-folder per simulation. See also the
  [AirfRANS documentation](https://airfrans.readthedocs.io).
- **VKI-LS59** ([PLAID benchmark](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59)): downloaded from the
  Hugging Face hub by `scripts/transform_vki_ls59.py`.

## 4. Transform a dataset

AirfRANS (the `geom` stage can be split over several processes with `--shard i --nshard n`):

```bash
python scripts/transform_airfrans.py geom --data-dir /path/to/AirfRANS/Dataset --out data/airfrans --check
python scripts/transform_airfrans.py tbal --data-dir /path/to/AirfRANS/Dataset --out data/airfrans
```

VKI-LS59:

```bash
python scripts/transform_vki_ls59.py --out data/vki_ls59 --check
```

Each simulation is written to `<out>/<name>.pkl` with the original coordinates `X_original`, the coordinates after
`T_geom` (`X_Tgeom`) and after `T_bal ∘ T_geom` (`X_Tbal`), the fields and the global parameters, together with
`manifest.json` (train/test split), the eCDF (`ecdf.pkl`) and per-simulation diagnostics (`diagnostics.json`).
With `--check`, the Jacobian determinant of the RBF map is evaluated at every mesh node and cell centroid
(`min_det`, `n_folded`, ...), to verify that the map is one-to-one.

For VKI-LS59, the geometry enters the models through the coefficients of a 7-mode POD of the signed-distance fields;
`--pod-fit train` (default) builds the POD basis from the training shapes only, `--pod-fit all` from every shape of
the PLAID dataset (geometry only, no flow fields).

The notebook `notebooks/transform_one_airfoil.ipynb` shows both steps on a single AirfRANS simulation.

## 5. Models

```bash
python scripts/compute_norm_stats.py --data-dir data/airfrans --dataset airfrans
python scripts/train.py --config configs/airfrans/inr.yaml --data.coordinate_type tbal --data.target_channel 0
python scripts/predict.py --run runs/exp_<time>_<id>
```

The configurations of the INR, GINO and Transolver++ for both datasets are in [`configs`](configs); any entry can be
overridden from the command line. Our MMGP implementation is in [`baselines/mmgp`](baselines/mmgp); MARIO was trained
with the authors' code: https://github.com/giovannicatalani/MARIO.

## 6. Citation

```bibtex
@article{tamburini2026decoupling,
  title   = {Decoupling Geometry and Physics: Information-Preserving Transformations for Lightweight Operator Learning},
  author  = {Tamburini, Emanuele and Belme, Anca Claudia and Gallinari, Patrick and Cinnella, Paola},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=SemYm9xelS}
}
```

## License

MIT (see [LICENSE](LICENSE)). The Transolver++ model code is from
[thuml/Transolver_plus](https://github.com/thuml/Transolver_plus) (MIT), see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
