"""
Transform the VKI-LS59 dataset (T = T_bal o T_geom) into one pickle per simulation.

Steps:
  1. load VKI-LS59 from the Hugging Face hub (PLAID-datasets/VKI-LS59) with PLAID;
  2. train/test split of the simulations with flow fields: random, seed 42, 15% test (570 / 100);
  3. T_geom for every mesh (with --check: det J at all nodes and quad centroids);
  4. T_bal: one eCDF per coordinate, pooled over the training meshes;
  5. geometric parameters: POD of the signed-distance fields (mean-centred, 7 modes); --pod-fit chooses the shapes
     defining the basis (train: training shapes only; all: every mesh of the PLAID dataset, geometry only).

Output (OUT = --out): OUT/sample_<id>.pkl with X_original, X_Tgeom, X_Tbal (N, 2), sdf (N,), fields (N, 2) =
(Mach, nu_t), params (9,) = (pod_1..pod_7, angle_in, mach_out); OUT/manifest.json; OUT/ecdf.pkl; OUT/diagnostics.json.

usage: python scripts/transform_vki_ls59.py --out /path/to/out [--check] [--pod-fit train|all]
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np

from dgp.datasets import vki_ls59 as V
from dgp.morphing import StrictECDF


def split(n_samples=670, test_ratio=0.15, seed=42):
    names = [f"sample_{i}.pkl" for i in range(n_samples)]
    random.seed(seed)
    random.shuffle(names)
    n_test = int(len(names) * test_ratio)
    key = lambda s: int(s.split("_")[1].split(".")[0])                  # noqa: E731
    return sorted(names[n_test:], key=key), sorted(names[:n_test], key=key)


def sid(name):
    return int(name.split("_")[1].split(".")[0])


def main(a):
    os.makedirs(a.out, exist_ok=True)
    dataset, problem = V.load_plaid(processes=a.procs)
    train, test = split()
    if a.limit:
        train, test = train[:a.limit], test[:max(1, a.limit // 5)]
    ids = sorted(sid(n) for n in train + test)

    # T_geom
    geom, diags = {}, []
    for i in ids:
        t0 = time.time()
        s = V.sample_arrays(dataset[i])
        V.check_structure(s["tags"])
        o = V.transform_geometry(s["X"], jacobian=a.check, quads=s["quads"] if a.check else None)
        d = dict(o["diag"], id=i, seconds=time.time() - t0)
        if a.check:
            d.update(min_det_nodes=float(o["det_nodes"].min()), min_det_centroids=float(o["det_cent"].min()))
        geom[i] = (s, o["X_Tgeom"])
        diags.append(d)
        print(json.dumps(d), flush=True)

    # T_bal
    ex, ey = StrictECDF(V.X0, V.X1), StrictECDF(V.Y0, V.Y1)
    for n in train:
        Xg = geom[sid(n)][1]
        ex.update(Xg[:, 0])
        ey.update(Xg[:, 1])
    ex.finalize()
    ey.finalize()

    # POD of the signed-distance fields
    pod_ids = [sid(n) for n in train] if a.pod_fit == "train" else list(range(len(dataset)))
    S = np.column_stack([geom[i][0]["sdf"] if i in geom else V.sample_arrays(dataset[i], with_fields=False)["sdf"] for i in pod_ids])
    mean = S.mean(axis=1, keepdims=True)
    U, sig, _ = np.linalg.svd(S - mean, full_matrices=False)
    U_r = U[:, :a.pod_modes]
    explained = float((sig[:a.pod_modes] ** 2).sum() / (sig ** 2).sum())

    for i in ids:
        s, Xg = geom[i]
        pod = U_r.T @ (s["sdf"] - mean[:, 0])
        rec = dict(X_original=s["X"], X_Tgeom=Xg, X_Tbal=V.apply_tbal(Xg, ex, ey), sdf=s["sdf"], fields=s["fields"],
                   params=np.r_[pod, s["angle_in"], s["mach_out"]])
        with open(os.path.join(a.out, f"sample_{i}.pkl"), "wb") as f:
            pickle.dump(rec, f, protocol=4)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(dict(train_samples=train, test_samples=test), f, indent=1)
    with open(os.path.join(a.out, "ecdf.pkl"), "wb") as f:
        pickle.dump(dict(ecdf_x=ex.state(), ecdf_y=ey.state()), f)
    with open(os.path.join(a.out, "diagnostics.json"), "w") as f:
        json.dump(dict(pod_fit=a.pod_fit, pod_shapes=len(pod_ids), pod_explained_variance=explained, samples=diags), f, indent=1)
    print(f"{len(ids)} simulations written to {a.out}; POD ({a.pod_fit}, {len(pod_ids)} shapes) explains {100 * explained:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="det J at all nodes and quad centroids")
    ap.add_argument("--pod-fit", choices=["train", "all"], default="train")
    ap.add_argument("--pod-modes", type=int, default=7)
    ap.add_argument("--procs", type=int, default=4, help="processes for the Hugging Face -> PLAID conversion")
    ap.add_argument("--limit", type=int, default=0, help="only a few simulations (quick test)")
    main(ap.parse_args())
