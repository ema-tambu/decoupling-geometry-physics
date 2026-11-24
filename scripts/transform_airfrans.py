"""
Transform the AirfRANS dataset (T = T_bal o T_geom) into one pickle per simulation.

Two stages, run in order:
  geom  per simulation: box cut, alignment, RBF morphing and radial wall correction (T_geom); with --check, det J
        at every mesh node and cell centroid. Can be split over several processes with --shard i --nshard n.
  tbal  radial eCDF pooled over the meshes of the --prior split (default scarce_train, contained in the training
        set of both the full and the scarce task), then X_Tbal for every simulation and the final dataset.

Output (OUT = --out):
  OUT/<name>.pkl, OUT/manifest.json     final dataset: params (6,), X_original, X_Tgeom, X_Tbal (N, 2),
                                         fields (N, 3) = (u, v, p), wall (N,) bool
  OUT/geom/<name>.pkl                    intermediate, per simulation (can be deleted afterwards)
  OUT/ecdf.pkl, OUT/diagnostics.json     eCDF and per-simulation diagnostics

usage:
  python scripts/transform_airfrans.py geom --data-dir /path/to/AirfRANS/Dataset --out /path/to/out [--check]
  python scripts/transform_airfrans.py tbal --data-dir /path/to/AirfRANS/Dataset --out /path/to/out
"""
import argparse
import json
import os
import pickle
import shutil
import time

import numpy as np

from dgp.datasets import airfrans as A
from dgp.morphing import StrictECDF

SPLITS = ("full_train", "full_test")      # scarce_train is a subset of full_train


def stage_geom(args):
    os.makedirs(os.path.join(args.out, "geom"), exist_ok=True)
    man = A.manifest(args.data_dir)
    names = [n for sp in SPLITS for n in man[sp][:args.limit or None]]
    cfg = A.MorphConfig(kernel=args.kernel, radius=args.radius, reg=args.reg)
    for name in names[args.shard::args.nshard]:
        target = os.path.join(args.out, "geom", name + ".pkl")
        if os.path.exists(target):
            continue
        t0 = time.time()
        s = A.load_sample(args.data_dir, name, with_cells=args.check)
        out = A.transform_geometry(s, cfg, jacobian=args.check)
        n_coll = len(out["theta"]) - len(np.unique(np.round(np.c_[out["theta"], out["r"]], 12), axis=0))
        diag = dict(out["diag"], name=name, collisions=int(n_coll), seconds=time.time() - t0)
        rec = dict(name=name, params=s["params"], X_original=s["X"], X_Tgeom=out["X_Tgeom"], theta=out["theta"],
                   r=out["r"], wall=s["wall"], fields=s["fields"], diag=diag, config=out["config"])
        with open(target + ".tmp", "wb") as f:
            pickle.dump(rec, f)
        os.replace(target + ".tmp", target)
        print(json.dumps(diag), flush=True)


def stage_tbal(args):
    man = A.manifest(args.data_dir)
    geom = os.path.join(args.out, "geom")
    available = {f[:-4] for f in os.listdir(geom) if f.endswith(".pkl")}
    load = lambda n: pickle.load(open(os.path.join(geom, n + ".pkl"), "rb"))   # noqa: E731

    prior = [n for n in man[args.prior] if n in available]
    if len(prior) < len(man[args.prior]):
        print(f"warning: {len(man[args.prior]) - len(prior)} meshes of {args.prior} missing from {geom}")
    if not prior:
        prior = sorted(n for n in available if n in man["full_train"]) or sorted(available)
        print(f"warning: eCDF estimated from the {len(prior)} available meshes instead (test run only)")
    ecdf = StrictECDF(A.R_WALL, A.R_OUT)
    for n in prior:
        ecdf.update(load(n)["r"])
    ecdf.finalize()
    with open(os.path.join(args.out, "ecdf.pkl"), "wb") as f:
        pickle.dump(dict(prior=args.prior, **ecdf.state()), f)

    out_dir = args.out
    diags = []
    for n in sorted(available):
        rec = load(n)
        r_bal = ecdf(rec["r"])
        res = dict(params=rec["params"], X_original=rec["X_original"], X_Tgeom=rec["X_Tgeom"],
                   X_Tbal=np.column_stack([rec["theta"], r_bal]), fields=rec["fields"], wall=rec["wall"], diag=rec["diag"])
        with open(os.path.join(out_dir, n + ".pkl"), "wb") as f:
            pickle.dump(res, f)
        d = dict(rec["diag"])
        d["tbal_monotone"] = bool(np.all(np.diff(r_bal[np.argsort(rec["r"], kind="stable")]) >= 0))
        diags.append(d)
    # manifest restricted to the transformed simulations
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({k: [n for n in v if n in available] for k, v in man.items()}, f, indent=1)
    with open(os.path.join(args.out, "diagnostics.json"), "w") as f:
        json.dump(dict(prior=args.prior, n_prior=len(prior), ecdf_empty_bins=ecdf.n_empty_bins, samples=diags), f, indent=1)
    print(f"{len(diags)} simulations written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["geom", "tbal"])
    ap.add_argument("--data-dir", required=True, help="raw AirfRANS dataset (folder with manifest.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="det J at all nodes and cell centroids (slower)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first n simulations of each split (quick test)")
    ap.add_argument("--kernel", default="compact_poly")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--reg", type=float, default=1e-6)
    ap.add_argument("--prior", default="scarce_train", help="split whose meshes define the radial eCDF")
    a = ap.parse_args()
    {"geom": stage_geom, "tbal": stage_tbal}[a.stage](a)
