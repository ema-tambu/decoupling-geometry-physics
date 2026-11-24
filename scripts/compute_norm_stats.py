"""
Mean and standard deviation of the input features and target fields over the training split of a transformed
dataset, written to <data-dir>/norm_stats.json (read by the training and inference scripts).
AirfRANS: one entry per task (airfrans_full, airfrans_scarce); VKI-LS59: vki_ls59.

usage: python scripts/compute_norm_stats.py --data-dir /path/to/transformed/airfrans --dataset airfrans
       python scripts/compute_norm_stats.py --data-dir /path/to/transformed/vki --dataset vki_ls59
"""
import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from dgp.data.dataset import mesh_features


def stats(data_dir, names, dataset_name):
    n, s_in, s2_in, s_out, s2_out = 0, 0.0, 0.0, 0.0, 0.0
    for name in tqdm(names):
        with open(os.path.join(data_dir, name if name.endswith(".pkl") else name + ".pkl"), "rb") as f:
            d = pickle.load(f)
        x, y = mesh_features(d, dataset_name), np.asarray(d["fields"])
        s_in = s_in + x.sum(0, dtype=np.float64)
        s2_in = s2_in + (x ** 2).sum(0, dtype=np.float64)
        s_out = s_out + y.sum(0, dtype=np.float64)
        s2_out = s2_out + (y ** 2).sum(0, dtype=np.float64)
        n += len(x)
    m_in, m_out = s_in / n, s_out / n
    sd_in, sd_out = np.sqrt(s2_in / n - m_in ** 2), np.sqrt(s2_out / n - m_out ** 2)
    sd_in[sd_in == 0.0] = 1.0
    sd_out[sd_out == 0.0] = 1.0
    return dict(input_mean=m_in.tolist(), input_std=sd_in.tolist(), target_mean=m_out.tolist(), target_std=sd_out.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", choices=["airfrans", "vki_ls59"], required=True)
    a = ap.parse_args()
    with open(os.path.join(a.data_dir, "manifest.json")) as f:
        man = json.load(f)
    if a.dataset == "airfrans":
        out = {f"airfrans_{v}": stats(a.data_dir, man[f"{v}_train"], "airfrans") for v in ("full", "scarce") if man.get(f"{v}_train")}
    else:
        out = {"vki_ls59": stats(a.data_dir, man["train_samples"], "vki_ls59")}
    with open(os.path.join(a.data_dir, "norm_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("written", os.path.join(a.data_dir, "norm_stats.json"), list(out))
