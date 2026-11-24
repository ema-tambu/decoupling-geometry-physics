import argparse
import ast

import yaml

from dgp.data.dataset import load_norm_stats


def load_config(config_path=None):
    """
    Loads a YAML configuration (--config path) and applies nested command-line overrides, e.g.
        python scripts/train.py --config configs/airfrans/inr_tbal.yaml --data.target_channel 1 --system.seed 2
    The normalisation statistics are read from <data.base_folder>/norm_stats.json (scripts/compute_norm_stats.py).
    """
    parser = argparse.ArgumentParser(description="YAML config with nested overrides (--group.key value)")
    parser.add_argument("--config", default=config_path, required=config_path is None)
    known_args, unknown_args = parser.parse_known_args()
    with open(known_args.config, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault('system', {})['config_path'] = known_args.config

    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                val_str = unknown_args[i + 1]
                i += 2
            else:                                   # boolean flag without value
                val_str = "True"
                i += 1
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = val_str
            sub_dict = config
            keys = key.split('.')
            for k in keys[:-1]:
                sub_dict = sub_dict.setdefault(k, {})
            sub_dict[keys[-1]] = val
        else:
            i += 1

    if 'normalization' not in config:
        config['normalization'] = load_norm_stats(config)
    return config
