# utils/paths.py
import os
import yaml
from pathlib import Path

class ConfigError(Exception):
    pass

def _norm(p):
    if p is None:
        return ""
    # normalize slashes for cross-platform use
    return os.path.normpath(str(p)).replace("\\", "/")

def load_config(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ----- validate required paths (no hardcoded defaults) -----
    p = cfg.get("paths", {})
    req = ["input_roots", "images_root", "logs_root", "conversion_log", "tmp_root", "cache_root"]
    missing = [k for k in req if not p.get(k)]
    if missing:
        raise ConfigError(f"config.yaml: paths missing required keys: {missing}")

    t = cfg.get("train_io", {})
    treq = ["data_csv", "images_root", "runs_root", "run_index"]
    tmissing = [k for k in treq if not t.get(k)]
    if tmissing:
        raise ConfigError(f"config.yaml: train_io missing required keys: {tmissing}")

    # ----- normalize paths -----
    p["input_roots"] = [_norm(x) for x in p.get("input_roots", [])]
    for k in ("images_root", "logs_root", "conversion_log", "tmp_root", "cache_root"):
        p[k] = _norm(p[k])

    for k in ("data_csv", "images_root", "runs_root", "run_index", "checkpoints_root", "tensorboard_root"):
        if k in t and t[k] is not None:
            t[k] = _norm(t[k])

    cfg["paths"] = p
    cfg["train_io"] = t
    return cfg
