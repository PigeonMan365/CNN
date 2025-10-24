#!/usr/bin/env python3
"""
Dual-mode converter (compress + truncate), OS-agnostic and limit-free.

- Reads paths.input_roots (list) and paths.images_root from config.yaml
- Walks <input_root>/{benign,malware}/
- For each file, writes PNGs to: images_root/<mode>/<label>/<sha256>.png
- Does NOT write conversion_log.csv; main.py rebuilds from disk afterwards.
"""

from pathlib import Path
import sys

# ---- Robust import for convert_file regardless of how this script is invoked ----
# Try package import first, then local fallback after adding this folder to sys.path.
try:
    from preprocessing.converter import convert_file  # when imported as a package
except Exception:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from converter import convert_file  # when run as a script

try:
    import yaml
except Exception:
    yaml = None

LABELS = ("benign", "malware")
MODES = ("compress", "truncate")

def load_cfg(path="config.yaml"):
    if yaml is None:
        print("[dual_convert] PyYAML not installed. pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    p = Path(path)
    if not p.exists():
        print(f"[dual_convert] Missing {path}", file=sys.stderr)
        sys.exit(2)
    return yaml.safe_load(p.read_text())

def main():
    cfg = load_cfg("config.yaml")
    paths = cfg.get("paths", {})
    input_roots = paths.get("input_roots", ["dataset/input"])
    images_root = Path(paths.get("images_root", "dataset/output")).resolve()
    images_root.mkdir(parents=True, exist_ok=True)

    # pre-scan: show how many inputs we actually see
    found = {("benign", r): 0 for r in input_roots} | {("malware", r): 0 for r in input_roots}
    for in_root in input_roots:
        base = Path(in_root).resolve()
        for label in LABELS:
            src_dir = base / label
            if src_dir.exists():
                found[(label, in_root)] = sum(1 for f in src_dir.iterdir() if f.is_file())
    print("[dual_convert] input file counts:")
    for (label, root), n in found.items():
        print(f"  {label:7s} @ {root}: {n}")

    total = 0
    for in_root in input_roots:
        base = Path(in_root).resolve()
        for label in LABELS:
            src_dir = base / label
            if not src_dir.exists():
                continue
            for f in src_dir.iterdir():
                if not f.is_file():
                    continue
                for mode in MODES:
                    out = convert_file(f, mode, images_root, label)
                    if out is not None:
                        total += 1

    print(f"[dual_convert] Wrote {total} PNGs under {images_root}")

if __name__ == "__main__":
    main()
