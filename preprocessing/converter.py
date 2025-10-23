"""
Convert binaries to grayscale images (256x256) using either:
 - 'truncate': take first H*W bytes (pad with zeros if shorter)
 - 'compress': average bytes into H*W bins (simple downsampling proxy)

Outputs PNGs to images_root/<mode>/<label>/<sha256>.png and appends rows to a
single conversion_log.csv (append-only). No file-type filtering; every file
under input_roots/{benign,malware} is considered. Originals are never modified.
"""

# --- add this block near the top of the file ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------
# preprocessing/converter.py
import argparse
import csv
import hashlib
from pathlib import Path

import yaml
from PIL import Image
import numpy as np


def sha256_file_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def to_png_grid_from_bytes(b: bytes, size: int = 256, mode: str = "truncate") -> Image.Image:
    n = size * size
    if mode == "truncate":
        buf = b[:n] + b"\x00" * max(0, n - len(b))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(size, size)
    elif mode == "compress":
        step = max(1, len(b) // n)
        arr = np.frombuffer(b[::step][:n], dtype=np.uint8)
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), constant_values=0)
        arr = arr.reshape(size, size)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return Image.fromarray(arr, mode="L")


def load_existing_relpaths(conv_log: Path) -> set:
    rels = set()
    if conv_log.exists():
        with conv_log.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames and "rel_path" in r.fieldnames:
                for row in r:
                    rp = (row.get("rel_path") or "").strip()
                    if rp:
                        rels.add(rp)
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["compress", "truncate"], required=True)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip saving image AND writing CSV row if rel_path already exists in conversion_log")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_roots = [Path(p) for p in cfg["paths"]["input_roots"]]
    images_root = Path(cfg["paths"]["images_root"]).resolve()
    conv_log = Path(cfg["paths"]["conversion_log"]).resolve()
    images_root.mkdir(parents=True, exist_ok=True)
    conv_log.parent.mkdir(parents=True, exist_ok=True)

    existing_rel = load_existing_relpaths(conv_log) if args.skip_existing else set()

    fieldnames = ["rel_path", "label", "mode", "sha256"]
    exists = conv_log.exists()
    with conv_log.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()

        for input_root in input_roots:
            for cls_name, lbl in (("benign", 0), ("malware", 1)):
                in_dir = (input_root / cls_name)
                if not in_dir.exists():
                    continue
                for src in in_dir.rglob("*"):
                    if not src.is_file():
                        continue
                    raw = src.read_bytes()
                    digest = sha256_file_bytes(raw)

                    out_rel = Path(args.mode) / cls_name / f"{digest}.png"   # RELATIVE to images_root
                    out_abs = images_root / out_rel
                    out_abs.parent.mkdir(parents=True, exist_ok=True)

                    if args.skip_existing and out_rel.as_posix() in existing_rel:
                        # Already logged → skip everything
                        continue

                    # Save image only if it doesn't exist yet
                    if not out_abs.exists():
                        to_png_grid_from_bytes(raw, size=256, mode=args.mode).save(out_abs)

                    # Append CSV row (idempotent: we won't reach here if skip-existing matched)
                    w.writerow({
                        "rel_path": out_rel.as_posix(),
                        "label": lbl,
                        "mode": args.mode,
                        "sha256": digest,
                    })


if __name__ == "__main__":
    main()
