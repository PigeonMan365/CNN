"""
dual_convert.py
- Runs BOTH conversion modes over everything it finds in paths.input_roots.
- By default, processes all discovered files (no limit).
- Simply orchestrates converter.py twice (truncate, then compress).
- Paths come from config.yaml; no hardcoded defaults.
"""

# --- add this block near the top of the file ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------

import argparse
import subprocess
import sys
from utils.paths import load_config


def run(cmd):
    print("> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="Run both compress and truncate conversions over input_roots (no filtering)."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = no limit)")
    args = ap.parse_args()

    # Validate config exists/parsable (also normalizes paths)
    _ = load_config(args.config)

    # Run truncate then compress to populate images_root and conversion_log
    for mode in ("truncate", "compress"):
        cmd = [sys.executable, "preprocessing/converter.py", "--config", args.config, "--mode", mode]
        if args.limit and args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        run(cmd)


if __name__ == "__main__":
    main()
