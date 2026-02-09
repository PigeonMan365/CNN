#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entrypoint for inference pipeline.

Commands:
    python main_infer.py preprocess   - Preprocess input files
    python main_infer.py infer        - Run inference with all models
    python main_infer.py report       - Generate report and metrics
    python main_infer.py visualize    - Launch 3D CNN visualization UI
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add test_models directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_infer import run_preprocessing
from infer import infer_all_models
from report import generate_report


def cmd_preprocess(args):
    """Preprocess input files."""
    print("[main] Running preprocessing...")
    count = run_preprocessing(args.config)
    if count == 0:
        print("[main] Warning: No files were processed")
        return 1
    return 0


def cmd_infer(args):
    """Run inference with all discovered models."""
    print("[main] Running inference...")
    results = infer_all_models(args.config)
    if not results:
        print("[main] Warning: No predictions generated")
        return 1

    print(f"[main] Generated {len(results)} prediction(s)")
    return 0


def cmd_report(args):
    """Generate report and metrics."""
    print("[main] Generating report...")

    print("[main] Collecting predictions...")
    predictions = infer_all_models(args.config)

    if not predictions:
        print("[main] Error: No predictions available. Run 'infer' first.")
        return 1

    summary = generate_report(args.config, predictions)

    if not summary:
        print("[main] Warning: Report generation may have issues")
        return 1

    return 0


def cmd_visualize(args):
    print("[main] Launching visualization UI...")

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    viz_script = Path(root) / "visualize_model.py"
    if not viz_script.exists():
        print(f"[main] Error: Visualization script not found: {viz_script}")
        return 1

    cmd = f"streamlit run \"{viz_script}\" -- --config \"{args.config}\""
    os.system(cmd)

    return 0


def parse_args(argv=None):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Inference pipeline for malware detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  preprocess  - Convert binary files in input/ to PNGs
  infer       - Run inference with all models in model/
  report      - Generate CSV report and metrics (if labels available)
  visualize   - Launch 3D CNN visualization UI

Examples:
  python main_infer.py preprocess
  python main_infer.py infer
  python main_infer.py report
  python main_infer.py visualize
        """
    )

    ap.add_argument(
        "command",
        choices=["preprocess", "infer", "report", "visualize"],
        help="Command to execute"
    )

    ap.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )

    return ap.parse_args(argv)


def main(argv=None):
    """Main entrypoint."""
    args = parse_args(argv)

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[main] Error: Config file not found: {config_path}")
        print(f"[main] Please create {config_path} or specify with --config")
        return 1

    # Dispatch to command handler
    handlers = {
        "preprocess": cmd_preprocess,
        "infer": cmd_infer,
        "report": cmd_report,
        "visualize": cmd_visualize
    }

    handler = handlers[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
