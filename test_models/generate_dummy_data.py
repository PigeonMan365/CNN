#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate dummy data for testing the inference pipeline.

This script:
1. Generates random binary/text files with random content
2. Places them in input/{benign,malware}/ directories
3. Optionally runs preprocessing to convert them to PNG images

Usage:
    python generate_dummy_data.py
    python generate_dummy_data.py --num-files 10
    python generate_dummy_data.py --preprocess
    python generate_dummy_data.py --cleanup
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:
    raise ImportError("Missing PyYAML. Install: pip install pyyaml")


LABELS = ("benign", "malware")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(p.read_text()) or {}


def generate_random_file(output_path: Path, size_kb: float = 10.0, seed: Optional[int] = None, 
                         binary: bool = False) -> Path:
    """
    Generate a file with random content.
    
    Args:
        output_path: Where to save the file
        size_kb: Target file size in kilobytes (default 10 KB)
        seed: Random seed for reproducibility (default: None for random)
        binary: If True, generate binary data; if False, generate text
    
    Returns:
        Path to the generated file
    """
    if seed is not None:
        random.seed(seed)
    
    target_bytes = int(size_kb * 1024)
    
    if binary:
        # Generate random binary data
        content = bytes(random.randint(0, 255) for _ in range(target_bytes))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
    else:
        # Generate random text content: letters (upper and lower), numbers, and some special chars
        chars = string.ascii_letters + string.digits + " \n\t"
        content = ''.join(random.choice(chars) for _ in range(target_bytes))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
    
    return output_path


def setup_dummy_files(input_root: Path, num_files_per_label: int = 100, 
                     seed: int = 42, binary: bool = False) -> List[Path]:
    """
    Set up dummy files in the input directory structure.
    
    Args:
        input_root: Base input directory (e.g., input)
        num_files_per_label: Number of files to generate per label
        seed: Random seed for file generation
        binary: If True, generate binary files; if False, generate text files
    
    Returns:
        List of generated file paths
    """
    generated_files = []
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    for label in LABELS:
        label_dir = input_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_files_per_label):
            # Generate files with different sizes (1-50 KB)
            size_kb = random.uniform(1.0, 50.0)
            ext = ".bin" if binary else ".txt"
            file_name = f"dummy_{label}_{i:03d}{ext}"
            file_path = label_dir / file_name
            
            # Use different seed for each file to ensure unique content
            file_seed = seed + hash((label, i)) % 10000
            generate_random_file(file_path, size_kb=size_kb, seed=file_seed, binary=binary)
            generated_files.append(file_path)
    
    return generated_files


def cleanup_files(input_root: Path, generated_files: List[Path]) -> None:
    """
    Remove generated test files and empty directories.
    
    Args:
        input_root: Base input directory
        generated_files: List of file paths to remove
    """
    print("[generate] Cleaning up generated files...")
    
    for file_path in generated_files:
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed {file_path.name}")
    
    # Remove empty label directories
    for label in LABELS:
        label_dir = input_root / label
        if label_dir.exists() and not any(label_dir.iterdir()):
            label_dir.rmdir()
            print(f"  Removed empty directory {label_dir}")


def main():
    """CLI entry point for generating dummy data."""
    parser = argparse.ArgumentParser(
        description="Generate dummy data for testing inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 files per label (default)
  python generate_dummy_data.py
  
  # Generate 20 files per label
  python generate_dummy_data.py --num-files 20
  
  # Generate binary files instead of text
  python generate_dummy_data.py --binary
  
  # Generate files and run preprocessing
  python generate_dummy_data.py --preprocess
  
  # Clean up existing dummy files
  python generate_dummy_data.py --cleanup
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    parser.add_argument(
        "--num-files",
        type=int,
        default=100,
        help="Number of files to generate per label (default: 10)"
    )
    
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Generate binary files instead of text files"
    )
    
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing after generating files"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove existing dummy files before generating new ones"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[generate] Error: {e}")
        print(f"[generate] Please create {args.config} or specify with --config")
        return 1
    
    paths = cfg.get("paths", {})
    input_root = Path(paths.get("input_root", "input")).resolve()
    
    print(f"[generate] Input directory: {input_root}")
    print(f"[generate] Will generate {args.num_files} file(s) per label ({len(LABELS)} labels)")
    print(f"[generate] File type: {'binary' if args.binary else 'text'}")
    
    # Cleanup existing files if requested
    if args.cleanup:
        print("\n[generate] Cleaning up existing dummy files...")
        for label in LABELS:
            label_dir = input_root / label
            if label_dir.exists():
                dummy_files = list(label_dir.glob("dummy_*"))
                for f in dummy_files:
                    f.unlink()
                    print(f"  Removed {f.name}")
                if not any(label_dir.iterdir()):
                    label_dir.rmdir()
                    print(f"  Removed empty directory {label_dir}")
    
    # Generate dummy files
    print("\n[generate] Generating dummy files...")
    try:
        generated_files = setup_dummy_files(
            input_root,
            num_files_per_label=args.num_files,
            seed=args.seed,
            binary=args.binary
        )
        print(f"[generate] Generated {len(generated_files)} file(s)")
        
        # Show summary
        for label in LABELS:
            label_dir = input_root / label
            if label_dir.exists():
                count = len(list(label_dir.glob("dummy_*")))
                print(f"  {label}: {count} file(s)")
        
    except Exception as e:
        print(f"[generate] Error generating files: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run preprocessing if requested
    if args.preprocess:
        print("\n[generate] Running preprocessing...")
        try:
            from preprocess_infer import run_preprocessing
            count = run_preprocessing(args.config)
            print(f"[generate] Preprocessing completed: {count} file(s) processed")
        except Exception as e:
            print(f"[generate] Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n[generate] ✓ Dummy data generation completed!")
    print(f"[generate] Files are in: {input_root}")
    if args.preprocess:
        output_root = Path(paths.get("output_root", "output")).resolve()
        print(f"[generate] Preprocessed images are in: {output_root}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

