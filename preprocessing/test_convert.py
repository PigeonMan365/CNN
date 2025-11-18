#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test conversion system by generating random test files and converting them.

This script:
1. Generates text files with random letters and numbers
2. Places them in dataset/input/{benign,malware}/ directories
3. Runs the conversion process to convert them to PNG images
4. Verifies the conversion was successful

Usage:
    python preprocessing/test_convert.py
    python main.py test convert
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.convert import run_all, convert_file, rebuild_conversion_log
from preprocessing.convert import MODES, LABELS


def generate_random_text_file(output_path: Path, size_kb: float = 10.0, seed: int = None) -> Path:
    """
    Generate a text file with random letters and numbers.
    
    Args:
        output_path: Where to save the file
        size_kb: Target file size in kilobytes (default 10 KB)
        seed: Random seed for reproducibility (default: None for random)
    
    Returns:
        Path to the generated file
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate random content: letters (upper and lower), numbers, and some special chars
    chars = string.ascii_letters + string.digits + " \n\t"
    target_bytes = int(size_kb * 1024)
    
    # Generate random content
    content = ''.join(random.choice(chars) for _ in range(target_bytes))
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')
    
    return output_path


def setup_test_files(input_root: Path, num_files_per_label: int = 50, seed: int = 42) -> List[Path]:
    """
    Set up test files in the input directory structure.
    
    Args:
        input_root: Base input directory (e.g., dataset/input)
        num_files_per_label: Number of files to generate per label
        seed: Random seed for file generation
    
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
            file_name = f"test_{label}_{i:03d}.txt"
            file_path = label_dir / file_name
            
            # Use different seed for each file
            file_seed = seed + hash((label, i)) % 10000
            generate_random_text_file(file_path, size_kb=size_kb, seed=file_seed)
            generated_files.append(file_path)
    
    return generated_files


def verify_conversions(images_root: Path, generated_files: List[Path]) -> bool:
    """
    Verify that all generated files were converted successfully.
    
    Args:
        images_root: Base output directory (e.g., dataset/output)
        generated_files: List of input files that should have been converted
    
    Returns:
        True if all conversions successful, False otherwise
    """
    from preprocessing.convert import sha256_file
    
    success_count = 0
    total_expected = len(generated_files) * len(MODES)
    
    print(f"\n[TEST] Verifying {total_expected} expected conversions...")
    
    for input_file in generated_files:
        # Determine label from path
        label = input_file.parent.name
        if label not in LABELS:
            continue
        
        # Compute expected SHA256
        expected_sha = sha256_file(input_file)
        
        # Check both modes
        for mode in MODES:
            expected_png = images_root / label / mode / f"{expected_sha}.png"
            if expected_png.exists():
                success_count += 1
                print(f"  ✓ {label}/{mode}/{expected_sha[:8]}...png")
            else:
                print(f"  ✗ MISSING: {label}/{mode}/{expected_sha[:8]}...png")
    
    print(f"\n[TEST] Conversion results: {success_count}/{total_expected} successful")
    
    if success_count == total_expected:
        print("[TEST] ✓ All conversions successful!")
        return True
    else:
        print(f"[TEST] ✗ {total_expected - success_count} conversion(s) failed")
        return False


def run_test(config_path: str = "config.yaml", num_files: int = 50, cleanup: bool = True) -> bool:
    """
    Run the full conversion test.
    
    Args:
        config_path: Path to config.yaml
        num_files: Number of test files to generate per label
        cleanup: If True, remove generated test files after test
    
    Returns:
        True if test passed, False otherwise
    """
    import yaml
    
    print("[TEST] Starting conversion test...")
    print(f"[TEST] Will generate {num_files} test files per label ({len(LABELS)} labels)")
    
    # Load config to get paths
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with cfg_path.open('r') as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    
    paths = cfg.get("paths", {})
    ti = cfg.get("train_io", {})
    input_roots = paths.get("input_roots", ["dataset/input"])
    input_root = Path(input_roots[0]).resolve()
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    
    print(f"[TEST] Input directory: {input_root}")
    print(f"[TEST] Output directory: {images_root}")
    
    # Step 1: Generate test files
    print("\n[TEST] Step 1: Generating random test files...")
    generated_files = setup_test_files(input_root, num_files_per_label=num_files, seed=42)
    print(f"[TEST] Generated {len(generated_files)} test files")
    
    # Step 2: Run conversion
    print("\n[TEST] Step 2: Running conversion process...")
    try:
        run_all(config_path=config_path, rebuild_only=False, skip_convert=False)
        print("[TEST] Conversion completed")
    except Exception as e:
        print(f"[TEST] ✗ Conversion failed: {e}")
        return False
    
    # Step 3: Verify conversions
    print("\n[TEST] Step 3: Verifying conversions...")
    success = verify_conversions(images_root, generated_files)
    
    # Step 4: Cleanup (optional)
    if cleanup:
        print("\n[TEST] Cleaning up test files...")
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
    
    return success


def main():
    """CLI entry point for test conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test conversion system with random files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--num-files", type=int, default=5, help="Number of test files per label (default: 5)")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test files after test")
    
    args = parser.parse_args()
    
    success = run_test(
        config_path=args.config,
        num_files=args.num_files,
        cleanup=not args.no_cleanup
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

