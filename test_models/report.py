#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate predictions and compute metrics.

Collects predictions from inference, saves to CSV, and computes metrics if labels are available.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

try:
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        confusion_matrix, precision_recall_curve, roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[report] Warning: scikit-learn not available. Metrics will be limited.", file=__import__("sys").stderr)

try:
    import yaml
except ImportError:
    raise ImportError("Missing PyYAML. Install: pip install pyyaml")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(p.read_text()) or {}


def discover_labels(input_root: Path) -> Dict[str, int]:
    """
    Discover labels from input/{benign,malware}/ structure.
    
    Returns:
        Dict mapping SHA256 (or filename) to label (0=benign, 1=malware)
    """
    labels = {}
    
    # Check for labeled subdirectories
    for label_name, label_value in [("benign", 0), ("malware", 1)]:
        label_dir = input_root / label_name
        if label_dir.exists():
            # Compute SHA256 for each file and map to label
            import hashlib
            for file_path in label_dir.rglob("*"):
                if file_path.is_file():
                    # Compute SHA256
                    h = hashlib.sha256()
                    with file_path.open("rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    sha = h.hexdigest()
                    labels[sha] = label_value
    
    return labels


def save_predictions_csv(predictions: List[Dict], csv_path: Path, labels: Optional[Dict[str, int]] = None):
    """
    Save predictions to CSV file.
    
    CSV format:
        file, model_name, mode, probability, prediction, [label]
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["file", "model_name", "mode", "probability", "prediction"]
        if labels:
            fieldnames.append("label")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            row = {
                "file": pred["file"],
                "model_name": pred["model_name"],
                "mode": pred["mode"],
                "probability": f"{pred['probability']:.6f}",
                "prediction": pred["prediction"]
            }
            if labels:
                row["label"] = labels.get(pred["file"], "")
            writer.writerow(row)
    
    print(f"[report] Saved {len(predictions)} prediction(s) to {csv_path}")


def compute_metrics(predictions: List[Dict], labels: Dict[str, int]) -> Dict:
    """
    Compute classification metrics if labels are available.
    
    Returns:
        Dict with metrics: pr_auc, roc_auc, f1, confusion_matrix, etc.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
    
    # Group predictions by model
    metrics_by_model = {}
    
    # Collect all unique models
    models = set(pred["model_name"] for pred in predictions)
    
    for model_name in models:
        model_preds = [p for p in predictions if p["model_name"] == model_name]
        
        # Extract y_true and y_pred_proba
        y_true = []
        y_proba = []
        
        for pred in model_preds:
            file_id = pred["file"]
            if file_id in labels:
                y_true.append(labels[file_id])
                y_proba.append(pred["probability"])
        
        if len(y_true) == 0:
            continue
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Compute metrics
        try:
            roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float('nan')
        except Exception:
            roc_auc = float('nan')
        
        try:
            pr_auc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float('nan')
        except Exception:
            pr_auc = float('nan')
        
        f1 = f1_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics_by_model[model_name] = {
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "n_samples": int(len(y_true))
        }
    
    # Aggregate metrics across all models
    if metrics_by_model:
        all_roc_aucs = [m["roc_auc"] for m in metrics_by_model.values() if m["roc_auc"] is not None]
        all_pr_aucs = [m["pr_auc"] for m in metrics_by_model.values() if m["pr_auc"] is not None]
        all_f1s = [m["f1"] for m in metrics_by_model.values()]
        
        summary = {
            "by_model": metrics_by_model,
            "aggregate": {
                "mean_roc_auc": float(np.mean(all_roc_aucs)) if all_roc_aucs else None,
                "mean_pr_auc": float(np.mean(all_pr_aucs)) if all_pr_aucs else None,
                "mean_f1": float(np.mean(all_f1s)) if all_f1s else None,
                "n_models": len(metrics_by_model)
            }
        }
    else:
        summary = {
            "by_model": {},
            "aggregate": {"n_models": 0}
        }
    
    return summary


def generate_report(config_path: str = "config.yaml", 
                   predictions: Optional[List[Dict]] = None) -> Dict:
    """
    Generate inference report: save CSV and compute metrics if labels exist.
    
    Args:
        config_path: Path to config file
        predictions: List of prediction dicts. If None, will be loaded from inference results.
    
    Returns:
        Dict with report summary
    """
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    
    input_root = Path(paths.get("input_root", "input")).resolve()
    logs_root = Path(paths.get("logs_root", "logs")).resolve()
    
    csv_path = logs_root / "infer_report.csv"
    summary_path = logs_root / "infer_summary.json"
    
    # If predictions not provided, we need to load them from somewhere
    # For now, assume they're passed in or we need to re-run inference
    if predictions is None:
        print("[report] No predictions provided. Run inference first.")
        return {}
    
    # Discover labels
    labels = discover_labels(input_root)
    has_labels = len(labels) > 0
    
    if has_labels:
        print(f"[report] Found labels for {len(labels)} file(s)")
    else:
        print("[report] No labels found (input/{benign,malware}/ not present)")
    
    # Save predictions CSV
    save_predictions_csv(predictions, csv_path, labels if has_labels else None)
    
    # Compute metrics if labels available
    summary = {}
    if has_labels and SKLEARN_AVAILABLE:
        print("[report] Computing metrics...")
        metrics = compute_metrics(predictions, labels)
        summary = {
            "has_labels": True,
            "n_labeled_files": len(labels),
            "metrics": metrics
        }
    else:
        summary = {
            "has_labels": has_labels,
            "n_labeled_files": len(labels) if has_labels else 0,
            "metrics": None if not has_labels else {"error": "scikit-learn not available"}
        }
    
    # Save summary JSON
    logs_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[report] Saved summary to {summary_path}")
    
    # Print summary
    if has_labels and "metrics" in summary and summary["metrics"]:
        metrics = summary["metrics"]
        if "aggregate" in metrics:
            agg = metrics["aggregate"]
            print("\n[report] Aggregate Metrics:")
            if agg.get("mean_roc_auc") is not None:
                print(f"  Mean ROC-AUC: {agg['mean_roc_auc']:.4f}")
            if agg.get("mean_pr_auc") is not None:
                print(f"  Mean PR-AUC: {agg['mean_pr_auc']:.4f}")
            if agg.get("mean_f1") is not None:
                print(f"  Mean F1: {agg['mean_f1']:.4f}")
            print(f"  Models evaluated: {agg.get('n_models', 0)}")
    
    return summary


if __name__ == "__main__":
    import sys
    # This would typically be called from main_infer.py with predictions
    print("[report] Run from main_infer.py: python main_infer.py report")
    sys.exit(0)

