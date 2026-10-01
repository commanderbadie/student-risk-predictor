"""
src/evaluate.py — Load saved models and print a full evaluation report.

Run:
    py src/evaluate.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_and_save

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_INFO   = os.path.join(MODELS_DIR, "model_info.json")


def main():
    print("=== Evaluation Report ===\n")

    # Load model info
    with open(MODEL_INFO) as f:
        info = json.load(f)

    print(f"Trained at     : {info['trained_at']}")
    print(f"Dataset size   : {info['dataset_size']}")
    print(f"Feature count  : {info['feature_count']}")

    print("\n--- Classifier Comparison ---")
    for name, metrics in info["classifier"]["all_models"].items():
        print(f"  {name:30s}  Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}")

    print(f"\nBest Classifier: {info['classifier']['algorithm']}")
    m = info["classifier"]["metrics"]
    print(f"  Accuracy  : {m['accuracy']:.3f}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1 (macro): {m['f1']:.3f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.3f}")

    print("\n--- Regressor Comparison ---")
    for name, metrics in info["regressor"]["all_models"].items():
        print(f"  {name:30s}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}")

    print(f"\nBest Regressor: {info['regressor']['algorithm']}")
    r = info["regressor"]["metrics"]
    print(f"  MAE  : {r['mae']:.3f}")
    print(f"  RMSE : {r['rmse']:.3f}")
    print(f"  R2   : {r['r2']:.3f}")


if __name__ == "__main__":
    main()
