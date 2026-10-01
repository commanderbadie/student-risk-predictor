"""
src/train.py — Trains and evaluates ML models for the Student Risk Predictor.

Models trained:
  Classification  (risk_level: Low / Medium / High)
    1. Logistic Regression      — baseline
    2. Random Forest Classifier — ensemble
    3. Gradient Boosting Classifier — best expected

  Regression (final_gpa: 0.0–4.0)
    1. Linear Regression        — baseline
    2. Gradient Boosting Regressor

Saves:
  models/risk_classifier.pkl
  models/gpa_regressor.pkl
  models/model_info.json
  data/processed/confusion_matrix.png
  data/processed/roc_curve.png
  data/processed/feature_importance.png

Run:
    py src/train.py
"""

import os
import sys
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_and_save, load_raw, Preprocessor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RISK_MODEL_PATH = os.path.join(MODELS_DIR, "risk_classifier.pkl")
GPA_MODEL_PATH = os.path.join(MODELS_DIR, "gpa_regressor.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

CLASSES = ["High", "Low", "Medium"]  # alphabetical order from sklearn


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", color="white")
    ax.tick_params(colors="white")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0D1B2A")
    plt.close()


def save_roc_curve(model, X_test, y_test, labels, path):
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_test, classes=labels)
    y_score = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    colors = ["#00BCD4", "#FF7043", "#66BB6A"]
    for i, (cls, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curves (One-vs-Rest)", color="white")
    ax.tick_params(colors="white")
    leg = ax.legend(facecolor="#1A2E45", labelcolor="white")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0D1B2A")
    plt.close()


def save_feature_importance(model, feature_names, path, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    bars = ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#00BCD4",
    )
    ax.set_xlabel("Importance", color="white")
    ax.set_title(f"Top {top_n} Feature Importances", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1A2E45")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0D1B2A")
    plt.close()


# ── Main training routine ─────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=== Loading and preprocessing data ===")
    X, y_risk, y_gpa, preprocessor = preprocess_and_save()
    feature_names = preprocessor.feature_names_

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, yr_train, yr_test, yg_train, yg_test = train_test_split(
        X, y_risk, y_gpa, test_size=0.20, random_state=42, stratify=y_risk
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Training classifiers ===")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_results = {}

    for name, clf in classifiers.items():
        cv_scores = cross_val_score(clf, X_train, yr_train, cv=cv, scoring="f1_macro")
        clf.fit(X_train, yr_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc  = accuracy_score(yr_test, y_pred)
        prec = precision_score(yr_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(yr_test, y_pred, average="macro", zero_division=0)
        f1   = f1_score(yr_test, y_pred, average="macro", zero_division=0)
        auc  = roc_auc_score(
            label_binarize(yr_test, classes=sorted(set(y_risk))),
            y_prob, multi_class="ovr", average="macro"
        )

        clf_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "cv_f1_mean": cv_scores.mean()}
        print(f"  {name:30s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  CV_F1={cv_scores.mean():.3f}")

    # Select best classifier by F1
    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1"])
    best_clf = classifiers[best_clf_name]
    best_clf.fit(X_train, yr_train)  # refit on full train
    print(f"\nBest classifier: {best_clf_name}")

    # Save confusion matrix and ROC
    y_pred_best = best_clf.predict(X_test)
    cm = confusion_matrix(yr_test, y_pred_best, labels=sorted(set(y_risk)))
    save_confusion_matrix(cm, sorted(set(y_risk)), os.path.join(PROCESSED_DIR, "confusion_matrix.png"))
    save_roc_curve(best_clf, X_test, yr_test, sorted(set(y_risk)), os.path.join(PROCESSED_DIR, "roc_curve.png"))

    # Feature importance (only tree-based models have this)
    if hasattr(best_clf, "feature_importances_"):
        save_feature_importance(best_clf, feature_names, os.path.join(PROCESSED_DIR, "feature_importance.png"))

    # Print full classification report
    print(f"\nClassification Report ({best_clf_name}):")
    print(classification_report(yr_test, y_pred_best, zero_division=0))

    # Save best classifier
    joblib.dump(best_clf, RISK_MODEL_PATH)
    print(f"Classifier saved -> {RISK_MODEL_PATH}")

    # ─────────────────────────────────────────────────────────────────────────
    # REGRESSION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Training regressors ===")
    regressors = {
        "Linear Regression":      LinearRegression(),
        "Gradient Boosting Reg":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    }

    reg_results = {}
    for name, reg in regressors.items():
        reg.fit(X_train, yg_train)
        y_pred_r = reg.predict(X_test)
        mae  = mean_absolute_error(yg_test, y_pred_r)
        rmse = np.sqrt(mean_squared_error(yg_test, y_pred_r))
        r2   = r2_score(yg_test, y_pred_r)
        reg_results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  {name:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_reg = regressors[best_reg_name]
    print(f"\nBest regressor: {best_reg_name}")

    joblib.dump(best_reg, GPA_MODEL_PATH)
    print(f"Regressor saved  -> {GPA_MODEL_PATH}")

    # ── Save model info JSON ──────────────────────────────────────────────────
    model_info = {
        "trained_at": datetime.now().isoformat(),
        "dataset_size": int(len(X)),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "classifier": {
            "algorithm": best_clf_name,
            "metrics": clf_results[best_clf_name],
            "all_models": clf_results,
        },
        "regressor": {
            "algorithm": best_reg_name,
            "metrics": reg_results[best_reg_name],
            "all_models": reg_results,
        },
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved -> {MODEL_INFO_PATH}")
    print("\n=== Training complete ===")


if __name__ == "__main__":
    train()
