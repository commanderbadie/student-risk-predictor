"""
src/predict.py — Inference module for the Student Risk Predictor.

Main entry point for the Streamlit app to call when making predictions.

Usage:
    from src.predict import predict_single, predict_batch

predict_single(input_dict) → dict with keys:
    risk_level      : "Low" / "Medium" / "High"
    probabilities   : {"Low": float, "Medium": float, "High": float}
    gpa_prediction  : float
    risk_color      : hex color string for UI rendering
    explanation     : plain-English explanation string

predict_batch(df) → pd.DataFrame with prediction columns appended
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import Preprocessor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RISK_MODEL_PATH  = os.path.join(MODELS_DIR, "risk_classifier.pkl")
GPA_MODEL_PATH   = os.path.join(MODELS_DIR, "gpa_regressor.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

RISK_COLORS = {
    "Low":    "#4CAF50",  # green
    "Medium": "#FF9800",  # orange
    "High":   "#F44336",  # red
}

_clf = None
_reg = None
_pre = None


def _load_models():
    global _clf, _reg, _pre
    if _clf is None:
        _clf = joblib.load(RISK_MODEL_PATH)
    if _reg is None:
        _reg = joblib.load(GPA_MODEL_PATH)
    if _pre is None:
        _pre = Preprocessor.load(PREPROCESSOR_PATH)


def _build_explanation(risk_level: str, gpa: float, row: dict) -> str:
    """Generate a plain-English explanation of the prediction."""
    lines = []
    risk_desc = {
        "Low":    "This student shows strong academic engagement and is on track.",
        "Medium": "This student shows some warning signs and may benefit from support.",
        "High":   "This student is at significant risk of academic difficulty or dropout.",
    }
    lines.append(risk_desc[risk_level])

    att = row.get("attendance_rate", None)
    if att is not None and att < 60:
        lines.append(f"Attendance is low ({att:.0f}%) — a key dropout indicator.")
    elif att is not None and att >= 80:
        lines.append(f"Good attendance ({att:.0f}%) is a positive factor.")

    md = row.get("missed_deadlines", None)
    if md is not None and md >= 4:
        lines.append(f"Missing {md} deadlines suggests disengagement.")

    sh = row.get("study_hours_per_week", None)
    if sh is not None and sh < 8:
        lines.append(f"Study time is low ({sh:.0f} hrs/week); increasing it can improve outcomes.")
    elif sh is not None and sh >= 20:
        lines.append(f"Strong study habits ({sh:.0f} hrs/week) support academic success.")

    ms = row.get("midterm_score", None)
    if ms is not None and ms < 50:
        lines.append(f"Midterm score of {ms:.0f}/100 indicates academic difficulty.")

    lines.append(f"Predicted GPA this semester: {gpa:.2f}/4.00.")
    return " ".join(lines)


def predict_single(input_dict: dict) -> dict:
    """
    Predict risk level and GPA for a single student.

    Parameters
    ----------
    input_dict : dict
        Keys match feature column names (see NUMERIC_FEATURES + CATEGORICAL_FEATURES
        in preprocessing.py). Missing keys are handled by imputation.

    Returns
    -------
    dict
        risk_level, probabilities, gpa_prediction, risk_color, explanation
    """
    _load_models()
    df = pd.DataFrame([input_dict])
    X = _pre.transform(df)

    risk_level = _clf.predict(X)[0]
    proba_arr = _clf.predict_proba(X)[0]
    classes = _clf.classes_
    probabilities = {cls: float(p) for cls, p in zip(classes, proba_arr)}

    gpa = float(np.clip(_reg.predict(X)[0], 0.0, 4.0))
    explanation = _build_explanation(risk_level, gpa, input_dict)

    return {
        "risk_level":     risk_level,
        "probabilities":  probabilities,
        "gpa_prediction": round(gpa, 2),
        "risk_color":     RISK_COLORS[risk_level],
        "explanation":    explanation,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict risk level and GPA for a batch DataFrame.
    Returns the original DataFrame with prediction columns appended.
    """
    _load_models()
    X = _pre.transform(df)

    risk_levels = _clf.predict(X)
    proba_arr   = _clf.predict_proba(X)
    classes     = _clf.classes_
    gpas        = np.clip(_reg.predict(X), 0.0, 4.0)

    result = df.copy()
    result["predicted_risk"] = risk_levels
    result["predicted_gpa"]  = gpas.round(2)
    for cls in classes:
        idx = list(classes).index(cls)
        result[f"prob_{cls}"] = proba_arr[:, idx].round(3)
    result["risk_color"] = [RISK_COLORS[r] for r in risk_levels]
    return result
