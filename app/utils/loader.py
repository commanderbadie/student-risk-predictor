"""
app/utils/loader.py — Lazy-load models, preprocessor, dataset, and model info.

All objects are cached after first load so repeated calls are free.
"""

import os
import sys
import json
import functools

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
DATA_RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")


@functools.lru_cache(maxsize=1)
def load_classifier():
    path = os.path.join(MODELS_DIR, "risk_classifier.pkl")
    return joblib.load(path)


@functools.lru_cache(maxsize=1)
def load_regressor():
    path = os.path.join(MODELS_DIR, "gpa_regressor.pkl")
    return joblib.load(path)


@functools.lru_cache(maxsize=1)
def load_preprocessor():
    from src.preprocessing import Preprocessor
    path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    return Preprocessor.load(path)


@functools.lru_cache(maxsize=1)
def load_model_info() -> dict:
    path = os.path.join(MODELS_DIR, "model_info.json")
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_raw_dataset() -> pd.DataFrame:
    path = os.path.join(DATA_RAW_DIR, "students.csv")
    return pd.read_csv(path)


def models_exist() -> bool:
    return (
        os.path.exists(os.path.join(MODELS_DIR, "risk_classifier.pkl"))
        and os.path.exists(os.path.join(MODELS_DIR, "gpa_regressor.pkl"))
        and os.path.exists(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        and os.path.exists(os.path.join(MODELS_DIR, "model_info.json"))
    )


def dataset_exists() -> bool:
    return os.path.exists(os.path.join(DATA_RAW_DIR, "students.csv"))
