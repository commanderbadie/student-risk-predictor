# Architecture — Student Academic Risk Predictor

## Overview

The system is composed of five independent layers that communicate through well-defined file artifacts and Python module interfaces.

```
+---------------------------+
|   Streamlit Frontend      |  app/app.py
|   6 pages, sidebar nav    |
+---------------------------+
           |
           | calls
           v
+---------------------------+
|   Inference API           |  src/predict.py
|   predict_single()        |
|   predict_batch()         |
+---------------------------+
           |
     loads at startup
           |
    +------+------+
    |             |
    v             v
+--------+   +----------+
|Preprocessor  Classifier|   models/preprocessor.pkl
|.pkl    |   |.pkl       |   models/risk_classifier.pkl
+--------+   +----------+   models/gpa_regressor.pkl
                 |
    (trained offline by)
                 |
+---------------------------+
|   Training Pipeline       |  src/train.py
|   src/preprocessing.py    |
|   generate_dataset.py     |
+---------------------------+
           |
           | reads/writes
           v
+---------------------------+
|   Data Layer              |
|   data/raw/students.csv   |
|   data/processed/         |
|   models/model_info.json  |
+---------------------------+

+---------------------------+
|   Persistence Layer       |  app/utils/db.py
|   data/predictions.db     |  SQLite
+---------------------------+
```

---

## Component Descriptions

### `generate_dataset.py`
Creates a 1,500-row synthetic student dataset using NumPy random distributions. A latent quality factor drives all correlated features. Outputs `data/raw/students.csv`. Fixed seed ensures reproducibility.

### `src/preprocessing.py` — `Preprocessor` class
Handles the full feature transformation pipeline:
- Median imputation for 13 numeric features (handles NaN from missing/year-1 students)
- Standard scaling (zero mean, unit variance)
- One-hot encoding of 3 categorical features (12 binary columns)
- Fit state serialised to `models/preprocessor.pkl` via joblib

The same fitted instance is used at both training time and inference time, guaranteeing identical transformations.

### `src/train.py`
Trains three classifiers and two regressors, cross-validates on the training split, evaluates on the held-out 20% test set, selects the best model by macro F1 / R², and saves:
- `models/risk_classifier.pkl`
- `models/gpa_regressor.pkl`
- `models/model_info.json` (all metrics, algorithm names, training date)
- PNG charts in `data/processed/`

### `src/predict.py`
Thin inference wrapper loaded by the Streamlit app. Exposes:
- `predict_single(input_dict) -> dict` — single student
- `predict_batch(df) -> DataFrame` — vectorised batch

Models are loaded once and cached in module-level globals.

### `app/app.py`
Multi-page Streamlit application using sidebar radio navigation. Each page is an isolated function. The app:
1. Calls `init_db()` on startup
2. Guards against missing models with a clear error message
3. Delegates all chart rendering to `app/components/charts.py`
4. Delegates all data loading to `app/utils/loader.py` (LRU-cached)
5. Delegates DB operations to `app/utils/db.py`

### `app/utils/db.py`
Auto-creates `data/predictions.db` on import. Exposes:
- `insert_prediction(...)` — called after every single prediction
- `get_all_predictions(limit)` — used by the History page
- `clear_predictions()` — wipe history button
- `prediction_count()` — displayed on the Home page

### `app/components/charts.py`
Factory functions that return `plotly.graph_objects.Figure` objects with a consistent dark navy/teal theme. The Streamlit app calls these and passes the result to `st.plotly_chart()`.

### `run.py`
Entry-point script that:
1. Checks whether trained model artifacts exist
2. If not, automatically runs `generate_dataset.py` then `src/train.py`
3. Launches the Streamlit app via subprocess

---

## Data Flow — Single Prediction

```
User fills form (app/app.py)
        |
        v
input_dict (Python dict of raw feature values)
        |
        v
predict_single(input_dict)     [src/predict.py]
        |
        +-- pd.DataFrame([input_dict])
        |
        +-- preprocessor.transform(df)   [models/preprocessor.pkl]
        |       impute -> scale -> OHE -> np.ndarray shape (1, 25)
        |
        +-- classifier.predict(X)        [models/risk_classifier.pkl]
        |       -> "Low" / "Medium" / "High"
        |
        +-- classifier.predict_proba(X)  -> probability dict
        |
        +-- regressor.predict(X)         [models/gpa_regressor.pkl]
        |       -> float (clipped 0.0-4.0)
        |
        +-- _build_explanation(...)      -> str
        |
        v
result dict {risk_level, probabilities, gpa_prediction, risk_color, explanation}
        |
        v
Streamlit renders badge, gauge chart, feature importance, explanation box
        |
        v
insert_prediction(...)                  [app/utils/db.py]
        -> data/predictions.db
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Synthetic data only | No external download dependency; fully reproducible |
| Shared `Preprocessor` artifact | Eliminates train-serve skew |
| Two ML tasks (classifier + regressor) | More impressive demo; one dataset, two insights |
| Gradient Boosting as final model | Best F1; provides `feature_importances_` for explainability |
| SQLite over CSV for history | Enables filtering, counting, and persistence across sessions |
| Streamlit sidebar navigation | Single-file app; no multi-page file structure complexity |
| LRU cache in `loader.py` | Models are loaded exactly once per Streamlit session |
| matplotlib `Agg` backend in train.py | Allows chart generation without a display server |
