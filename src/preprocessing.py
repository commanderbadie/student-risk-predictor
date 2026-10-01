"""
src/preprocessing.py — Data cleaning and feature engineering pipeline.

The Preprocessor class can be:
  - fit_transform(df) → np.ndarray  (used during training)
  - transform(df)     → np.ndarray  (used at prediction time)

The fitted preprocessor is saved to models/preprocessor.pkl so that
train-time and inference-time transformations are identical.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")

# Columns that are targets — never used as features
TARGET_COLS = ["final_gpa", "risk_level"]
ID_COL = "student_id"

# Ordered list of numeric and categorical feature names after dropping targets/id
NUMERIC_FEATURES = [
    "age",
    "year_of_study",
    "attendance_rate",
    "avg_assignment_score",
    "midterm_score",
    "study_hours_per_week",
    "extracurricular_activities",
    "part_time_job",
    "financial_aid",
    "previous_gpa",
    "missed_deadlines",
    "library_visits_per_month",
    "online_platform_usage_hrs",
]

CATEGORICAL_FEATURES = [
    "gender",
    "major",
    "parent_education_level",
]

# All expected categories (for consistent one-hot encoding)
CATEGORY_VALUES = {
    "gender": ["F", "M", "Other"],
    "major": ["Arts", "Business", "Computer Science", "Engineering", "Science"],
    "parent_education_level": ["Bachelor", "Graduate", "HighSchool", "None"],
}


class Preprocessor:
    """
    Handles imputation, encoding, and scaling for student feature data.

    After calling fit_transform() the instance stores:
      - numeric_imputer  : SimpleImputer (median strategy)
      - scaler           : StandardScaler
      - feature_names_   : list of final feature column names
    """

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features using known category lists."""
        encoded_parts = []
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                # Fill with mode placeholder
                series = pd.Series(["HighSchool"] * len(df), name=col)
            else:
                series = df[col].fillna(
                    CATEGORY_VALUES[col][0]
                )  # fill with first category
            dummies = pd.get_dummies(series, prefix=col).reindex(
                columns=[f"{col}_{v}" for v in CATEGORY_VALUES[col]], fill_value=0
            )
            encoded_parts.append(dummies)
        return pd.concat(encoded_parts, axis=1)

    def _select_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the numeric feature columns, adding NaN for any missing."""
        result = pd.DataFrame(index=df.index)
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                result[col] = np.nan
        return result

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor on training data and return transformed array.
        Also stores self.feature_names_.
        """
        num_df = self._select_numerics(df)
        cat_df = self._encode_categoricals(df)

        # Fit + transform numerics
        num_imputed = self.numeric_imputer.fit_transform(num_df)
        num_scaled = self.scaler.fit_transform(num_imputed)

        # Concatenate
        cat_arr = cat_df.values.astype(float)
        X = np.concatenate([num_scaled, cat_arr], axis=1)

        # Store feature names for later use (e.g., feature importance charts)
        self.feature_names_ = list(num_df.columns) + list(cat_df.columns)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the already-fitted preprocessor.
        Accepts a DataFrame with any subset of columns; missing columns → NaN.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform first.")

        num_df = self._select_numerics(df)
        cat_df = self._encode_categoricals(df)

        num_imputed = self.numeric_imputer.transform(num_df)
        num_scaled = self.scaler.transform(num_imputed)

        cat_arr = cat_df.values.astype(float)
        return np.concatenate([num_scaled, cat_arr], axis=1)

    # ------------------------------------------------------------------
    def save(self, path: str = PREPROCESSOR_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = PREPROCESSOR_PATH) -> "Preprocessor":
        return joblib.load(path)


# ── Standalone helper ─────────────────────────────────────────────────────────

def load_raw(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "students.csv")
    return pd.read_csv(csv_path)


def preprocess_and_save(csv_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, "Preprocessor"]:
    """
    Full pipeline: load raw CSV → preprocess → save processed CSV + preprocessor.
    Returns (X, y_risk, y_gpa, preprocessor).
    """
    df = load_raw(csv_path)

    # Extract targets before transforming
    y_risk = df["risk_level"].to_numpy()
    y_gpa = df["final_gpa"].to_numpy(dtype=float)

    # Transform features
    preprocessor = Preprocessor()
    X = preprocessor.fit_transform(df)

    # Save preprocessor
    preprocessor.save()

    # Save processed feature matrix as CSV for reference
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_df = pd.DataFrame(X, columns=preprocessor.feature_names_)
    processed_df["risk_level"] = y_risk
    processed_df["final_gpa"] = y_gpa
    processed_df.to_csv(os.path.join(processed_dir, "students_processed.csv"), index=False)

    print(f"Preprocessed shape: {X.shape}")
    print(f"Features          : {len(preprocessor.feature_names_)}")
    print(f"Risk label counts : {pd.Series(y_risk).value_counts().to_dict()}")
    print(f"Preprocessor saved -> {PREPROCESSOR_PATH}")

    return X, y_risk, y_gpa, preprocessor


if __name__ == "__main__":
    preprocess_and_save()
