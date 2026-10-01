"""tests/test_preprocessing.py — Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from generate_dataset import generate
from src.preprocessing import Preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def sample_df():
    return generate(n=200, seed=42)


@pytest.fixture(scope="module")
def fitted_preprocessor(sample_df):
    pre = Preprocessor()
    pre.fit_transform(sample_df)
    return pre


class TestPreprocessor:
    def test_fit_transform_shape(self, sample_df):
        pre = Preprocessor()
        X = pre.fit_transform(sample_df)
        assert X.ndim == 2
        assert X.shape[0] == len(sample_df)
        # 13 numeric + 3+5+4 OHE = 25 features total
        assert X.shape[1] == 25

    def test_no_nans_after_transform(self, sample_df):
        pre = Preprocessor()
        X = pre.fit_transform(sample_df)
        assert not np.isnan(X).any(), "Transformed array contains NaN values"

    def test_feature_names_stored(self, fitted_preprocessor):
        assert len(fitted_preprocessor.feature_names_) == 25

    def test_transform_single_row(self, fitted_preprocessor):
        """transform() must handle a single-row DataFrame."""
        row = {
            "age": 19, "gender": "F", "major": "Business", "year_of_study": 1,
            "attendance_rate": 80.0, "avg_assignment_score": 75.0,
            "midterm_score": 70.0, "study_hours_per_week": 15.0,
            "extracurricular_activities": 1, "part_time_job": 0, "financial_aid": 1,
            "parent_education_level": "HighSchool", "previous_gpa": np.nan,
            "missed_deadlines": 1, "library_visits_per_month": 3,
            "online_platform_usage_hrs": 2.0,
        }
        df_row = pd.DataFrame([row])
        X = fitted_preprocessor.transform(df_row)
        assert X.shape == (1, 25)
        assert not np.isnan(X).any()

    def test_transform_before_fit_raises(self):
        pre = Preprocessor()
        with pytest.raises(RuntimeError):
            pre.transform(pd.DataFrame([{"age": 19}]))

    def test_consistency_fit_vs_transform(self, sample_df):
        """fit_transform and transform should produce same result on same data."""
        pre = Preprocessor()
        X1 = pre.fit_transform(sample_df)
        X2 = pre.transform(sample_df)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_missing_column_handled(self, fitted_preprocessor):
        """Missing columns in input should be filled with NaN and imputed."""
        df_partial = pd.DataFrame([{"age": 20, "gender": "M"}])
        X = fitted_preprocessor.transform(df_partial)
        assert X.shape[1] == 25
        assert not np.isnan(X).any()

    def test_unknown_categorical_falls_back(self, fitted_preprocessor):
        """Unknown categorical value should fall back to first known category."""
        df_row = pd.DataFrame([{
            "age": 20, "gender": "Unknown_Gender",
            "major": "Philosophy",  # unknown
            "parent_education_level": "PhD",  # unknown
        }])
        X = fitted_preprocessor.transform(df_row)
        assert X.shape[1] == 25
        assert not np.isnan(X).any()
