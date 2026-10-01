"""tests/test_predict.py — Tests for the predict module."""

import numpy as np
import pytest

from src.predict import predict_single, predict_batch
import pandas as pd


VALID_KEYS = {
    "risk_level", "probabilities", "gpa_prediction", "risk_color", "explanation"
}


class TestPredictSingle:
    def test_output_keys(self, sample_input):
        result = predict_single(sample_input)
        assert set(result.keys()) == VALID_KEYS

    def test_risk_level_valid(self, sample_input):
        result = predict_single(sample_input)
        assert result["risk_level"] in {"Low", "Medium", "High"}

    def test_gpa_in_range(self, sample_input):
        result = predict_single(sample_input)
        assert 0.0 <= result["gpa_prediction"] <= 4.0

    def test_probabilities_sum_to_one(self, sample_input):
        result = predict_single(sample_input)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities sum to {prob_sum}"

    def test_probabilities_non_negative(self, sample_input):
        result = predict_single(sample_input)
        for cls, p in result["probabilities"].items():
            assert p >= 0.0

    def test_risk_color_is_hex(self, sample_input):
        result = predict_single(sample_input)
        assert result["risk_color"].startswith("#")
        assert len(result["risk_color"]) == 7

    def test_explanation_is_string(self, sample_input):
        result = predict_single(sample_input)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    def test_high_risk_student(self, high_risk_input):
        """A student with very poor attendance and grades should be High risk."""
        result = predict_single(high_risk_input)
        assert result["risk_level"] == "High"

    def test_low_risk_student(self, low_risk_input):
        """A student with excellent performance should be Low risk."""
        result = predict_single(low_risk_input)
        assert result["risk_level"] == "Low"

    def test_missing_optional_fields(self):
        """Prediction should still work when optional fields are missing."""
        minimal = {
            "age": 19,
            "gender": "M",
            "major": "Science",
            "year_of_study": 1,
            "attendance_rate": 70.0,
            "midterm_score": 60.0,
            # previous_gpa intentionally omitted (NaN will be imputed)
        }
        result = predict_single(minimal)
        assert result["risk_level"] in {"Low", "Medium", "High"}
        assert 0.0 <= result["gpa_prediction"] <= 4.0

    def test_boundary_gpa_zero(self):
        """Edge case: all worst possible values."""
        worst = {
            "age": 17, "gender": "Other", "major": "Arts",
            "year_of_study": 1, "attendance_rate": 0.0,
            "avg_assignment_score": 0.0, "midterm_score": 0.0,
            "study_hours_per_week": 0.0, "extracurricular_activities": 0,
            "part_time_job": 1, "financial_aid": 0,
            "parent_education_level": "None", "previous_gpa": np.nan,
            "missed_deadlines": 10, "library_visits_per_month": 0,
            "online_platform_usage_hrs": 0.0,
        }
        result = predict_single(worst)
        assert result["gpa_prediction"] >= 0.0

    def test_boundary_perfect_student(self):
        """Edge case: all best possible values."""
        best = {
            "age": 20, "gender": "F", "major": "Engineering",
            "year_of_study": 4, "attendance_rate": 100.0,
            "avg_assignment_score": 100.0, "midterm_score": 100.0,
            "study_hours_per_week": 40.0, "extracurricular_activities": 5,
            "part_time_job": 0, "financial_aid": 1,
            "parent_education_level": "Graduate", "previous_gpa": 4.0,
            "missed_deadlines": 0, "library_visits_per_month": 20,
            "online_platform_usage_hrs": 15.0,
        }
        result = predict_single(best)
        assert result["gpa_prediction"] <= 4.0


class TestPredictBatch:
    def test_batch_returns_dataframe(self, sample_input):
        df = pd.DataFrame([sample_input] * 5)
        result = predict_batch(df)
        assert isinstance(result, pd.DataFrame)

    def test_batch_correct_row_count(self, sample_input):
        n = 10
        df = pd.DataFrame([sample_input] * n)
        result = predict_batch(df)
        assert len(result) == n

    def test_batch_has_prediction_columns(self, sample_input):
        df = pd.DataFrame([sample_input] * 3)
        result = predict_batch(df)
        assert "predicted_risk" in result.columns
        assert "predicted_gpa" in result.columns

    def test_batch_risk_levels_valid(self, sample_input):
        df = pd.DataFrame([sample_input] * 5)
        result = predict_batch(df)
        assert result["predicted_risk"].isin({"Low", "Medium", "High"}).all()

    def test_batch_gpa_in_range(self, sample_input):
        df = pd.DataFrame([sample_input] * 5)
        result = predict_batch(df)
        assert result["predicted_gpa"].between(0.0, 4.0).all()
