"""tests/test_data_generation.py — Tests for dataset generation."""

import os
import numpy as np
import pandas as pd
import pytest

from generate_dataset import generate


class TestDataGeneration:
    def test_shape(self):
        """Dataset should have exactly N rows and 19 columns."""
        df = generate(n=100)
        assert df.shape[0] == 100
        assert df.shape[1] == 19

    def test_columns_present(self):
        expected = [
            "student_id", "age", "gender", "major", "year_of_study",
            "attendance_rate", "avg_assignment_score", "midterm_score",
            "study_hours_per_week", "extracurricular_activities",
            "part_time_job", "financial_aid", "parent_education_level",
            "previous_gpa", "missed_deadlines", "library_visits_per_month",
            "online_platform_usage_hrs", "final_gpa", "risk_level",
        ]
        df = generate(n=50)
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_reproducibility(self):
        """Same seed should produce identical DataFrames."""
        df1 = generate(n=200, seed=99)
        df2 = generate(n=200, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate(n=100, seed=1)
        df2 = generate(n=100, seed=2)
        assert not df1["final_gpa"].equals(df2["final_gpa"])

    def test_target_no_nulls(self):
        """Target columns must never be null."""
        df = generate(n=500)
        assert df["risk_level"].isnull().sum() == 0
        assert df["final_gpa"].isnull().sum() == 0

    def test_risk_levels_valid(self):
        df = generate(n=500)
        assert set(df["risk_level"].unique()).issubset({"Low", "Medium", "High"})

    def test_all_risk_levels_present(self):
        """With 500 samples, all three risk levels should appear."""
        df = generate(n=500)
        assert "Low" in df["risk_level"].values
        assert "Medium" in df["risk_level"].values
        assert "High" in df["risk_level"].values

    def test_gpa_range(self):
        df = generate(n=500)
        assert df["final_gpa"].between(0.0, 4.0).all()

    def test_attendance_range(self):
        df = generate(n=500)
        valid = df["attendance_rate"].dropna()
        assert valid.between(0.0, 100.0).all()

    def test_student_ids_unique(self):
        df = generate(n=500)
        assert df["student_id"].is_unique

    def test_previous_gpa_nan_for_year1(self):
        """Year 1 students should have NaN previous GPA."""
        df = generate(n=1000)
        year1 = df[df["year_of_study"] == 1]
        assert year1["previous_gpa"].isnull().all()

    def test_previous_gpa_present_for_year234(self):
        df = generate(n=1000)
        upper_years = df[df["year_of_study"] > 1]
        assert upper_years["previous_gpa"].isnull().sum() == 0
