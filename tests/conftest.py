"""tests/conftest.py — shared pytest fixtures."""

import os
import sys
import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def sample_input():
    """A valid, complete student input dictionary."""
    return {
        "age": 20,
        "gender": "M",
        "major": "Computer Science",
        "year_of_study": 2,
        "attendance_rate": 75.0,
        "avg_assignment_score": 70.0,
        "midterm_score": 65.0,
        "study_hours_per_week": 12.0,
        "extracurricular_activities": 2,
        "part_time_job": 0,
        "financial_aid": 1,
        "parent_education_level": "Bachelor",
        "previous_gpa": 2.8,
        "missed_deadlines": 2,
        "library_visits_per_month": 4,
        "online_platform_usage_hrs": 3.0,
    }


@pytest.fixture(scope="session")
def high_risk_input():
    """An input profile that should produce a High risk prediction."""
    return {
        "age": 21,
        "gender": "F",
        "major": "Arts",
        "year_of_study": 3,
        "attendance_rate": 30.0,
        "avg_assignment_score": 30.0,
        "midterm_score": 25.0,
        "study_hours_per_week": 2.0,
        "extracurricular_activities": 0,
        "part_time_job": 1,
        "financial_aid": 0,
        "parent_education_level": "None",
        "previous_gpa": 1.0,
        "missed_deadlines": 8,
        "library_visits_per_month": 0,
        "online_platform_usage_hrs": 0.5,
    }


@pytest.fixture(scope="session")
def low_risk_input():
    """An input profile that should produce a Low risk prediction."""
    return {
        "age": 20,
        "gender": "F",
        "major": "Engineering",
        "year_of_study": 3,
        "attendance_rate": 95.0,
        "avg_assignment_score": 90.0,
        "midterm_score": 92.0,
        "study_hours_per_week": 30.0,
        "extracurricular_activities": 2,
        "part_time_job": 0,
        "financial_aid": 1,
        "parent_education_level": "Graduate",
        "previous_gpa": 3.8,
        "missed_deadlines": 0,
        "library_visits_per_month": 12,
        "online_platform_usage_hrs": 8.0,
    }
