"""tests/test_db.py — Tests for the SQLite database layer."""

import os
import pytest

import app.utils.db as db_module


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file so tests don't pollute the real DB."""
    temp_db_path = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db_path)
    db_module.init_db()
    yield temp_db_path


class TestDatabase:
    def test_init_creates_db(self, temp_db):
        assert os.path.exists(temp_db)

    def test_insert_and_retrieve(self):
        db_module.insert_prediction(
            risk_level="High",
            gpa_prediction=1.5,
            probabilities={"High": 0.7, "Medium": 0.2, "Low": 0.1},
            input_dict={"age": 20, "attendance_rate": 40.0},
            student_name="Test Student",
        )
        records = db_module.get_all_predictions()
        assert len(records) == 1
        assert records[0]["risk_level"] == "High"
        assert records[0]["gpa_prediction"] == 1.5
        assert records[0]["student_name"] == "Test Student"

    def test_multiple_inserts(self):
        for _ in range(5):
            db_module.insert_prediction(
                risk_level="Low",
                gpa_prediction=3.5,
                probabilities={"High": 0.05, "Medium": 0.1, "Low": 0.85},
                input_dict={"age": 19},
            )
        records = db_module.get_all_predictions()
        assert len(records) == 5

    def test_get_all_ordered_by_most_recent(self):
        for gpa in [1.0, 2.0, 3.0]:
            db_module.insert_prediction("Medium", gpa, {}, {})
        records = db_module.get_all_predictions()
        assert records[0]["gpa_prediction"] == 3.0
        assert records[-1]["gpa_prediction"] == 1.0

    def test_clear_predictions(self):
        db_module.insert_prediction("Low", 3.8, {}, {})
        db_module.clear_predictions()
        records = db_module.get_all_predictions()
        assert len(records) == 0

    def test_prediction_count_empty(self):
        assert db_module.prediction_count() == 0

    def test_prediction_count_after_insert(self):
        db_module.insert_prediction("Medium", 2.5, {}, {})
        db_module.insert_prediction("High", 1.2, {}, {})
        assert db_module.prediction_count() == 2

    def test_anonymous_default_name(self):
        db_module.insert_prediction("Low", 3.5, {}, {}, student_name=None)
        records = db_module.get_all_predictions()
        assert records[0]["student_name"] == "Anonymous"

    def test_probabilities_stored_correctly(self):
        probs = {"High": 0.6, "Medium": 0.3, "Low": 0.1}
        db_module.insert_prediction("High", 1.8, probs, {})
        records = db_module.get_all_predictions()
        assert abs(records[0]["prob_high"] - 0.6) < 1e-4
        assert abs(records[0]["prob_medium"] - 0.3) < 1e-4
        assert abs(records[0]["prob_low"] - 0.1) < 1e-4

    def test_limit_respected(self):
        for _ in range(10):
            db_module.insert_prediction("Low", 3.0, {}, {})
        records = db_module.get_all_predictions(limit=3)
        assert len(records) == 3
