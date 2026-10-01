"""
generate_dataset.py — Generates a synthetic student academic dataset.

Run:
    py generate_dataset.py

Outputs:
    data/raw/students.csv   (1 500 records, 19 columns)

Feature descriptions:
    student_id                 Unique integer identifier
    age                        Student age (17–25)
    gender                     M / F / Other
    major                      Declared major field
    year_of_study              Academic year (1–4)
    attendance_rate            % of classes attended (0–100)
    avg_assignment_score       Mean assignment score (0–100)
    midterm_score              Midterm exam score (0–100)
    study_hours_per_week       Self-reported study hours (0–40)
    extracurricular_activities Number of extracurricular activities (0–5)
    part_time_job              1 if student holds a part-time job
    financial_aid              1 if student receives financial aid
    parent_education_level     Highest parental education level
    previous_gpa               Prior semester GPA (NaN for first-year students)
    missed_deadlines           Number of missed assignment deadlines (0–10)
    library_visits_per_month   Library visits as engagement proxy (0–20)
    online_platform_usage_hrs  Weekly LMS / online learning hours (0–15)
    final_gpa                  TARGET: current semester GPA (0.0–4.0)
    risk_level                 TARGET: Low / Medium / High dropout risk
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N = 1500
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "students.csv")


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographic features ──────────────────────────────────────────────────
    student_id = np.arange(1001, 1001 + n)
    year_of_study = rng.choice([1, 2, 3, 4], size=n, p=[0.30, 0.28, 0.24, 0.18])
    age = year_of_study + 17 + rng.integers(0, 3, size=n)
    gender = rng.choice(["M", "F", "Other"], size=n, p=[0.48, 0.48, 0.04])
    major = rng.choice(
        ["Computer Science", "Business", "Engineering", "Arts", "Science"],
        size=n,
        p=[0.25, 0.22, 0.20, 0.15, 0.18],
    )
    parent_ed = rng.choice(
        ["None", "HighSchool", "Bachelor", "Graduate"],
        size=n,
        p=[0.10, 0.35, 0.38, 0.17],
    )
    financial_aid = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    part_time_job = rng.choice([0, 1], size=n, p=[0.62, 0.38])

    # ── Latent "student quality" factor (drives most academic outcomes) ───────
    # Higher parent education slightly boosts quality; part-time job slightly reduces it
    parent_boost = np.where(
        parent_ed == "Graduate", 0.3,
        np.where(parent_ed == "Bachelor", 0.15,
                 np.where(parent_ed == "HighSchool", 0.0, -0.2))
    )
    job_penalty = part_time_job * -0.15
    quality = rng.normal(0.5, 0.20, size=n) + parent_boost + job_penalty
    quality = np.clip(quality, 0.05, 0.98)

    # ── Academic behaviour features (correlated with quality) ─────────────────
    attendance_rate = np.clip(quality * 80 + rng.normal(10, 8, size=n), 0, 100)
    study_hours = np.clip(quality * 30 + rng.normal(2, 4, size=n), 0, 40)
    avg_assignment_score = np.clip(quality * 70 + rng.normal(15, 7, size=n), 0, 100)
    midterm_score = np.clip(quality * 65 + rng.normal(20, 8, size=n), 0, 100)
    missed_deadlines = np.clip(
        rng.poisson(lam=(1 - quality) * 5, size=n), 0, 10
    ).astype(int)
    extracurricular = np.clip(
        rng.poisson(lam=quality * 2.5, size=n), 0, 5
    ).astype(int)
    library_visits = np.clip(
        rng.poisson(lam=quality * 8, size=n), 0, 20
    ).astype(int)
    online_hrs = np.clip(quality * 10 + rng.normal(1, 2, size=n), 0, 15)

    # ── Previous GPA (NaN for first-year students) ────────────────────────────
    prev_gpa = quality * 3.5 + rng.normal(0.1, 0.2, size=n)
    prev_gpa = np.clip(prev_gpa, 0.0, 4.0)
    prev_gpa = np.where(year_of_study == 1, np.nan, prev_gpa)

    # ── TARGET: final GPA ─────────────────────────────────────────────────────
    # Weighted formula; previous GPA most predictive when available
    gpa_base = (
        0.25 * (midterm_score / 100 * 4)
        + 0.25 * (avg_assignment_score / 100 * 4)
        + 0.15 * (attendance_rate / 100 * 4)
        + 0.15 * (study_hours / 40 * 4)
        - 0.10 * (missed_deadlines / 10 * 2)
        + 0.10 * quality * 4
    )
    # Blend in previous GPA where available
    has_prev = ~np.isnan(prev_gpa)
    gpa_base[has_prev] = 0.60 * gpa_base[has_prev] + 0.40 * prev_gpa[has_prev]
    final_gpa = np.clip(gpa_base + rng.normal(0, 0.12, size=n), 0.0, 4.0).round(2)

    # ── TARGET: risk_level ────────────────────────────────────────────────────
    # Risk is driven by GPA, attendance, and missed deadlines
    risk_score = (
        (4.0 - final_gpa) / 4.0 * 50       # low GPA → high risk
        + (100 - attendance_rate) / 100 * 30  # low attendance → high risk
        + missed_deadlines / 10 * 20          # missed deadlines → high risk
    )
    risk_score = risk_score + rng.normal(0, 3, size=n)  # add noise
    risk_level = np.where(
        risk_score >= 45, "High",
        np.where(risk_score >= 25, "Medium", "Low"),
    )

    # ── Add 2% random missing values to select non-target columns ─────────────
    df = pd.DataFrame(
        {
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "major": major,
            "year_of_study": year_of_study,
            "attendance_rate": attendance_rate.round(1),
            "avg_assignment_score": avg_assignment_score.round(1),
            "midterm_score": midterm_score.round(1),
            "study_hours_per_week": study_hours.round(1),
            "extracurricular_activities": extracurricular,
            "part_time_job": part_time_job,
            "financial_aid": financial_aid,
            "parent_education_level": parent_ed,
            "previous_gpa": prev_gpa.round(2),
            "missed_deadlines": missed_deadlines,
            "library_visits_per_month": library_visits,
            "online_platform_usage_hrs": online_hrs.round(1),
            "final_gpa": final_gpa,
            "risk_level": risk_level,
        }
    )

    # Inject ~2% missing values in non-critical numeric columns
    noise_cols = [
        "attendance_rate", "study_hours_per_week",
        "library_visits_per_month", "online_platform_usage_hrs",
    ]
    for col in noise_cols:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_FILE, index=False)
    print(f"Dataset saved -> {OUT_FILE}")
    print(f"Shape         : {df.shape}")
    print(f"\nRisk level distribution:\n{df['risk_level'].value_counts()}")
    print(f"\nGPA stats:\n{df['final_gpa'].describe().round(3)}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    main()
