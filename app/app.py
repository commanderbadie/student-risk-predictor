"""
app/app.py — Student Academic Risk Predictor — Streamlit Application

Five pages (sidebar navigation):
  1. Home
  2. Single Student Prediction
  3. Batch Prediction
  4. EDA Dashboard
  5. Model Performance
  6. Prediction History

Run:
    streamlit run app/app.py
    -- or --
    py run.py
"""

import os
import sys
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

# ── Resolve project root ───────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from app.utils.db      import init_db, insert_prediction, get_all_predictions, clear_predictions, prediction_count
from app.utils.loader  import load_classifier, load_regressor, load_preprocessor, load_model_info, load_raw_dataset, models_exist, dataset_exists
from app.components.charts import (
    probability_gauge, feature_importance_chart, confusion_matrix_chart,
    roc_curve_chart, distribution_chart, correlation_heatmap,
    risk_breakdown_chart, risk_donut, actual_vs_predicted,
)
from src.predict import predict_single, predict_batch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (extra polish on top of config.toml theme) ─────────────────────
st.markdown("""
<style>
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .metric-card {
        background: #1A2E45;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #263A52;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00BCD4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90A4AE;
    }
    .explanation-box {
        background: #1A2E45;
        border-left: 4px solid #00BCD4;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 10px;
        color: #E0E0E0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .section-header {
        border-bottom: 1px solid #263A52;
        padding-bottom: 4px;
        margin-bottom: 14px;
        color: #00BCD4;
    }
    div[data-testid="stSidebar"] {
        background: #0A1628 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialise DB on startup ───────────────────────────────────────────────────
init_db()

# ── Auto-train on first deploy if models are missing ──────────────────────────
if not models_exist():
    with st.spinner("First-time setup: generating dataset and training models... (this takes ~30 seconds)"):
        import subprocess
        # Generate dataset
        subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "generate_dataset.py")],
            cwd=PROJECT_ROOT, check=True
        )
        # Train models
        subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "train.py")],
            cwd=PROJECT_ROOT, check=True
        )
    st.success("Setup complete! Loading app...")
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 Student Risk Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=[
            "🏠  Home",
            "🔍  Single Prediction",
            "📂  Batch Prediction",
            "📊  EDA Dashboard",
            "🤖  Model Performance",
            "📋  Prediction History",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<span style='color:#90A4AE; font-size:0.8rem;'>Built with Streamlit · scikit-learn</span>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("🎓 Student Academic Risk Predictor")
    st.markdown(
        "An AI-powered system that predicts a student's **dropout/failure risk** "
        "and **estimated GPA** using academic behaviour and demographic features."
    )
    st.markdown("---")

    # Key metrics row
    info = load_model_info()
    clf_m = info["classifier"]["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['dataset_size']:,}</div>
            <div class="metric-label">Training Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{clf_m['accuracy']:.0%}</div>
            <div class="metric-label">Classifier Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{clf_m['roc_auc']:.2f}</div>
            <div class="metric-label">ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        n_pred = prediction_count()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_pred}</div>
            <div class="metric-label">Predictions Made</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**1. Input Student Data**\nEnter attendance, grades, study habits, and demographic information.")
    with col_b:
        st.info("**2. AI Analysis**\nA Gradient Boosting model trained on 1,500+ student records analyses the profile.")
    with col_c:
        st.info("**3. Risk Assessment**\nReceive a Low / Medium / High risk classification, predicted GPA, and an explanation.")

    st.markdown("### Quick navigation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🔍 **Single Prediction** — Assess one student profile")
    with c2:
        st.markdown("📂 **Batch Prediction** — Upload a CSV of multiple students")
    with c3:
        st.markdown("📊 **EDA Dashboard** — Explore the training dataset")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def page_single_prediction():
    st.title("🔍 Single Student Prediction")
    st.markdown("Fill in the student's profile and click **Predict** to assess their academic risk.")

    with st.form("prediction_form"):
        st.markdown('<p class="section-header">Student Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            student_name = st.text_input("Student Name (optional)", value="", placeholder="e.g. John Doe")
            age          = st.slider("Age", 17, 25, 19)
            gender       = st.selectbox("Gender", ["M", "F", "Other"])
        with c2:
            major        = st.selectbox("Major", ["Computer Science", "Business", "Engineering", "Arts", "Science"])
            year_of_study = st.selectbox("Year of Study", [1, 2, 3, 4])
            parent_ed    = st.selectbox("Parent Education Level", ["None", "HighSchool", "Bachelor", "Graduate"])
        with c3:
            financial_aid = st.checkbox("Receiving Financial Aid", value=False)
            part_time_job = st.checkbox("Has Part-Time Job", value=False)

        st.markdown("---")
        st.markdown('<p class="section-header">Academic Performance</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            attendance    = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, step=0.5)
            midterm       = st.slider("Midterm Score (0–100)", 0.0, 100.0, 65.0, step=0.5)
            assignment    = st.slider("Avg Assignment Score (0–100)", 0.0, 100.0, 70.0, step=0.5)
        with c5:
            study_hrs     = st.slider("Study Hours / Week", 0.0, 40.0, 12.0, step=0.5)
            missed        = st.slider("Missed Deadlines", 0, 10, 2)
            extracurr     = st.slider("Extracurricular Activities", 0, 5, 1)
        with c6:
            prev_gpa_known = st.checkbox("Previous GPA known?", value=(year_of_study > 1))
            prev_gpa      = st.slider("Previous GPA (0.0–4.0)", 0.0, 4.0, 2.5, step=0.05,
                                      disabled=not prev_gpa_known)
            library       = st.slider("Library Visits / Month", 0, 20, 4)
            online_hrs    = st.slider("Online Platform (hrs/week)", 0.0, 15.0, 3.0, step=0.5)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if submitted:
        # Build input dict
        input_dict = {
            "age":                       age,
            "gender":                    gender,
            "major":                     major,
            "year_of_study":             year_of_study,
            "attendance_rate":           attendance,
            "avg_assignment_score":      assignment,
            "midterm_score":             midterm,
            "study_hours_per_week":      study_hrs,
            "extracurricular_activities": extracurr,
            "part_time_job":             int(part_time_job),
            "financial_aid":             int(financial_aid),
            "parent_education_level":    parent_ed,
            "previous_gpa":              prev_gpa if prev_gpa_known else np.nan,
            "missed_deadlines":          missed,
            "library_visits_per_month":  library,
            "online_platform_usage_hrs": online_hrs,
        }

        with st.spinner("Analysing student profile..."):
            result = predict_single(input_dict)

        # Display results
        st.markdown("---")
        st.markdown("### Prediction Result")
        col_risk, col_gpa, col_blank = st.columns([1, 1, 2])

        risk   = result["risk_level"]
        color  = result["risk_color"]
        gpa    = result["gpa_prediction"]

        with col_risk:
            st.markdown(
                f'<div class="metric-card">'
                f'<span class="risk-badge" style="background:{color}; color:#fff;">{risk.upper()} RISK</span>'
                f'<div class="metric-label" style="margin-top:8px;">Risk Level</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_gpa:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{gpa:.2f}</div>'
                f'<div class="metric-label">Predicted GPA (out of 4.00)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(probability_gauge(result["probabilities"]), use_container_width=True)
        with col_chart2:
            clf = load_classifier()
            pre = load_preprocessor()
            if hasattr(clf, "feature_importances_"):
                st.plotly_chart(
                    feature_importance_chart(pre.feature_names_, clf.feature_importances_.tolist(), top_n=10),
                    use_container_width=True,
                )

        st.markdown(
            f'<div class="explanation-box">💡 <strong>Explanation:</strong><br>{result["explanation"]}</div>',
            unsafe_allow_html=True,
        )

        # Save to DB
        insert_prediction(
            risk_level=risk,
            gpa_prediction=gpa,
            probabilities=result["probabilities"],
            input_dict=input_dict,
            student_name=student_name.strip() or None,
        )
        st.success("Prediction saved to history.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def page_batch_prediction():
    st.title("📂 Batch Student Prediction")
    st.markdown(
        "Upload a CSV file containing multiple student records. "
        "The system will predict risk level and GPA for every row."
    )

    # Template download
    template_cols = [
        "age", "gender", "major", "year_of_study", "attendance_rate",
        "avg_assignment_score", "midterm_score", "study_hours_per_week",
        "extracurricular_activities", "part_time_job", "financial_aid",
        "parent_education_level", "previous_gpa", "missed_deadlines",
        "library_visits_per_month", "online_platform_usage_hrs",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV Template",
        data=csv_template,
        file_name="student_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to begin batch prediction.")
        return

    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if df_in.empty:
        st.warning("The uploaded file is empty.")
        return

    st.markdown(f"**{len(df_in)} records loaded.** Preview:")
    st.dataframe(df_in.head(5), use_container_width=True)

    if st.button("🔮 Run Batch Prediction", use_container_width=True):
        with st.spinner(f"Predicting {len(df_in)} students..."):
            try:
                df_out = predict_batch(df_in)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        st.success(f"Done! {len(df_out)} predictions completed.")

        # Risk distribution donut
        risk_counts = df_out["predicted_risk"].value_counts().to_dict()
        col_d, col_t = st.columns([1, 2])
        with col_d:
            st.plotly_chart(risk_donut(risk_counts), use_container_width=True)
        with col_t:
            # Color-coded table
            display_cols = [c for c in df_out.columns if not c.startswith("prob_") and c != "risk_color"]
            st.dataframe(df_out[display_cols], use_container_width=True)

        # Download results
        csv_out = df_out.drop(columns=["risk_color"]).to_csv(index=False)
        st.download_button(
            "📥 Download Predictions CSV",
            data=csv_out,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def page_eda():
    st.title("📊 EDA Dashboard")
    st.markdown("Explore the training dataset to understand student distributions and feature relationships.")

    if not dataset_exists():
        st.error("Dataset not found. Run `py generate_dataset.py` first.")
        return

    df = load_raw_dataset()
    n_students = len(df)
    n_high = (df["risk_level"] == "High").sum()

    # Summary stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Students", n_students)
    s2.metric("High Risk", int(n_high), delta=f"{n_high/n_students:.0%}")
    s3.metric("Avg GPA", f"{df['final_gpa'].mean():.2f}")
    s4.metric("Avg Attendance", f"{df['attendance_rate'].mean():.1f}%")

    st.markdown("---")

    # Distribution section
    st.subheader("Feature Distributions")
    numeric_cols = [
        "attendance_rate", "midterm_score", "avg_assignment_score",
        "study_hours_per_week", "final_gpa", "missed_deadlines",
        "library_visits_per_month", "online_platform_usage_hrs",
    ]
    col_sel = st.selectbox("Select feature", numeric_cols)
    color_opt = st.checkbox("Colour by Risk Level", value=True)
    st.plotly_chart(
        distribution_chart(df, col_sel, "risk_level" if color_opt else None),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Risk Level Breakdowns")
    tab1, tab2, tab3 = st.tabs(["By Major", "By Year of Study", "By Gender"])
    with tab1:
        st.plotly_chart(risk_breakdown_chart(df, "major"), use_container_width=True)
    with tab2:
        st.plotly_chart(risk_breakdown_chart(df, "year_of_study"), use_container_width=True)
    with tab3:
        st.plotly_chart(risk_breakdown_chart(df, "gender"), use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    drop_cols = ["student_id", "risk_level", "gender", "major", "parent_education_level"]
    corr_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    st.plotly_chart(correlation_heatmap(corr_df), use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Dataset Sample")
    st.dataframe(df.sample(20, random_state=1).reset_index(drop=True), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.title("🤖 Model Performance")
    st.markdown("Detailed evaluation of all trained models on the held-out test set.")

    info = load_model_info()
    clf_info = info["classifier"]
    reg_info = info["regressor"]

    # ── Classifier section ────────────────────────────────────────────────────
    st.subheader("Classification — Risk Level Prediction")
    st.markdown(f"**Best Model:** `{clf_info['algorithm']}`")

    # Metrics table
    all_clf = clf_info["all_models"]
    rows = []
    for name, m in all_clf.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f}",
            "F1 (macro)": f"{m['f1']:.3f}",
            "ROC-AUC": f"{m['roc_auc']:.3f}",
            "CV F1": f"{m['cv_f1_mean']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion matrix and ROC from saved images (or recompute)
    cm_path  = os.path.join(PROJECT_ROOT, "data", "processed", "confusion_matrix.png")
    roc_path = os.path.join(PROJECT_ROOT, "data", "processed", "roc_curve.png")
    fi_path  = os.path.join(PROJECT_ROOT, "data", "processed", "feature_importance.png")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves", use_container_width=True)

    if os.path.exists(fi_path):
        st.image(fi_path, caption="Feature Importances", use_container_width=True)

    # ── Regressor section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Regression — GPA Prediction")
    st.markdown(f"**Best Model:** `{reg_info['algorithm']}`")

    all_reg = reg_info["all_models"]
    reg_rows = [
        {"Model": name, "MAE": f"{m['mae']:.3f}", "RMSE": f"{m['rmse']:.3f}", "R²": f"{m['r2']:.3f}"}
        for name, m in all_reg.items()
    ]
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # Actual vs Predicted chart (run inference on test portion)
    st.markdown("---")
    st.subheader("Actual vs Predicted GPA")
    with st.spinner("Computing predictions on test set..."):
        try:
            df_full = load_raw_dataset()
            from sklearn.model_selection import train_test_split
            _, df_test = train_test_split(df_full, test_size=0.20, random_state=42, stratify=df_full["risk_level"])
            pre = load_preprocessor()
            reg = load_regressor()
            X_test = pre.transform(df_test)
            y_true = df_test["final_gpa"].to_numpy(dtype=float)
            y_pred = np.clip(reg.predict(X_test), 0.0, 4.0)
            st.plotly_chart(actual_vs_predicted(y_true, y_pred), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate chart: {e}")

    # Model metadata
    st.markdown("---")
    st.subheader("Training Metadata")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Dataset Size", info["dataset_size"])
    meta_cols[1].metric("Train / Test Split", f"{info['train_size']} / {info['test_size']}")
    meta_cols[2].metric("Features", info["feature_count"])
    meta_cols[3].metric("Trained At", info["trained_at"][:10])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

def page_history():
    st.title("📋 Prediction History")
    st.markdown("All predictions made through the Single Prediction page are stored here.")

    records = get_all_predictions(limit=500)
    if not records:
        st.info("No predictions yet. Go to **Single Student Prediction** to make your first prediction.")
        return

    df = pd.DataFrame(records)

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with col_f2:
        name_filter = st.text_input("Search by student name", "")

    filtered = df[df["risk_level"].isin(risk_filter)]
    if name_filter.strip():
        filtered = filtered[filtered["student_name"].str.contains(name_filter, case=False, na=False)]

    st.markdown(f"Showing **{len(filtered)}** of **{len(df)}** records.")

    # Display table
    display_df = filtered[["id", "timestamp", "student_name", "risk_level", "gpa_prediction", "prob_high", "prob_medium", "prob_low"]].copy()
    display_df.columns = ["ID", "Timestamp", "Student", "Risk Level", "Predicted GPA", "P(High)", "P(Medium)", "P(Low)"]
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Summary donut
    risk_counts = df["risk_level"].value_counts().to_dict()
    col_d, col_stats = st.columns([1, 1])
    with col_d:
        st.plotly_chart(risk_donut(risk_counts), use_container_width=True)
    with col_stats:
        st.metric("Total Predictions", len(df))
        st.metric("High Risk Predictions", risk_counts.get("High", 0))
        st.metric("Average Predicted GPA", f"{df['gpa_prediction'].mean():.2f}")

    # Clear history
    st.markdown("---")
    if st.button("🗑️ Clear All History", type="secondary"):
        clear_predictions()
        st.success("History cleared.")
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════

PAGE_DISPATCH = {
    "🏠  Home":               page_home,
    "🔍  Single Prediction":  page_single_prediction,
    "📂  Batch Prediction":   page_batch_prediction,
    "📊  EDA Dashboard":      page_eda,
    "🤖  Model Performance":  page_model_performance,
    "📋  Prediction History": page_history,
}

PAGE_DISPATCH[page]()
