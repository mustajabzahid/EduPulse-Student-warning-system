import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="EduPulse — At-Risk Student Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@600;700&family=Public+Sans:wght@400;500;600&display=swap');
    .stApp {
        background-color: #eef2f7;
        background-image:
            radial-gradient(at 14% 8%, rgba(20, 184, 166, 0.09) 0px, transparent 48%),
            radial-gradient(at 88% 12%, rgba(15, 23, 42, 0.06) 0px, transparent 46%);
        color: #0f172a;
        font-family: 'Public Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.02em; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 18, 38, 0.90) 0%, rgba(11, 22, 44, 0.88) 60%, rgba(14, 30, 59, 0.90) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #f8fafc;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0; }
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.70);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.80);
    }
    @keyframes edupulseFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], div[data-testid="stForm"], div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.70);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 24px -6px rgba(11, 22, 44, 0.08), 0 1px 2px rgba(11, 22, 44, 0.04);
        animation: edupulseFadeUp 0.28s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }
    .edupulse-hero {
        background: linear-gradient(135deg, #081226 0%, #0b162c 65%, #102a43 100%);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -12px rgba(11, 22, 44, 0.35);
    }
    .edupulse-hero p { color: #cbd5e1; margin: 6px 0 0 0; font-size: 0.92rem; }
    .edupulse-hero .kicker { color: #5eead4; font-size: 0.78rem; font-weight: 600; margin-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_RANGES = {
    "attendance_pct": (30.0, 100.0),
    "quiz_avg": (0.0, 10.0),
    "assignment_avg": (0.0, 10.0),
    "mid_marks": (0.0, 30.0),
    "lab_marks": (0.0, 20.0),
    "study_hours_week": (0.0, 40.0),
    "prev_cgpa": (1.0, 4.0),
}

NUMERIC_FEATURES = list(FEATURE_RANGES.keys())
CATEGORICAL_FEATURES = ["part_time_job"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLS = ["student_id"] + ALL_FEATURES
PALETTE = {"Pass": "#0d9488", "Fail": "#dc2626"}


def build_default_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(202604)
    n = 410
    part_time = rng.choice(["Yes", "No"], size=n, p=[0.34, 0.66])
    engagement = rng.normal(0, 1, size=n)
    job_penalty = np.where(part_time == "Yes", -0.35, 0.1)

    attendance = np.clip(np.round(78 + engagement * 12 + job_penalty * 6 + rng.normal(0, 5, n), 1), 30, 100)
    quiz_avg = np.clip(np.round(6.9 + engagement * 1.4 + rng.normal(0, 0.7, n), 1), 0, 10)
    assignment_avg = np.clip(np.round(7.3 + engagement * 1.3 + rng.normal(0, 0.65, n), 1), 0, 10)
    mid_marks = np.clip(np.round(20.2 + engagement * 4.4 + rng.normal(0, 2.0, n), 1), 0, 30)
    lab_marks = np.clip(np.round(14.4 + engagement * 2.6 + rng.normal(0, 1.3, n), 1), 0, 20)
    study_hours = np.clip(np.round(13.5 + engagement * 4.8 + job_penalty * 2.5 + rng.normal(0, 2.2, n), 1), 0, 40)
    prev_cgpa = np.clip(np.round(2.82 + engagement * 0.52 + rng.normal(0, 0.25, n), 2), 1.0, 4.0)

    logit = (
        14.2
        - 0.068 * attendance
        - 0.16 * mid_marks
        - 0.24 * quiz_avg
        - 0.20 * assignment_avg
        - 0.14 * lab_marks
        - 0.075 * study_hours
        - 1.05 * prev_cgpa
        + np.where(part_time == "Yes", 0.55, -0.15)
        + rng.normal(0, 0.55, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    final_result = np.where(rng.random(n) < prob, "Fail", "Pass")

    df = pd.DataFrame(
        {
            "student_id": [f"STU-{1001 + i}" for i in range(n)],
            "attendance_pct": attendance,
            "quiz_avg": quiz_avg,
            "assignment_avg": assignment_avg,
            "mid_marks": mid_marks,
            "lab_marks": lab_marks,
            "study_hours_week": study_hours,
            "prev_cgpa": prev_cgpa,
            "part_time_job": part_time,
            "final_result": final_result,
        }
    )

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan
    df.loc[df.index[-8:], "final_result"] = np.nan
    dups = df.iloc[[4, 19, 37, 64, 102, 148, 195, 241, 289, 333]]
    return pd.concat([df, dups], ignore_index=True)


def validate_academic_ranges(df: pd.DataFrame):
    """Flag out-of-range values rather than silently treating them as valid."""
    validated_df = df.copy()
    range_flags = []
    for col, (min_v, max_v) in FEATURE_RANGES.items():
        numeric_series = pd.to_numeric(validated_df[col], errors="coerce")
        non_numeric_count = int((validated_df[col].notna() & numeric_series.isna()).sum())
        out_mask = numeric_series.notna() & ((numeric_series < min_v) | (numeric_series > max_v))
        out_count = int(out_mask.sum())
        if out_count > 0 or non_numeric_count > 0:
            range_flags.append(
                {
                    "column": col,
                    "expected_range": f"{min_v}–{max_v}",
                    "out_of_range": out_count,
                    "non_numeric": non_numeric_count,
                }
            )
            numeric_series = numeric_series.mask(out_mask, np.nan)
        validated_df[col] = numeric_series

    valid_jobs = {"Yes", "No"}
    job_series = validated_df["part_time_job"].astype("object")
    bad_job_mask = job_series.notna() & (~job_series.astype(str).str.strip().isin(valid_jobs))
    if int(bad_job_mask.sum()) > 0:
        range_flags.append(
            {
                "column": "part_time_job",
                "expected_range": "Yes / No",
                "out_of_range": int(bad_job_mask.sum()),
                "non_numeric": 0,
            }
        )
        validated_df.loc[bad_job_mask, "part_time_job"] = np.nan
    return validated_df, range_flags


def clean_and_audit(raw_df: pd.DataFrame):
    orig_rows = len(raw_df)
    dedup_df = raw_df.drop_duplicates().copy()
    dup_count = orig_rows - len(dedup_df)

    validated_df, range_flags = validate_academic_ranges(dedup_df)
    missing_counts = validated_df.isna().sum()
    missing_target = int(validated_df["final_result"].isna().sum())

    model_df = validated_df.dropna(subset=["final_result"]).copy()
    display_df = model_df.copy()
    for col in NUMERIC_FEATURES:
        display_df[col] = display_df[col].fillna(display_df[col].median())
    display_df["part_time_job"] = display_df["part_time_job"].fillna(
        display_df["part_time_job"].mode().iloc[0]
    )

    return {
        "original_rows": orig_rows,
        "duplicate_count": dup_count,
        "cleaned_rows": len(model_df),
        "missing_counts": missing_counts,
        "missing_target_excluded": missing_target,
        "range_flags": range_flags,
        "model_df": model_df,
        "display_df": display_df,
    }


@st.cache_resource(show_spinner=False)
def train_pipelines(model_df: pd.DataFrame):
    X = model_df[ALL_FEATURES].copy()
    y = (model_df["final_result"].astype(str).str.strip().str.lower() == "fail").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    lr_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    rf_pipe = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    results = {}
    for name, pipe in [("Logistic Regression", lr_pipe), ("Random Forest", rf_pipe)]:
        preds = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, preds),
            "fail_precision": precision_score(y_test, preds, pos_label=1, zero_division=0),
            "fail_recall": recall_score(y_test, preds, pos_label=1, zero_division=0),
            "cm": confusion_matrix(y_test, preds, labels=[1, 0]),
        }
    return results


def risk_tier(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color="#334155"),
        margin=dict(t=48, r=18, b=46, l=54),
    )
    fig.update_xaxes(gridcolor="#eef2f6")
    fig.update_yaxes(gridcolor="#eef2f6")
    return fig


def advisor_notes(row: dict) -> list[str]:
    notes = []
    if row["attendance_pct"] < 75:
        notes.append(
            f"Prioritize raising attendance from {row['attendance_pct']:.1f}% above the 75% threshold."
        )
    if row["mid_marks"] < 18:
        notes.append(
            f"Arrange targeted midterm remediation (current midterm: {row['mid_marks']:.1f}/30, benchmark ≥ 18/30)."
        )
    if row["quiz_avg"] < 6:
        notes.append(
            f"Introduce spaced-repetition quizzes to lift quiz average from {row['quiz_avg']:.1f}/10 toward 6.0+/10."
        )
    if row["assignment_avg"] < 6:
        notes.append(
            f"Set 48-hour pre-deadline draft reviews to lift assignment average from {row['assignment_avg']:.1f}/10 toward 6.0+/10."
        )
    if row["lab_marks"] < 12:
        notes.append(
            f"Pair with peer lab tutoring to close practical gaps (current lab score: {row['lab_marks']:.1f}/20, benchmark ≥ 12/20)."
        )
    if row["study_hours_week"] < 10:
        notes.append(
            f"Build a 12+ hr/week study schedule (currently {row['study_hours_week']:.1f} hrs/week, benchmark ≥ 10 hrs/week)."
        )
    if row["prev_cgpa"] < 2.4:
        notes.append(
            f"Provide foundational academic coaching (previous CGPA: {row['prev_cgpa']:.2f}/4.00, below 2.40)."
        )
    if row["part_time_job"] == "Yes" and (row["study_hours_week"] < 12 or row["attendance_pct"] < 80):
        notes.append("Review part-time shift conflicts with core lecture schedules.")
    return notes or ["Metrics are aligned with passing benchmarks; maintain current study cadence."]


st.sidebar.title("EduPulse")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Analytics",
        "AI Prediction",
        "Student Risk Register",
        "Batch Analysis",
        "What-if Simulator",
    ],
)

uploaded = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    try:
        raw_df = pd.read_csv("Track 4 - student_performance.csv")
    except FileNotFoundError:
        raw_df = build_default_dataframe()

missing_req = [c for c in REQUIRED_COLS + ["final_result"] if c not in raw_df.columns]
if missing_req:
    st.error(f"Missing required column(s): {', '.join(missing_req)}")
    st.stop()

audit = clean_and_audit(raw_df)
if audit["range_flags"]:
    st.warning(f"Flagged out-of-range/invalid values in dataset: {audit['range_flags']}")

models = train_pipelines(audit["model_df"])
active_model_name = st.sidebar.selectbox("Active Model", ["Random Forest", "Logistic Regression"])
active_pipe = models[active_model_name]["pipeline"]

scored_df = audit["display_df"].copy()
scored_df["fail_probability"] = active_pipe.predict_proba(audit["model_df"][ALL_FEATURES])[:, 1]
scored_df["risk_tier"] = scored_df["fail_probability"].apply(risk_tier)

if page == "Overview":
    st.markdown(
        f"""
        <div class="edupulse-hero">
            <div class="kicker">Early-Warning Academic Telemetry · {active_model_name}</div>
            <h1 style="margin:0;color:#fff;font-size:1.75rem;">Academic Cohort Overview</h1>
            <p>Screening {audit['cleaned_rows']} cleaned records with {models[active_model_name]['fail_recall']*100:.1f}% Fail recall on held-out stratified evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    pass_count = int((scored_df["final_result"] == "Pass").sum())
    fail_count = int((scored_df["final_result"] == "Fail").sum())
    c1.metric("Cleaned Students", audit["cleaned_rows"])
    c2.metric("Pass Rate", f"{(pass_count / max(1, len(scored_df))) * 100:.1f}%")
    c3.metric("Fail Count", fail_count)
    c4.metric("High-Risk Students", int((scored_df["risk_tier"] == "High").sum()))

    col1, col2 = st.columns(2)
    fig1 = px.box(
        scored_df,
        x="final_result",
        y="attendance_pct",
        color="final_result",
        points="all",
        color_discrete_map=PALETTE,
        title="Attendance vs Final Result (75% Benchmark)",
    )
    fig1.add_hline(y=75, line_dash="dash", line_color="#d97706")
    col1.plotly_chart(style_plotly(fig1), use_container_width=True)

    fig2 = px.histogram(
        scored_df,
        x="prev_cgpa",
        color="final_result",
        barmode="overlay",
        color_discrete_map=PALETTE,
        title="Previous CGPA Distribution (2.40 Risk Cutoff)",
    )
    fig2.add_vline(x=2.4, line_dash="dash", line_color="#d97706")
    col2.plotly_chart(style_plotly(fig2), use_container_width=True)

    col3, col4 = st.columns(2)
    fig3 = px.histogram(
        scored_df,
        x="part_time_job",
        color="final_result",
        barmode="group",
        color_discrete_map=PALETTE,
        title="Part-Time Job vs Final Result",
    )
    col3.plotly_chart(style_plotly(fig3), use_container_width=True)

    tier_counts = scored_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig4 = px.bar(
        x=tier_counts.values,
        y=tier_counts.index,
        orientation="h",
        color=tier_counts.index,
        color_discrete_map={"Low": "#0d9488", "Medium": "#d97706", "High": "#dc2626"},
        title="Predicted Risk Tier Distribution",
    )
    col4.plotly_chart(style_plotly(fig4), use_container_width=True)

    st.subheader("Data Cleaning Summary")
    st.write(
        {
            "Original Rows": audit["original_rows"],
            "Exact Duplicates Removed": audit["duplicate_count"],
            "Cleaned Training Rows": audit["cleaned_rows"],
            "Missing final_result Excluded": audit["missing_target_excluded"],
            "Pass Count": pass_count,
            "Fail Count": fail_count,
            "Range Flags": len(audit["range_flags"]),
        }
    )

elif page == "Analytics":
    st.title("Exploratory Academic Analytics")
    c1, c2 = st.columns(2)
    fig_quad = px.scatter(
        scored_df,
        x="attendance_pct",
        y="mid_marks",
        color="final_result",
        color_discrete_map=PALETTE,
        title="Attendance × Midterm Risk Quadrant (<75% & <18/30)",
    )
    fig_quad.add_shape(type="rect", x0=28, x1=75, y0=0, y1=18, fillcolor="rgba(220,38,38,0.06)", line_width=0)
    c1.plotly_chart(style_plotly(fig_quad), use_container_width=True)

    corr_df = scored_df[NUMERIC_FEATURES].copy()
    corr_df["fail_flag"] = (scored_df["final_result"] == "Fail").astype(int)
    corr_mat = corr_df.corr().round(2)
    fig_corr = px.imshow(
        corr_mat,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale=["#0f766e", "#f8fafc", "#dc2626"],
        title="Academic Feature Correlation Matrix",
    )
    c2.plotly_chart(style_plotly(fig_corr), use_container_width=True)

    fig_study = px.scatter(
        scored_df,
        x="study_hours_week",
        y="prev_cgpa",
        color="final_result",
        color_discrete_map=PALETTE,
        title="Study Hours vs Previous CGPA",
    )
    fig_study.add_vline(x=10, line_dash="dot", line_color="#d97706")
    fig_study.add_hline(y=2.4, line_dash="dot", line_color="#d97706")
    st.plotly_chart(style_plotly(fig_study), use_container_width=True)

elif page == "AI Prediction":
    st.title("AI Prediction, Evaluation & Explainability")
    m1, m2 = st.columns(2)
    for col, name in [(m1, "Random Forest"), (m2, "Logistic Regression")]:
        res = models[name]
        col.subheader(name)
        col.write(
            f"Accuracy: {res['accuracy']:.3f} | Fail Precision: {res['fail_precision']:.3f} | Fail Recall: {res['fail_recall']:.3f}"
        )
        col.dataframe(
            pd.DataFrame(res["cm"], index=["Actual Fail", "Actual Pass"], columns=["Pred Fail", "Pred Pass"])
        )

    st.subheader("Individual Student Prediction")
    with st.form("single_pred"):
        att = st.number_input("Attendance % (30–100)", min_value=30.0, max_value=100.0, value=68.0, step=0.5)
        quiz = st.number_input("Quiz Avg (0–10)", min_value=0.0, max_value=10.0, value=5.4, step=0.1)
        assign = st.number_input("Assignment Avg (0–10)", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
        mid = st.number_input("Mid Marks (0–30)", min_value=0.0, max_value=30.0, value=15.5, step=0.5)
        lab = st.number_input("Lab Marks (0–20)", min_value=0.0, max_value=20.0, value=11.5, step=0.5)
        hrs = st.number_input("Study Hours / Week (0–40)", min_value=0.0, max_value=40.0, value=8.5, step=0.5)
        cgpa = st.number_input("Previous CGPA (1.00–4.00)", min_value=1.0, max_value=4.0, value=2.35, step=0.05)
        job = st.selectbox("Part-Time Job", ["No", "Yes"], index=1)
        submitted = st.form_submit_button("Predict Student Risk")

    if submitted:
        sample = {
            "attendance_pct": att,
            "quiz_avg": quiz,
            "assignment_avg": assign,
            "mid_marks": mid,
            "lab_marks": lab,
            "study_hours_week": hrs,
            "prev_cgpa": cgpa,
            "part_time_job": job,
        }
        prob = float(active_pipe.predict_proba(pd.DataFrame([sample]))[0, 1])
        tier = risk_tier(prob)
        gauge_color = "#dc2626" if tier == "High" else ("#d97706" if tier == "Medium" else "#0d9488")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%"},
                title={"text": f"Predicted Failure Probability ({tier} Risk)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(13,148,136,0.14)"},
                        {"range": [35, 65], "color": "rgba(217,119,6,0.14)"},
                        {"range": [65, 100], "color": "rgba(220,38,38,0.14)"},
                    ],
                },
            )
        )
        st.plotly_chart(style_plotly(fig_gauge), use_container_width=True)
        for note in advisor_notes(sample):
            st.write(f"- {note}")
    else:
        st.info("Enter the student's academic metrics and press **Predict Student Risk** to view the prediction.")

    st.info("Methodological note: Feature importance and coefficients reflect predictive associations, not causal effects.")

elif page == "Student Risk Register":
    st.title("Student Risk Register")
    st.dataframe(scored_df.sort_values("fail_probability", ascending=False), use_container_width=True)

elif page == "Batch Analysis":
    st.title("Batch Class Prediction")
    batch_file = st.file_uploader("Upload Class CSV for Batch Prediction", type=["csv"], key="batch")
    if batch_file is not None:
        bdf = pd.read_csv(batch_file)
        miss = [c for c in REQUIRED_COLS if c not in bdf.columns]
        if miss:
            st.error(f"Missing columns: {', '.join(miss)}")
        else:
            validated_bdf, batch_flags = validate_academic_ranges(bdf)
            if batch_flags:
                st.warning(f"Flagged out-of-range/invalid values in batch CSV: {batch_flags}")
            probs = active_pipe.predict_proba(validated_bdf[ALL_FEATURES])[:, 1]
            bdf["fail_probability"] = np.round(probs * 100, 1)
            bdf["risk_tier"] = [risk_tier(p) for p in probs]
            bdf = bdf.sort_values("fail_probability", ascending=False)
            st.dataframe(bdf, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                bdf.to_csv(index=False),
                "edupulse_batch_predictions.csv",
                "text/csv",
            )

elif page == "What-if Simulator":
    st.title("What-if Attendance Simulator")
    st.info("Hypothetical model prediction: Attendance changes while all other inputs stay constant.")
    att_sim = st.slider("Simulated Attendance (%)", 30, 100, 68)
    fixed = {
        "attendance_pct": float(att_sim),
        "quiz_avg": 5.4,
        "assignment_avg": 5.8,
        "mid_marks": 15.5,
        "lab_marks": 11.5,
        "study_hours_week": 8.5,
        "prev_cgpa": 2.35,
        "part_time_job": "Yes",
    }
    sim_prob = float(active_pipe.predict_proba(pd.DataFrame([fixed]))[0, 1])
    st.metric("Simulated Failure Probability", f"{sim_prob * 100:.1f}%", delta=f"{risk_tier(sim_prob)} Risk")
    curve_x = list(range(30, 101, 5))
    curve_df = pd.DataFrame([{**fixed, "attendance_pct": float(a)} for a in curve_x])
    curve_y = active_pipe.predict_proba(curve_df)[:, 1] * 100
    fig_curve = px.line(x=curve_x, y=curve_y, markers=True, title="Hypothetical Attendance Sensitivity Curve")
    fig_curve.add_hrect(y0=0, y1=35, fillcolor="rgba(13,148,136,0.06)", line_width=0)
    fig_curve.add_hrect(y0=35, y1=65, fillcolor="rgba(217,119,6,0.06)", line_width=0)
    fig_curve.add_hrect(y0=65, y1=100, fillcolor="rgba(220,38,38,0.06)", line_width=0)
    fig_curve.add_vline(x=75, line_dash="dash", line_color="#d97706")
    st.plotly_chart(style_plotly(fig_curve), use_container_width=True)
