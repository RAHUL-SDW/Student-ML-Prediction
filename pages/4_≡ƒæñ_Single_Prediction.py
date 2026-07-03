"""pages/4_👤_Single_Prediction.py"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
from modules import meta_cognitive as mc
from modules import recommender as rec
from modules import preprocessing as pp

st.set_page_config(page_title="Single Prediction", page_icon="👤", layout="wide")
st.title("👤 Single Student Prediction")


# ── helpers ─────────────────────────────────────────────────────────────────

def load_models_from_disk():
    """Try to load previously saved models from disk."""
    try:
        d_model  = joblib.load("models/dropout_best.pkl")
        p_model  = joblib.load("models/placement_best.pkl")
        scaler   = joblib.load("models/scaler.pkl")
        with open("models/feat_cols.txt") as f:
            feat_cols = f.read().strip().split(",")
        return d_model, p_model, scaler, feat_cols
    except Exception:
        return None, None, None, None


def gauge_color(pct: float, invert=False):
    """Red→orange→green (or inverted)."""
    if invert:
        pct = 100 - pct
    if pct >= 70:
        return "#4CAF50"
    elif pct >= 45:
        return "#FF9800"
    else:
        return "#F44336"


def metric_card(label, value, suffix="%", sublabel="", color="#4F8EF7"):
    return f"""
    <div style="background:#1e1e2e;border:1px solid #2a2a4a;border-radius:12px;
                padding:1rem;text-align:center;margin:4px;">
      <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase;
                  letter-spacing:1px;">{label}</div>
      <div style="color:{color};font-size:2.4rem;font-weight:700;
                  line-height:1.1;">{value}{suffix}</div>
      <div style="color:#6e7681;font-size:0.75rem;">{sublabel}</div>
    </div>"""


def risk_badge(label, color):
    return f"""<span style="background:{color}22;color:{color};border:1px solid {color};
               border-radius:20px;padding:3px 14px;font-size:0.85rem;">{label}</span>"""


# ── load models ──────────────────────────────────────────────────────────────
d_model  = st.session_state.get("dropout_best_model")
p_model  = st.session_state.get("placement_best_model")
scaler   = st.session_state.get("scaler")
feat_cols = st.session_state.get("feat_cols")

if d_model is None:
    d_model, p_model, scaler, feat_cols = load_models_from_disk()

if d_model is None:
    st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
    st.stop()

# ── input form ───────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Student Profile")

with st.form("student_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Academic**")
        cgpa        = st.slider("CGPA",        4.0, 10.0, 7.5, 0.1)
        attendance  = st.slider("Attendance (%)", 20.0, 100.0, 78.0, 0.5)
        sem_gpa     = st.slider("Semester GPA",   4.0, 10.0, 7.3, 0.1)
        int_marks   = st.slider("Internal Marks", 20.0, 90.0, 65.0, 0.5)
        backlogs    = st.number_input("Backlogs", 0, 10, 0)

    with c2:
        st.markdown("**🛠️ Skills**")
        coding      = st.slider("Coding Skill",        0, 100, 65)
        communication = st.slider("Communication Skill", 0, 100, 60)
        technical   = st.slider("Technical Skill",     0, 100, 62)
        aptitude    = st.slider("Aptitude Score",       0, 100, 68)
        english     = st.slider("English Proficiency", 0, 100, 63)

    with c3:
        st.markdown("**💼 Experience**")
        internship    = st.selectbox("Internship",      [0, 1], format_func=lambda x: "Yes" if x else "No")
        projects      = st.number_input("Projects",      0, 10, 1)
        certifications = st.number_input("Certifications", 0, 10, 1)

    submitted = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

# ── prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_dict = {
        "cgpa": cgpa, "attendance": attendance, "sem_gpa": sem_gpa,
        "internal_marks": int_marks, "backlogs": backlogs,
        "coding_skill": coding, "communication_skill": communication,
        "technical_skill": technical, "aptitude_score": aptitude,
        "english_proficiency": english,
        "internship": internship, "projects": projects, "certifications": certifications,
    }

    # Build feature row
    row = pd.DataFrame([input_dict])
    row = pp.engineer_features(row)
    row_feat = row[[c for c in feat_cols if c in row.columns]]
    # fill any missing engineered cols with 50
    for c in feat_cols:
        if c not in row_feat.columns:
            row_feat[c] = 50.0

    row_scaled = scaler.transform(row_feat[feat_cols])

    drop_proba   = float(d_model.predict_proba(row_scaled)[0][1])
    place_proba  = float(p_model.predict_proba(row_scaled)[0][1])
    drop_pct     = round(drop_proba * 100, 1)
    place_pct    = round(place_proba * 100, 1)

    meta = mc.evaluate_prediction(input_dict, drop_proba, place_proba)

    # ── results header ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Prediction Results")

    primary_cols = st.columns(2)
    drop_color  = gauge_color(drop_pct, invert=True)
    place_color = gauge_color(place_pct)

    drop_label  = "High Risk"   if drop_pct > 60 else ("Medium Risk" if drop_pct > 35 else "Low Risk")
    place_label = "High Chance" if place_pct > 65 else ("Moderate" if place_pct > 40 else "Low Chance")

    with primary_cols[0]:
        st.markdown(
            metric_card("Dropout Risk", drop_pct, sublabel=drop_label, color=drop_color)
            + f"<div style='text-align:center;margin-top:4px;'>{risk_badge(drop_label, drop_color)}</div>",
            unsafe_allow_html=True
        )
    with primary_cols[1]:
        st.markdown(
            metric_card("Placement Probability", place_pct, sublabel=place_label, color=place_color)
            + f"<div style='text-align:center;margin-top:4px;'>{risk_badge(place_label, place_color)}</div>",
            unsafe_allow_html=True
        )

    # ── meta-cognitive scores ─────────────────────────────────────────────────
    st.markdown("### 🧠 Meta-Cognitive Evaluation")

    mc_cols = st.columns(5)
    mc_labels = [
        ("Prediction Confidence",  "prediction_confidence",  False),
        ("Academic Consistency",   "academic_consistency",   False),
        ("Skill Readiness",        "skill_readiness",        False),
        ("Feature Reliability",    "feature_reliability",    False),
        ("Final Reliability",      "final_reliability_score", False),
    ]
    for col, (lbl, key, inv) in zip(mc_cols, mc_labels):
        val = meta[key]
        color = gauge_color(val, invert=inv)
        with col:
            st.markdown(metric_card(lbl, val, color=color), unsafe_allow_html=True)

    rel_label = mc.reliability_label(meta["final_reliability_score"])
    rel_color = gauge_color(meta["final_reliability_score"])
    st.markdown(
        f"<div style='text-align:center;margin-top:8px;'>"
        f"Overall Reliability: {risk_badge(f'{rel_label} Reliability', rel_color)}"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── recommendations ───────────────────────────────────────────────────────
    st.markdown("### 💡 Personalised Recommendations")
    recs = rec.generate_recommendations(input_dict, drop_pct, place_pct)
    for issue, advice in recs:
        with st.expander(issue):
            st.write(advice)
