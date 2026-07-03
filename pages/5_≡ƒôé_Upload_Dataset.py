"""pages/5_📂_Upload_Dataset.py — Batch predictions with % output"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
from modules import preprocessing as pp

st.set_page_config(page_title="Upload Dataset", page_icon="📂", layout="wide")
st.title("📂 Batch Dataset Prediction")


def load_models():
    try:
        d  = joblib.load("models/dropout_best.pkl")
        p  = joblib.load("models/placement_best.pkl")
        sc = joblib.load("models/scaler.pkl")
        with open("models/feat_cols.txt") as f:
            fc = f.read().strip().split(",")
        return d, p, sc, fc
    except Exception:
        return None, None, None, None


d_model  = st.session_state.get("dropout_best_model")
p_model  = st.session_state.get("placement_best_model")
scaler   = st.session_state.get("scaler")
feat_cols = st.session_state.get("feat_cols")
if d_model is None:
    d_model, p_model, scaler, feat_cols = load_models()

if d_model is None:
    st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
    st.stop()


def risk_label(pct, invert=False):
    if invert:  # dropout: high % = bad
        if pct >= 60: return "🔴 High Risk"
        if pct >= 35: return "🟡 Medium Risk"
        return "🟢 Low Risk"
    else:        # placement: high % = good
        if pct >= 65: return "🟢 High"
        if pct >= 40: return "🟡 Moderate"
        return "🔴 Low"


st.markdown("Upload a CSV with student records — predictions are shown as **percentages**.")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.markdown(f"**Uploaded:** `{df_raw.shape[0]}` students × `{df_raw.shape[1]}` columns")
    st.dataframe(df_raw.head(5), use_container_width=True)

    if st.button("⚡ Run Batch Predictions", type="primary", use_container_width=True):
        with st.spinner("Preprocessing & predicting…"):
            df = pp.clean(df_raw.copy())
            df = pp.encode(df)
            df = pp.engineer_features(df)
            for c in feat_cols:
                if c not in df.columns:
                    df[c] = 50.0
            X = df[feat_cols].fillna(df[feat_cols].median())
            X_scaled = scaler.transform(X)
            drop_proba  = d_model.predict_proba(X_scaled)[:, 1]
            place_proba = p_model.predict_proba(X_scaled)[:, 1]

        # Build results — all % not 0/1
        results = df_raw.copy()
        results["Dropout Risk %"]       = (drop_proba  * 100).round(1)
        results["Placement Prob %"]     = (place_proba * 100).round(1)
        results["Dropout Label"]        = [risk_label(p, invert=True)  for p in results["Dropout Risk %"]]
        results["Placement Label"]      = [risk_label(p, invert=False) for p in results["Placement Prob %"]]
        results["Prediction Confidence %"] = (
            (np.abs(drop_proba - 0.5) + np.abs(place_proba - 0.5)) * 100
        ).round(1)

        st.success(f"✅ Predictions complete for {len(results)} students")

        # Colour-coded display
        def colour_dropout(val):
            if isinstance(val, float):
                if val >= 60: return "background-color:#3a1a1a; color:#F7644F"
                if val >= 35: return "background-color:#3a2e00; color:#F7C948"
                return "background-color:#1a3a1a; color:#6BD490"
            return ""

        def colour_placement(val):
            if isinstance(val, float):
                if val >= 65: return "background-color:#1a3a1a; color:#6BD490"
                if val >= 40: return "background-color:#3a2e00; color:#F7C948"
                return "background-color:#3a1a1a; color:#F7644F"
            return ""

        display_cols = ["Dropout Risk %","Dropout Label","Placement Prob %","Placement Label","Prediction Confidence %"]
        if "student_id" in results.columns:
            display_cols = ["student_id"] + display_cols
        if "cgpa" in results.columns:
            display_cols = display_cols[:1] + ["cgpa","attendance"] + display_cols[1:]

        styled = (results[display_cols]
                  .style
                  .applymap(colour_dropout,  subset=["Dropout Risk %"])
                  .applymap(colour_placement, subset=["Placement Prob %"]))

        st.dataframe(styled, use_container_width=True, height=420)

        # Download
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Predictions CSV", csv_bytes,
                           file_name="predictions.csv", mime="text/csv",
                           use_container_width=True)

        # Summary
        st.markdown("### 📊 Batch Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students",   len(results))
        c2.metric("🔴 High Dropout Risk",
                  int((drop_proba >= 0.6).sum()),
                  f"{(drop_proba >= 0.6).mean()*100:.1f}% of class")
        c3.metric("🟢 Likely Placed",
                  int((place_proba >= 0.65).sum()),
                  f"{(place_proba >= 0.65).mean()*100:.1f}% of class")
        c4.metric("Avg Placement %", f"{place_proba.mean()*100:.1f}%")

else:
    st.info("📄 Upload a CSV to begin.")
    st.markdown("**Expected columns (all optional except `cgpa`):**")
    st.code(
        "cgpa, attendance, sem_gpa, internal_marks, backlogs,\n"
        "coding_skill, communication_skill, technical_skill,\n"
        "aptitude_score, english_proficiency,\n"
        "internship, projects, certifications"
    )
    # Provide sample download
    sample = pd.DataFrame([
        {"cgpa":8.1,"attendance":88,"backlogs":0,"coding_skill":75,
         "communication_skill":70,"technical_skill":72,"internship":1,"projects":2,"certifications":2},
        {"cgpa":5.9,"attendance":62,"backlogs":3,"coding_skill":42,
         "communication_skill":45,"technical_skill":40,"internship":0,"projects":0,"certifications":0},
        {"cgpa":7.2,"attendance":78,"backlogs":1,"coding_skill":60,
         "communication_skill":58,"technical_skill":55,"internship":1,"projects":1,"certifications":1},
    ])
    st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False).encode(),
                       "sample_students.csv", "text/csv")
