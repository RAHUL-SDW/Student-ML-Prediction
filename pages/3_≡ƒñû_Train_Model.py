"""pages/3_🤖_Train_Model.py  —  Train & Compare all 6 models"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from generate_dataset import generate
from modules import preprocessing as pp
from modules import model_trainer as mt
from modules import eda

st.set_page_config(page_title="Train Model", page_icon="🤖", layout="wide")
st.title("🤖 Model Training & Comparison")
st.caption("4 base models + Hard Voting + Soft Voting — best selected automatically by F1 Score")

@st.cache_data
def load_data():
    os.makedirs("data", exist_ok=True)
    path = "data/students_raw.csv"
    if not os.path.exists(path):
        df = generate(); df.to_csv(path, index=False)
    return pd.read_csv(path)

df = load_data()

# ── colour rows: best model gets green ──────────────────────────────────────
def style_best(df_metrics, best):
    styles = pd.DataFrame("", index=df_metrics.index, columns=df_metrics.columns)
    if best in df_metrics.index:
        styles.loc[best] = "background-color:#1a3a1a; color:#6BD490; font-weight:700"
    # voting rows: subtle blue tint
    for name in ["Hard Voting","Soft Voting"]:
        if name in df_metrics.index:
            styles.loc[name] = "background-color:#0d1b2a; color:#7eb8f7"
    return styles


if st.button("🚀 Train All 6 Models", type="primary", use_container_width=True):

    with st.spinner("Preprocessing…"):
        X, y_drop, y_place, scaler, feat_cols = pp.preprocess_pipeline(df)
    st.session_state.update({"scaler": scaler, "feat_cols": feat_cols})

    for label, y, prefix in [
        ("Dropout Risk",          y_drop,  "dropout"),
        ("Placement Probability", y_place, "placement"),
    ]:
        st.markdown(f"---\n### 📌 {label}")
        X_tr, X_te, y_tr, y_te = pp.split_data(X, y)

        with st.spinner(f"Training {label} models (6 models)…"):
            trained, results, best = mt.train_all(X_tr, y_tr, X_te, y_te)

        # Persist in session
        st.session_state[f"{prefix}_models"]     = trained
        st.session_state[f"{prefix}_results"]    = results
        st.session_state[f"{prefix}_best"]       = best
        st.session_state[f"{prefix}_best_model"] = trained[best]

        # Save to disk
        os.makedirs("models", exist_ok=True)
        for mname, mobj in trained.items():
            safe = mname.lower().replace(" ", "_")
            mt.save_model(mobj, f"models/{prefix}_{safe}.pkl")
        mt.save_model(trained[best], f"models/{prefix}_best.pkl")
        pp.save_scaler(scaler, "models/scaler.pkl")
        with open("models/feat_cols.txt", "w") as f: f.write(",".join(feat_cols))
        with open(f"models/{prefix}_best_name.txt","w") as f: f.write(best)

        st.success(f"✅  Best model: **{best}**")

        # Metrics table
        metrics_df = pd.DataFrame({
            m: {
                "Accuracy":  r["accuracy"],
                "Precision": r["precision"],
                "Recall":    r["recall"],
                "F1 Score":  r["f1"],
            }
            for m, r in results.items()
        }).T

        st.dataframe(
            metrics_df.style.apply(style_best, best=best, axis=None)
                            .format("{:.4f}"),
            use_container_width=True
        )

        # Chart + Confusion matrix side by side
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(eda.model_comparison_chart(results, "f1"), use_container_width=True)
        with c2:
            st.plotly_chart(
                eda.confusion_matrix_chart(results[best]["confusion_matrix"],
                                           f"{best} — Confusion Matrix"),
                use_container_width=True,
            )

        # Feature importance
        imp = mt.get_feature_importance(trained[best], feat_cols)
        st.plotly_chart(
            eda.feature_importance_chart(imp, f"Feature Importance — {label} ({best})"),
            use_container_width=True,
        )

        # Per-model confusion matrices in expander
        with st.expander("🔍 View all confusion matrices"):
            cols = st.columns(3)
            for i, (mname, r) in enumerate(results.items()):
                with cols[i % 3]:
                    st.plotly_chart(
                        eda.confusion_matrix_chart(r["confusion_matrix"], mname),
                        use_container_width=True,
                    )

elif any(k in st.session_state for k in ["dropout_results","placement_results"]):
    st.info("Models already trained this session. Click **Train All 6 Models** to retrain.")
else:
    st.info("👆 Click **Train All 6 Models** to begin. Training takes about 10–15 seconds.")

    # Show architecture diagram
    st.markdown("### 🏗️ Hybrid AI Architecture")
    st.markdown("""
    ```
    Dataset
       │
       ▼
    Preprocessing (clean → encode → feature engineer → scale)
       │
       ├──► Logistic Regression ──┐
       ├──► Decision Tree         ├──► Hard Voting Ensemble
       ├──► Random Forest         ├──► Soft Voting Ensemble
       └──► XGBoost ──────────────┘
                                  │
                        Compare by F1 Score
                                  │
                         Auto-select Best Model
                                  │
                        Predict + Meta-Cognitive Evaluation
    ```
    """)
