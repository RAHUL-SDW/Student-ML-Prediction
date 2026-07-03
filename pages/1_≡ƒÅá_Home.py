"""pages/1_🏠_Home.py"""
import streamlit as st

st.set_page_config(page_title="Home | Student AI Framework", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2rem; }
.hero h1 { color: #e0e0e0; font-size: 2.4rem; font-weight: 700; margin:0; }
.hero p  { color: #a0b4cc; font-size: 1.1rem; margin-top: 0.5rem; }
.badge { display:inline-block; background:#0f3460; color:#4F8EF7;
         border:1px solid #4F8EF7; border-radius:20px;
         padding:4px 14px; font-size:0.8rem; margin:4px; }
.card  { background:#1e1e2e; border:1px solid #2a2a4a; border-radius:12px;
         padding:1.4rem; height:100%; }
.card h4 { color:#c9d1d9; margin-bottom:0.4rem; }
.card p  { color:#8b949e; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Adaptive Meta-Cognitive Hybrid AI</h1>
  <p>Student Dropout & Placement Risk Prediction Framework</p>
  <span class="badge">XGBoost</span>
  <span class="badge">Random Forest</span>
  <span class="badge">Meta-Cognitive Evaluation</span>
  <span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, icon, title, desc in [
    (c1, "📊", "Dataset Analysis", "Explore correlations, distributions & feature importance"),
    (c2, "🤖", "Train Model",      "Compare 4 ML models & auto-select the best"),
    (c3, "👤", "Predict Student",  "Single-student prediction with meta-cognitive scores"),
    (c4, "📂", "Batch Upload",     "Upload CSV & download bulk predictions"),
]:
    with col:
        st.markdown(f"""
        <div class="card">
          <h4>{icon} {title}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### ℹ️ How to use this app")
st.markdown("""
1. **Dataset Analysis** — view EDA charts on the sample dataset  
2. **Train Model** — train all four classifiers and pick the best  
3. **Single Prediction** — enter one student's profile and get instant results  
4. **Upload Dataset** — batch-predict an entire class and download CSV  
""")

st.info("Navigate using the **sidebar** on the left. Start with **Dataset Analysis**.")
