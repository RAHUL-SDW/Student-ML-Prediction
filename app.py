"""
app.py  ─  Entry point for the Streamlit multi-page app.
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Student AI Framework",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0d1117;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #c9d1d9;
}
/* Global dark card feel */
.stApp {
    background: #0d1117;
    color: #c9d1d9;
}
/* Remove default top padding */
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student AI Prediction Framework")
st.markdown("""
Welcome! Use the **sidebar** to navigate between pages:

| Page | Purpose |
|------|---------|
| 🏠 Home              | Overview & guide |
| 📊 Dataset Analysis  | Explore the dataset with charts |
| 🤖 Train Model       | Train & compare ML models |
| 👤 Single Prediction | Predict for one student |
| 📂 Upload Dataset    | Batch predictions via CSV upload |
""")

st.info("👈 Select a page from the sidebar to get started.")
