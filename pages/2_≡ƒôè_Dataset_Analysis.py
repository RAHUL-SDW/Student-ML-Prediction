"""pages/2_📊_Dataset_Analysis.py"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_dataset import generate
from modules import eda

st.set_page_config(page_title="Dataset Analysis", page_icon="📊", layout="wide")
st.title("📊 Dataset Analysis")

@st.cache_data
def load_data():
    os.makedirs("data", exist_ok=True)
    path = "data/students_raw.csv"
    if not os.path.exists(path):
        df = generate()
        df.to_csv(path, index=False)
    import pandas as pd
    return pd.read_csv(path)

df = load_data()

st.markdown(f"**Dataset shape:** {df.shape[0]} students × {df.shape[1]} features")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Data Preview", "📈 Distributions", "🔥 Correlation", "🥧 Labels", "📌 Summary Stats"]
)

with tab1:
    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.attendance_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.cgpa_distribution(df), use_container_width=True)

with tab3:
    st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        fig = eda.dropout_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = eda.placement_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    import pandas as pd
    num_cols = df.select_dtypes("number").columns.tolist()
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
