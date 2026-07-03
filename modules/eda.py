"""
modules/eda.py
Generates EDA charts using Plotly (Streamlit-compatible).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np


def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=["student_id"] if "student_id" in df.columns else [], errors="ignore"
    )
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(height=600)
    return fig


def attendance_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="attendance", nbins=30, color_discrete_sequence=["#4F8EF7"],
        title="Attendance Distribution",
        labels={"attendance": "Attendance (%)"},
    )
    fig.add_vline(x=75, line_dash="dash", line_color="red",
                  annotation_text="75% threshold")
    return fig


def cgpa_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="cgpa", nbins=25, color_discrete_sequence=["#6BD490"],
        title="CGPA Distribution",
        labels={"cgpa": "CGPA"},
    )
    return fig


def dropout_distribution(df: pd.DataFrame):
    if "dropout_risk" not in df.columns:
        return None
    counts = df["dropout_risk"].map({0: "Safe", 1: "At Risk"}).value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values,
        color_discrete_sequence=["#6BD490", "#F7644F"],
        title="Dropout Risk Distribution",
    )
    return fig


def placement_distribution(df: pd.DataFrame):
    if "placement" not in df.columns:
        return None
    counts = df["placement"].map({0: "Not Placed", 1: "Placed"}).value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values,
        color_discrete_sequence=["#4F8EF7", "#F7C948"],
        title="Placement Distribution",
    )
    return fig


def feature_importance_chart(imp_df: pd.DataFrame, title="Feature Importance"):
    top = imp_df.head(12)
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
        title=title,
        labels={"importance": "Importance Score", "feature": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def model_comparison_chart(results: dict, metric="f1"):
    names = list(results.keys())
    values = [results[m][metric] for m in names]
    fig = px.bar(
        x=names, y=values,
        color=values, color_continuous_scale="Viridis",
        labels={"x": "Model", "y": metric.upper()},
        title=f"Model Comparison — {metric.upper()} Score",
    )
    fig.update_traces(text=[f"{v:.3f}" for v in values], textposition="outside")
    return fig


def confusion_matrix_chart(cm_list: list, title="Confusion Matrix"):
    cm = np.array(cm_list)
    labels = ["Negative", "Positive"]
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", showscale=False,
    )
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return fig
