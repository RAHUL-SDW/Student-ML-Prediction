"""
modules/preprocessing.py
Handles data cleaning, encoding, feature engineering, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLS = [
    "cgpa", "attendance", "sem_gpa", "internal_marks", "backlogs",
    "coding_skill", "communication_skill", "technical_skill",
    "aptitude_score", "english_proficiency",
    "internship", "projects", "certifications",
    "academic_consistency", "skill_readiness", "overall_performance",
]

CATEGORICAL_COLS = ["gender", "branch"]
LABEL_TARGETS = ["dropout_risk", "placement"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Academic Consistency: closeness of CGPA and SemGPA, penalised by backlogs
    if "cgpa" in df.columns and "sem_gpa" in df.columns:
        diff = (df["cgpa"] - df["sem_gpa"]).abs()
        df["academic_consistency"] = np.clip(
            100 - (diff * 10) - (df.get("backlogs", 0) * 5), 0, 100
        )
    else:
        df["academic_consistency"] = 50.0

    # Skill Readiness: average of skill columns (normalised 0-100)
    skill_cols = [c for c in ["coding_skill", "communication_skill",
                               "technical_skill", "aptitude_score",
                               "english_proficiency"] if c in df.columns]
    if skill_cols:
        df["skill_readiness"] = df[skill_cols].mean(axis=1)
    else:
        df["skill_readiness"] = 50.0

    # Overall Performance: weighted composite
    cgpa_norm = df.get("cgpa", 7) / 10 * 100
    att_w = df.get("attendance", 75)
    df["overall_performance"] = (
        0.35 * cgpa_norm + 0.25 * att_w + 0.20 * df["skill_readiness"]
        + 0.10 * df.get("internship", 0) * 100
        + 0.10 * df.get("projects", 0) * 33.3
    ).clip(0, 100)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


def preprocess_pipeline(df: pd.DataFrame, scaler=None, fit_scaler=True):
    """Full preprocessing pipeline. Returns (X_scaled, y_dropout, y_placement, scaler, feature_cols)."""
    df = clean(df)
    df = encode(df)
    df = engineer_features(df)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].copy()

    y_dropout = df["dropout_risk"] if "dropout_risk" in df.columns else None
    y_placement = df["placement"] if "placement" in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feat_cols)
    X_scaled = X_scaled.fillna(0.0)  # safety: fill any residual NaNs
    return X_scaled, y_dropout, y_placement, scaler, feat_cols


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def save_scaler(scaler, path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path)
