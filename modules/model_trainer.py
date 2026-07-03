"""
modules/model_trainer.py
Trains four base classifiers + Hard Voting + Soft Voting ensembles.
Compares all six models; auto-selects best by F1.
"""

import os, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier


BASE_MODEL_NAMES   = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
VOTING_MODEL_NAMES = ["Hard Voting", "Soft Voting"]
ALL_MODEL_NAMES    = BASE_MODEL_NAMES + VOTING_MODEL_NAMES


def build_base_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                              random_state=42, eval_metric="logloss", verbosity=0),
    }


def build_voting(estimators, voting="soft"):
    return VotingClassifier(estimators=estimators, voting=voting)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    # probability for confidence display
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = y_pred.astype(float)
    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_all(X_train, y_train, X_test, y_test):
    """
    Train 4 base models + Hard Voting + Soft Voting.
    Returns (trained_models_dict, results_dict, best_model_name).
    """
    base    = build_base_models()
    trained = {}
    results = {}

    # ── Base models ──────────────────────────────────────────────────────────
    for name, model in base.items():
        model.fit(X_train, y_train)
        trained[name] = model
        results[name] = evaluate(model, X_test, y_test)

    # ── Voting ensembles ─────────────────────────────────────────────────────
    estimators = [(n, m) for n, m in trained.items()]

    hard_voter = build_voting(estimators, voting="hard")
    hard_voter.fit(X_train, y_train)
    trained["Hard Voting"] = hard_voter
    results["Hard Voting"] = evaluate(hard_voter, X_test, y_test)

    soft_voter = build_voting(estimators, voting="soft")
    soft_voter.fit(X_train, y_train)
    trained["Soft Voting"] = soft_voter
    results["Soft Voting"] = evaluate(soft_voter, X_test, y_test)

    best_name = max(results, key=lambda k: results[k]["f1"])
    return trained, results, best_name


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Works for tree-based models and LogReg; returns 0 for voting."""
    actual = model
    # Unwrap voting classifier to its best base estimator
    if isinstance(model, VotingClassifier):
        # Use the RF inside voting
        for name, est in model.estimators_:
            if "Random Forest" in name or "forest" in name.lower():
                actual = est
                break
        else:
            actual = model.estimators_[0][1]

    if hasattr(actual, "feature_importances_"):
        imp = actual.feature_importances_
    elif hasattr(actual, "coef_"):
        imp = np.abs(actual.coef_[0])
    else:
        imp = np.ones(len(feature_cols))

    df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df.sort_values("importance", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
